#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import torch
#import pytorch_lightning as pl
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F

import torch.utils
import torch.utils.checkpoint

from .utils.constants_torch import use_cuda
import numpy as np

from einops import repeat, reduce, rearrange
from .utils.sampling import DownsampleLayer
from .utils.utils_mtm import *

# model ===============================================
class TokenMixingLayer(nn.Module):

    def __init__(self, d_model=64, r_hid=4, drop=0.2, norm_first=False, temporal_depth=2):
        super().__init__()
                
        self.temporal = nn.ModuleList([
            TemporalAttn(d_model, drop, norm_first)
            for _ in range(temporal_depth)
        ])
        
        self.mlp3 = MLP(d_model, r_hid, drop, norm_first)
        


    def forward(self, x, x_mask, cls_tok, pos, pe, idx_c, pe_type='rel'):
        x = torch.concat([cls_tok, x], dim=1)
        x_mask = F.pad(x_mask, (0, 0, 1, 0), 'constant', False)
        p_mask = pos < 0
        pos = F.pad(pos + 1, (0, 0, 1, 0), 'constant', 0)
        p_mask = F.pad(p_mask, (0, 0, 1, 0), 'constant', False)
        
        
        imp = None
        for layer in self.temporal:
            x, imp = layer(x, x_mask, pos, pe, pe_type)                  
                
        x = self.mlp3(x)
        return x[:, 1:, :, :], x[:, [0], :, :], imp


class modelMTM(nn.Module):

    def __init__(self, num_chn, d_static, num_cls, ratios, d_model=96, r_hid=4, drop=0.2, norm_first=True, down_mode='concat', 
                 vocab_size=16, embedding_size=4, temporal_depth=2, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.d_static = d_static
        self.ratios = ratios
        self.register_buffer('rpe', precompute_rpe(d_model))
        self.register_buffer('ape', precompute_ape(d_model))
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        
        self.chn_emb = nn.Embedding(num_chn, d_model)
        nn.init.xavier_uniform_(self.chn_emb.weight)
        self.cls_tok = nn.Parameter(torch.rand(num_chn, d_model))
        self.register_buffer('_c_arange', torch.arange(num_chn), persistent=False)

        # 传入 temporal_depth
        self.inp_layer = TokenMixingLayer(d_model, r_hid, drop, norm_first, temporal_depth)
        self.mixers = nn.ModuleList()
        self.samplers = nn.ModuleList()
        for r in ratios:
            self.mixers.append(TokenMixingLayer(d_model, r_hid, drop, norm_first, temporal_depth))
            self.samplers.append(DownsampleLayer(d_model, r, down_mode))

        self.cls_head = CLSHead(d_model, d_static, num_cls, drop)

  
    def forward(self, signal, kmer, x_mask, t, x_static):
        kmer_embed = self.embedding(kmer)
        x = torch.cat([signal, kmer_embed], dim=-1)

        bsz, nt, nc = x_mask.shape
        nt = nt + 1
        dev = x.device
        idx_t = repeat(t, "b t -> b t c", c=nc)
        idx_b = repeat(torch.arange(bsz, device=dev), "b -> b t c", t=nt, c=nc)
        idx_c = repeat(self._c_arange, "c -> b t c", b=bsz, t=nt)
        c_feat = self.chn_emb(self._c_arange)

        x = apply_abs_pe(x.nan_to_num(0)[..., None] * c_feat, idx_t, self.ape)
        cls_tok = repeat(self.cls_tok, "c d -> b 1 c d", b=bsz)

        x, cls_tok, imp = self.inp_layer(x, x_mask, cls_tok, idx_t, self.rpe, idx_c)

        for sampler, mixer in zip(self.samplers, self.mixers):
            x, x_mask, idx_t = sampler(x, x_mask, idx_b, idx_t, idx_c, imp)
            x, cls_tok, imp = mixer(x, x_mask, cls_tok, idx_t, self.rpe, idx_c)

        outputs = [reduce(cls_tok, "b 1 c d -> b d", 'max')]
        if self.d_static > 0:
            outputs.append(x_static)
        outputs = self.cls_head(torch.cat(outputs, -1))
        return outputs


########################################
class AggrAttRNN(nn.Module):
    def __init__(self, seq_len=11, num_layers=1, num_classes=1,
                 dropout_rate=0.5, hidden_size=32, binsize=20,
                 model_type="attbigru",
                 device=0):
        super(AggrAttRNN, self).__init__()
        self.model_type = model_type
        self.device = device

        self.seq_len = seq_len
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.hidden_size = hidden_size

        self.feas_ccs = binsize + 1
        if self.model_type == "attbilstm":
            self.rnn_cell = "lstm"
            self.rnn = nn.LSTM(self.feas_ccs, self.hidden_size, self.num_layers,
                               dropout=0, batch_first=True, bidirectional=True)
        elif self.model_type == "attbigru":
            self.rnn_cell = "gru"
            self.rnn = nn.GRU(self.feas_ccs, self.hidden_size, self.num_layers,
                              dropout=0, batch_first=True, bidirectional=True)
        else:
            raise ValueError("--model_type not set right!")

        self._att3 = Attention(self.hidden_size * 2, self.hidden_size * 2, self.hidden_size)

        self.dropout1 = nn.Dropout(p=dropout_rate)
        self.fc1 = nn.Linear(self.hidden_size * 2, self.num_classes)  # 2 for bidirection

        # self.softmax = nn.Softmax(1)

    def get_model_type(self):
        return self.model_type

    def init_hidden(self, batch_size, num_layers, hidden_size):
        # Set initial states
        h0 = torch.randn(num_layers * 2, batch_size, hidden_size, requires_grad=True)
        if use_cuda and self.device != "cpu":
            h0 = h0.cuda(self.device)
        if self.rnn_cell == "lstm":
            c0 = torch.randn(num_layers * 2, batch_size, hidden_size, requires_grad=True)
            if use_cuda and self.device != "cpu":
                c0 = c0.cuda(self.device)
            return h0, c0
        return h0

    def forward(self, offsets, histos):

        offsets = torch.reshape(offsets, (-1, self.seq_len, 1)).float()  # (N, L, 1)

        out = torch.cat((histos.float(), offsets), 2)

        out, n_states = self.rnn(out, self.init_hidden(out.size(0),
                                                       self.num_layers,
                                                       self.hidden_size))  # (N, L, nhid*2)
        # attention_net3 ======
        # h_n: (num_layer * 2, N, nhid), h_0, c_0 -> h_n, c_n not affected by batch_first
        # h_n (last layer) = out[:, -1, :self.hidden_size] concats out1[:, 0, self.hidden_size:]
        h_n = n_states[0] if self.rnn_cell == "lstm" else n_states
        h_n = h_n.reshape(self.num_layers, 2, -1, self.hidden_size)[-1]  # last layer (2, N, nhid)
        h_n = h_n.transpose(0, 1).reshape(-1, 1, 2 * self.hidden_size)
        out, att_weights = self._att3(h_n, out)

        out = self.dropout1(out)
        out = self.fc1(out)
        # out = self.softmax(out)

        return out


def mask_3d(inputs, seq_len, mask_value=0.):
    batches = inputs.size()[0]
    assert batches == len(seq_len)
    max_idx = max(seq_len)
    for n, idx in enumerate(seq_len):
        if idx < max_idx.item():
            if len(inputs.size()) == 3:
                inputs[n, idx.int():, :] = mask_value
            else:
                assert len(inputs.size()) == 2, "The size of inputs must be 2 or 3, received {}".format(inputs.size())
                inputs[n, idx.int():] = mask_value
    return inputs


# bahdanau attention
class Attention(nn.Module):
    """
    Inputs:
        last_hidden: (batch_size, hidden_size)  # query, (2, N, C) -> (N, 1, C*2)
        encoder_outputs: (batch_size, max_time, hidden_size)  # key, (N, L, C*2)
    Returns:
        context_vector: (N, 2*C)
        attention_weights: (batch_size, max_time)
    """
    def __init__(self, query_size, key_size, hidden_size=128):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.Wa = nn.Linear(query_size, hidden_size, bias=False)
        self.Ua = nn.Linear(key_size, hidden_size, bias=False)
        self.va = nn.Linear(hidden_size, 1, bias=False)
        self.attw_softmax = nn.Softmax(1)

    def forward(self, last_hidden, encoder_outputs):

        attention_energies = self._score(last_hidden, encoder_outputs).squeeze(2)  # (N, L, 1) -> (N, L)

        # if seq_len is not None:
        #     attention_energies = mask_3d(attention_energies, seq_len, -float('inf'))

        attention_weights = self.attw_softmax(attention_energies).unsqueeze(2)  # (N, L) -> (N, L, 1)

        values = torch.transpose(encoder_outputs, 1, 2)  # (N, 2*C, L)
        context_vector = torch.matmul(values, attention_weights).squeeze(2)  # (N, 2*C, 1) -> (N, 2*C)

        return context_vector, attention_weights.squeeze(2)

    def _score(self, last_hidden, encoder_outputs):
        """
        Computes an attention score
        :param last_hidden: (batch_size, hidden_dim)  # (2, N, C) -> (N, 1, C*2)
        :param encoder_outputs: (batch_size, max_time, hidden_dim)  # (N, L, C*2)
        :return: a score (batch_size, max_time)
        """
        out = torch.tanh(self.Wa(last_hidden) + self.Ua(encoder_outputs))  # (N, L, nhid)
        return self.va(out)  # (N, L, 1)