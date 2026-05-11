#!/usr/bin/env python3
"""
export_torchscript.py
=====================
Convert a deepsignal3 .ckpt checkpoint to TorchScript (.pt) format
so it can be loaded by the Rust ds3-model crate via `tch::CModule`.

Usage
-----
# Export a modelMTM checkpoint (default):
python export_torchscript.py \
    --model_path  ../model/human_r1041_5khz_CG_epoch5.ckpt \
    --output_path ../model/human_r1041_5khz_CG_epoch5.pt  \
    --model_class mtm

# Export a ModelBiLSTM checkpoint:
python export_torchscript.py \
    --model_path  ../model/plant_r1041_5khz_C_epoch4.ckpt  \
    --output_path ../model/plant_r1041_5khz_C_epoch4.pt   \
    --model_class bilstm

Notes
-----
* Run this once per checkpoint; the .pt file can be reused indefinitely.
* The export uses `torch.jit.trace()` with dummy inputs.  If you see
  TracerWarnings about dynamic control flow, the traced model should still
  be correct for all real inputs of the same shape.
* Match --seq_len / --signal_len / model hyper-params exactly to training.
"""

import argparse
import sys
import torch


# ── Add deepsignal3 package to path ───────────────────────────────────────────
# Adjust if the deepsignal3 Python package lives elsewhere.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from deepsignal3.models import AggrAttRNN, ModelBiLSTM, modelMTM  # type: ignore


# ─────────────────────────────────────────────────────────────────────────────
# MTM export
# ─────────────────────────────────────────────────────────────────────────────

def _trace_device(args):
    """
    Select the device used for tracing.

    torch.jit.trace freezes the device of every runtime op (e.g. torch.arange)
    into the saved model.  The traced model must therefore be traced on the
    *same* kind of device as it will run on at inference time:
      - traced on CPU → internal arange tensors frozen as cpu → works only on CPU
      - traced on CUDA → internal arange tensors frozen as cuda:0 → works on GPU

    Rule: trace on CUDA when CUDA is available, unless --force-cpu-export is set.
    """
    if args.force_cpu_export or not torch.cuda.is_available():
        dev = torch.device("cpu")
    else:
        dev = torch.device("cuda")
    print(f"Tracing device: {dev}")
    return dev


def export_mtm(args):
    """Trace and save a modelMTM checkpoint as TorchScript."""
    print(f"Loading modelMTM checkpoint: {args.model_path}")
    model = modelMTM(
        num_chn  = args.mtm_num_base_features + args.n_embed,
        d_static = args.mtm_d_static,
        num_cls  = args.class_num,
        ratios   = args.mtm_ratios,
        d_model  = args.mtm_hid_rnn,
        r_hid    = args.mtm_r_hid,
        drop     = 0.0,   # always 0 for inference
        norm_first    = args.mtm_norm_first,
        down_mode     = args.mtm_down_mode,
        vocab_size    = args.n_vocab,
        embedding_size= args.n_embed,
        temporal_depth= args.mtm_temporal_depth,
    )

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=True)
    # Strip DDP / torch.compile prefixes
    clean = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(clean, strict=True)
    model.eval()

    dev = _trace_device(args)
    model = model.to(dev)

    # ── Build representative dummy inputs on the same device ──────────────
    B   = 2          # batch size (tracing; any B ≥ 1 works)
    L   = args.seq_len
    S   = args.signal_len
    LS  = L * S      # flattened signal time steps

    signals  = torch.randn(B, LS, 1, device=dev)
    kmer_exp = torch.randint(0, args.n_vocab, (B, LS), device=dev)
    x_mask   = torch.zeros(B, LS, 1 + args.n_embed, dtype=torch.bool, device=dev)
    tpos     = torch.arange(LS, device=dev).unsqueeze(0).expand(B, -1)
    x_static = torch.zeros(B, 1, dtype=torch.long, device=dev)

    print("Tracing model …")
    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (signals, kmer_exp, x_mask, tpos, x_static),
            strict=False,
        )

    traced.save(args.output_path)
    print(f"Saved TorchScript model → {args.output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# BiLSTM export
# ─────────────────────────────────────────────────────────────────────────────

def export_bilstm(args):
    """Trace and save a ModelBiLSTM checkpoint as TorchScript."""
    print(f"Loading ModelBiLSTM checkpoint: {args.model_path}")
    model = ModelBiLSTM(
        seq_len        = args.seq_len,
        signal_len     = args.signal_len,
        num_layers1    = args.layernum1,
        num_layers2    = args.layernum2,
        num_classes    = args.class_num,
        dropout_rate   = 0.0,
        hidden_size    = args.hid_rnn,
        vocab_size     = args.n_vocab,
        embedding_size = args.n_embed,
        is_base        = args.is_base,
        is_signallen   = args.is_signallen,
        is_trace       = args.is_trace,
        module         = "both_bilstm",
    )

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=True)
    clean = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(clean, strict=True)
    model.eval()

    dev = _trace_device(args)
    model = model.to(dev)

    # ── Dummy inputs on the same device ──────────────────────────────────
    B = 2
    L = args.seq_len
    S = args.signal_len

    kmers   = torch.randint(0, args.n_vocab, (B, L), device=dev).long()
    means   = torch.randn(B, L, device=dev)
    stds    = torch.rand(B, L, device=dev).abs()
    lens    = torch.randint(1, 20, (B, L), device=dev).float()
    signals = torch.randn(B, L, S, device=dev)

    print("Tracing model …")
    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (kmers, means, stds, lens, signals),
            strict=False,
        )

    traced.save(args.output_path)
    print(f"Saved TorchScript model → {args.output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# AggrAttRNN export
# ─────────────────────────────────────────────────────────────────────────────

class _AggrAttRNNForExport(AggrAttRNN):
    """
    Thin wrapper that replaces the random `init_hidden` with zeros so that
    `torch.jit.trace` captures a static-shape model safe for all batch sizes.

    GRU/LSTM with a zero initial hidden state is equivalent to the default
    PyTorch behaviour when `h_0` is omitted, so inference results are identical.
    """

    def forward(self, offsets, histos):
        B = offsets.size(0)
        offsets = offsets.reshape(B, self.seq_len, 1).float()
        out = torch.cat((histos.float(), offsets), 2)          # (B, L, binsize+1)

        h0 = torch.zeros(self.num_layers * 2, B, self.hidden_size)
        if self.rnn_cell == "lstm":
            c0 = torch.zeros_like(h0)
            out, n_states = self.rnn(out, (h0, c0))
        else:
            out, n_states = self.rnn(out, h0)                  # (B, L, nhid*2)

        # Bahdanau attention using last hidden layer
        h_n = n_states[0] if self.rnn_cell == "lstm" else n_states
        h_n = h_n.reshape(self.num_layers, 2, B, self.hidden_size)[-1]
        h_n = h_n.transpose(0, 1).reshape(B, 1, 2 * self.hidden_size)
        out, _ = self._att3(h_n, out)                          # (B, nhid*2)

        out = self.fc1(out)                                    # (B, 1)
        return out.clamp(0.0, 1.0)


def export_aggr(args):
    """Trace and save an AggrAttRNN checkpoint as TorchScript."""
    print(f"Loading AggrAttRNN checkpoint: {args.model_path}")
    model = _AggrAttRNNForExport(
        seq_len      = args.aggr_seq_len,
        num_layers   = args.aggr_num_layers,
        num_classes  = 1,
        dropout_rate = 0.0,
        hidden_size  = args.aggr_hidden_size,
        binsize      = args.bin_size,
        model_type   = args.aggr_model_type,
        device       = "cpu",
    )

    ckpt = torch.load(args.model_path, map_location="cpu", weights_only=True)
    # Strip DDP / torch.compile prefixes
    clean = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(clean, strict=True)
    model.eval()

    # Dummy inputs matching Rust calling convention:
    #   offsets : (B, seq_len)           f32  – position distances
    #   histos  : (B, seq_len, bin_size) f32  – normalised histograms
    B = 2
    L = args.aggr_seq_len
    dummy_offsets = torch.zeros(B, L)
    dummy_histos  = torch.zeros(B, L, args.bin_size)

    print("Tracing AggrAttRNN …")
    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (dummy_offsets, dummy_histos),
            strict=False,
        )

    traced.save(args.output_path)
    print(f"Saved TorchScript AggrAttRNN → {args.output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Export a deepsignal3 .ckpt to TorchScript (.pt) for Rust inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument("--model_path",  "-m", required=True, help=".ckpt checkpoint path")
    p.add_argument("--output_path", "-o", required=True, help="output .pt file path")
    p.add_argument("--model_class", default="mtm", choices=["mtm", "bilstm", "aggr"],
                   help="model architecture")
    p.add_argument("--force-cpu-export", action="store_true", dest="force_cpu_export",
                   help="trace on CPU even when a GPU is available (produces a CPU-only model)")

    # Shared hyper-params
    p.add_argument("--seq_len",      type=int,   default=21)
    p.add_argument("--signal_len",   type=int,   default=15)
    p.add_argument("--class_num",    type=int,   default=2)
    p.add_argument("--n_vocab",      type=int,   default=16)
    p.add_argument("--n_embed",      type=int,   default=4)

    # BiLSTM-specific
    p.add_argument("--hid_rnn",      type=int,   default=256,
                   help="hidden size for BiLSTM")
    p.add_argument("--layernum1",    type=int,   default=3)
    p.add_argument("--layernum2",    type=int,   default=1)
    p.add_argument("--is_base",      type=bool,  default=True)
    p.add_argument("--is_signallen", type=bool,  default=True)
    p.add_argument("--is_trace",     type=bool,  default=False)

    # MTM-specific
    p.add_argument("--mtm_hid_rnn",           type=int,          default=128,
                   help="d_model (hidden size) for MTM")
    p.add_argument("--mtm_num_base_features", type=int,          default=1)
    p.add_argument("--mtm_d_static",          type=int,          default=1)
    p.add_argument("--mtm_ratios",  nargs="+", type=int,          default=[2, 2, 2, 2])
    p.add_argument("--mtm_r_hid",             type=int,          default=4)
    p.add_argument("--mtm_norm_first",        type=bool,         default=True)
    p.add_argument("--mtm_down_mode",         default="concat",
                   choices=["concat", "avg", "max"])
    p.add_argument("--mtm_temporal_depth",    type=int,          default=2)

    # AggrAttRNN-specific
    p.add_argument("--aggr_seq_len",     type=int, default=11,
                   help="context window length (default 11)")
    p.add_argument("--aggr_num_layers",  type=int, default=1,
                   help="RNN layer count (default 1)")
    p.add_argument("--aggr_hidden_size", type=int, default=32,
                   help="RNN hidden size (default 32)")
    p.add_argument("--bin_size",         type=int, default=20,
                   help="histogram bin count (default 20)")
    p.add_argument("--aggr_model_type",  default="attbigru",
                   choices=["attbigru", "attbilstm"],
                   help="RNN cell type (default attbigru)")

    args = p.parse_args()

    if args.model_class == "mtm":
        export_mtm(args)
    elif args.model_class == "bilstm":
        export_bilstm(args)
    else:
        export_aggr(args)


if __name__ == "__main__":
    main()
