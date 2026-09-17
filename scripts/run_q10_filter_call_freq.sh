#!/usr/bin/env bash
# Filter PyraMeth per-read calls by Q10 read IDs, run call_freq with and without
# the site-level aggregate model, and explain why the unfiltered .all.bed is large.
#
# Intended to run on gpu1, where the data live under /lhd_root and /home/lhd_root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

OUTDIR="${OUTDIR:-/lhd_root/xyf/pyrameth/q10_freq}"
PLATFORM="${PLATFORM:-4khz}"
PROB_CF="${PROB_CF:-0.0}"
COV_CF="${COV_CF:-4}"
BIN_SIZE="${BIN_SIZE:-20}"
PYTHON="${PYTHON:-python3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-pyrameth}"

activate_conda_env() {
    if command -v pyrameth >/dev/null 2>&1; then
        return 0
    fi
    local conda_sh="" candidate
    for candidate in \
        "${HOME}/miniconda3/etc/profile.d/conda.sh" \
        "${HOME}/anaconda3/etc/profile.d/conda.sh" \
        "/home/lhd_root/xyf/miniconda3/etc/profile.d/conda.sh" \
        "/home/lhd_root/xyf/anaconda3/etc/profile.d/conda.sh" \
        "/home/lhd_root/miniconda3/etc/profile.d/conda.sh" \
        "/opt/conda/etc/profile.d/conda.sh"
    do
        if [[ -f "${candidate}" ]]; then
            conda_sh="${candidate}"
            break
        fi
    done
    if [[ -z "${conda_sh}" ]]; then
        return 0
    fi
    # shellcheck source=/dev/null
    source "${conda_sh}"
    if conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
        conda activate "${CONDA_ENV_NAME}"
        PYTHON="${CONDA_PREFIX}/bin/python"
        echo "[job] activated conda env ${CONDA_ENV_NAME} (${PYTHON})"
    fi
}

activate_conda_env

SITE_MODEL="${SITE_MODEL:-${ROOT}/pyrameth/model/r1041_${PLATFORM}_5mC_site.ckpt}"

CALLS_HBA20="${CALLS_HBA20:-/lhd_root/xyf/pyrameth/HBA20-2.dorado_0.7.1.pyrameth.all.read_calls.tsv.gz}"
CALLS_HBC1="${CALLS_HBC1:-/lhd_root/xyf/pyrameth/HBC1-2.dorado_0.7.1.pyrameth.all.read_calls.tsv.gz}"
IDS_HBA20="${IDS_HBA20:-/home/lhd_root/xyf/workspace/methylation/HBA20-2.dorado_0.7.1.sup.Q10.read_ids.txt.gz}"
IDS_HBC1="${IDS_HBC1:-/home/lhd_root/xyf/workspace/methylation/HBC1-2.dorado_0.7.1.sup.Q10.read_ids.txt.gz}"
ALL_BED="${ALL_BED:-/lhd_root/xyf/pyrameth/HBA20-2.dorado_0.7.1.pyrameth.all.bed}"
REF_BED="${REF_BED:-/lhd_root/xyf/workspace/methylation/wgbs/HBA20-2.dorado_0.7.1.sup.per_read.Q10.ds.CG.freq.bed}"

mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/run.log"
exec > >(tee -a "${LOG}") 2>&1

run_filter() {
    if command -v pyrameth >/dev/null 2>&1; then
        pyrameth filter_reads "$@"
    else
        "${PYTHON}" -m pyrameth.filter_reads "$@"
    fi
}

run_bed_stats() {
    if command -v pyrameth >/dev/null 2>&1; then
        pyrameth bed_stats "$@"
    else
        "${PYTHON}" -m pyrameth.bed_stats "$@"
    fi
}

run_call_freq() {
    if command -v pyrameth >/dev/null 2>&1; then
        pyrameth call_freq "$@"
    else
        "${PYTHON}" -m pyrameth.call_mods_freq "$@"
    fi
}

echo "============================================================"
echo "[job] start $(date -Is)"
echo "[job] host $(hostname)"
echo "[job] ROOT=${ROOT}"
echo "[job] OUTDIR=${OUTDIR}"
echo "[job] PLATFORM=${PLATFORM}"
echo "[job] SITE_MODEL=${SITE_MODEL}"
echo "[job] PYTHON=${PYTHON} ($("${PYTHON}" -V 2>&1 || true))"
echo "============================================================"
df -h "${OUTDIR}" "${CALLS_HBA20%/*}" 2>/dev/null || df -h
command -v nvidia-smi >/dev/null && nvidia-smi || true
free -h || true

need_files=(
    "${CALLS_HBA20}" "${CALLS_HBC1}"
    "${IDS_HBA20}" "${IDS_HBC1}"
    "${SITE_MODEL}"
)
for path in "${need_files[@]}"; do
    if [[ ! -f "${path}" ]]; then
        echo "[job] missing required file: ${path}" >&2
        exit 1
    fi
    echo "[job] found $(du -h "${path}" | awk '{print $1, $2}')"
done

if [[ ! -f "${ALL_BED}" ]]; then
    echo "[job] WARNING: unfiltered bed not found, skip size diagnosis: ${ALL_BED}"
    ALL_BED=""
fi
if [[ ! -f "${REF_BED}" ]]; then
    echo "[job] WARNING: comparison bed not found, skip size diagnosis: ${REF_BED}"
    REF_BED=""
fi

if [[ -n "${ALL_BED}" && -n "${REF_BED}" ]]; then
    echo "[job] 0/3 diagnosing why ${ALL_BED} is larger than ${REF_BED}"
    run_bed_stats \
        -i "${ALL_BED}" \
        -i "${REF_BED}" \
        -o "${OUTDIR}/HBA20-2.bed_size_compare.txt"
else
    echo "[job] skip bed size comparison (one or both beds missing)"
fi

filter_one() {
    local sample="$1" calls="$2" ids="$3"
    local out="${OUTDIR}/${sample}.dorado_0.7.1.pyrameth.Q10.read_calls.tsv.gz"
    echo "[job] filter ${sample}"
    run_filter -i "${calls}" --read_ids "${ids}" -o "${out}"
}

echo "[job] 1/3 filtering per-read calls by Q10 read IDs"
filter_one "HBA20-2" "${CALLS_HBA20}" "${IDS_HBA20}" &
pid_hba20=$!
filter_one "HBC1-2" "${CALLS_HBC1}" "${IDS_HBC1}" &
pid_hbc1=$!
set +e
wait "${pid_hba20}"
status_hba20=$?
wait "${pid_hbc1}"
status_hbc1=$?
set -e
if [[ "${status_hba20}" -ne 0 || "${status_hbc1}" -ne 0 ]]; then
    echo "[job] filter failed: HBA20-2=${status_hba20} HBC1-2=${status_hbc1}" >&2
    exit 1
fi

call_freq_one() {
    local sample="$1"
    local calls="${OUTDIR}/${sample}.dorado_0.7.1.pyrameth.Q10.read_calls.tsv.gz"
    local count_bed="${OUTDIR}/${sample}.dorado_0.7.1.pyrameth.Q10.count.freq.bed"
    local aggre_bed="${OUTDIR}/${sample}.dorado_0.7.1.pyrameth.Q10.aggre.freq.bed"

    echo "[job] call_freq count-mode ${sample}"
    run_call_freq \
        -i "${calls}" \
        -o "${count_bed}" \
        --bed --sort \
        --prob_cf "${PROB_CF}"

    echo "[job] call_freq aggregate-mode ${sample}"
    run_call_freq \
        -i "${calls}" \
        -o "${aggre_bed}" \
        --aggre_model "${SITE_MODEL}" \
        --cov_cf "${COV_CF}" \
        --bin_size "${BIN_SIZE}" \
        --sort \
        --prob_cf "${PROB_CF}"
}

echo "[job] 2/3 call_freq with and without aggregate model"
call_freq_one "HBA20-2"
call_freq_one "HBC1-2"

echo "[job] 3/3 comparing Q10 frequency beds against the reference"
if [[ -n "${REF_BED}" ]]; then
    run_bed_stats \
        -i "${OUTDIR}/HBA20-2.dorado_0.7.1.pyrameth.Q10.count.freq.bed" \
        -i "${OUTDIR}/HBA20-2.dorado_0.7.1.pyrameth.Q10.aggre.freq.bed" \
        -i "${REF_BED}" \
        -o "${OUTDIR}/HBA20-2.Q10.freq_vs_ref.txt"
fi
if [[ -n "${ALL_BED}" ]]; then
    run_bed_stats \
        -i "${ALL_BED}" \
        -i "${OUTDIR}/HBA20-2.dorado_0.7.1.pyrameth.Q10.count.freq.bed" \
        -o "${OUTDIR}/HBA20-2.all_vs_Q10.txt"
fi

echo "[job] outputs:"
ls -lh "${OUTDIR}"
echo "[job] done $(date -Is)"
