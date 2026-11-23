#!/usr/bin/env bash
# Regenerate the paper's tables and figures from the exported results (CPU only).
#
#   bash scripts/reproduce_paper.sh                  # uses results/paper -> outputs/paper
#   RESULTS=results/paper OUT=outputs/paper bash scripts/reproduce_paper.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RESULTS="${RESULTS:-results/paper}"
OUT="${OUT:-outputs/paper}"

python scripts/reproduce_layerwise.py --results "$RESULTS" --out "$OUT"
python scripts/reproduce_sgi.py --results "$RESULTS" --out "$OUT"

# Table 7 needs the scaling sweeps (see README); skipped if they were not run.
if [ -f results/scaling_encoders/probe_scores.parquet ] || [ -f results/scaling_mamba/probe_scores.parquet ]; then
    python scripts/reproduce_scaling.py --results results/scaling_encoders results/scaling_mamba --out "$OUT"
fi

echo ""
echo "Done. See $OUT/layerwise/SUMMARY.md and $OUT/sgi/SUMMARY.md"
