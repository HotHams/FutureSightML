#!/bin/bash
# Retrain all formats with fixed splits + team-only AUC evaluation
# Run from project root: bash scripts/retrain_all.sh

set -e
cd "$(dirname "$0")/.."

FORMATS=(
    gen9ou
    gen9uu
    gen9ru
    gen9nu
    gen9ubers
    gen9vgc2026regf
    gen9doublesou
    gen1ou
    gen2ou
    gen3ou
    gen3ubers
)

LOG_DIR="data/logs"
mkdir -p "$LOG_DIR"

echo "=============================================="
echo "RETRAINING ALL FORMATS ($(date))"
echo "=============================================="

for fmt in "${FORMATS[@]}"; do
    echo ""
    echo "=============================================="
    echo "Training $fmt ($(date))"
    echo "=============================================="
    python scripts/train_model.py \
        --format "$fmt" \
        --model both \
        --limit 50000 \
        2>&1 | tee "$LOG_DIR/retrain_${fmt}.log"
    echo "$fmt DONE at $(date)"
done

echo ""
echo "=============================================="
echo "ALL TRAINING COMPLETE ($(date))"
echo "=============================================="

# Summary: extract team-only AUC lines from each log
echo ""
echo "=============================================="
echo "RESULTS SUMMARY"
echo "=============================================="
for fmt in "${FORMATS[@]}"; do
    echo "--- $fmt ---"
    grep -E "TEAM-ONLY|TEST results" "$LOG_DIR/retrain_${fmt}.log" 2>/dev/null || echo "  (no results found)"
done
