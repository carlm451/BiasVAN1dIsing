#!/bin/bash
#
# Launch targeted re-run of failed (K, h) points across multiple GPUs.
#
# Usage:
#   bash scripts/launch_rerun.sh N [MAX_STEP] [BATCH_SIZE] [THRESHOLD]
#
# Examples:
#   bash scripts/launch_rerun.sh 64                    # defaults: 50k steps, 8k batch
#   bash scripts/launch_rerun.sh 128 50000 8000 0.01   # explicit params
#   bash scripts/launch_rerun.sh 64 100000 16000 0.05  # aggressive re-run
#
# First run with --dry-run to see which points will be re-trained:
#   source venv/bin/activate
#   python experiments/rerun_failed.py --N 64 --dry-run

set -euo pipefail

N_GPUS=4
N=${1:?Usage: launch_rerun.sh N [MAX_STEP] [BATCH_SIZE] [THRESHOLD]}
MAX_STEP=${2:-50000}
BATCH_SIZE=${3:-8000}
THRESHOLD=${4:-0.01}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: venv/bin/activate not found."
    exit 1
fi

# Dry run first to show what will be re-trained
echo "============================================="
echo "Targeted Re-run: N=$N"
echo "  max_step=$MAX_STEP, batch_size=$BATCH_SIZE, threshold=$THRESHOLD"
echo "  GPUs: $N_GPUS"
echo "============================================="
echo ""
echo "Points to re-run:"
python experiments/rerun_failed.py --N "$N" --threshold "$THRESHOLD" --dry-run
echo ""

# Launch one process per GPU
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

PIDS=()
for GPU in $(seq 0 $((N_GPUS - 1))); do
    LOG_FILE="$LOG_DIR/rerun_N${N}_chunk${GPU}.log"
    echo "Launching chunk $GPU on GPU $GPU -> $LOG_FILE"
    CUDA_VISIBLE_DEVICES=$GPU python -u experiments/rerun_failed.py \
        --N "$N" \
        --threshold "$THRESHOLD" \
        --max_step "$MAX_STEP" \
        --batch_size "$BATCH_SIZE" \
        --chunk "$GPU" \
        --n_chunks "$N_GPUS" \
        --device cuda \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $N_GPUS chunks launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."

FAILED=0
for i in $(seq 0 $((N_GPUS - 1))); do
    if wait ${PIDS[$i]}; then
        echo "  Chunk $i completed successfully"
    else
        echo "  ERROR: Chunk $i failed. See $LOG_DIR/rerun_N${N}_chunk${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "ERROR: $FAILED/$N_GPUS chunks failed."
    exit 1
fi

echo ""
echo "All chunks completed. Merging patches..."
python experiments/rerun_failed.py --N "$N" --merge-only

echo ""
echo "Done! Updated results/sweep_Kh_N${N}.npz"
echo "(Original backed up to results/sweep_Kh_N${N}_pre_rerun.npz)"
