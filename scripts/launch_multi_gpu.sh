#!/bin/bash
#
# Launch multi-GPU VAN sweep using task-level parallelism.
# Each GPU runs an independent chunk of the (K, h) grid.
#
# Usage:
#   bash scripts/launch_multi_gpu.sh [N] [BATCH_SIZE] [MAX_STEP] [N_SEEDS]
#
# Defaults: N=32, BATCH_SIZE=4000, MAX_STEP=5000, N_SEEDS=1
#
# Logs are written to results/logs/chunk_*.log

set -euo pipefail

# Configuration
N_GPUS=4
N=${1:-32}
BATCH_SIZE=${2:-4000}
MAX_STEP=${3:-5000}
N_SEEDS=${4:-1}

# Paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
else
    echo "ERROR: venv/bin/activate not found. Run scripts/setup_unicorn.sh first."
    exit 1
fi

# Verify CUDA
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print(f'Found {torch.cuda.device_count()} GPU(s)')"
if [ $? -ne 0 ]; then
    echo "ERROR: CUDA check failed"
    exit 1
fi

# Create log directory
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"

echo "============================================="
echo "Multi-GPU VAN Sweep"
echo "  N=$N, batch_size=$BATCH_SIZE, max_step=$MAX_STEP, n_seeds=$N_SEEDS"
echo "  GPUs: $N_GPUS"
echo "  Logs: $LOG_DIR/chunk_*.log"
echo "============================================="

# Launch one process per GPU
PIDS=()
for GPU in $(seq 0 $((N_GPUS - 1))); do
    LOG_FILE="$LOG_DIR/chunk_${GPU}.log"
    echo "Launching chunk $GPU on GPU $GPU -> $LOG_FILE"
    CUDA_VISIBLE_DEVICES=$GPU python -u experiments/sweep_TH.py \
        --N "$N" \
        --chunk "$GPU" \
        --n_chunks "$N_GPUS" \
        --device cuda \
        --batch_size "$BATCH_SIZE" \
        --max_step "$MAX_STEP" \
        --n_seeds "$N_SEEDS" \
        > "$LOG_FILE" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "All $N_GPUS chunks launched. PIDs: ${PIDS[*]}"
echo "Waiting for completion..."

# Wait for all processes and track failures
FAILED=0
for i in $(seq 0 $((N_GPUS - 1))); do
    if wait ${PIDS[$i]}; then
        echo "  Chunk $i completed successfully"
    else
        echo "  ERROR: Chunk $i failed (exit code $?). See $LOG_DIR/chunk_${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

if [ $FAILED -gt 0 ]; then
    echo ""
    echo "ERROR: $FAILED/$N_GPUS chunks failed. Check logs in $LOG_DIR/"
    exit 1
fi

echo ""
echo "All chunks completed. Merging results..."
python experiments/merge_chunks.py --N "$N"

echo ""
echo "Done! Results saved to results/sweep_Kh_N${N}.npz"
