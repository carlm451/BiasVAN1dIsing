# Running VAN Sweeps

How to run (K, h) grid sweeps on any hardware.

## Single-task mode (any machine)

```bash
source venv/bin/activate

# CPU (Mac/Linux, no GPU required)
python experiments/sweep_TH.py --N 16 --n_seeds 1

# Single GPU (auto-detected)
python experiments/sweep_TH.py --N 32 --n_seeds 1 --batch_size 4000

# Explicit device selection
python experiments/sweep_TH.py --N 32 --device cuda:0 --batch_size 4000
```

## Multi-GPU mode (task-parallel)

The (K, h) grid is split into interleaved chunks, one per GPU. Each GPU runs an independent process with no inter-GPU communication.

```bash
# 4 GPUs (default launcher)
bash scripts/launch_multi_gpu.sh 32

# Custom: 4 GPUs, 8000 batch size, 10000 steps, 3 seeds
bash scripts/launch_multi_gpu.sh 32 8000 10000 3
```

### Adapting to your hardware

Edit `N_GPUS=4` in `scripts/launch_multi_gpu.sh`, or run chunks manually:

```bash
# Example: 2-GPU machine
CUDA_VISIBLE_DEVICES=0 python -u experiments/sweep_TH.py \
    --N 32 --chunk 0 --n_chunks 2 --device cuda --batch_size 4000 &
CUDA_VISIBLE_DEVICES=1 python -u experiments/sweep_TH.py \
    --N 32 --chunk 1 --n_chunks 2 --device cuda --batch_size 4000 &
wait
python experiments/merge_chunks.py --N 32
```

Mixed CPU+GPU (e.g., 4 GPU chunks + 2 CPU workers):

```bash
# GPU workers
for GPU in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU python -u experiments/sweep_TH.py \
        --N 32 --chunk $GPU --n_chunks 6 --device cuda &
done
# CPU workers for remaining chunks
python -u experiments/sweep_TH.py --N 32 --chunk 4 --n_chunks 6 --device cpu &
python -u experiments/sweep_TH.py --N 32 --chunk 5 --n_chunks 6 --device cpu &
wait
python experiments/merge_chunks.py --N 32
```

## Batch size guidance

| Hardware | Batch size | Notes |
|----------|-----------|-------|
| CPU | 1000 (default) | Limited by memory bandwidth |
| Single GPU (16-20GB) | 4000-8000 | Reduces REINFORCE variance |
| Large GPU (40GB+) | 8000-16000 | |

Larger batches reduce REINFORCE gradient variance, improving convergence speed and result quality.

## Output format

All modes produce the same `results/sweep_Kh_N{N}.npz` format containing:
- `K_grid`, `h_grid`: parameter grids (h is the full symmetric grid)
- `exact_f`, `exact_m`: exact transfer matrix results
- `nmf_f`, `nmf_m`: naive mean field results
- `van_bias_f_mean`, `van_bias_f_std`: VAN with bias (mean/std over seeds)
- `van_nobias_f_mean`, `van_nobias_f_std`: VAN without bias
- `delta_f_*`: free energy errors vs exact

Plotting scripts and inference scripts work identically regardless of how the sweep was run.

## First-time setup on a GPU machine

```bash
bash scripts/setup_unicorn.sh      # install deps, verify GPUs, run tests
python scripts/verify_gpu.py       # quick training check on GPU
bash scripts/launch_multi_gpu.sh 32 # run a real sweep
```
