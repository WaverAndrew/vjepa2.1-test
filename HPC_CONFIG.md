# HPC Configuration

## Cluster

| Property | Value |
|----------|-------|
| Login node | `lnode01` |
| Account | `3206024` |
| Home dir | `/home/3206024` |
| Repo path | `/home/3206024/vjepa2.1-test` |
| Scratch | `/scratch/HD-EPIC/` |

## Environment setup (run on login node)

```bash
module load cuda/12.1
module load miniconda3
source $(conda info --base)/etc/profile.d/conda.sh
conda activate venv
```

## Interactive GPU session

```bash
srun --account=3206024 \
     --partition=gpunew \
     --gres=gpu:H100:1 \
     --time=2:00:00 \
     --cpus-per-task=8 \
     --mem=64G \
     --pty bash
```

## SLURM defaults (use in all #SBATCH headers)

```
--account=3206024
--partition=gpunew
--gres=gpu:H100:1
--cpus-per-task=8
--mem=64G
```

## Dataset: HD-EPIC

```
/scratch/HD-EPIC/
  Videos/
    P01/   P04/   P05/   P06/   P07/   P08/   P09/
      *.mp4                  (egocentric video recordings)
      *_vrs_to_mp4_log.json  (skip — conversion logs)
      *_mp4_to_vrs_time_ns.csv  (skip — timing metadata)
```

Video naming convention: `P{participant}-{YYYYMMDD}-{HHMMSS}.mp4`

## torch.hub checkpoint cache

Pre-download on login node (needs internet), compute nodes may be firewalled:

```bash
export TORCH_HOME=/scratch/3206024/torch_hub_cache
bash scripts/download_checkpoints.sh /scratch/3206024/torch_hub_cache
```

Then add to all SLURM scripts:
```bash
export TORCH_HOME=/scratch/3206024/torch_hub_cache
```
