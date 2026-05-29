#!/usr/bin/env bash
#SBATCH --job-name=vw-netdiag
#SBATCH --output=runs/_slurm/netdiag_%j.out
#SBATCH --error=runs/_slurm/netdiag_%j.err
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:05:00

set -u
echo "=== host ==="
hostname
date

echo; echo "=== DNS: huggingface.co ==="
nslookup huggingface.co 2>&1 || echo "(nslookup failed: $?)"

echo; echo "=== DNS: hf-mirror.com ==="
nslookup hf-mirror.com 2>&1 || echo "(nslookup failed: $?)"

echo; echo "=== HEAD huggingface.co ==="
curl -sSI --max-time 10 https://huggingface.co 2>&1 | head -5 || echo "(curl failed: $?)"

echo; echo "=== HEAD hf-mirror.com ==="
curl -sSI --max-time 10 https://hf-mirror.com 2>&1 | head -5 || echo "(curl failed: $?)"

echo; echo "=== sample byte download from HF ==="
curl -sSI --max-time 10 "https://huggingface.co/datasets/Tevatron/wiki-ss-corpus/resolve/main/README.md" 2>&1 | head -10 || echo "(curl readme failed: $?)"

echo; echo "=== DNS: google.com (sanity) ==="
nslookup google.com 2>&1 | head -5 || echo "(failed)"

echo; echo "done"
