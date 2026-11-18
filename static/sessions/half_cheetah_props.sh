#!/bin/bash
#SBATCH -A grp_hbenamor
#SBATCH -N 1
#SBATCH -c 8
#SBATCH -t 1-00:00:00
#SBATCH -p public
#SBATCH -q public
#SBATCH --mem=8G
#SBATCH --gpus-per-node=a100:1
#SBATCH -o sessions/errors/halfcheetah_props.%j.out
#SBATCH -e sessions/errors/halfcheetah_props.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sbottu@asu.edu
#SBATCH --export=NONE
set -euo pipefail

# --- Activate env via ASU Sol modules + mamba ---
if ! command -v module >/dev/null 2>&1; then
  echo "ERROR: 'module' command is not available on this node."
  exit 1
fi
module load mamba/latest
if ! command -v mamba >/dev/null 2>&1; then
  echo "ERROR: 'mamba' not found after loading module."
  exit 1
fi
set +u
source activate props-factor
set -u

# --- Start Ollama server on the compute node ---
if ! command -v ollama >/dev/null 2>&1; then
  echo "ERROR: 'ollama' not found in PATH on this node."
  exit 1
fi
export OLLAMA_HOST="127.0.0.1:11434"
ollama serve > "sessions/errors/ollama.${SLURM_JOB_ID}.log" 2>&1 &
OLLAMA_PID=$!

cleanup() {
  kill "$OLLAMA_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

for i in {1..60}; do
  if curl -fsS "http://${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    echo "Ollama is up."
    break
  fi
  sleep 1
done

# --- Run your program ---
NTFY_TOPIC="slurmsbatchjobs"

if python main.py --config configs/halfcheetah/halfcheetah_props.yaml; then
  curl -s -d "halfcheetah_props (Job $SLURM_JOB_ID) finished successfully" \
       -H "Title: SLURM Done ✅" \
       -H "Priority: default" \
       -H "Tags: white_check_mark" \
       ntfy.sh/$NTFY_TOPIC
else
  curl -s -d "halfcheetah_props (Job $SLURM_JOB_ID) FAILED" \
       -H "Title: SLURM Failed ❌" \
       -H "Priority: urgent" \
       -H "Tags: x" \
       ntfy.sh/$NTFY_TOPIC
  exit 1
fi