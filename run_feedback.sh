#!/bin/bash
#SBATCH --job-name=acsl-feedback
#SBATCH --account=AIFAC_F02_1251
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=01:00:00
#SBATCH --output=/leonardo_work/AIFAC_F02_1251/SDPO_acsl_rl/slurm_logs/%x-%j.out
# Does the model improve when re-prompted with the feedback of its failed attempts (as SDPO's teacher is)?
# Usage: sbatch run_feedback.sh                         (base 3B model, validation set)
#        sbatch --export=ALL,MODEL=/path/to/model,SPLIT=val run_feedback.sh
#        sbatch --export=ALL,WITH_ATTEMPT=1 run_feedback.sh    (also show the failed code)
set -euo pipefail

BASE=/leonardo_work/AIFAC_F02_1251
MODEL=${MODEL:-$BASE/models/Qwen2.5-Coder-3B-Instruct}
SPLIT=${SPLIT:-val}
WITH_ATTEMPT=${WITH_ATTEMPT:-0}
NAME=$(basename "$MODEL")_$SPLIT
DATA=datasets/acsl/$SPLIT.parquet
RESULTS=output_logs/eval

cd "$BASE/SDPO_acsl_rl"
trap 'echo "FAILED (exit $?) at line $LINENO: $BASH_COMMAND"' ERR

source "$BASE/sdpo_verl_env/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$BASE/cache/hf"
export XDG_CACHE_HOME="$BASE/cache" TRITON_CACHE_DIR="$BASE/cache/triton" VLLM_CACHE_ROOT="$BASE/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor" FLASHINFER_WORKSPACE_BASE="$BASE/cache" VLLM_USE_FLASHINFER_SAMPLER=0
export PYTHONUNBUFFERED=1 VLLM_USE_V1=1
export LD_LIBRARY_PATH="$(ls -d "$BASE"/cuda-compat/usr/local/cuda-*/compat | tail -1):${LD_LIBRARY_PATH:-}"
set +u; source "$BASE/env.sh"; set -u
module load gcc/12.2.0
ulimit -c 0

# Attempts to re-prompt: the model's own answers without feedback (reused if already there)
[ -f "$RESULTS/base_$NAME.json" ] || python scripts/eval_acsl.py run --model "$MODEL" --data "$DATA" \
  --out "$RESULTS/base_$NAME.json"
if [ "$WITH_ATTEMPT" = 1 ]; then FLAG=--with-attempt; OUT=feedback_attempt_$NAME; else FLAG=; OUT=feedback_$NAME; fi
python scripts/eval_acsl.py feedback --model "$MODEL" --data "$DATA" --attempts "$RESULTS/base_$NAME.json" \
  --out "$RESULTS/$OUT.json" $FLAG
echo "Done"