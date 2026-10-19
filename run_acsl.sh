#!/bin/bash
#SBATCH --job-name=sdpo-verl
#SBATCH --account=AIFAC_F02_1251
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=480G
#SBATCH --time=06:00:00
#SBATCH --output=/leonardo_work/AIFAC_F02_1251/SDPO_acsl_rl/slurm_logs/%x-%j.out
# Usage: sbatch run_acsl.sh acsl|acsl_grpo [hydra overrides]
# Evaluates the base model before training (once per model) and the last checkpoint after. EVAL=0 skips both.
set -euo pipefail

BASE=/leonardo_work/AIFAC_F02_1251
CONFIG=${1:?config name}; shift
export EXPERIMENT=${EXPERIMENT:-"${CONFIG}_$(date +%Y%m%d)_${SLURM_JOB_ID:-local}"}
export TASK=datasets/acsl USER=${USER:-$(whoami)}
MODEL=${MODEL:-$BASE/models/Qwen2.5-Coder-3B-Instruct}
EVAL=${EVAL:-1}

cd "$BASE/SDPO_acsl_rl"
mkdir -p "output_logs/$EXPERIMENT"
exec > >(tee -a "output_logs/$EXPERIMENT/training.log") 2>&1
trap 'echo "FAILED (exit $?) at line $LINENO: $BASH_COMMAND"' ERR

source "$BASE/sdpo_verl_env/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME="$BASE/cache/hf"
# Keep every compile cache out of the small $HOME quota
export XDG_CACHE_HOME="$BASE/cache" TRITON_CACHE_DIR="$BASE/cache/triton" VLLM_CACHE_ROOT="$BASE/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor" FLASHINFER_WORKSPACE_BASE="$BASE/cache" VLLM_USE_FLASHINFER_SAMPLER=0
export WANDB_MODE=offline PYTHONUNBUFFERED=1 VLLM_USE_V1=1 PYTHONPATH="$PWD:${PYTHONPATH:-}"
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray_$SLURM_JOB_ID"
export LD_LIBRARY_PATH="$(ls -d "$BASE"/cuda-compat/usr/local/cuda-*/compat | tail -1):${LD_LIBRARY_PATH:-}"
set +u; source "$BASE/env.sh"; set -u
module load gcc/12.2.0
ulimit -c 0
unset ROCR_VISIBLE_DEVICES   # set by Slurm; verl refuses it next to CUDA_VISIBLE_DEVICES

RESULTS=output_logs/eval
BASE_RESULT="$RESULTS/base_$(basename "$MODEL").json"
if [ "$EVAL" = 1 ] && [ ! -f "$BASE_RESULT" ]; then
  python scripts/eval_acsl.py run --model "$MODEL" --out "$BASE_RESULT"
fi

python -m verl.trainer.main_ppo --config-name "$CONFIG" \
  ray_kwargs.ray_init.num_cpus=$SLURM_CPUS_PER_TASK actor_rollout_ref.model.path="$MODEL" "$@"

CKPT="checkpoints/$EXPERIMENT"
if [ "$EVAL" = 1 ] && [ -f "$CKPT/latest_checkpointed_iteration.txt" ]; then
  STEP=$(cat "$CKPT/latest_checkpointed_iteration.txt")
  HF="$CKPT/global_step_$STEP/hf"
  [ -d "$HF" ] || python -m verl.model_merger merge --backend fsdp \
    --local_dir "$CKPT/global_step_$STEP/actor" --target_dir "$HF"
  TRAINED="$RESULTS/${EXPERIMENT}_step$STEP.json"
  python scripts/eval_acsl.py run --model "$HF" --tokenizer "$MODEL" --out "$TRAINED"
  python scripts/eval_acsl.py compare "$BASE_RESULT" "$TRAINED" --plot "${TRAINED%.json}.png"
fi
echo "Done"