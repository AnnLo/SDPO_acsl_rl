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
#SBATCH --output=/leonardo_work/AIFAC_F02_1251/anna/SDPO_acsl_rl/slurm_logs/%x-%j.out
# Usage: sbatch run_acsl.sh acsl|acsl_grpo [hydra overrides]
#        TASK=datasets/acsl_code sbatch run_acsl.sh acsl custom_reward_function.reward_kwargs.verify=False
# Evaluates the base model before training and the last checkpoint after, on the test split of $TASK and on every
# set in EVAL_SETS (default: McEval C). Base results are computed once per model and evaluation set. EVAL=0 skips all.
set -euo pipefail

BASE=/leonardo_work/AIFAC_F02_1251/anna
CONFIG=${1:?config name}; shift
export EXPERIMENT=${EXPERIMENT:-"${CONFIG}_$(date +%Y%m%d)_${SLURM_JOB_ID:-local}"}
export TASK=${TASK:-datasets/acsl} USER=${USER:-$(whoami)}
MODEL=${MODEL:-$BASE/models/Qwen2.5-Coder-3B-Instruct}
EVAL=${EVAL:-1}
EVAL_SETS=${EVAL_SETS:-"datasets/mceval_c"}   # extra evaluation sets (folders with test.parquet), space-separated
EVAL_N=${EVAL_N:-16}                          # samples per problem in the evaluations

cd "$BASE/SDPO_acsl_rl"
mkdir -p "output_logs/$EXPERIMENT"
exec > >(tee -a "output_logs/$EXPERIMENT/training.log") 2>&1
trap 'echo "FAILED (exit $?) at line $LINENO: $BASH_COMMAND"' ERR

source "$BASE/sdpo_verl_env/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HOME="$BASE/cache/hf"
# Keep every compile cache out of the small $HOME quota
export XDG_CACHE_HOME="$BASE/cache" TRITON_CACHE_DIR="$BASE/cache/triton" VLLM_CACHE_ROOT="$BASE/cache/vllm"
export TORCHINDUCTOR_CACHE_DIR="$BASE/cache/inductor" FLASHINFER_WORKSPACE_BASE="$BASE/cache" VLLM_USE_FLASHINFER_SAMPLER=0
# Compute nodes have no internet: wandb logs offline into $WANDB_DIR; sync it live from a login node with
#   scripts/wandb_live_sync.sh output_logs/<experiment>/wandb
export WANDB_MODE=${WANDB_MODE:-offline} WANDB_DIR="$PWD/output_logs/$EXPERIMENT"
export PYTHONUNBUFFERED=1 VLLM_USE_V1=1 PYTHONPATH="$PWD:${PYTHONPATH:-}"
export RAY_TMPDIR="${TMPDIR:-/tmp}/ray_$SLURM_JOB_ID"
export LD_LIBRARY_PATH="$(ls -d "$BASE"/cuda-compat/usr/local/cuda-*/compat | tail -1):${LD_LIBRARY_PATH:-}"
set +u; source "$BASE/env.sh"; set -u
module load gcc/12.2.0
ulimit -c 0
unset ROCR_VISIBLE_DEVICES   # set by Slurm; verl refuses it next to CUDA_VISIBLE_DEVICES
EVAL_ENV="env VLLM_WORKER_MULTIPROC_METHOD=spawn"   # vLLM 0.11 forks its engine, which fails once CUDA is initialised

RESULTS=output_logs/eval
mkdir -p "$RESULTS"
SETS="$TASK $EVAL_SETS"                    # the training task's test split first, then the extra sets
for SET in $SETS; do
  [ -f "$SET/test.parquet" ] || { echo "Missing $SET/test.parquet"; exit 1; }
done
echo "task: $TASK   eval sets: $SETS   experiment: $EXPERIMENT"

# Base model, once per model and evaluation set (the set's name is in the file name, so different datasets or
# prompt modes get their own base result)
if [ "$EVAL" = 1 ]; then
  for SET in $SETS; do
    BASE_RESULT="$RESULTS/base_$(basename "$MODEL")_$(basename "$SET").json"
    [ -f "$BASE_RESULT" ] || $EVAL_ENV python scripts/eval_acsl.py run --model "$MODEL" --data "$SET/test.parquet" \
      --n "$EVAL_N" --out "$BASE_RESULT"
  done
fi

python -m verl.trainer.main_ppo --config-name "$CONFIG" \
  ray_kwargs.ray_init.num_cpus=$SLURM_CPUS_PER_TASK actor_rollout_ref.model.path="$MODEL" "$@"

CKPT="checkpoints/$EXPERIMENT"
if [ "$EVAL" = 1 ] && [ -f "$CKPT/latest_checkpointed_iteration.txt" ]; then
  STEP=$(cat "$CKPT/latest_checkpointed_iteration.txt")
  HF="$CKPT/global_step_$STEP/hf"
  [ -d "$HF" ] || python -m verl.model_merger merge --backend fsdp \
    --local_dir "$CKPT/global_step_$STEP/actor" --target_dir "$HF"
  for SET in $SETS; do
    NAME=$(basename "$SET")
    BASE_RESULT="$RESULTS/base_$(basename "$MODEL")_$NAME.json"
    TRAINED="$RESULTS/${EXPERIMENT}_step${STEP}_$NAME.json"
    $EVAL_ENV python scripts/eval_acsl.py run --model "$HF" --tokenizer "$MODEL" --data "$SET/test.parquet" \
      --n "$EVAL_N" --out "$TRAINED"
    echo "=== $NAME: base vs. step $STEP"
    python scripts/eval_acsl.py compare "$BASE_RESULT" "$TRAINED" --plot "${TRAINED%.json}.png"
  done
fi
echo "Done"