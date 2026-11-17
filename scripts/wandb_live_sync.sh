#!/bin/bash
# Uploads offline wandb runs to wandb.ai every few minutes, so they can be followed live while the job runs.
# Run on a LOGIN node (compute nodes have no internet), ideally in tmux, one per experiment or for a parent folder:
#   scripts/wandb_live_sync.sh output_logs/<experiment>/wandb          # one run
#   scripts/wandb_live_sync.sh output_logs                             # every run below output_logs
# Stops by itself once the Slurm job of the newest run has ended and a last sync is done (or Ctrl+C).
set -uo pipefail
DIR=${1:?folder with offline wandb runs}
EVERY=${EVERY:-120}   # seconds between syncs
JOB=${JOB:-}          # optional Slurm job id: stop when it is no longer in the queue

while true; do
  for RUN in $(find "$DIR" -maxdepth 4 -type d -name "offline-run-*" 2>/dev/null); do
    wandb sync --include-offline "$RUN" >/dev/null 2>&1 && echo "$(date +%H:%M) synced $RUN" \
      || echo "$(date +%H:%M) sync failed for $RUN (run 'wandb login' once on this node if it keeps failing)"
  done
  if [ -n "$JOB" ] && ! squeue -h -j "$JOB" 2>/dev/null | grep -q .; then
    echo "job $JOB has ended; last sync done"; break
  fi
  sleep "$EVERY"
done