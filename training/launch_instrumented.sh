#!/bin/bash
# Instrumented relaunch wrapper: resumes training from the latest checkpoint,
# captures the *real* exit status/signal (the old `python3 | tee` pipeline
# masked this behind tee's exit code), and polls process/GPU/memory state
# every 10s to a sidecar log so a silent death leaves forensic data instead
# of nothing.
#
# Usage: launch_instrumented.sh <seed> <out_dir>
set -uo pipefail

SEED="$1"
OUT_DIR="$2"
cd /home/abshe/MyCodes/alchemy/training || exit 1

LOG="${OUT_DIR}.log"
STATUS_FILE="${OUT_DIR}.exitstatus"
RESOURCE_LOG="${OUT_DIR}.resources.log"

# --- resource-polling sidecar ---
poll_resources() {
    while true; do
        ts=$(date '+%Y-%m-%d %H:%M:%S')
        mem=$(free -m | awk '/^Mem:/{print $2,$3,$4,$6,$7}')
        py_pid=$(pgrep -f "train_painn.py --seed ${SEED} " | head -1)
        if [[ -n "$py_pid" ]]; then
            py_stat=$(ps -o pid,stat,%cpu,%mem,rss,etimes -p "$py_pid" --no-headers 2>/dev/null)
        else
            py_stat="NO_PYTHON_PROCESS_FOUND"
        fi
        gpu=$(nvidia-smi --query-gpu=utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits 2>/dev/null)
        echo "[$ts] mem_total_used_free_avail_MiB=($mem) py=($py_stat) gpu_util%,temp,watts=($gpu)" >> "$RESOURCE_LOG"
        sleep 10
    done
}
poll_resources &
POLL_PID=$!

cleanup() {
    kill "$POLL_PID" 2>/dev/null
}
trap cleanup EXIT

# --- main training run, real exit status captured via PIPESTATUS ---
python3 -u train_painn.py \
    --seed "$SEED" \
    --resume \
    --out-dir "$OUT_DIR" \
    --data-dir qm9_data \
    --split-file split.npz \
    2>&1 | tee -a "$LOG"
PY_EXIT="${PIPESTATUS[0]}"

{
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') process exited ==="
    echo "python3 exit code: $PY_EXIT"
    if [[ "$PY_EXIT" -gt 128 ]]; then
        SIG=$((PY_EXIT - 128))
        echo "killed by signal $SIG ($(kill -l "$SIG" 2>/dev/null))"
    fi
} > "$STATUS_FILE"
cat "$STATUS_FILE"

cleanup
