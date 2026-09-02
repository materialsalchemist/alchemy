#!/bin/bash
# Launch a PaiNN training run (resumed from its latest checkpoint) fully
# detached from the controlling terminal.
#
# Root cause of both prior silent deaths: the tmux *pane* hosting each run
# was destroyed at the exact death timestamp (confirmed via `last -F` utmp
# records for the pane's pty, matched to the second against each run's last
# log line) while the tmux *server* itself stayed up the whole time. Closing
# a pane kills everything in its foreground process group, including a
# training process launched with a plain `tmux send-keys`.
#
# `setsid` gives the process its own session with no controlling terminal at
# all, so there is no tty left for a pane/session close to hang up -- it
# survives the pane, the tmux server, and even a logout, unconditionally.
# `nohup` is added as a second, redundant layer (ignores SIGHUP outright).
#
# Usage (run inside tmux, or anywhere -- detachment no longer depends on tmux):
#   ./run_detached.sh <seed> <out_dir>
# Example:
#   ./run_detached.sh 1 run_seed1
#   ./run_detached.sh 2 run_seed2
set -euo pipefail

SEED="$1"
OUT_DIR="$2"
cd "$(dirname "${BASH_SOURCE[0]}")"

setsid nohup bash launch_instrumented.sh "$SEED" "$OUT_DIR" \
    > "${OUT_DIR}.nohup.log" 2>&1 < /dev/null &

PID=$!
echo "launched seed $SEED -> $OUT_DIR, detached session (PID $PID)"
echo "log:            ${OUT_DIR}.log"
echo "resource log:   ${OUT_DIR}.resources.log"
echo "exit status:    ${OUT_DIR}.exitstatus  (written only once the run ends)"
echo
echo "you can now close this pane/terminal/tmux session -- the run will not die with it."
echo "check status any time with: pgrep -af 'train_painn.py --seed $SEED '"
