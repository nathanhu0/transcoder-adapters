#!/bin/bash

LOG_DIR="logs"

# 1. Find the highest number among the .out files
# We look for files matching [0-9]*.out, strip the extension, sort numerically, and grab the top one.
LATEST_NUM=$(ls "$LOG_DIR"/[0-9]*.out 2>/dev/null | sed "s|$LOG_DIR/||;s/.out//" | sort -n | tail -1)

# 2. Check if a log was actually found
if [ -z "$LATEST_NUM" ]; then
    echo "Error: No logs found in $LOG_DIR/"
    exit 1
fi

OUT_FILE="$LOG_DIR/$LATEST_NUM.out"
ERR_FILE="$LOG_DIR/$LATEST_NUM.err"

echo "Watching latest logs: $LATEST_NUM (.out and .err)"

# 3. Launch tmux
# Create a new session named 'log_watcher', run the first tail, split it, and run the second tail.
# tmux new-session -d -s "log_watcher_$LATEST_NUM" "tail -f $OUT_FILE"
# tmux split-window -h "tail -f $ERR_FILE"
# tmux select-layout even-horizontal
# tmux attach-session -t "log_watcher_$LATEST_NUM"
tail -f "$OUT_FILE" "$ERR_FILE"
