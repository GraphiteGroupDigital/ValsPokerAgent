#!/bin/bash
# Resume new_model_eval games in parallel using tmux

set -euo pipefail

WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

BATCH_ARG="${1:-configs/new_model_eval/batch_1}"
TOTAL_HANDS="${2:-100}"

if [[ "$BATCH_ARG" = /* ]]; then
    BATCH_DIR="$BATCH_ARG"
else
    BATCH_DIR="${WORKSPACE}/${BATCH_ARG}"
fi

if [ ! -d "$BATCH_DIR" ]; then
    echo "Error: Batch directory not found: $BATCH_DIR"
    exit 1
fi

SESSION_NAME="resume_$(basename "$BATCH_DIR")"

echo "Resuming new_model_eval games from configs in:"
echo "   $BATCH_DIR"
echo "Will play until ${TOTAL_HANDS} total hands per game"
echo ""

CONFIGS=()
while IFS= read -r cfg; do
    CONFIGS+=("$cfg")
done < <(find "$BATCH_DIR" -maxdepth 1 -type f -name "*.yaml" | sort)

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "Error: No .yaml configs found in $BATCH_DIR"
    exit 1
fi

LOGS=()
for cfg in "${CONFIGS[@]}"; do
    LOG_PATH=$(grep -E "^[[:space:]]*output_file:" "$cfg" | awk '{print $2}')
    if [ -z "${LOG_PATH:-}" ]; then
        echo "Warning: No output_file found in $(basename "$cfg"), skipping"
        continue
    fi

    if [[ "$LOG_PATH" != /* ]]; then
        LOG_PATH="${WORKSPACE}/${LOG_PATH}"
    fi

    if [ ! -f "$LOG_PATH" ]; then
        echo "Warning: Log file not found for $(basename "$cfg"): $LOG_PATH (skipping)"
        continue
    fi

    LOGS+=("$LOG_PATH")
done

if [ "${#LOGS[@]}" -eq 0 ]; then
    echo "Error: No valid log files found for configs in $BATCH_DIR"
    exit 1
fi

echo "Found ${#LOGS[@]} log(s) to resume:"
for log in "${LOGS[@]}"; do
    echo "  - $(basename "$log")"
done
echo ""

FIRST_LOG="${LOGS[0]}"
FIRST_NAME=$(basename "$FIRST_LOG" .json)

tmux new-session -d -s "$SESSION_NAME" -n "$FIRST_NAME"
tmux send-keys -t "${SESSION_NAME}:0" \
    "cd ${WORKSPACE} && uv run resume_game.py \"$FIRST_LOG\" --to-total=${TOTAL_HANDS}" C-m

echo "✅ Started resuming: $FIRST_NAME in tmux window 0"

INDEX=1
for LOG_FILE in "${LOGS[@]:1}"; do
    NAME=$(basename "$LOG_FILE" .json)
    tmux new-window -t "$SESSION_NAME" -n "$NAME"
    tmux send-keys -t "${SESSION_NAME}:${INDEX}" \
        "cd ${WORKSPACE} && uv run resume_game.py \"$LOG_FILE\" --to-total=${TOTAL_HANDS}" C-m

    echo "✅ Started resuming: $NAME in tmux window ${INDEX}"
    INDEX=$((INDEX + 1))
done

echo ""
echo "All resume jobs started in tmux session: ${SESSION_NAME}"
echo ""
echo "To monitor progress:"
echo "  tmux attach -t ${SESSION_NAME}    # Attach to session"
echo "  Ctrl+b then 0-9                   # Switch between windows"
echo "  Ctrl+b then d                     # Detach (games keep running)"
echo ""
echo "To check logs (after resuming):"
echo "  ls -lh ${WORKSPACE}/logs/new_model_eval/*.json"
echo ""

 
