#!/bin/bash
# run a batch of new-model eval games in parallel using tmux

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage:"
    echo "  $0 BATCH_NUMBER [SESSION_NAME]"
    echo "  $0 /path/to/batch_dir [SESSION_NAME]"
    exit 1
fi

WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

RAW_BATCH_ARG="$1"
SESSION_NAME="${2:-}"

if [[ "$RAW_BATCH_ARG" =~ ^[0-9]+$ ]]; then
    BATCH_NUM="$RAW_BATCH_ARG"
    BATCH_DIR="${WORKSPACE}/configs/new_model_eval/batch_${BATCH_NUM}"
    DEFAULT_SESSION_NAME="poker_batch_${BATCH_NUM}"
else
    if [[ "$RAW_BATCH_ARG" = /* ]]; then
        BATCH_DIR="$RAW_BATCH_ARG"
    else
        BATCH_DIR="${WORKSPACE}/${RAW_BATCH_ARG}"
    fi
    BATCH_BASENAME=$(basename "$BATCH_DIR")
    DEFAULT_SESSION_NAME="poker_${BATCH_BASENAME}"
fi

SESSION_NAME="${SESSION_NAME:-$DEFAULT_SESSION_NAME}"

if [ ! -d "$BATCH_DIR" ]; then
    echo "Error: Batch directory not found: $BATCH_DIR"
    exit 1
fi

echo "🂡 Running new-model eval batch from: $BATCH_DIR"
echo "🂢 tmux session: $SESSION_NAME"
echo ""

CONFIGS=()
while IFS= read -r cfg; do
    CONFIGS+=("$cfg")
done < <(find "$BATCH_DIR" -maxdepth 1 -type f -name "*.yaml" | sort)

if [ "${#CONFIGS[@]}" -eq 0 ]; then
    echo "Error: No .yaml configs found in $BATCH_DIR"
    exit 1
fi

echo "Found ${#CONFIGS[@]} config(s):"
for cfg in "${CONFIGS[@]}"; do
    echo "  - $(basename "$cfg")"
done
echo ""

FIRST_CONFIG="${CONFIGS[0]}"
FIRST_NAME=$(basename "$FIRST_CONFIG" .yaml)

tmux new-session -d -s "$SESSION_NAME" -n "$FIRST_NAME"
tmux send-keys -t "${SESSION_NAME}:0" \
    "cd ${WORKSPACE} && uv run llm_agent_play.py --config-path=\"${BATCH_DIR}\" --config-name=\"${FIRST_NAME}\"" C-m

echo "✅ Started: $FIRST_NAME in tmux window 0"

INDEX=1
for cfg in "${CONFIGS[@]:1}"; do
    NAME=$(basename "$cfg" .yaml)
    tmux new-window -t "$SESSION_NAME" -n "$NAME"
    tmux send-keys -t "${SESSION_NAME}:${INDEX}" \
        "cd ${WORKSPACE} && uv run llm_agent_play.py --config-path=\"${BATCH_DIR}\" --config-name=\"${NAME}\"" C-m

    echo "✅ Started: $NAME in tmux window ${INDEX}"
    INDEX=$((INDEX + 1))
done

echo ""
echo "All games started in parallel in tmux session: ${SESSION_NAME}"
echo ""
echo "To attach:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo "Inside tmux:"
echo "  Ctrl+b then 0-4   # switch between game windows"
echo "  Ctrl+b then d     # detach (games keep running)"
echo ""


