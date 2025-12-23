# Vals.ai Poker Agent Benchmark 

Complete workflow for running poker simulations and analysis

## Setup dependencies with uv and virtual environment

```bash
# Setup
uv sync
source .venv/bin/activate
```

## Pipeline Overview

```
1. Run Games (You Handle Matchmaking)
   │
   ├─> llm_agent_play.py → logs/*.json
   │
2. Generate Analysis
   │
   ├─> calculate_trueskill_ratings.py → analysis/results/trueskill_rankings.json
   ├─> analyze_model_actions.py → model_cost_stats.json, model_latency_stats.json
   └─> plot_all_chips.py → analysis/results/chips_over_time/*.png
```

---

## 1. Running Games

### Single Game

```bash
uv run llm_agent_play.py --config configs/final/seed_42/seating_0.yaml
```

**Output:** `logs/final/seed_42/seating_0.json`

### Parallel Batch (tmux)

```bash
# Run entire batch in parallel
./run_batch_tmux.sh 1

# Monitor
tmux attach -t poker_batch_1
# Ctrl+b then 0-9 to switch windows
# Ctrl+b then d to detach
```

### Resume Interrupted Games

```bash
# Single game
uv run resume_game.py logs/final/seed_42/seating_0.json --to-total 100

# Entire batch
./resume_seatings.sh configs/new_model_eval/batch_1 150
```

**Note:** Update `WORKSPACE` path in `resume_seatings.sh` line 6:
```bash
WORKSPACE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
```

---

## 2. Analysis Scripts

### A. TrueSkill Rankings

**Generate skill-based ratings and visualizations:**

```bash
python analysis/calculate_trueskill_ratings.py --logs-dir logs/final
```

**Outputs:**
- `analysis/results/trueskill_rankings.json` - Full rankings with statistics
- `analysis/results/trueskill_ratings.png` - Bar chart
- `analysis/results/trueskill_top_models.png` - Top 10 models
- `analysis/results/trueskill_comprehensive.png` - Multi-panel view
- `analysis/results/trueskill_pareto_frontier.png` - Performance vs confidence

**Key metrics in JSON:**
- `rating`: TrueSkill rating (μ)
- `rating_sigma`: Uncertainty (σ)
- `conservative_rating`: μ - 3σ
- `average_chips`: Mean chips at end of games
- `games_played`: Number of games participated

### B. Cost & Latency Analysis

**Generate token usage, costs, and latency metrics:**

```bash
python analysis/analyze_model_actions.py logs/final
```

**Outputs:**
- `model_cost_stats.json` - Cost per hand, tokens in/out
- `model_latency_stats.json` - Average latency per API call
- `analysis/results/hand_tokens_output.png` - Output tokens per hand
- `analysis/results/hand_tokens_total.png` - Total tokens per hand
- `analysis/results/model_play_styles.png` - Radar chart of betting patterns

**Cost metrics:**
- `cost`: USD per hand
- `average_tokens_in_per_hand`: Input tokens
- `average_tokens_out_per_hand`: Output tokens
- `average_tokens_in_per_call`: Per individual API call
- `average_tokens_out_per_call`: Per individual API call

**Latency metrics:**
- `average_latency_seconds`: Mean response time
- `min_latency_seconds`: Fastest call
- `max_latency_seconds`: Slowest call
- `total_api_calls`: Number of calls made

### C. Chip Progression Charts

**Generate chip trajectory visualizations:**

```bash
python analysis/plot_all_chips.py logs/final
```

**Outputs:**
- `analysis/results/chips_over_time/*.png` - Individual game trajectories
- `analysis/results/chips_over_time_overlays/*.png` - Per-model across all games

### D. Individual Game Analysis 

```bash
# Detailed analysis of single game
python analysis/analyze_log.py logs/final/seed_42/seating_0.json --verbose

# Batch analysis for seed
python analysis/analyze_seed_batch.py logs/final/seed_42
```

---

### Metrics 

**Performance:**
- `trueskill_rating`: Skill rating (default: μ=25, σ=8.33)
- `accuracy`: Scaled TrueSkill: `(rating - 1000) / 10`
- `average_profit`: Average chips won/lost per game
- `latency`: Average API response time in seconds

**Cost:**
- `cost_per_hand`: USD cost per poker hand

**Gameplay:**
- `hands_played_pct`: Percentage of hands where player saw flop
- `aggressive_actions_per_game`: Average bets + raises per game
- `average_raise_size`: Mean raise amount
- `percent_*`: Action distribution (bet/call/check/fold/raise)

---

## Complete Workflow Example

```bash
# 1. Run all evaluations 
./run_batch_tmux.sh 1

# 2. Generate all analysis
python analysis/calculate_trueskill_ratings.py --logs-dir logs/final
python analysis/analyze_model_actions.py logs/final
python analysis/plot_all_chips.py logs/final

# 3. Convert to correct JSON foramt
python generate_benchmark_view.py

---

## Troubleshooting

### Missing Analysis Files

```bash
# Regenerate all
python analysis/calculate_trueskill_ratings.py --logs-dir logs/final
python analysis/analyze_model_actions.py logs/final
```

### Incomplete Games

```bash
./resume_seatings.sh configs/new_model_eval/batch_1 100
```

### Validate Logs

```bash
python validate_logs.py seed_42
```

```bash
python generate_benchmark_view.py \
  --logs-dir logs/custom \
  --exclude-dirs archived temp \
  --starting-chips 200 \
  --output custom_benchmark.json
```

### Analyze Specific Seed

```bash
python analysis/calculate_trueskill_ratings.py \
  --logs-dir logs \
  --seed-folders seed_42 seed_1042 seed_2042
```
