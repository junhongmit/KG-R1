# KG-R1 Baselines

This folder isolates lightweight baseline experiments from the RL / in-process `vllm` stack.

Current baselines:

- `vanilla.py`: direct-answer prompting against an external OpenAI-compatible API
- `cot.py`: chain-of-thought prompting with final-answer extraction
- `sc.py`: self-consistency over multiple chain-of-thought samples with majority-vote aggregation

Shared features:

- Resume and progress tracking via `predictions.jsonl.progress.json`
- Threaded parallel execution for API-bound inference
- Automatic one-call evaluation after generation
- Same parquet data source used by the existing KG-R1 evaluation scripts

Examples:

```bash
cd /orcd/home/002/junhong/LLM/KG-R1

python -m baselines.vanilla \
  --dataset cwq \
  --num_workers 8

python -m baselines.cot \
  --dataset cwq \
  --num_workers 8

python -m baselines.sc \
  --dataset cwq \
  --num_workers 8 \
  --n_rollouts 8
```

Naming options:

- `--run_name`: full output folder name
- `--experiment_name`: stable prefix for timestamped runs
- `--run_postfix`: suffix appended to the timestamped run name, e.g. `_qwen3_235B_0`

Example:

```bash
python -m baselines.sc \
  --dataset cwq \
  --n_rollouts 8 \
  --experiment_name cwq-sc \
  --run_postfix _qwen3_235B_0
```

Environment variables:

- `KGR1_BASELINE_OPENAI_API_BASE` (default: `http://localhost:7878/v1`)
- `KGR1_BASELINE_OPENAI_API_KEY` (default: `EMPTY`)
- `KGR1_BASELINE_OPENAI_MODEL` (optional; auto-discovers first served model if unset)

Standalone evaluation:

```bash
python -m baselines.evaluate \
  --input_file baselines/results/self_consistency/cwq/<run_name>/predictions.jsonl \
  --k_values 1 2 4 8
```

Evaluation notes:

- `exact_match/mean`, `f1/mean`, etc. are computed on the primary response.
- For self-consistency, the primary response is the majority-vote aggregated answer.
- `pass@k` metrics are computed over the candidate sampled answers.
