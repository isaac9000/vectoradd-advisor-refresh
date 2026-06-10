# VectorAdd Autoresearch — Epoch Refresh

An advisor-worker agent pair that iteratively optimizes a CUDA kernel for float16 vector addition on NVIDIA H100. Each iteration the **advisor** reviews experiment history and proposes a strategic direction; the **worker** implements it, evaluates on an H100 via Modal, and logs the result.

After each epoch the run history is committed to git and wiped, and the best kernel from that epoch becomes the baseline for the next — giving the agents a fresh context window to explore without the noise of accumulated dead ends.

## Task

Add two `(N, N)` float16 matrices element-wise:

```
C = A + B
```

`custom_kernel` receives a `(A, B)` tuple and returns a new tensor:

| Argument | Shape | Dtype |
|---|---|---|
| A | `N × N` | `float16` |
| B | `N × N` | `float16` |
| output | `N × N` | `float16` |

**Correctness test shapes** (must pass before benchmarking):

| N |
|---|
| 256 |
| 512 |
| 1024 |
| 2048 |

**Benchmark shapes:**

| N | Elements |
|---|---|
| 1024 | 1024 × 1024 |
| 2048 | 2048 × 2048 |
| 4096 | 4096 × 4096 |
| 8192 | 8192 × 8192 |

Ranked by geometric mean latency across all four benchmark shapes (lower is better). Score = 3000 / geomean_us.

## Setup

```bash
uv sync
```

Create a `.env` file in the repo root:

```
ANTHROPIC_API_KEY=...
MODAL_TOKEN_ID=...
MODAL_TOKEN_SECRET=...
AUTORESEARCH_MODEL=claude-sonnet-4-6   # optional, this is the default
```

Deploy the H100 evaluator (once, before any agent runs):

```bash
uv run modal deploy eval_modal_vectoradd.py
```

## Running the agent

Two epochs of 10 iterations each, starting from scratch:

```bash
uv run vectoradd/agent.py --epoch-sizes 10 10
```

Start from the provided Triton baseline:

```bash
uv run vectoradd/agent.py --baseline vectoradd/starting_point.py --epoch-sizes 15 10
```

Use different models for advisor and worker:

```bash
uv run vectoradd/agent.py --baseline vectoradd/starting_point.py --advisor-model claude-opus-4-8 --worker-model claude-sonnet-4-6 --epoch-sizes 15 10
```

Or use the provided script (checks for H100 then launches in tmux):

```bash
./run_agent.sh
```

Evaluate a kernel file without running the agent:

```bash
cd vectoradd
python run_eval.py submission.py -o results.json
python run_eval.py submission.py -o results.json --mode test   # correctness only
```

## Epoch refresh

Each epoch runs for `N` iterations, then:

1. The epoch directory (history, TSV, plots, snapshots) is committed to git.
2. All run artifacts are deleted — the next epoch's agents start with a blank slate.
3. `best_submission.py` from the epoch is copied to `submission.py` as the next epoch's baseline.
4. Advisor and worker agents are rebuilt with fresh memory and new thread IDs.

Epoch directories are named by timestamp (not by epoch number) so agents cannot infer their position in the run from the filesystem.

## Structure

```
eval_modal_vectoradd.py   — deployable Modal H100 evaluator
run_agent.sh              — H100 check + tmux agent launcher
vectoradd/
├── agent.py              — advisor-worker agentic loop with epoch refresh
├── advisor_prompt.md     — advisor system prompt: strategy, comparison discipline
├── worker_prompt.md      — worker system prompt: mandatory sequence, rules
├── submission.py         — the kernel file the worker edits each iteration
├── starting_point.py     — original Triton baseline
├── run_eval.py           — submits submission.py to the deployed Modal evaluator
├── tools.py              — log_experiment and get_experiment_history tools
└── runs/                 — one directory per run, containing one directory per epoch
```

Each epoch directory (named by timestamp) contains:
- `experiment_history.md` — full log of every attempt with code and result (deleted after epoch commit)
- `results.tsv` — tab-separated summary for plotting (deleted after epoch commit)
- `progress.png` — latency scatter plot updated each experiment; shows keep/discard/crash points, best-time step line, and cumulative LLM call count (deleted after epoch commit)
- `iterations.png` — best latency per advisor iteration (deleted after epoch commit)
- `best_submission.py` — snapshot of the fastest kernel found in this epoch (kept; promoted to next epoch baseline)
- `proposals.md` — advisor proposals for every iteration (deleted after epoch commit)
- `snapshot_iter{N}.py` — per-iteration snapshots of submission.py before worker edits (deleted after epoch commit)

## LLM Call Counter

The agent tracks how many times the LLM is invoked across both the advisor and worker agents (each tool-calling turn and each plain response counts as one call). This is reported:

- **Per-iteration** in the console: `[advisor]` and `[worker]` call counts accumulated into a running total
- **At each checkpoint** (every `--checkpoint-every` iterations): `LLM calls (total): T`
- **In the final report**: `LLM calls (total): T`
- **On `progress.png`**: displayed as a badge in the bottom-right corner of every plot, updated live as experiments are logged
