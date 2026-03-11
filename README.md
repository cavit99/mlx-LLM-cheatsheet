# mlx-LLM-cheatsheet

Source-backed cheat sheet and lightweight verifier for current `mlx` and `mlx-lm`
behavior.

## Baseline

- MLX source tag: `v0.31.0` (published February 28, 2026)
- MLX-LM source tag: `v0.31.0` (published March 7, 2026)
- Runtime spot checks performed: March 11, 2026

This repo now targets the current upstream source contracts, not older folklore
or ad hoc experiments.

Note: the PyPI `mlx-lm==0.31.0` wheel is yanked due to a batched KV cache
cross-contamination bug. This repo tracks the upstream source tag and installs
`mlx-lm` from GitHub for that reason.

## Files

- `cheatsheet.md`: current MLX and MLX-LM reference
- `validation.py`: correctness-focused verifier for the claims in the cheat sheet
- `VALIDATION.md`: claim coverage map showing what is runtime-validated, source-validated, or advice
- `requirements.txt`: baseline dependencies for reproducing the checks

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

## Validate

```bash
python validation.py
```

The validator does not download any models. It only checks API surface and
small tensor behaviors locally.

Optional live-model validation:

```bash
MLX_LM_LOCAL_MODEL=/path/to/mlx-model python validation.py
```

If a local model path is provided, the validator also checks real `mlx-lm`
loading, one-step generation, prompt-cache round-trips, and the current
`batch_generate(max_tokens=1)` edge case.

## Scope

- Official MLX semantics from upstream docs and source
- Current MLX-LM implementation patterns that matter for LLM code
- Explicit separation between documented contracts and observed-but-fragile
  behavior
