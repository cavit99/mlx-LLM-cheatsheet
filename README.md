# mlx-LLM-cheatsheet

Source-backed cheat sheet and lightweight verifier for current `mlx` and `mlx-lm`
behavior.

## Baseline

- MLX source tag: `v0.31.1` (published March 12, 2026)
- MLX-LM source tag: `v0.31.0` (published March 7, 2026)
- Python support from current package metadata:
  `mlx 0.31.1` requires `>=3.10` and ships wheels through `cp314`;
- Runtime spot checks performed: March 12, 2026

Note: the PyPI `mlx-lm==0.31.0` wheel is yanked due to a batched KV cache
cross-contamination bug. This repo tracks the upstream source tag and installs
`mlx-lm` from GitHub for that reason.

## Files

- `cheatsheet.md`: current MLX and MLX-LM reference, including memory sizing and profiling guidance
- `validation.py`: correctness-focused verifier for the claims in the cheat sheet
- `VALIDATION.md`: claim coverage map showing what is runtime-validated, source-validated, or advice
- `pyproject.toml` and `uv.lock`: primary project metadata and locked dependency set
- `requirements.txt`: compatibility fallback for pip-based environments
- `skills/mlx`: discoverable Codex skill for MLX / MLX-LM work

## Environment

Use any isolated Python environment supported by the current MLX baseline. The
practical floor for this repo is Python `3.10+`, because that is what
`mlx 0.31.1` requires. The recommended path below uses `uv`.

```bash
uv sync
```

`uv` will create and manage the project environment automatically.

If you already have a dedicated environment, use that instead. The primary
source of truth is `pyproject.toml` plus `uv.lock`; `requirements.txt` is kept
as a compatibility fallback for plain `pip` workflows.

Validation for this repo revision was run on Python `3.11.6`.

## Validate

```bash
uv run python validation.py
```

The validator does not download any models. It only checks API surface and
small tensor behaviors locally.

Optional live-model validation:

```bash
MLX_LM_LOCAL_MODEL=/path/to/mlx-model uv run python validation.py
```

If a local model path is provided, the validator also checks real `mlx-lm`
loading, one-step generation, prompt-cache round-trips, and the current
`batch_generate(max_tokens=1)` edge case.

## Codex Skill

This repo includes a visible Codex skill at [skills/mlx](/Users/caviterginsoy/Coding/mlx-LLM-cheatsheet/skills/mlx/SKILL.md).

That follows the common GitHub repo pattern used by the official
[`openai/skills`](https://github.com/openai/skills) repository and matches the
Codex skill installer expectation of a repo path like `skills/<skill-name>`.

Install it from GitHub with:

```bash
python ~/.codex/skills/.system/skill-installer/scripts/install-skill-from-github.py \
  --repo cavit99/mlx-LLM-cheatsheet \
  --path skills/mlx
```

Or copy/sync it locally from this repo:

```bash
mkdir -p ~/.codex/skills/mlx
rsync -a --delete skills/mlx/ ~/.codex/skills/mlx/
```

## Scope

- Official MLX semantics from upstream docs and source
- Current MLX-LM implementation patterns that matter for LLM code
- Current top-level MLX memory profiling APIs and Apple silicon working-set guidance
- Explicit separation between documented contracts and observed-but-fragile
  behavior
