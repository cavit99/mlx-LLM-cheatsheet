#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_python() {
  local candidate
  local found_any=false

  for candidate in python3 python; do
    if ! command -v "$candidate" >/dev/null 2>&1; then
      continue
    fi
    found_any=true
    if "$candidate" - <<'PY' >/dev/null 2>&1
import mlx.core  # noqa: F401
PY
    then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  if [[ "$found_any" == false ]]; then
    echo "python3 or python is required for mlx_probe.sh" >&2
  else
    echo "No available python3/python interpreter can import 'mlx'." >&2
    echo "Activate an MLX-capable environment, then rerun mlx_probe.sh." >&2
  fi
  exit 1
}

PYTHON_BIN="$(find_python)"

exec "$PYTHON_BIN" "$SCRIPT_DIR/mlx_probe.py" "$@"
