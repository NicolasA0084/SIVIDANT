#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$project_dir/.venv/bin/python"

if [[ ! -x "$python_bin" ]]; then
    echo "Missing .venv. Follow the installation steps in README.md." >&2
    exit 1
fi

cd -- "$project_dir"
exec "$python_bin" "$project_dir/main.py" "$@"
