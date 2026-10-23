#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$project_dir/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
    printf '%s\n' 'No se encontró .venv/bin/python. Consulta docs/INSTALACION.md.' >&2
    exit 2
fi
exec "$python_bin" -B "$project_dir/main.py" "$@"
