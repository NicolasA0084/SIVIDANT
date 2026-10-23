#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
export PYTHONPATH="$project_dir/src${PYTHONPATH:+:$PYTHONPATH}"
"$project_dir/.venv/bin/python" -B -m unittest discover -s tests -v
"$project_dir/.venv/bin/python" -B - <<'PY'
from pathlib import Path
files = list(Path('src').rglob('*.py'))
for path in files:
    compile(path.read_text(encoding='utf-8'), str(path), 'exec')
print(f'Sintaxis verificada: {len(files)} módulos; sin generar bytecode.')
PY
