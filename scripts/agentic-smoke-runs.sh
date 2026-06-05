#!/usr/bin/env bash
set -euo pipefail

RUNS="${1:-5}"
MAX_TURNS="${2:-150}"

if ! [[ "$RUNS" =~ ^[0-9]+$ ]]; then
  echo "error: run count must be an integer" >&2
  exit 1
fi
if ! [[ "$MAX_TURNS" =~ ^[0-9]+$ ]]; then
  echo "error: max_turns must be an integer" >&2
  exit 1
fi
if (( RUNS < 1 )); then
  echo "error: run count must be >= 1" >&2
  exit 1
fi
if (( RUNS > 100 )); then
  echo "error: run count safety cap exceeded (max 100)" >&2
  exit 1
fi

successes=0
failures=0

echo "Running ${RUNS} fully agentic runs (max_turns=${MAX_TURNS})..."
for ((i=1; i<=RUNS; i++)); do
  seed="$(( (RANDOM << 16) ^ RANDOM ))"
  echo
  echo "=== Run ${i}/${RUNS} (seed=${seed}) ==="
  output="$(uv run dungeon-agent --mode agent-only --seed "${seed}" --max-turns "${MAX_TURNS}" 2>&1 || true)"
  status_line="$(printf "%s\n" "${output}" | grep -E "run-complete:" | tail -n 1 || true)"
  if printf "%s\n" "${status_line}" | grep -q "status=success"; then
    successes=$((successes + 1))
    echo "result: success"
  else
    failures=$((failures + 1))
    echo "result: failure"
  fi
  echo "${status_line:-run-complete line not found}"
done

echo
echo "=== Aggregate ==="
echo "runs=${RUNS} successes=${successes} failures=${failures}"
