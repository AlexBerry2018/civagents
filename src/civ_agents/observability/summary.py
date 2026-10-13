"""Сводка по журналу вызовов модели.

    uv run python -m civ_agents.observability.summary runs/<run_id>
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def load_calls(path: str | Path) -> list[dict[str, Any]]:
    """Прочитать журнал. Можно передать папку прогона или сам файл."""
    path = Path(path)
    if path.is_dir():
        path = path / "llm_calls.jsonl"
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def percentile(values: list[float], q: float) -> float:
    """Перцентиль методом ближайшего ранга (q от 0 до 100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(q / 100 * len(ordered)) - 1))
    return ordered[index]


def summarize(calls: list[dict[str, Any]]) -> dict[str, Any]:
    latencies = [c["latency_s"] for c in calls]
    slowest = max(calls, key=lambda c: c["latency_s"], default=None)

    by_agent: dict[str, list[float]] = defaultdict(list)
    for call in calls:
        by_agent[str(call.get("agent", "—"))].append(call["latency_s"])

    return {
        "calls": len(calls),
        "ok": sum(c["ok"] for c in calls),
        "errors": dict(Counter(c["error_type"] for c in calls if not c["ok"])),
        "retried": sum(c["attempts"] > 1 for c in calls),
        "prompt_tokens": sum(c["prompt_tokens"] or 0 for c in calls),
        "completion_tokens": sum(c["completion_tokens"] or 0 for c in calls),
        "cached_tokens": sum(c["cached_tokens"] or 0 for c in calls),
        "latency_total_s": round(sum(latencies), 2),
        "latency_p50_s": percentile(latencies, 50),
        "latency_p95_s": percentile(latencies, 95),
        "slowest": slowest and {k: slowest.get(k) for k in ("call_id", "turn", "agent", "latency_s")},
        "with_warnings": sum(bool(c["warnings"]) for c in calls),
        "by_agent": {
            agent: {"calls": len(values), "latency_total_s": round(sum(values), 2)}
            for agent, values in sorted(by_agent.items())
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print("Использование: python -m civ_agents.observability.summary runs/<run_id>", file=sys.stderr)
        return 1
    try:
        calls = load_calls(args[0])
    except FileNotFoundError as error:
        print(f"Журнал не найден: {error.filename}", file=sys.stderr)
        return 1
    print(json.dumps(summarize(calls), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())