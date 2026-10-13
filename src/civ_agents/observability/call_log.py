"""Журнал вызовов модели: одна строка JSON на вызов (формат JSONL).

Подключается к клиенту как наблюдатель:
    OpenAICompatClient(profile, observer=CallLog.for_run(run, save_prompts=False))
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from civ_agents.llm.base import CallRecord
from civ_agents.llm.errors import LLMContentError
from civ_agents.observability.context import current_context
from civ_agents.observability.run import Run


class CallLog:
    """Наблюдатель, который пишет каждый CallRecord в файл.

    Основной файл — компактные метрики вызова (всегда).
    Файл промптов — полные сообщения и ответы (только при save_prompts, он большой).
    """

    def __init__(self, path: str | Path, *, run_id: str, prompts_path: str | Path | None = None) -> None:
        self.path = Path(path)
        self.prompts_path = Path(prompts_path) if prompts_path else None
        self.run_id = run_id
        self._lock = threading.Lock()  # на случай вызовов из нескольких потоков

    @classmethod
    def for_run(cls, run: Run, *, save_prompts: bool = False) -> CallLog:
        return cls(run.calls_path, run_id=run.run_id, prompts_path=run.prompts_path if save_prompts else None)

    def __call__(self, record: CallRecord) -> None:
        call_id = uuid.uuid4().hex[:12]
        context = current_context()
        response = record.response or getattr(record.error, "response", None)

        entry: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "call_id": call_id,
            **context,
            "model": record.model,
            "ok": record.ok,
            "error_type": type(record.error).__name__ if record.error else None,
            "error": str(record.error) if record.error else None,
            "latency_s": round(record.latency_s, 3),
            "attempts": record.attempts,
            "prompt_tokens": response.usage.prompt_tokens if response else None,
            "completion_tokens": response.usage.completion_tokens if response else None,
            "cached_tokens": response.usage.cached_tokens if response else None,
            "finish_reason": response.finish_reason.value if response else None,
            "tool_calls": [call.name for call in response.tool_calls] if response else [],
            "n_messages": len(record.messages),
            "prompt_chars": sum(len(m.content) for m in record.messages),
            "warnings": record.warnings,
        }
        self._append(self.path, entry)

        if self.prompts_path is not None:
            self._append(
                self.prompts_path,
                {
                    "call_id": call_id,
                    **context,
                    "messages": [m.model_dump() for m in record.messages],
                    "tools": [t.name for t in record.tools],
                    "response": response.model_dump(mode="json") if response else None,
                    "content_error": isinstance(record.error, LLMContentError),
                },
            )

    def _append(self, path: Path, entry: dict[str, Any]) -> None:
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as file:
                file.write(line + "\n")