"""Прогон: папка с результатами одной партии или эксперимента."""

from __future__ import annotations

import secrets
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class Run:
    """Папка прогона `runs/<run_id>/` и пути к файлам внутри неё."""

    run_id: str
    dir: Path

    @classmethod
    def create(cls, runs_dir: str | Path, name: str | None = None) -> Run:
        """Создать новую папку прогона.

        run_id сортируется по времени: 20261013-141600-baseline-a1f3.
        Случайный суффикс исключает совпадение при двух запусках в одну секунду.
        """
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        parts = [stamp, name, secrets.token_hex(2)] if name else [stamp, secrets.token_hex(2)]
        run_id = "-".join(parts)
        directory = Path(runs_dir) / run_id
        directory.mkdir(parents=True, exist_ok=False)
        return cls(run_id=run_id, dir=directory)

    @property
    def config_path(self) -> Path:
        return self.dir / "config.yaml"

    @property
    def calls_path(self) -> Path:
        return self.dir / "llm_calls.jsonl"

    @property
    def prompts_path(self) -> Path:
        return self.dir / "llm_prompts.jsonl"