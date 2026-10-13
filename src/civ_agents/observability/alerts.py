"""Алерты: правила проверяются после каждого хода.

Сработавший алерт пишется в runs/<run_id>/alerts.jsonl, в лог (WARNING → Loki)
и в метрику civ_alerts (→ Grafana, где на неё можно повесить уведомление).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from civ_agents.agents.government import Government, TurnReport
from civ_agents.observability import otel

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    turn: int
    name: str
    message: str


class AlertMonitor:
    def __init__(self, government: Government, turn_budget_s: float, path: str | Path | None = None,
                 window: int = 5, invalid_ratio: float = 0.3) -> None:
        self.government = government
        self.turn_budget_s = turn_budget_s
        self.path = Path(path) if path else None
        self.window = window
        self.invalid_ratio = invalid_ratio
        self.alerts: list[Alert] = []
        self._llm_errors_seen = 0

    def check(self, report: TurnReport) -> list[Alert]:
        fired: list[Alert] = []
        # 1. Зависание хода: ход дольше 80% бюджета.
        if report.seconds > 0.8 * self.turn_budget_s:
            fired.append(Alert(report.turn, "slow_turn",
                               f"turn took {report.seconds:.1f}s (budget {self.turn_budget_s:.0f}s)"))
        # 2. Модель недоступна / ответы негодны: новые ошибки LLM за ход.
        errors = sum(a.stats.llm_errors for a in self.government.advisors.values())
        if errors > self._llm_errors_seen:
            fired.append(Alert(report.turn, "llm_errors", f"{errors - self._llm_errors_seen} new LLM errors"))
        self._llm_errors_seen = errors
        # 3. Рост недопустимых действий за последние ходы.
        recent = self.government.reports[-self.window:]
        sent = sum(r.actions_sent for r in recent)
        failed = sum(r.actions_failed for r in recent)
        stats = [a.stats for a in self.government.advisors.values()]
        invalid = sum(s.invalid_actions for s in stats)
        valid = sum(s.valid_actions for s in stats)
        if invalid + valid >= 10 and invalid / (invalid + valid) > self.invalid_ratio:
            fired.append(Alert(report.turn, "invalid_actions_high",
                               f"{invalid}/{invalid + valid} proposed actions invalid"))
        if sent + failed >= 5 and failed / (sent + failed) > self.invalid_ratio:
            fired.append(Alert(report.turn, "executor_rejections_high", f"{failed}/{sent + failed} rejected"))
        # 4. Ход без единого действия, когда агенты вызывались.
        if report.agents_called and report.actions_sent == 0:
            fired.append(Alert(report.turn, "idle_turn", "agents were called but no action was sent"))

        for alert in fired:
            logger.warning("ALERT %s (turn %s): %s", alert.name, alert.turn, alert.message)
            otel.record_alert(alert.name)
            if self.path:
                with self.path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(vars(alert), ensure_ascii=False) + "\n")
        self.alerts.extend(fired)
        return fired
