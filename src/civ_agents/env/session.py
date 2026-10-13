"""Партия Freeciv: запуск, действия, конец хода, горизонт, запись в папку прогона.

    session = GameSession(settings, run)
    view = session.start()
    while not session.done:
        for action in choose(view):          # агенты решают
            view = session.act(action)       # недопустимое не уйдёт в игру
        view = session.end_turn()
    session.close()
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from civ_agents.env.actions import Action, ActionCatalog, InvalidAction
from civ_agents.env.server import wait_for_server
from civ_agents.env.state import GameState
from civ_agents.env.summary import Role, areas_from_llm_info, build_summary

if TYPE_CHECKING:
    from civ_agents.config import Settings
    from civ_agents.observability.run import Run

logger = logging.getLogger(__name__)


class GameEnv(Protocol):
    """Всё, что сессии нужно от игры. Реализации: CivRealmEnv и поддельная среда в тестах."""

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]: ...

    def step(self, action: tuple[str, Any, str] | None) -> tuple[dict, float, bool, bool, dict]: ...

    def llm_info(self, observations: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]: ...

    def results(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


class CivRealmEnv:
    """Настоящая игра через CivRealm.

    Действия идут в базовую среду во внутреннем формате. LLMWrapper используется
    только как источник описаний окрестностей (get_llm_info): его step переводит
    имена всех действий, что ломает действия науки и правительства (см. ADR).
    """

    def __init__(self, server_url: str, server_wait_s: float) -> None:
        wait_for_server(server_url, timeout_s=server_wait_s)  # CivRealm опрашивает сервер при импорте

        # Старый ray внутри CivRealm использует pkg_resources (ADR 0001) — предупреждение известно.
        warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

        import gymnasium
        import civrealm  # noqa: F401 — регистрирует среды
        from civrealm.configs import fc_args
        from civrealm.envs.freeciv_wrapper import LLMWrapper

        self._client_port = fc_args["client_port"]
        # Проверка gymnasium отключена: пространство наблюдений CivRealm объявлено неточно.
        self._wrapper = LLMWrapper(gymnasium.make("civrealm/FreecivBase-v0", disable_env_checker=True))

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        return self._wrapper.reset(client_port=self._client_port)

    def step(self, action: tuple[str, Any, str] | None) -> tuple[dict, float, bool, bool, dict]:
        return self._wrapper.env.step(action)  # мимо перевода имён LLMWrapper

    def llm_info(self, observations: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
        return self._wrapper.get_llm_info(observations, info)

    def results(self) -> dict[str, Any]:
        return self._wrapper.unwrapped.get_game_results()

    def close(self) -> None:
        self._wrapper.close()


@dataclass
class TurnView:
    """Всё, что агентам нужно знать о текущем моменте игры."""

    state: GameState
    catalog: ActionCatalog
    areas: dict[tuple[int, int], str]
    max_turns: int | None = None

    @property
    def turn(self) -> int:
        return self.state.turn

    def summary(self, role: Role = "all", max_tokens: int = 1500) -> str:
        return build_summary(self.state, self.catalog, self.areas, role=role, max_turns=self.max_turns, max_tokens=max_tokens)


@dataclass
class SessionStats:
    actions_sent: int = 0
    actions_rejected: int = 0
    actions_skipped: int = 0  # безвредные повторы, которые не отправлялись в игру
    turns_played: int = 0
    terminated: bool = False
    truncated: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


class GameSession:
    def __init__(self, settings: Settings, run: Run | None = None, *, env: GameEnv | None = None) -> None:
        self._settings = settings
        self._run = run
        self._env = env  # в тестах передаётся поддельная среда
        self._view: TurnView | None = None
        self.stats = SessionStats()

    # --- Жизненный цикл ---------------------------------------------------------

    def start(self) -> TurnView:
        if self._env is None:
            game = self._settings.game
            self._env = CivRealmEnv(str(game.server_url), game.server_wait_s)
        observations, info = self._env.reset()
        self._update(observations, info)
        self._record_turn()
        return self.view

    @property
    def view(self) -> TurnView:
        if self._view is None:
            raise RuntimeError("Партия не начата: вызовите start()")
        return self._view

    @property
    def done(self) -> bool:
        """Партия окончена: игра завершилась, оборвалась или достигнут горизонт max_turns."""
        return (
            self.stats.terminated
            or self.stats.truncated
            or self.view.turn > self._settings.game.max_turns
        )

    def act(self, action: Action) -> TurnView:
        """Отправить проверенное действие. Недопустимое не уходит в игру: InvalidAction."""
        researching = self.view.state.me.researching
        if action.ctrl == "tech" and researching and action.readable == f"research {researching}":
            # Повторный выбор текущей технологии сбрасывает накопленные колбы (проверено на игре).
            self.stats.actions_skipped += 1
            self._log("actions.jsonl", {"turn": self.view.turn, "action": str(action), "accepted": False,
                                        "reason": "already researching"})
            return self.view
        if not self.view.catalog.is_valid(action):
            self.stats.actions_rejected += 1
            self._log("actions.jsonl", {"turn": self.view.turn, "action": str(action), "accepted": False})
            raise InvalidAction(f"{action} is not available now")
        self._step(action.to_civrealm())
        self.stats.actions_sent += 1
        self._log("actions.jsonl", {"turn": self.view.turn, "action": str(action), "accepted": True})
        return self.view

    def end_turn(self) -> TurnView:
        """Закончить ход (в CivRealm — шаг с действием None)."""
        previous = self.view.turn
        self._step(None)
        if self.view.turn != previous:
            self.stats.turns_played += 1
            self._record_turn()
        return self.view

    def close(self) -> dict[str, Any]:
        """Завершить партию, сохранить и вернуть результаты."""
        results: dict[str, Any] = {}
        if self._env is not None:
            self._env.close()  # CivRealm подсчитывает результаты при закрытии, поэтому сначала close
            try:
                results = self._env.results()
            except Exception:  # noqa: BLE001 — результаты не должны мешать закрытию
                logger.exception("Не удалось получить результаты партии")
        summary = {
            "turns_played": self.stats.turns_played,
            "last_turn": self._view.turn if self._view else None,
            "actions_sent": self.stats.actions_sent,
            "actions_rejected": self.stats.actions_rejected,
            "actions_skipped": self.stats.actions_skipped,
            "terminated": self.stats.terminated,
            "truncated": self.stats.truncated,
            "game_results": results,
        }
        if self._run is not None:
            path = self._run.dir / "game_results.json"
            path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return summary

    # --- Внутреннее ---------------------------------------------------------------

    def _step(self, action: tuple[str, Any, str] | None) -> None:
        observations, _reward, terminated, truncated, info = self._env.step(action)
        self.stats.terminated |= bool(terminated)
        self.stats.truncated |= bool(truncated)
        if truncated:
            logger.warning("Партия оборвана (truncated) на ходу %s", info.get("turn"))
        if observations and info:  # при обрыве CivRealm возвращает пустые словари
            self._update(observations, info)

    def _update(self, observations: dict[str, Any], info: dict[str, Any]) -> None:
        state = GameState.from_observation(observations, info)
        llm_info = info.get("llm_info") or self._env.llm_info(observations, info)
        self._view = TurnView(
            state=state,
            catalog=ActionCatalog(info.get("available_actions", {})),
            areas=areas_from_llm_info(llm_info, state),
            max_turns=self._settings.game.max_turns,
        )

    def _record_turn(self) -> None:
        self._log("turns.jsonl", {"turn": self.view.turn, "state": self.view.state.model_dump(mode="json")})

    def _log(self, filename: str, entry: dict[str, Any]) -> None:
        if self._run is None:
            return
        with (self._run.dir / filename).open("a", encoding="utf-8") as file:
            file.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")