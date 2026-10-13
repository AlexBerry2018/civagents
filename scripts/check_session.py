"""Проверка GameSession: 3 хода простыми правилами."""

import logging

from civ_agents.config import load_settings, save_snapshot
from civ_agents.env.actions import InvalidAction
from civ_agents.env.session import GameSession
from civ_agents.observability.run import Run

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

settings = load_settings()
settings.game.max_turns = 6
run = Run.create(settings.paths.runs_dir, name="session-check")
save_snapshot(settings, run.config_path)

session = GameSession(settings, run)
view = session.start()
while not session.done:
    print(f"\n=== Ход {view.turn} ===")
    for unit in view.state.units:
        options = view.catalog.options("unit", unit.id)
        choice = "build city" if "build city" in options else "keep activity" if "keep activity" in options else None
        if choice:
            try:
                view = session.act(view.catalog.resolve("unit", unit.id, choice))
                print(f"  {unit.type} {unit.id}: {choice}")
            except InvalidAction as error:
                print(f"  отклонено: {error}")
    if view.turn == 1:
        tech = view.state.research_options[0]
        view = session.act(view.catalog.resolve("tech", "cur_player", f"research {tech}"))
        print(f"  наука: research {tech} (было: {view.state.me.researching})")
    view = session.end_turn()

print("\n", session.close())
print(f"Папка прогона: {run.dir}")