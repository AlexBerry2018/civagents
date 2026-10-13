"""Проверка каталога действий на реальной игре."""

from civ_agents.env.server import wait_for_server

wait_for_server()

import gymnasium  # noqa: E402
import civrealm  # noqa: E402, F401
from civrealm.configs import fc_args  # noqa: E402

from civ_agents.env.actions import ActionCatalog  # noqa: E402

env = gymnasium.make("civrealm/FreecivBase-v0")
observations, info = env.reset(client_port=fc_args["client_port"])
catalog = ActionCatalog(info["available_actions"])

print(f"Всего допустимых действий: {len(catalog)}")
for unit_id in catalog.actors("unit"):
    unit_type = observations["unit"][unit_id]["type_rule_name"]
    print(f"\n{unit_type} {unit_id}: {', '.join(catalog.options('unit', unit_id))}")
print(f"\nНаука: {catalog.options('tech', 'cur_player')[:8]} ...")
print(f"Правительство: {catalog.options('gov', 0)[:3]} ...")

first = catalog.actors("unit")[0]
action = catalog.resolve("unit", str(first), "keep activity")
print(f"\nresolve('unit', '{first}', 'keep activity') -> {action.to_civrealm()}")
env.close()