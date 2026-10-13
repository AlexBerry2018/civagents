"""Сводка реального хода и первый ответ модели на неё."""

from civ_agents.env.server import wait_for_server

wait_for_server()

import gymnasium  # noqa: E402
import civrealm  # noqa: E402, F401
from civrealm.configs import fc_args  # noqa: E402
from civrealm.envs.freeciv_wrapper import LLMWrapper  # noqa: E402

from civ_agents.config import load_settings  # noqa: E402
from civ_agents.env.actions import ActionCatalog  # noqa: E402
from civ_agents.env.state import GameState  # noqa: E402
from civ_agents.env.summary import areas_from_llm_info, build_summary  # noqa: E402
from civ_agents.llm.openai_compat import OpenAICompatClient  # noqa: E402
from civ_agents.llm.types import Message  # noqa: E402

env = LLMWrapper(gymnasium.make("civrealm/FreecivBase-v0"))
observations, info = env.reset(client_port=fc_args["client_port"])

state = GameState.from_observation(observations, info)
catalog = ActionCatalog(info["available_actions"])
areas = areas_from_llm_info(info["llm_info"], state)
summary = build_summary(state, catalog, areas, max_turns=50)
print(summary)

client = OpenAICompatClient(load_settings().llm.active)
response = client.chat([
    Message.system(
        "You play Freeciv as the ruler of a civilization. Goal: survive and grow in the first 50 turns. "
        "Choose ONE action from the listed actions. "
        "Answer in exactly one line: <actor type> <actor id>: <action>. Example: unit 105: build city"
    ),
    Message.user(summary),
])
print("\nМодель:", response.text)
print(f"Промпт: {response.usage.prompt_tokens} токенов, время {response.latency_s:.1f} с")
env.close()