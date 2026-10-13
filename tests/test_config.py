import copy
import os
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from civ_agents.config import load_settings, save_snapshot

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_CONFIG = PROJECT_ROOT / "configs" / "settings.yaml"

VALID = {
    "llm": {
        "active_profile": "local",
        "profiles": {
            "local": {
                "base_url": "http://localhost:11434/v1",
                "model": "qwen3:4b-instruct",
                "timeout_s": 60,
                "max_output_tokens": 256,
                "context_window": 8192,
            },
            "other": {
                "base_url": "http://localhost:11434/v1",
                "model": "qwen3:4b",
                "timeout_s": 120,
                "max_output_tokens": 1024,
                "context_window": 8192,
            },
        },
    },
    "game": {"civrealm_config": "configs/civrealm/baseline.yaml", "max_turns": 50, "username": "myagent"},
    "limits": {"turn_time_budget_s": 60, "max_llm_calls_per_turn": 10},
    "paths": {"runs_dir": "runs"},
}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch):
    """Тесты не зависят от переменных окружения и локального .env."""
    for key in list(os.environ):
        if key.startswith("CIV_"):
            monkeypatch.delenv(key)
    monkeypatch.setattr("civ_agents.config.load_dotenv", lambda **kwargs: False)


def write_config(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def valid_data() -> dict:
    return copy.deepcopy(VALID)


def test_real_config_loads():
    settings = load_settings(REAL_CONFIG)
    assert settings.llm.active.model


def test_valid_config_values(tmp_path):
    settings = load_settings(write_config(tmp_path, valid_data()))
    assert settings.game.max_turns == 50
    assert settings.llm.active.model == "qwen3:4b-instruct"
    assert settings.logging.level == "INFO"  # значение по умолчанию


def test_typo_in_field_name_fails(tmp_path):
    data = valid_data()
    data["game"]["max_turn"] = data["game"].pop("max_turns")
    with pytest.raises(ValidationError, match="max_turn"):
        load_settings(write_config(tmp_path, data))


def test_missing_section_fails(tmp_path):
    data = valid_data()
    del data["limits"]
    with pytest.raises(ValidationError, match="limits"):
        load_settings(write_config(tmp_path, data))


def test_unknown_active_profile_fails(tmp_path):
    data = valid_data()
    data["llm"]["active_profile"] = "nope"
    with pytest.raises(ValidationError, match="nope"):
        load_settings(write_config(tmp_path, data))


def test_output_tokens_must_fit_context(tmp_path):
    data = valid_data()
    data["llm"]["profiles"]["local"]["max_output_tokens"] = 9000
    with pytest.raises(ValidationError, match="context_window"):
        load_settings(write_config(tmp_path, data))


def test_negative_number_fails(tmp_path):
    data = valid_data()
    data["game"]["max_turns"] = -1
    with pytest.raises(ValidationError):
        load_settings(write_config(tmp_path, data))


def test_env_overrides_yaml(tmp_path, monkeypatch):
    monkeypatch.setenv("CIV_GAME__MAX_TURNS", "30")
    monkeypatch.setenv("CIV_LLM__ACTIVE_PROFILE", "other")
    settings = load_settings(write_config(tmp_path, valid_data()))
    assert settings.game.max_turns == 30
    assert settings.llm.active.model == "qwen3:4b"
    assert settings.game.username == "myagent"  # остальное из YAML не затёрто


def test_invalid_env_value_fails(tmp_path, monkeypatch):
    monkeypatch.setenv("CIV_GAME__MAX_TURNS", "-5")
    with pytest.raises(ValidationError):
        load_settings(write_config(tmp_path, valid_data()))


def test_civ_config_selects_file(tmp_path, monkeypatch):
    data = valid_data()
    data["game"]["username"] = "from_env_path"
    monkeypatch.setenv("CIV_CONFIG", str(write_config(tmp_path, data)))
    assert load_settings().game.username == "from_env_path"


def test_missing_file_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "nope.yaml")


def test_snapshot_roundtrip(tmp_path):
    settings = load_settings(write_config(tmp_path, valid_data()))
    snapshot = save_snapshot(settings, tmp_path / "run_001" / "config.yaml")
    assert snapshot.is_file()
    assert load_settings(snapshot) == settings