"""Настройки системы.

Источники по убыванию приоритета:
  1. переменные окружения CIV_* (в т.ч. из .env);
  2. YAML-файл (configs/settings.yaml или путь из CIV_CONFIG).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, HttpUrl, ValidationError, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict

DEFAULT_CONFIG_PATH = Path("configs/settings.yaml")


class _Section(BaseModel):
    """Базовый класс раздела: лишние поля (опечатки) запрещены."""

    model_config = ConfigDict(extra="forbid")


class LLMProfile(_Section):
    base_url: HttpUrl
    model: str = Field(min_length=1)
    timeout_s: float = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    context_window: int = Field(gt=0)

    @model_validator(mode="after")
    def _output_fits_context(self) -> LLMProfile:
        if self.max_output_tokens >= self.context_window:
            raise ValueError("max_output_tokens должен быть меньше context_window")
        return self


class LLMSettings(_Section):
    active_profile: str
    profiles: dict[str, LLMProfile] = Field(min_length=1)

    @model_validator(mode="after")
    def _active_profile_exists(self) -> LLMSettings:
        if self.active_profile not in self.profiles:
            raise ValueError(
                f"active_profile '{self.active_profile}' не найден, "
                f"есть: {sorted(self.profiles)}"
            )
        return self

    @property
    def active(self) -> LLMProfile:
        """Текущий профиль модели."""
        return self.profiles[self.active_profile]


class GameSettings(_Section):
    civrealm_config: Path
    max_turns: int = Field(gt=0)
    username: str = Field(min_length=1)


class LimitsSettings(_Section):
    turn_time_budget_s: float = Field(gt=0)
    max_llm_calls_per_turn: int = Field(gt=0)


class PathsSettings(_Section):
    runs_dir: Path


class LoggingSettings(_Section):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    save_prompts: bool = False


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CIV_",
        env_nested_delimiter="__",
        extra="forbid",
    )

    llm: LLMSettings
    game: GameSettings
    limits: LimitsSettings
    paths: PathsSettings
    logging: LoggingSettings = LoggingSettings()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Первый в кортеже — самый сильный.
        # Данные из YAML передаются как аргументы (init_settings),
        # поэтому переменные окружения ставим выше них.
        return (env_settings, init_settings)


def resolve_config_path(path: str | Path | None = None) -> Path:
    """Путь к YAML: аргумент → CIV_CONFIG → путь по умолчанию."""
    if path is not None:
        return Path(path)
    return Path(os.environ.get("CIV_CONFIG", DEFAULT_CONFIG_PATH))


def load_settings(path: str | Path | None = None) -> Settings:
    """Загрузить и проверить настройки. Ошибка — сразу при старте."""
    load_dotenv(override=False)  # .env → окружение, реальные переменные не перетираются
    config_path = resolve_config_path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Файл настроек не найден: {config_path}")

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return Settings(**data)


def save_snapshot(settings: Settings, path: str | Path) -> Path:
    """Сохранить итоговые настройки в YAML (снимок для папки прогона)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(settings.model_dump(mode="json"), allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    return path


def main() -> int:
    """`uv run python -m civ_agents.config` — напечатать итоговую конфигурацию."""
    try:
        settings = load_settings()
    except (FileNotFoundError, ValidationError) as error:
        print(f"Ошибка настроек:\n{error}", file=sys.stderr)
        return 1
    print(yaml.safe_dump(settings.model_dump(mode="json"), allow_unicode=True, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())