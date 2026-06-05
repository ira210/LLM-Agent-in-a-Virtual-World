from dungeon_agent.settings import Settings, load_settings_with_overrides


def test_settings_from_env_parses_types() -> None:
    settings = Settings.from_env(
        {
            "DUNGEON_AGENT_MODEL_NAME": "gpt-5-mini",
            "DUNGEON_AGENT_MODEL_TIMEOUT_SECONDS": "30",
            "DUNGEON_AGENT_SEED": "42",
            "DUNGEON_AGENT_MAX_TURNS": "123",
            "DUNGEON_AGENT_ALLOW_HUMAN_HINTS": "false",
            "DUNGEON_AGENT_DRY_RUN": "true",
            "DUNGEON_AGENT_DEBUG_OUTPUT": "true",
        }
    )
    assert settings.model.model_name == "gpt-5-mini"
    assert settings.model.request_timeout_seconds == 30
    assert settings.runtime.seed == 42
    assert settings.runtime.max_turns == 123
    assert settings.runtime.allow_human_hints is False
    assert settings.runtime.dry_run is True
    assert settings.runtime.debug_output is True


def test_load_settings_with_overrides() -> None:
    settings = load_settings_with_overrides(
        env={},
        seed=7,
        max_turns=10,
        model_name="test-model",
        dry_run=False,
        debug_output=True,
    )
    assert settings.model.model_name == "test-model"
    assert settings.runtime.seed == 7
    assert settings.runtime.max_turns == 10
    assert settings.runtime.dry_run is False
    assert settings.runtime.debug_output is True
