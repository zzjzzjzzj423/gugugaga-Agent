from simple_cc import config
from simple_cc.__main__ import build_runtime
from simple_cc.config import Settings
from simple_cc.tools import TOOL_DEFINITIONS
from tests.fakes import ScriptedProvider


EXPECTED_TOOLS = {definition["name"] for definition in TOOL_DEFINITIONS}


def test_s01_through_s17_tools_are_registered(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    app = build_runtime(Settings.from_env(tmp_path), provider=ScriptedProvider())
    try:
        assert EXPECTED_TOOLS == {
            spec.name for spec in app.runtime.registry.specs()
        }
    finally:
        app.close()


def test_memory_content_is_searchable_and_visible_to_prompt_state(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    app = build_runtime(Settings.from_env(tmp_path), provider=ScriptedProvider())
    config.MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    config.MEMORY_INDEX.write_text(
        "Always use pathlib for filesystem paths", encoding="utf-8"
    )
    try:
        assert "pathlib" in app.runtime.state_builder()["memories"]
        assert "search_memory" not in {
            spec.name for spec in app.runtime.registry.specs()
        }
    finally:
        app.close()
