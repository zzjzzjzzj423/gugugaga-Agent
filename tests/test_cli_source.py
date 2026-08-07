from __future__ import annotations

from types import SimpleNamespace

import pytest

from simple_cc import agent, config, context, subagents, teams
from simple_cc import __main__ as cli
from simple_cc.config import Settings


ALL_DERIVED_PATHS = {
    "SKILLS_DIR": "skills",
    "TRANSCRIPT_DIR": ".transcripts",
    "TOOL_RESULTS_DIR": ".task_outputs/tool-results",
    "TASKS_DIR": ".tasks",
    "MAILBOX_DIR": ".mailboxes",
    "MEMORY_DIR": ".memory",
    "MEMORY_INDEX": ".memory/MEMORY.md",
    "DURABLE_PATH": ".scheduled_tasks.json",
}


class NoopThread:
    def __init__(self, *args, **kwargs):
        pass

    def start(self):
        pass


class FakeApp:
    def __init__(self):
        self.runtime = self
        self.run_count = 0
        self.close_count = 0

    def run_turn(self, query):
        self.run_count += 1
        return "unexpected"

    def close(self):
        self.close_count += 1


def configure_env(monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")


def scripted_input(*items):
    values = iter(items)

    def read(_prompt):
        item = next(values)
        if isinstance(item, BaseException):
            raise item
        return item

    return read


def test_configure_workspace_synchronizes_every_s20_derived_path(tmp_path):
    original = config.WORKDIR
    try:
        selected = config.configure_workspace(tmp_path / "workspace")
        assert config.WORKDIR == selected
        for name, relative in ALL_DERIVED_PATHS.items():
            assert getattr(config, name) == selected / relative
    finally:
        config.configure_workspace(original)


@pytest.mark.parametrize(
    ("command", "workspace_argument"),
    [("q", None), ("exit", "chosen"), ("/exit", "chosen")],
)
def test_main_selects_default_or_explicit_workspace_and_accepts_exit_aliases(
    tmp_path, monkeypatch, command, workspace_argument
):
    configure_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    app = FakeApp()
    captured = {}

    def fake_build(settings, approval_callback=None, provider=None):
        captured["settings"] = settings
        config.configure_workspace(settings.workspace)
        return app

    monkeypatch.setattr(cli, "build_runtime", fake_build)
    monkeypatch.setattr(cli.threading, "Thread", NoopThread)
    monkeypatch.setattr("builtins.input", scripted_input(command, EOFError()))
    argv = [] if workspace_argument is None else ["--workspace", str(tmp_path / workspace_argument)]

    assert cli.main(argv) == 0
    expected = tmp_path if workspace_argument is None else (tmp_path / workspace_argument)
    assert captured["settings"].workspace == expected.resolve()
    assert config.WORKDIR == expected.resolve()
    assert app.run_count == 0
    assert app.close_count == 1


def test_ctrl_c_closes_runtime(tmp_path, monkeypatch):
    configure_env(monkeypatch)
    app = FakeApp()
    monkeypatch.setattr(cli, "build_runtime", lambda *args, **kwargs: app)
    monkeypatch.setattr(cli.threading, "Thread", NoopThread)
    monkeypatch.setattr("builtins.input", scripted_input(KeyboardInterrupt()))

    assert cli.main(["--workspace", str(tmp_path)]) == 0
    assert app.close_count == 1


def test_build_runtime_installs_one_provider_at_all_source_boundaries(
    tmp_path, monkeypatch
):
    configure_env(monkeypatch)
    provider = object()
    settings = Settings.from_env(tmp_path)

    app = cli.build_runtime(settings, provider=provider)
    try:
        assert app.runtime.provider is provider
        assert agent.client is provider
        assert context.client is provider
        assert subagents.client is provider
        assert teams._team_provider is provider
    finally:
        app.close()


def test_missing_process_model_is_not_rehydrated_from_repository_dotenv(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text(
        "SILICONFLOW_API_KEY=dotenv-test-key\nSILICONFLOW_MODEL=dotenv-test-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("SILICONFLOW_API_KEY", "process-test-key")
    monkeypatch.delenv("SILICONFLOW_MODEL", raising=False)

    with pytest.raises(ValueError, match="SILICONFLOW_MODEL"):
        Settings.from_env(tmp_path)
