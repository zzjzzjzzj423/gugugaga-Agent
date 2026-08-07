from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

import pytest

from simple_cc import agent, config, context, cron, subagents, teams, tools
from simple_cc import __main__ as cli
from simple_cc.config import Settings
from simple_cc.models import ToolCall
from simple_cc.permissions import PermissionPolicy
from simple_cc.provider import ProviderResponse, TextBlock, ToolUseBlock


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
        self.messages = []
        self.context = {}
        self.permissions = PermissionPolicy()
        self.approval_callback = None
        self.stop_event = threading.Event()
        self.autorun_thread = None
        self.run_count = 0
        self.close_count = 0

    def run_turn(self, query):
        self.run_count += 1
        return "unexpected"

    def close(self):
        self.close_count += 1


class ContentBlockProvider:
    def __init__(self, responses=None):
        self.responses = list(responses or [])
        self.requests = []

    def create(self, messages, system, tools, max_tokens, model=None):
        self.requests.append(
            {
                "messages": copy.deepcopy(messages),
                "system": system,
                "tools": copy.deepcopy(tools),
                "max_tokens": max_tokens,
                "model": model,
            }
        )
        if self.responses:
            return self.responses.pop(0)
        return ProviderResponse(
            content=[TextBlock(text="Waiting for more work.")],
            stop_reason="end_turn",
        )


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


def test_importing_cli_does_not_load_cron_state_from_launch_directory(
    tmp_path,
):
    launch = tmp_path / "launch"
    launch.mkdir()
    (launch / ".scheduled_tasks.json").write_text(
        json.dumps(
            [
                {
                    "id": "launch_job",
                    "cron": "* * * * *",
                    "prompt": "must not load before workspace selection",
                    "recurring": True,
                    "durable": True,
                }
            ]
        ),
        encoding="utf-8",
    )
    environment = os.environ.copy()
    project_root = str(Path(__file__).resolve().parents[1])
    environment["PYTHONPATH"] = os.pathsep.join(
        [project_root, environment.get("PYTHONPATH", "")]
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import json; import simple_cc.__main__; "
                "from simple_cc import cron; "
                "print(json.dumps(sorted(cron.scheduled_jobs)))"
            ),
        ],
        cwd=launch,
        env=environment,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )

    assert json.loads(result.stdout.strip()) == []


def test_build_runtime_initializes_cron_after_workspace_selection(
    tmp_path, monkeypatch
):
    configure_env(monkeypatch)
    selected = tmp_path / "selected"
    settings = Settings.from_env(selected)
    observed = []
    real_configure = config.configure_workspace

    def configure(workspace):
        observed.append(("configure", Path(workspace).resolve()))
        return real_configure(workspace)

    def initialize():
        observed.append(("initialize", config.WORKDIR))

    assert callable(getattr(cli, "initialize_cron", None)), (
        "runtime construction must explicitly initialize cron"
    )
    monkeypatch.setattr(config, "configure_workspace", configure)
    monkeypatch.setattr(cli, "initialize_cron", initialize)

    app = cli.build_runtime(settings, provider=ContentBlockProvider())
    try:
        assert observed[:2] == [
            ("configure", selected.resolve()),
            ("initialize", selected.resolve()),
        ]
    finally:
        app.close()


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


def test_production_runtime_uses_fixed_registry_and_s20_tool_history(
    tmp_path, monkeypatch
):
    configure_env(monkeypatch)
    provider = ContentBlockProvider(
        [
            ProviderResponse(
                content=[
                    ToolUseBlock(
                        id="toolu_write",
                        name="write_file",
                        input={"path": "runtime.txt", "content": "fixed"},
                    )
                ],
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content=[TextBlock(text="Done.")],
                stop_reason="end_turn",
            ),
        ]
    )
    app = cli.build_runtime(Settings.from_env(tmp_path), provider=provider)
    try:
        assert app.runtime.run_turn("Create runtime.txt") == "Done."
        assert (tmp_path / "runtime.txt").read_text() == "fixed"
        assert {spec.name for spec in app.runtime.registry.specs()} == {
            definition["name"] for definition in tools.TOOL_DEFINITIONS
        }
        second_history = provider.requests[1]["messages"]
        assert second_history[-1] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_write",
                    "content": "Wrote 5 bytes to runtime.txt",
                }
            ],
        }
        assert not any(message.get("role") == "tool" for message in second_history)
        assert not any("tool_calls" in message for message in second_history)
    finally:
        app.close()


def test_production_runtime_denial_blocks_fixed_bash_handler_and_reaches_model(
    tmp_path, monkeypatch
):
    configure_env(monkeypatch)
    provider = ContentBlockProvider(
        [
            ProviderResponse(
                content=[
                    ToolUseBlock(
                        id="toolu_bash",
                        name="bash",
                        input={
                            "command": "echo bypassed> permission-marker.txt"
                        },
                    )
                ],
                stop_reason="tool_use",
            ),
            ProviderResponse(
                content=[TextBlock(text="Used a safer approach.")],
                stop_reason="end_turn",
            ),
        ]
    )
    approval_requests: list[ToolCall] = []

    def deny(call: ToolCall) -> bool:
        approval_requests.append(call)
        return False

    app = cli.build_runtime(
        Settings.from_env(tmp_path),
        approval_callback=deny,
        provider=provider,
    )
    try:
        assert app.runtime.run_turn("Create the marker with bash") == (
            "Used a safer approach."
        )
        assert not (tmp_path / "permission-marker.txt").exists()
        assert approval_requests == [
            ToolCall(
                "toolu_bash",
                "bash",
                {"command": "echo bypassed> permission-marker.txt"},
            )
        ]
        assert provider.requests[1]["messages"][-1] == {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "toolu_bash",
                    "content": (
                        "Permission denied for tool 'bash'. "
                        "Choose a safer approach."
                    ),
                }
            ],
        }
    finally:
        app.close()


@pytest.mark.parametrize("exit_input", ["q", KeyboardInterrupt()])
def test_cli_exit_stops_and_joins_source_teammates(
    tmp_path, monkeypatch, exit_input
):
    configure_env(monkeypatch)
    monkeypatch.setattr(teams, "IDLE_POLL_INTERVAL", 1)
    monkeypatch.setattr(teams, "IDLE_TIMEOUT", 60)
    provider = ContentBlockProvider()
    original_build = cli.build_runtime
    captured = {}
    teammate_name = f"exit-worker-{type(exit_input).__name__}"

    def build(settings, approval_callback=None, provider_override=None):
        app = original_build(settings, provider=provider)
        assert teams.spawn_teammate_thread(
            teammate_name, "developer", "Wait for work"
        ).startswith("Teammate")
        deadline = time.monotonic() + 2
        while not any(
            thread.name == f"teammate-{teammate_name}"
            for thread in threading.enumerate()
        ):
            assert time.monotonic() < deadline
            time.sleep(0.01)
        captured["thread"] = next(
            thread
            for thread in threading.enumerate()
            if thread.name == f"teammate-{teammate_name}"
        )
        return app

    monkeypatch.setattr(cli, "build_runtime", build)
    monkeypatch.setattr("builtins.input", scripted_input(exit_input))

    try:
        assert cli.main(["--workspace", str(tmp_path)]) == 0
        assert callable(getattr(teams, "stop_all_teammates", None)), (
            "source teammate lifecycle must expose stop_all_teammates"
        )
        assert not captured["thread"].is_alive()
        assert teammate_name not in teams.active_teammates
    finally:
        if captured.get("thread") and captured["thread"].is_alive():
            teams.BUS.send(
                "lead",
                teammate_name,
                "Shut down.",
                "shutdown_request",
                {"request_id": "test-cleanup"},
            )
            captured["thread"].join(timeout=2)


class BlockingLateWriteProvider:
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.requests = []

    def create(self, messages, system, tools, max_tokens, model=None):
        self.requests.append(copy.deepcopy(messages))
        self.entered.set()
        assert self.release.wait(timeout=3)
        return ProviderResponse(
            content=[
                ToolUseBlock(
                    id="toolu_cli_late_write",
                    name="write_file",
                    input={"path": "cli-late-write.txt", "content": "too late"},
                )
            ],
            stop_reason="tool_use",
        )


@pytest.mark.parametrize("exit_input", ["q", KeyboardInterrupt()])
def test_cli_exit_during_provider_call_discards_late_tool_response(
    tmp_path, monkeypatch, exit_input
):
    configure_env(monkeypatch)
    provider = BlockingLateWriteProvider()
    original_build = cli.build_runtime
    captured = {}

    def build(settings, approval_callback=None, provider_override=None):
        del provider_override
        app = original_build(
            settings,
            approval_callback=approval_callback,
            provider=provider,
        )
        assert teams.spawn_teammate_thread(
            "late-cli-worker", "developer", "Wait before writing"
        ).startswith("Teammate")
        captured["app"] = app
        return app

    def exit_during_provider(_prompt):
        assert provider.entered.wait(timeout=1)

        def release_after_shutdown():
            assert teams._teammate_stop_event.wait(timeout=2)
            provider.release.set()

        captured["releaser"] = threading.Thread(target=release_after_shutdown)
        captured["releaser"].start()
        if isinstance(exit_input, BaseException):
            raise exit_input
        return exit_input

    monkeypatch.setattr(cli, "build_runtime", build)
    monkeypatch.setattr("builtins.input", exit_during_provider)

    try:
        assert cli.main(["--workspace", str(tmp_path)]) == 0
        captured["releaser"].join(timeout=2)
        assert not captured["releaser"].is_alive()
        assert not (tmp_path / "cli-late-write.txt").exists()
        assert len(provider.requests) == 1
        assert captured["app"]._close_outcome.stopped
        assert captured["app"]._close_outcome.live_threads == ()
    finally:
        provider.release.set()
        if captured.get("app"):
            captured["app"].close()


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
