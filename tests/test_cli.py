from simple_cc.__main__ import build_runtime, create_parser, handle_command
from simple_cc.config import Settings
from simple_cc.models import ModelResponse
from tests.fakes import ScriptedProvider


def make_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    return Settings.from_env(tmp_path)


def test_parser_accepts_workspace_and_model(tmp_path):
    args = create_parser().parse_args(["--workspace", str(tmp_path), "--model", "demo"])
    assert args.workspace == str(tmp_path)
    assert args.model == "demo"


def test_build_runtime_and_status_command(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, monkeypatch)
    app = build_runtime(settings, provider=ScriptedProvider([ModelResponse("ok")]))
    handled, output = handle_command("/status", app)
    assert handled
    assert "Workspace" in output
    assert "test-model" in output


def test_exit_command_requests_shutdown(tmp_path, monkeypatch):
    app = build_runtime(make_settings(tmp_path, monkeypatch), provider=ScriptedProvider())
    handled, output = handle_command("/exit", app)
    assert handled and output == "__exit__"

