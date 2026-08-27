import pytest

from gugugaga.__main__ import build_runtime, create_parser, handle_command, main
from gugugaga.config import Settings
from gugugaga.context_modes import ContextModeError
from gugugaga.models import ModelResponse
from tests.fakes import ScriptedProvider


def make_settings(tmp_path, monkeypatch):
    monkeypatch.setenv("SILICONFLOW_API_KEY", "test-key")
    monkeypatch.setenv("SILICONFLOW_MODEL", "test-model")
    return Settings.from_env(tmp_path)


def test_parser_accepts_workspace_and_model(tmp_path):
    args = create_parser().parse_args([
        "--workspace", str(tmp_path), "--model", "demo",
        "--context-mode", "pi", "--context-window-tokens", "262144",
    ])
    assert args.workspace == str(tmp_path)
    assert args.model == "demo"
    assert args.context_mode == "pi"
    assert args.context_window_tokens == 262_144


def test_build_runtime_and_status_command(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, monkeypatch)
    app = build_runtime(settings, provider=ScriptedProvider([ModelResponse("ok")]))
    handled, output = handle_command("/status", app)
    assert handled
    assert "Workspace" in output
    assert "test-model" in output
    assert "Context mode: CC" in output
    assert "Successful compactions: 0" in output


def test_exit_command_requests_shutdown(tmp_path, monkeypatch):
    app = build_runtime(make_settings(tmp_path, monkeypatch), provider=ScriptedProvider())
    handled, output = handle_command("/exit", app)
    assert handled and output == "__exit__"


@pytest.mark.parametrize("mode", ["cc", "hermes", "pi"])
def test_build_runtime_selects_and_reports_each_context_mode(
    tmp_path, monkeypatch, mode
):
    app = build_runtime(
        make_settings(tmp_path / mode, monkeypatch),
        provider=ScriptedProvider([ModelResponse("ok")]),
        context_mode=mode,
    )
    try:
        assert app.runtime.context_status()["mode"] == mode
    finally:
        app.close()


def test_build_runtime_rejects_unknown_context_mode(tmp_path, monkeypatch):
    with pytest.raises(ContextModeError) as invalid:
        build_runtime(
            make_settings(tmp_path, monkeypatch),
            provider=ScriptedProvider(),
            context_mode="typo",
        )
    assert invalid.value.code == "INVALID_CONTEXT_MODE"


def test_cli_reports_stable_error_for_invalid_context_mode(
    tmp_path, monkeypatch, capsys
):
    make_settings(tmp_path, monkeypatch)
    assert main(["--workspace", str(tmp_path), "--context-mode", "CC"]) == 2
    assert "INVALID_CONTEXT_MODE" in capsys.readouterr().err

