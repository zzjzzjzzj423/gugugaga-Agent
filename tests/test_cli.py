import pytest

import gugugaga.__main__ as cli
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


@pytest.mark.parametrize("flag", ["--context-threshold-ratio", "--hermes-threshold-ratio"])
def test_parser_accepts_shared_compaction_ratio_alias(flag):
    args = create_parser().parse_args([flag, "0.6"])
    assert args.hermes_threshold_ratio == 0.6


def test_build_runtime_and_status_command(tmp_path, monkeypatch):
    settings = make_settings(tmp_path, monkeypatch)
    app = build_runtime(settings, provider=ScriptedProvider([ModelResponse("ok")]))
    handled, output = handle_command("/status", app)
    assert handled
    assert "Workspace" in output
    assert "test-model" in output
    assert "Context mode: CC" in output
    assert "Automatic summary trigger: 98304 tokens (75%" in output
    assert "Successful compactions: 0" in output


@pytest.mark.parametrize("mode", ["cc", "hermes", "pi"])
def test_discovered_window_sets_shared_threshold(tmp_path, monkeypatch, mode):
    settings = make_settings(tmp_path, monkeypatch)

    class FakeSiliconFlow(ScriptedProvider):
        pass

    monkeypatch.setattr(cli, "SiliconFlowProvider", FakeSiliconFlow)
    monkeypatch.setattr(
        cli, "resolve_context_window",
        lambda provider, model, path: (1_000_000, "models.dev"),
    )
    app = build_runtime(settings, provider=FakeSiliconFlow(), context_mode=mode)
    try:
        status = app.runtime.context_status()
        assert status["context_window_tokens"] == 1_000_000
        assert status["automatic_trigger_tokens"] == 500_000
        assert status["context_window_source"] == "models.dev"
    finally:
        app.close()


def test_manual_window_skips_catalog_lookup(tmp_path, monkeypatch):
    monkeypatch.setenv("GUGUGAGA_CONTEXT_WINDOW_TOKENS", "262144")
    settings = make_settings(tmp_path, monkeypatch)

    def unexpected(*args):
        raise AssertionError("catalog must not be fetched")

    monkeypatch.setattr(cli, "resolve_context_window", unexpected)
    app = build_runtime(settings, provider=ScriptedProvider())
    try:
        status = app.runtime.context_status()
        assert status["context_window_tokens"] == 262_144
        assert status["automatic_trigger_tokens"] == 196_608
        assert status["context_window_source"] == "environment"
    finally:
        app.close()


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

