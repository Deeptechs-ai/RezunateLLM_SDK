"""Tests for the command line."""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from rezunate_guard import __version__, cli, constants


@pytest.fixture
def home(tmp_path, monkeypatch):
    """A fake home for both `~/.claude` and `~/.rezunate`."""
    claude = tmp_path / "claude"
    claude.mkdir()
    monkeypatch.setenv(constants.CLAUDE_CONFIG_DIR_ENV, str(claude))
    monkeypatch.setenv(constants.HOME_ENV, str(tmp_path / "rezunate"))
    return tmp_path


@pytest.fixture
def project(tmp_path, monkeypatch):
    workdir = tmp_path / "work"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    return workdir


def settings(home) -> dict:
    path = home / "claude" / "settings.json"
    return json.loads(path.read_text()) if path.exists() else {}


class TestInstall:
    def test_registers_the_hook(self, home, project):
        assert cli.main(["install"]) == 0
        entries = settings(home)["hooks"]["PostToolUse"]
        assert entries[0]["matcher"] == "*"
        assert "rezunate_guard" in entries[0]["hooks"][0]["command"]

    def test_the_command_does_not_rely_on_PATH(self, home, project):
        """Installed as a library, the script lands in a venv `bin` that Claude Code may
        not have on PATH — and a hook it cannot launch leaves the file unredacted."""
        cli.main(["install"])
        command = settings(home)["hooks"]["PostToolUse"][0]["hooks"][0]["command"]

        interpreter = command.split()[0]
        assert Path(interpreter).is_absolute()
        assert Path(interpreter).exists()
        assert command.endswith("-m rezunate_guard hook")

    def test_the_registered_command_actually_runs(self, home, project):
        """The command is only useful if it starts and answers on stdout."""
        cli.main(["install"])
        command = settings(home)["hooks"]["PostToolUse"][0]["hooks"][0]["command"]

        payload = {"hook_event_name": "PostToolUse", "tool_response": {"stdout": "hello"}}
        finished = subprocess.run(
            command.split(), input=json.dumps(payload), capture_output=True, text=True
        )
        assert finished.returncode == 0, finished.stderr

    def test_leaves_unrelated_settings_alone(self, home, project):
        path = home / "claude" / "settings.json"
        path.write_text(json.dumps({"theme": "dark", "tui": "fullscreen"}))

        cli.main(["install"])
        after = settings(home)
        assert after["theme"] == "dark"
        assert after["tui"] == "fullscreen"

    def test_leaves_other_peoples_hooks_alone(self, home, project):
        path = home / "claude" / "settings.json"
        path.write_text(
            json.dumps(
                {
                    "hooks": {
                        "PostToolUse": [
                            {"matcher": "Write", "hooks": [{"type": "command", "command": "other"}]}
                        ],
                        "PreToolUse": [
                            {"matcher": "Bash", "hooks": [{"type": "command", "command": "audit"}]}
                        ],
                    }
                }
            )
        )

        cli.main(["install"])
        hooks = settings(home)["hooks"]
        assert len(hooks["PostToolUse"]) == 2
        assert hooks["PreToolUse"][0]["hooks"][0]["command"] == "audit"

    def test_installing_twice_is_harmless(self, home, project):
        cli.main(["install"])
        cli.main(["install"])
        assert len(settings(home)["hooks"]["PostToolUse"]) == 1

    def test_project_scope_writes_locally(self, home, project):
        cli.main(["install", "--scope", "project"])
        assert (project / ".claude" / "settings.json").exists()
        assert settings(home) == {}

    def test_refuses_a_malformed_hooks_block(self, home, project, capsys):
        (home / "claude" / "settings.json").write_text(json.dumps({"hooks": "nonsense"}))
        assert cli.main(["install"]) == 1
        assert "not an object" in capsys.readouterr().err


class TestUninstall:
    def test_removes_our_hook(self, home, project):
        cli.main(["install"])
        assert cli.main(["uninstall"]) == 0
        assert "hooks" not in settings(home)

    def test_keeps_other_hooks(self, home, project):
        cli.main(["install"])
        current = settings(home)
        current["hooks"]["PostToolUse"].append(
            {"matcher": "Write", "hooks": [{"type": "command", "command": "other"}]}
        )
        (home / "claude" / "settings.json").write_text(json.dumps(current))

        cli.main(["uninstall"])
        remaining = settings(home)["hooks"]["PostToolUse"]
        assert len(remaining) == 1
        assert remaining[0]["hooks"][0]["command"] == "other"

    def test_uninstalling_when_absent_is_harmless(self, home, project):
        assert cli.main(["uninstall"]) == 0

    def test_preserves_unrelated_settings(self, home, project):
        (home / "claude" / "settings.json").write_text(json.dumps({"theme": "dark"}))
        cli.main(["install"])
        cli.main(["uninstall"])
        assert settings(home)["theme"] == "dark"


class TestInit:
    def test_writes_a_config(self, home, project):
        assert cli.main(["init"]) == 0
        assert (project / constants.CONFIG_FILENAME).exists()

    def test_refuses_to_clobber(self, home, project, capsys):
        (project / constants.CONFIG_FILENAME).write_text("scan:\n  - mine/**\n")
        assert cli.main(["init"]) == 1
        assert "already exists" in capsys.readouterr().err
        assert "mine" in (project / constants.CONFIG_FILENAME).read_text()

    def test_force_overwrites(self, home, project):
        (project / constants.CONFIG_FILENAME).write_text("scan:\n  - mine/**\n")
        assert cli.main(["init", "--force"]) == 0
        assert "mine" not in (project / constants.CONFIG_FILENAME).read_text()


class TestLogin:
    def test_saves_the_key_privately(self, home, project):
        assert cli.main(["login", "sk-test-123"]) == 0
        path = home / "rezunate" / "credentials"
        assert path.read_text().strip() == "sk-test-123"
        assert path.stat().st_mode & 0o077 == 0, "credentials must not be readable by others"

    def test_rejects_an_empty_key(self, home, project, monkeypatch, capsys):
        monkeypatch.setattr("builtins.input", lambda *a: "")
        assert cli.main(["login"]) == 1

    def test_overwrites_a_previous_key(self, home, project):
        cli.main(["login", "old"])
        cli.main(["login", "new"])
        assert (home / "rezunate" / "credentials").read_text().strip() == "new"


class TestStatus:
    """Every silent-failure mode has to be reported, and has to exit non-zero."""

    def test_fresh_install_reports_everything_missing(self, home, project, capsys):
        assert cli.main(["status"]) == 1
        out = capsys.readouterr().out
        assert "MISSING" in out
        assert "NOT INSTALLED" in out
        assert "rezunate-guard login" in out
        assert "rezunate-guard init" in out

    def test_reports_an_inert_config(self, home, project, capsys):
        cli.main(["login", "k"])
        cli.main(["install"])
        cli.main(["init"])

        assert cli.main(["status"]) == 1
        assert "nothing is protected" in capsys.readouterr().out.lower()

    def test_reports_a_broken_config(self, home, project, capsys):
        cli.main(["login", "k"])
        cli.main(["install"])
        (project / constants.CONFIG_FILENAME).write_text("scan: [unclosed\n")

        assert cli.main(["status"]) == 1
        assert "could not be read" in capsys.readouterr().out

    def test_a_working_setup_reports_active(self, home, project, capsys):
        cli.main(["login", "k"])
        cli.main(["install"])
        (project / "clients").mkdir(exist_ok=True)
        (project / constants.CONFIG_FILENAME).write_text("scan:\n  - clients\n")

        assert cli.main(["status"]) == 0
        out = capsys.readouterr().out
        assert "Protection is active" in out
        assert "clients" in out

    def test_an_env_key_counts(self, home, project, monkeypatch, capsys):
        monkeypatch.setenv(constants.API_KEY_ENV, "from-env")
        cli.main(["install"])
        (project / "clients").mkdir(exist_ok=True)
        (project / constants.CONFIG_FILENAME).write_text("scan:\n  - clients\n")

        assert cli.main(["status"]) == 0
        assert constants.API_KEY_ENV in capsys.readouterr().out


class TestCheck:
    def test_explains_each_path(self, home, project, capsys):
        (project / "clients").mkdir(exist_ok=True)
        (project / constants.CONFIG_FILENAME).write_text("scan:\n  - clients\n")
        assert cli.main(["check", "clients/a.md", "src/b.py"]) == 0

        out = capsys.readouterr().out
        assert "scan " in out
        assert "skip " in out
        assert "protected folder" in out


class TestHookEntryPoint:
    """`rezunate-guard hook` is what Claude Code actually runs. Every other hook test
    calls `rezunate_guard.hook` directly, so the shipped path had no coverage."""

    def test_the_hook_subcommand_runs_and_exits_zero(self, project, capsys):
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "cwd": str(project),
            "tool_input": {"file_path": str(project / "notes.md")},
            "tool_response": {
                "type": "text",
                "file": {"filePath": str(project / "notes.md"), "content": "hello"},
            },
        }
        with (
            patch("sys.stdin", io.StringIO(json.dumps(payload))),
            pytest.raises(SystemExit) as exit,
        ):
            cli.main(["hook"])
        assert exit.value.code == 0
        assert capsys.readouterr().out == "", "an unprotected file must not be rewritten"

    def test_version_prints_the_packaged_version(self, capsys):
        with pytest.raises(SystemExit) as exit:
            cli.main(["--version"])
        assert exit.value.code == 0
        assert capsys.readouterr().out.strip() == __version__
