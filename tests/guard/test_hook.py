"""Tests for the hook, and mostly for the ways it is allowed to fail."""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from rezunate_guard import constants, hook
from rezunate_guard.scanner import ScanError


@pytest.fixture
def project(tmp_path):
    """A project protecting `clients/`, with one file inside and one outside."""
    (tmp_path / constants.CONFIG_FILENAME).write_text("scan:\n  - clients\n", encoding="utf-8")
    (tmp_path / "clients").mkdir()
    (tmp_path / "clients/acme.md").write_text("Ali Hassan, +971 50 123 4567", encoding="utf-8")
    (tmp_path / "notes.md").write_text("nothing sensitive", encoding="utf-8")
    return tmp_path


def read_payload(path: Path, content: str) -> dict:
    """A PostToolUse payload shaped the way Claude Code really sends one."""
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": "Read",
        "tool_input": {"file_path": str(path)},
        "tool_response": {
            "type": "text",
            "file": {"filePath": str(path), "content": content, "numLines": 1},
        },
    }


def updated_file(reply: dict) -> dict:
    return reply["hookSpecificOutput"]["updatedToolOutput"]["file"]


class TestPassThrough:
    """Files we don't protect must come back untouched, and cost nothing."""

    def test_unprotected_file_is_left_alone(self, project):
        payload = read_payload(project / "notes.md", "nothing sensitive")
        assert hook.respond(payload) is None

    def test_other_hook_events_are_ignored(self, project):
        payload = read_payload(project / "clients/acme.md", "secret")
        payload["hook_event_name"] = "PreToolUse"
        assert hook.respond(payload) is None

    def test_unknown_response_shape_on_an_unprotected_path_is_ignored(self, project):
        payload = read_payload(project / "notes.md", "hello")
        payload["tool_response"] = {"stdout": "hello", "stderr": ""}
        assert hook.respond(payload) is None

    def test_unprotected_file_never_calls_the_redactor(self, project, monkeypatch):
        def explode(texts):
            raise AssertionError("redactor ran on an unprotected file")

        monkeypatch.setattr(hook, "redact_all", explode)
        assert hook.respond(read_payload(project / "notes.md", "hello")) is None


class TestEveryReadPath:
    """The same protected file, reached by tools other than Read.

    This is the gap that shipped: matching on Read alone meant `cat` returned the file
    in full, and nothing in the session said so.
    """

    def bash_payload(self, command: str, stdout: str, cwd: str | None = None) -> dict:
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": command, "description": "read a file"},
            "tool_response": {
                "stdout": stdout,
                "stderr": "",
                "interrupted": False,
                "isImage": False,
            },
        }
        if cwd is not None:
            payload["cwd"] = cwd
        return payload

    def test_cat_on_a_protected_file_is_redacted(self, project, monkeypatch):
        monkeypatch.setattr(hook, "redact_all", lambda texts: dict.fromkeys(texts, "[NAME]"))
        payload = self.bash_payload(f"cat {project / 'clients/acme.md'}", "Ali Hassan")
        updated = hook.respond(payload)["hookSpecificOutput"]["updatedToolOutput"]
        assert updated["stdout"] == "[NAME]"

    def test_a_relative_path_resolves_against_the_session_cwd(self, project, monkeypatch):
        monkeypatch.setattr(hook, "redact_all", lambda texts: dict.fromkeys(texts, "[NAME]"))
        payload = self.bash_payload("cat clients/acme.md", "Ali Hassan", cwd=str(project))
        updated = hook.respond(payload)["hookSpecificOutput"]["updatedToolOutput"]
        assert updated["stdout"] == "[NAME]"

    def test_bash_shape_survives_intact(self, project, monkeypatch):
        """A reply that fails Bash's schema is discarded, and the raw stdout stands."""
        monkeypatch.setattr(hook, "redact_all", lambda texts: dict.fromkeys(texts, "[NAME]"))
        payload = self.bash_payload(f"cat {project / 'clients/acme.md'}", "Ali Hassan")
        updated = hook.respond(payload)["hookSpecificOutput"]["updatedToolOutput"]
        assert set(updated) == {"stdout", "stderr", "interrupted", "isImage"}
        assert updated["interrupted"] is False

    def test_stderr_is_redacted_too(self, project, monkeypatch):
        monkeypatch.setattr(hook, "redact_all", lambda texts: dict.fromkeys(texts, "[NAME]"))
        payload = self.bash_payload(f"grep x {project / 'clients/acme.md'}", "")
        payload["tool_response"]["stderr"] = "Ali Hassan"
        updated = hook.respond(payload)["hookSpecificOutput"]["updatedToolOutput"]
        assert updated["stderr"] == "[NAME]"

    def test_a_glob_protects_the_directory_it_names(self, project, monkeypatch):
        monkeypatch.setattr(hook, "redact_all", lambda texts: dict.fromkeys(texts, "[NAME]"))
        payload = self.bash_payload(f"head {project / 'clients'}/*.md", "Ali Hassan")
        updated = hook.respond(payload)["hookSpecificOutput"]["updatedToolOutput"]
        assert updated["stdout"] == "[NAME]"

    def test_a_command_touching_nothing_protected_is_left_alone(self, project, monkeypatch):
        def explode(texts):
            raise AssertionError("redactor ran on an unprotected command")

        monkeypatch.setattr(hook, "redact_all", explode)
        assert hook.respond(self.bash_payload("ls -la /tmp", "file.txt")) is None

    def test_a_bare_string_response_is_redacted(self, project, monkeypatch):
        """Some tools answer with a string rather than an object."""
        monkeypatch.setattr(hook, "redact_all", lambda texts: dict.fromkeys(texts, "[NAME]"))
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__files__read",
            "tool_input": {"path": str(project / "clients/acme.md")},
            "tool_response": "Ali Hassan",
        }
        updated = hook.respond(payload)["hookSpecificOutput"]["updatedToolOutput"]
        assert updated == "[NAME]"

    def test_nested_shapes_are_reached(self, project, monkeypatch):
        """An MCP server can nest content arbitrarily. We don't know its schema."""
        monkeypatch.setattr(hook, "redact_all", lambda texts: dict.fromkeys(texts, "[NAME]"))
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__files__read",
            "tool_input": {"path": str(project / "clients/acme.md")},
            "tool_response": {"content": [{"type": "text", "text": "Ali Hassan"}]},
        }
        updated = hook.respond(payload)["hookSpecificOutput"]["updatedToolOutput"]
        assert updated["content"][0]["text"] == "[NAME]"
        assert updated["content"][0]["type"] == "text"  # structure, not content

    def test_a_tag_key_holding_prose_is_still_redacted(self, project, monkeypatch):
        """`type` is skipped as a schema discriminator. A sentence under it is not one,
        and skipping it there would hand the model whatever it holds."""
        monkeypatch.setattr(hook, "redact_all", lambda texts: dict.fromkeys(texts, "[NAME]"))
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__x__read",
            "tool_input": {"path": str(project / "clients/acme.md")},
            "tool_response": {"type": "Ali Hassan, +971 50 123 4567"},
        }
        updated = hook.respond(payload)["hookSpecificOutput"]["updatedToolOutput"]
        assert updated["type"] == "[NAME]"

    def test_a_real_discriminator_survives_at_any_depth(self, project, monkeypatch):
        """Rewriting one fails schema validation, which sends the whole unredacted
        response through — a worse leak than the tag could ever be."""
        monkeypatch.setattr(hook, "redact_all", lambda texts: dict.fromkeys(texts, "[NAME]"))
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": "mcp__x__read",
            "tool_input": {"path": str(project / "clients/acme.md")},
            "tool_response": {"a": {"b": {"c": [{"type": "text", "text": "Ali Hassan"}]}}},
        }
        updated = hook.respond(payload)["hookSpecificOutput"]["updatedToolOutput"]
        assert updated["a"]["b"]["c"][0]["type"] == "text"
        assert updated["a"]["b"]["c"][0]["text"] == "[NAME]"

    def test_a_failed_scan_withholds_bash_output(self, project, monkeypatch):
        """Fail-closed has to hold for shapes other than Read's."""
        payload = self.bash_payload(f"cat {project / 'clients/acme.md'}", "Ali Hassan")
        reply = run_hook(payload)  # no API key, so the scan cannot run
        updated = reply["hookSpecificOutput"]["updatedToolOutput"]
        assert "Ali Hassan" not in json.dumps(reply)
        assert "withheld" in updated["stdout"]
        assert set(updated) == {"stdout", "stderr", "interrupted", "isImage"}


class TestPathsInACommand:
    """The heuristic that finds protected files in a shell command. It had no direct
    test, which is how a branch that could never run went unnoticed."""

    @pytest.mark.parametrize(
        ("command", "expected"),
        [
            ("cat clients/acme.md", ["clients/acme.md"]),
            ("head -n 20 clients/acme.md", ["clients/acme.md"]),
            ("cat notes.md", ["notes.md"]),
            ("grep --file=clients/pat.txt x", ["clients/pat.txt"]),
            ("FOO=clients/a.md cat $FOO", ["clients/a.md"]),
            ("cat clients/a.md; head clients/b.md", ["clients/a.md", "clients/b.md"]),
            ("cat 'clients/a.md'", ["clients/a.md"]),
            ("curl https://example.com/x.md", []),
            ("ls -la", []),
            ("echo hello", []),
        ],
    )
    def test_paths_are_picked_out_of_a_command(self, command, expected):
        assert hook._paths_in_command(command) == expected

    def test_a_flag_is_not_mistaken_for_a_path(self):
        assert hook._paths_in_command("grep -rn --color=never x") == []

    def test_a_path_built_at_runtime_is_not_found(self):
        """A known hole, recorded so it stays a decision rather than a surprise."""
        assert hook._paths_in_command('cat "$FILE"') == []


class TestContentThatArrivesOutsideTheResult:
    """Files whose contents never appear in the tool result as text.

    A PDF arrives as page images in a message of its own, invisible to PostToolUse, so
    withholding the result there is honest and useless. These are denied before the call
    runs — decided by reading the file, not by matching an extension.
    """

    def pre_payload(self, path, tool: str = "Read") -> dict:
        return {
            "hook_event_name": "PreToolUse",
            "tool_name": tool,
            "tool_input": {"file_path": str(path)},
        }

    def denial(self, reply: dict) -> str:
        assert reply["hookSpecificOutput"]["hookEventName"] == "PreToolUse"
        assert reply["hookSpecificOutput"]["permissionDecision"] == "deny"
        return reply["hookSpecificOutput"]["permissionDecisionReason"]

    def test_a_protected_pdf_is_denied_before_it_runs(self, project):
        pdf = project / "clients/resume.pdf"
        pdf.write_bytes(b"%PDF-1.7\n\x00\x01binary junk")
        reason = self.denial(hook.respond(self.pre_payload(pdf)))
        assert "resume.pdf" in reason

    def test_a_protected_binary_of_no_known_format_is_denied(self, project):
        """Nothing here knows what a .dat is, and it doesn't need to."""
        blob = project / "clients/export.dat"
        blob.write_bytes(bytes(range(256)) * 4)
        self.denial(hook.respond(self.pre_payload(blob)))

    def test_a_protected_docx_is_denied(self, project):
        """A zip container, so binary — covered without naming the format."""
        docx = project / "clients/cv.docx"
        docx.write_bytes(b"PK\x03\x04\x14\x00\x00\x00\x08\x00binary")
        self.denial(hook.respond(self.pre_payload(docx)))

    def test_a_protected_text_file_is_allowed_through_to_redaction(self, project):
        """Denying text files would replace redaction with refusal. It stays redaction."""
        assert hook.respond(self.pre_payload(project / "clients/acme.md")) is None

    def test_utf8_text_is_not_mistaken_for_binary(self, project):
        arabic = project / "clients/notes-ar.md"
        arabic.write_text("علي حسن، هاتف ٠٥٠١٢٣٤٥٦٧\n" * 400, encoding="utf-8")
        assert hook.respond(self.pre_payload(arabic)) is None

    def test_a_character_split_by_the_sniff_boundary_is_still_text(self, project):
        """The sample is a fixed byte count, so a multi-byte character can be cut in
        half at its end — which decodes as an error that says nothing about the file.

        Byte 8192 lands mid-character here by construction. Without the retry the file
        reads as binary and a perfectly ordinary text file is refused.
        """
        straddled = project / "clients/wide.md"
        straddled.write_text("a" * (hook.SNIFF_BYTES - 2) + "€€", encoding="utf-8")

        sample = straddled.read_bytes()[: hook.SNIFF_BYTES]
        with pytest.raises(UnicodeDecodeError):
            sample.decode("utf-8")  # the test is worthless if this ever stops raising

        assert hook._reaches_the_model_as_text(str(straddled)) is True
        assert hook.respond(self.pre_payload(straddled)) is None

    def test_a_dense_two_byte_script_is_still_text(self, project):
        """Arabic is two bytes a character, so one ASCII byte in front moves every
        boundary. Trimming a fixed number of bytes lands mid-character again and calls
        the file binary, which would deny a perfectly ordinary Arabic document."""
        notes = project / "clients/dense-ar.md"
        notes.write_text("a" + "\u0628" * 5000, encoding="utf-8")

        sample = notes.read_bytes()[: hook.SNIFF_BYTES]
        with pytest.raises(UnicodeDecodeError):
            sample.decode("utf-8")
        with pytest.raises(UnicodeDecodeError):
            sample[:-4].decode("utf-8")  # trimming a fixed four bytes is not enough here

        assert hook._reaches_the_model_as_text(str(notes)) is True
        assert hook.respond(self.pre_payload(notes)) is None

    def test_an_unprotected_pdf_is_left_alone(self, project):
        pdf = project / "brochure.pdf"
        pdf.write_bytes(b"%PDF-1.7\n\x00binary")
        assert hook.respond(self.pre_payload(pdf)) is None

    @pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode 000 file")
    def test_a_protected_file_we_cannot_read_is_denied(self, project):
        """If we can't tell what it holds, the safe answer is the one that stops it."""
        locked = project / "clients/locked.md"
        locked.write_text("Ali Hassan", encoding="utf-8")
        locked.chmod(0o000)
        try:
            self.denial(hook.respond(self.pre_payload(locked)))
        finally:
            locked.chmod(0o600)

    def test_cat_on_a_protected_pdf_is_denied_too(self, project):
        pdf = project / "clients/resume.pdf"
        pdf.write_bytes(b"%PDF-1.7\n\x00binary")
        payload = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": f"cat {pdf}"},
        }
        self.denial(hook.respond(payload))

    @pytest.mark.parametrize("pattern", ["*.pdf", "?.pdf", "[rs].pdf", "r.*"])
    def test_a_wildcard_does_not_get_a_protected_pdf_past_the_gate(self, project, pattern):
        """A glob is not a file, so `os.path.isfile` used to wave every one of these
        straight through — defeating the only protection PreToolUse offers."""
        (project / "clients/r.pdf").write_bytes(b"%PDF-1.7\n\x00binary")
        payload = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": f"cat {project / 'clients'}/{pattern}"},
        }
        self.denial(hook.respond(payload))

    def test_a_wildcard_outside_a_protected_directory_still_runs(self, project):
        (project / "public.pdf").write_bytes(b"%PDF-1.7\n\x00binary")
        payload = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": f"cat {project}/*.pdf"},
        }
        assert hook.respond(payload) is None

    def test_a_directory_is_not_denied(self, project):
        """`ls` on a protected directory returns text; PostToolUse handles it."""
        payload = {
            "hook_event_name": "PreToolUse",
            "tool_name": "Bash",
            "tool_input": {"command": f"ls -la {project / 'clients'}"},
        }
        assert hook.respond(payload) is None

    def test_an_unexpected_failure_denies_rather_than_withholds(self, project, monkeypatch, capsys):
        """A withheld reply is ignored by PreToolUse, so failing closed means denying."""

        def boom(payload):
            raise RuntimeError("config exploded")

        monkeypatch.setattr(hook, "_before_tool", boom)
        payload = json.dumps(self.pre_payload(project / "clients/acme.md"))
        monkeypatch.setattr(sys, "stdin", io.StringIO(payload))

        with pytest.raises(SystemExit) as exit_info:
            hook.main()

        assert exit_info.value.code == 0  # anything else and Claude Code runs the tool
        reply = json.loads(capsys.readouterr().out)
        assert reply["hookSpecificOutput"]["permissionDecision"] == "deny"


class TestRedaction:
    def test_protected_content_is_replaced(self, project, monkeypatch):
        monkeypatch.setattr(
            hook, "redact_all", lambda texts: dict.fromkeys(texts, "[NAME], [PHONE]")
        )
        reply = hook.respond(read_payload(project / "clients/acme.md", "Ali Hassan, +971"))
        assert updated_file(reply)["content"] == "[NAME], [PHONE]"

    def test_reply_names_the_right_event(self, project, monkeypatch):
        monkeypatch.setattr(hook, "redact_all", lambda texts: dict.fromkeys(texts, "clean"))
        reply = hook.respond(read_payload(project / "clients/acme.md", "dirty"))
        assert reply["hookSpecificOutput"]["hookEventName"] == "PostToolUse"

    def test_surrounding_fields_survive_untouched(self, project, monkeypatch):
        """Claude Code checks our reply against Read's schema; drop a field and it
        discards the whole thing and uses the original — unredacted — output."""
        monkeypatch.setattr(hook, "redact_all", lambda texts: dict.fromkeys(texts, "clean"))
        payload = read_payload(project / "clients/acme.md", "dirty")
        file = updated_file(hook.respond(payload))
        assert file["filePath"] == str(project / "clients/acme.md")
        assert file["numLines"] == 1

    def test_original_payload_is_not_mutated(self, project, monkeypatch):
        monkeypatch.setattr(hook, "redact_all", lambda texts: dict.fromkeys(texts, "clean"))
        payload = read_payload(project / "clients/acme.md", "dirty")
        hook.respond(payload)
        assert payload["tool_response"]["file"]["content"] == "dirty"


class TestFailClosed:
    """When we can't redact, the content must not go through."""

    def test_redactor_failure_withholds_content(self, project, monkeypatch):
        def boom(texts):
            raise RuntimeError("API unreachable")

        monkeypatch.setattr(hook, "redact_all", boom)
        payload = read_payload(project / "clients/acme.md", "Ali Hassan")
        with pytest.raises(RuntimeError):
            hook.respond(payload)

        # main() is the layer that catches it, so check the whole path.
        reply = run_hook(payload, redactor_raises=True)
        assert "Ali Hassan" not in json.dumps(reply)
        assert "withheld" in updated_file(reply)["content"]

    def test_missing_credentials_withhold_rather_than_leak(self, project):
        """No API key means no scan. The file must not go through unscanned."""
        payload = read_payload(project / "clients/acme.md", "Ali Hassan")
        reply = run_hook(payload)
        content = updated_file(reply)["content"]
        assert "Ali Hassan" not in json.dumps(reply)
        assert "withheld" in content
        # The notice reaches the model, so it has to say what the user should do.
        assert "rezunate-guard login" in content

    def test_an_unexpected_failure_does_not_quote_the_file(self, project, monkeypatch):
        """Our own errors are safe to show. A stray exception might carry file content
        in its message, so only the type gets reported."""

        def boom(texts):
            raise RuntimeError(f"failed while handling: {texts}")

        monkeypatch.setattr(hook, "redact_all", boom)
        payload = read_payload(project / "clients/acme.md", "Ali Hassan")
        reply = run_hook(payload, redactor_raises=True, secret="Ali Hassan")
        assert "Ali Hassan" not in json.dumps(reply)
        assert "RuntimeError" in updated_file(reply)["content"]

    def test_a_blocked_verdict_from_the_api_withholds_content(self, project, monkeypatch):
        """The other blocked test stubs the redactor, so it only proves the stub raised.
        This one comes back from the transport the way the real thing does."""
        monkeypatch.setattr(
            "rezunate_guard.scanner.send_batch",
            lambda texts: [{"entities": [], "blocked": True} for _ in texts],
        )
        payload = read_payload(project / "clients/acme.md", "Ali Hassan")
        with pytest.raises(hook.Blocked):
            hook.respond(payload)

        reply = hook._withhold(payload, "blocked")
        assert "Ali Hassan" not in json.dumps(reply)

    def test_blocked_content_is_withheld(self, project, monkeypatch):
        """A workspace guardrail set to block, not redact, must stop the read."""

        def blocked(texts):
            raise hook.Blocked("guardrail is set to block")

        monkeypatch.setattr(hook, "redact_all", blocked)
        with pytest.raises(hook.Blocked):
            hook.respond(read_payload(project / "clients/acme.md", "Ali Hassan"))

    def test_withheld_reply_keeps_the_response_shape(self, project):
        reply = run_hook(read_payload(project / "clients/acme.md", "secret"))
        updated = reply["hookSpecificOutput"]["updatedToolOutput"]
        assert updated["type"] == "text"
        assert set(updated["file"]) == {"filePath", "content", "numLines"}

    def test_protected_image_is_withheld(self, project, tmp_path):
        image = project / "clients/scan.png"
        image.write_bytes(b"\x89PNG")
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": str(image)},
            "tool_response": {"type": "image", "file": {"base64": "iVBORw0KGgo=", "type": "png"}},
        }
        reply = hook.respond(payload)
        assert "iVBORw0KGgo=" not in json.dumps(reply)

        # Replaced outright, not merged into — a merge would leave the base64 beside
        # the notice. This is the shape Claude Code itself uses to strip an image.
        updated = reply["hookSpecificOutput"]["updatedToolOutput"]
        assert updated["type"] == "text"
        assert set(updated["file"]) == {
            "filePath",
            "content",
            "numLines",
            "startLine",
            "totalLines",
        }

    @pytest.mark.parametrize(
        "response",
        [
            "Ali Hassan",
            {"stdout": "Ali Hassan", "stderr": ""},
            ["", {"text": "Ali Hassan"}],
            {"a": {"b": [{"deep": "Ali Hassan"}]}},
            {"type": "text", "file": {"filePath": "/x", "content": "Ali Hassan"}},
        ],
        ids=["bare-string", "bash", "list", "deeply-nested", "read"],
    )
    def test_any_response_holding_content_gets_a_notice(self, response):
        """`main` prints nothing for None, so returning None while content is present
        would leave the original output standing."""
        reply = hook._withhold({"tool_response": response}, "the scan failed")

        assert reply is not None
        assert "Ali Hassan" not in json.dumps(reply)
        assert "rezunate-guard" in json.dumps(reply)

    @pytest.mark.parametrize(
        "response",
        [None, 42, {}, [], {"filePath": "/x/y.md"}, {"type": "text"}, ""],
        ids=["none", "number", "empty-dict", "empty-list", "path-only", "tag-only", "empty"],
    )
    def test_a_response_with_no_content_is_left_alone(self, response):
        """None here means one thing only: there was nothing that could leak."""
        assert hook._withhold({"tool_response": response}, "the scan failed") is None

    def test_unprotected_image_passes(self, project):
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": str(project / "logo.png")},
            "tool_response": {"type": "image", "file": {"base64": "iVBORw0KGgo=", "type": "png"}},
        }
        assert hook.respond(payload) is None


def run_hook(payload: dict, redactor_raises: bool = False, secret: str = "boom") -> dict:
    """Run the hook as Claude Code would: a subprocess, JSON in, JSON out.

    `secret` goes into the raised exception's message, so a test can check that a stray
    failure does not echo file content back to the model.
    """
    source = "import sys; from rezunate_guard import hook\n"
    if redactor_raises:
        source += f"hook.redact_all = lambda ts: (_ for _ in ()).throw(RuntimeError({secret!r}))\n"
    source += "hook.main()\n"

    result = subprocess.run(
        [sys.executable, "-c", source],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"nonzero exit means the raw output is used: {result.stderr}"
    return json.loads(result.stdout)


class TestEndToEnd:
    """config -> scan -> mask -> reply, with only the network stubbed out."""

    def test_a_protected_read_comes_back_masked(self, project, monkeypatch):
        text = "Patient Ali Hassan, phone +971501234567, seen Tuesday."

        def fake_api(chunk):
            found = []
            for value, label in [("Ali Hassan", "person"), ("+971501234567", "phone")]:
                start = chunk.find(value)
                if start != -1:
                    found.append(
                        {
                            "start": start,
                            "end": start + len(value),
                            "label": label,
                            "text": value,
                            "score": 0.99,
                        }
                    )
            return {"entities": found, "blocked": False}

        monkeypatch.setattr(
            "rezunate_guard.scanner.send_batch",
            lambda texts: [fake_api(text) for text in texts],
        )

        reply = hook.respond(read_payload(project / "clients/acme.md", text))
        content = updated_file(reply)["content"]

        assert "Ali Hassan" not in content
        assert "+971501234567" not in content
        assert "[PERSON_" in content
        assert "[PHONE_" in content
        # Everything that wasn't PII has to survive, or the model loses the plot.
        assert content.startswith("Patient ")
        assert content.endswith(", seen Tuesday.")

    def test_a_clean_protected_file_is_unchanged(self, project, monkeypatch):
        monkeypatch.setattr(
            "rezunate_guard.scanner.send_batch",
            lambda texts: [{"entities": [], "blocked": False} for _ in texts],
        )
        text = "Meeting notes: ship the thing on Friday."
        reply = hook.respond(read_payload(project / "clients/acme.md", text))
        assert updated_file(reply)["content"] == text

    def test_the_same_person_masks_alike_in_two_files(self, project, monkeypatch):
        """No vault, no counter — the placeholder comes from the value, so two separate
        hook invocations agree."""

        def fake_api(chunk):
            start = chunk.find("Ali Hassan")
            if start == -1:
                return {"entities": [], "blocked": False}
            return {
                "entities": [
                    {
                        "start": start,
                        "end": start + 10,
                        "label": "person",
                        "text": "Ali Hassan",
                        "score": 0.99,
                    }
                ],
                "blocked": False,
            }

        monkeypatch.setattr(
            "rezunate_guard.scanner.send_batch",
            lambda texts: [fake_api(text) for text in texts],
        )

        first = updated_file(
            hook.respond(read_payload(project / "clients/a.md", "Ali Hassan called."))
        )["content"]
        second = updated_file(
            hook.respond(read_payload(project / "clients/b.md", "Later, Ali Hassan left."))
        )["content"]

        token = re.search(r"\[PERSON_[0-9a-f]+\]", first).group()
        assert token in second


class TestProcessContract:
    """Whatever happens, exit 0 and print either valid JSON or nothing."""

    @pytest.mark.parametrize(
        "stdin",
        ["", "not json at all", "[]", "null", '{"hook_event_name": "PostToolUse"}'],
    )
    def test_bad_input_still_exits_zero(self, stdin):
        result = subprocess.run(
            [sys.executable, "-m", "rezunate_guard.hook"],
            input=stdin,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0
        if result.stdout.strip():
            json.loads(result.stdout)


class TestServingARedactedCopy:
    """A file the model cannot be shown is swapped for a redacted copy of its text.

    Refusing the read is safe but leaves the user stuck, so anything we can extract is
    extracted, redacted and served instead. Everything we cannot is still refused.
    """

    def pre_payload(self, path, **extra) -> dict:
        return {
            "hook_event_name": "PreToolUse",
            "tool_name": "Read",
            "tool_input": {"file_path": str(path), **extra},
        }

    @pytest.fixture(autouse=True)
    def offline(self, monkeypatch):
        """Redaction stands in for the API, so no test here touches the network."""
        monkeypatch.setattr(
            hook, "redact_all", lambda texts: {t: t.replace("Ali Hassan", "[NAME]") for t in texts}
        )

    def redirect(self, reply: dict) -> dict:
        output = reply["hookSpecificOutput"]
        assert output["hookEventName"] == "PreToolUse"
        return output["updatedInput"]

    def test_a_protected_docx_is_read_as_redacted_text(self, project, make_docx):
        docx = make_docx(project / "clients/cv.docx")
        served = Path(self.redirect(hook.respond(self.pre_payload(docx)))["file_path"])
        assert served != docx
        assert "[NAME]" in served.read_text(encoding="utf-8")

    def test_the_original_text_is_not_in_the_copy(self, project, make_docx):
        docx = make_docx(project / "clients/cv.docx")
        served = Path(self.redirect(hook.respond(self.pre_payload(docx)))["file_path"])
        assert "Ali Hassan" not in served.read_text(encoding="utf-8")

    @pytest.mark.skipif(shutil.which("pdftotext") is None, reason="pdftotext is not installed")
    def test_a_protected_pdf_is_read_as_redacted_text(self, project, make_pdf):
        pdf = make_pdf(project / "clients/intake.pdf")
        served = Path(self.redirect(hook.respond(self.pre_payload(pdf)))["file_path"])
        body = served.read_text(encoding="utf-8")
        assert "[NAME]" in body and "Ali Hassan" not in body

    def test_the_reply_does_not_grant_permission(self, project, make_docx):
        """Saying "allow" here would wave the read past the prompt the user would
        normally get. The swap does not need it."""
        reply = hook.respond(self.pre_payload(make_docx(project / "clients/cv.docx")))
        assert "permissionDecision" not in reply["hookSpecificOutput"]

    def test_the_rest_of_the_call_is_left_alone(self, project, make_docx):
        docx = make_docx(project / "clients/cv.docx")
        updated = self.redirect(hook.respond(self.pre_payload(docx, offset=10, limit=5)))
        assert updated["offset"] == 10 and updated["limit"] == 5

    def test_an_unprotected_docx_is_not_touched(self, project, make_docx):
        assert hook.respond(self.pre_payload(make_docx(project / "cv.docx"))) is None

    def test_a_file_we_cannot_extract_is_still_denied(self, project):
        blob = project / "clients/export.dat"
        blob.write_bytes(bytes(range(256)) * 4)
        reply = hook.respond(self.pre_payload(blob))
        assert reply["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_a_failed_scan_denies_rather_than_serving_the_original(
        self, project, make_docx, monkeypatch
    ):
        """The copy is only safe if it was redacted. Serving unredacted text here would
        be worse than refusing, since it would look like the guard had done its job."""

        def explode(texts):
            raise ScanError("no API key")

        monkeypatch.setattr(hook, "redact_all", explode)
        reply = hook.respond(self.pre_payload(make_docx(project / "clients/cv.docx")))
        assert reply["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_blocked_content_denies(self, project, make_docx, monkeypatch):
        def blocked(texts):
            raise hook.Blocked("the workspace guardrail is set to block this content")

        monkeypatch.setattr(hook, "redact_all", blocked)
        reply = hook.respond(self.pre_payload(make_docx(project / "clients/cv.docx")))
        assert reply["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_cat_on_a_protected_docx_is_still_denied(self, project, make_docx):
        """A path inside a shell command cannot be swapped: rewriting the middle of a
        pipeline would change what the command does."""
        docx = make_docx(project / "clients/cv.docx")
        reply = hook.respond(
            {
                "hook_event_name": "PreToolUse",
                "tool_name": "Bash",
                "tool_input": {"command": f"cat {docx}"},
            }
        )
        assert reply["hookSpecificOutput"]["permissionDecision"] == "deny"

    def test_a_document_that_reads_as_text_is_still_extracted(self, project, make_pdf):
        """A metadata-heavy PDF can be pure ASCII for the whole sniff and still deliver
        its pages as images. Trusting the sniff withheld a file we could have redacted."""
        pdf = make_pdf(project / "clients/ascii.pdf")
        padded = b"%PDF-1.4\n" + b"% padding comment\n" * 600 + pdf.read_bytes()[9:]
        pdf.write_bytes(padded)

        assert hook._reaches_the_model_as_text(str(pdf)) is True
        served = self.redirect(hook.respond(self.pre_payload(pdf)))["file_path"]
        assert Path(served) != pdf

    def test_reading_the_copy_does_not_loop(self, project, make_docx):
        """The copy lives outside any project, so the hook has nothing to say about it."""
        docx = make_docx(project / "clients/cv.docx")
        served = self.redirect(hook.respond(self.pre_payload(docx)))["file_path"]
        assert hook.respond(self.pre_payload(served)) is None

    def test_a_problem_is_written_to_the_log(self, project, make_docx, monkeypatch):
        def explode(texts):
            raise ScanError("no API key")

        monkeypatch.setattr(hook, "redact_all", explode)
        hook.respond(self.pre_payload(make_docx(project / "clients/cv.docx")))
        assert "could not build a redacted copy" in constants.log_path().read_text()
