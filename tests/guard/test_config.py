"""Tests for the scan decision."""

from pathlib import Path

import pytest

from rezunate_guard import config as rezunate_config
from rezunate_guard import constants
from rezunate_guard.config import (
    CONFIG_TEMPLATE,
    GuardConfig,
    find_config_file,
    load_config,
    resolve_config,
    scan_decision,
    should_scan,
)


def write(root: Path, relative: str, content: str = "x") -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def config_for(root: Path, body: str) -> GuardConfig:
    write(root, constants.CONFIG_FILENAME, body)
    return load_config(root / constants.CONFIG_FILENAME)


class TestUnconfigured:
    """With no config file, the guard scans nothing — and must admit it."""

    def test_nothing_is_scanned(self, tmp_path):
        target = write(tmp_path, "clients/acme.md")
        assert should_scan(target) is False

    def test_inert_state_is_detectable(self, tmp_path):
        target = write(tmp_path, "clients/acme.md")
        assert resolve_config(target).protects_nothing is True

    def test_inert_decision_explains_itself(self, tmp_path):
        target = write(tmp_path, "clients/acme.md")
        assert "no folders configured" in scan_decision(target).reason

    def test_configured_guard_is_not_inert(self, tmp_path):
        config = config_for(tmp_path, "scan:\n  - clients\n")
        assert config.protects_nothing is False


class TestShippedTemplate:
    """What ``init`` writes must parse, and must protect nothing until edited."""

    def test_template_parses_without_error(self, tmp_path):
        assert config_for(tmp_path, CONFIG_TEMPLATE).error is None

    def test_template_is_inert_until_edited(self, tmp_path):
        assert config_for(tmp_path, CONFIG_TEMPLATE).protects_nothing is True

    def test_template_becomes_active_when_uncommented(self, tmp_path):
        write(tmp_path, "clients/acme.md")
        config = config_for(tmp_path, CONFIG_TEMPLATE.replace("  # - clients", "  - clients"))

        assert config.protects_nothing is False
        assert should_scan(tmp_path / "clients/acme.md", config) is True
        assert should_scan(tmp_path / "src/app.py", config) is False


class TestListedFolders:
    """A listed folder protects everything inside it, and nothing outside it."""

    def test_the_folder_and_everything_under_it(self, tmp_path):
        config = config_for(tmp_path, "scan:\n  - clients\n")
        assert should_scan(tmp_path / "clients/acme.md", config) is True
        assert should_scan(tmp_path / "clients/2026/q1/acme.md", config) is True
        assert should_scan(tmp_path / "src/app.py", config) is False

    def test_a_nested_folder_does_not_protect_its_parent(self, tmp_path):
        config = config_for(tmp_path, "scan:\n  - data/patients\n")
        assert should_scan(tmp_path / "data/patients/list.csv", config) is True
        assert should_scan(tmp_path / "data/other.csv", config) is False

    def test_several_folders(self, tmp_path):
        config = config_for(tmp_path, "scan:\n  - clients\n  - exports\n")
        assert should_scan(tmp_path / "clients/a.md", config) is True
        assert should_scan(tmp_path / "exports/b.csv", config) is True
        assert should_scan(tmp_path / "src/c.py", config) is False

    def test_a_name_matches_a_whole_segment_only(self, tmp_path):
        """``clients`` must not swallow ``clients-archive``."""
        config = config_for(tmp_path, "scan:\n  - clients\n")
        assert should_scan(tmp_path / "clients-archive/a.md", config) is False

    def test_a_folder_is_anchored_to_the_config(self, tmp_path):
        """A name never floats to any depth; it means that one folder."""
        config = config_for(tmp_path, "scan:\n  - clients\n")
        assert should_scan(tmp_path / "packages/web/clients/a.md", config) is False

    def test_everything_inside_is_scanned(self, tmp_path):
        """No built-in exclusions. A broad scan really does pull in everything."""
        config = config_for(tmp_path, "scan:\n  - app\n")
        assert should_scan(tmp_path / "app/node_modules/react/index.js", config) is True
        assert should_scan(tmp_path / "app/uv.lock", config) is True

    def test_no_hidden_rule_swallows_an_explicit_scan(self, tmp_path):
        """A shipped ``build/`` exclusion would have made this protect nothing."""
        config = config_for(tmp_path, "scan:\n  - build/exports\n")
        assert should_scan(tmp_path / "build/exports/customers.csv", config) is True

    def test_decisions_explain_themselves(self, tmp_path):
        config = config_for(tmp_path, "scan:\n  - clients\n")
        assert "in a protected folder" in scan_decision(tmp_path / "clients/a.md", config).reason
        assert "not in a protected folder" in scan_decision(tmp_path / "src/a.py", config).reason


class TestAbsoluteFolders:
    """An absolute path lets a config protect a folder outside its own project."""

    def test_an_absolute_folder_outside_the_project(self, tmp_path):
        project = tmp_path / "project"
        outside = tmp_path / "share" / "case-files"
        outside.mkdir(parents=True)
        config = config_for(project, f"scan:\n  - {outside}\n")

        assert should_scan(outside / "a.pdf", config) is True
        assert should_scan(outside / "deep/b.pdf", config) is True
        assert should_scan(tmp_path / "share/other.pdf", config) is False

    def test_a_tilde_is_expanded(self, tmp_path, isolated_home):
        config = config_for(tmp_path, "scan:\n  - ~/records\n")
        assert config.scan == (Path.home() / "records",)

    def test_dot_means_the_whole_project(self, tmp_path):
        config = config_for(tmp_path, "scan:\n  - '.'\n")
        assert should_scan(tmp_path / "src/app.py", config) is True
        assert should_scan(tmp_path / "clients/acme.md", config) is True


class TestFolderNormalisation:
    @pytest.mark.parametrize("entry", ["clients", "clients/", "clients//"])
    def test_trailing_slashes_are_trimmed(self, tmp_path, entry):
        config = config_for(tmp_path, f"scan:\n  - '{entry}'\n")
        assert config.scan == ((tmp_path / "clients").resolve(),)

    def test_comments_and_blanks_are_ignored(self, tmp_path):
        config = config_for(tmp_path, "scan:\n  - '# not a folder'\n  - ''\n  - clients\n")
        assert config.scan == ((tmp_path / "clients").resolve(),)

    def test_a_single_folder_may_be_written_as_a_string(self, tmp_path):
        config = config_for(tmp_path, "scan: clients\n")
        assert should_scan(tmp_path / "clients/a.md", config) is True


class TestNothingIsScannedUnlessNamed:
    """A config we cannot read protects nothing. Only the log says why."""

    @pytest.mark.parametrize(
        "body",
        [
            "scan: [unclosed\n",
            "- just\n- a\n- list\n",
            "scans:\n  - clients\n",
            "scan:\n  - '*.csv'\n",
            "scan:\n  - clients: \n",
        ],
        ids=["malformed", "not-a-mapping", "misspelled-key", "a-pattern", "stray-colon"],
    )
    def test_a_config_we_cannot_read_protects_nothing(self, tmp_path, body):
        config = config_for(tmp_path, body)
        assert config.error is not None
        assert config.protects_nothing is True
        assert should_scan(tmp_path / "clients/acme.md", config) is False

    def test_a_pattern_names_itself_in_the_error(self, tmp_path):
        assert "*.csv" in config_for(tmp_path, "scan:\n  - '*.csv'\n").error

    def test_a_misspelled_key_names_itself_in_the_error(self, tmp_path):
        assert "scans" in config_for(tmp_path, "scans:\n  - clients\n").error

    def test_unreadable_config_protects_nothing(self, tmp_path):
        path = write(tmp_path, constants.CONFIG_FILENAME, "scan:\n  - clients\n")
        path.chmod(0o000)
        try:
            config = load_config(path)
        finally:
            path.chmod(0o644)
        assert config.error is not None
        assert should_scan(tmp_path / "clients/a.md", config) is False

    def test_empty_config_is_inert_not_broken(self, tmp_path):
        config = config_for(tmp_path, "")
        assert config.error is None
        assert config.protects_nothing is True
        assert should_scan(tmp_path / "clients/acme.md", config) is False


class TestProblemsAreLogged:
    """The log is the only thing that tells a user they are unprotected."""

    def log_lines(self) -> list[str]:
        path = constants.log_path()
        return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []

    def test_a_broken_config_is_logged(self, tmp_path):
        config_for(tmp_path, "scans:\n  - clients\n")
        assert any("unknown setting 'scans'" in line for line in self.log_lines())

    def test_a_config_listing_nothing_is_logged(self, tmp_path):
        config_for(tmp_path, "")
        assert any("no folders listed" in line for line in self.log_lines())

    def test_a_missing_folder_is_logged(self, tmp_path):
        """A mis-indented list parses cleanly and names a folder that isn't there."""
        config_for(tmp_path, "scan:\n  clients\n  data\n")
        assert any("folder not found" in line for line in self.log_lines())

    def test_a_folder_that_exists_is_not_logged(self, tmp_path):
        (tmp_path / "clients").mkdir()
        config_for(tmp_path, "scan:\n  - clients\n")
        assert self.log_lines() == []


class TestReusingAParse:
    """One tool call asks about many paths, and they usually share a config."""

    def test_an_unchanged_config_is_parsed_once(self, tmp_path, monkeypatch):
        path = write(tmp_path, constants.CONFIG_FILENAME, "scan:\n  - clients\n")
        parses = []
        real = rezunate_config._parse
        monkeypatch.setattr(rezunate_config, "_parse", lambda p, t: parses.append(p) or real(p, t))

        for _ in range(10):
            load_config(path)
        assert len(parses) == 1

    def test_an_edited_config_is_parsed_again(self, tmp_path):
        path = write(tmp_path, constants.CONFIG_FILENAME, "scan:\n  - clients\n")
        assert load_config(path).scan == ((tmp_path / "clients").resolve(),)

        path.write_text("scan:\n  - exports\n", encoding="utf-8")
        assert load_config(path).scan == ((tmp_path / "exports").resolve(),)


class TestDiscovery:
    def test_finds_config_in_parent_folder(self, tmp_path):
        write(tmp_path, constants.CONFIG_FILENAME, "scan:\n  - clients\n")
        target = write(tmp_path, "deep/nested/file.md")
        assert find_config_file(target) == tmp_path / constants.CONFIG_FILENAME

    def test_nearest_config_wins(self, tmp_path):
        write(tmp_path, constants.CONFIG_FILENAME, "scan:\n  - '.'\n")
        write(tmp_path, "apps/api/" + constants.CONFIG_FILENAME, "scan:\n  - secrets\n")
        target = write(tmp_path, "apps/api/notes.md")

        assert find_config_file(target) == tmp_path / "apps/api" / constants.CONFIG_FILENAME
        # The outer config would have scanned this; the inner one does not.
        assert should_scan(target) is False

    def test_returns_none_when_absent(self, tmp_path):
        assert find_config_file(write(tmp_path, "file.md")) is None


class TestPathResolution:
    def test_file_outside_config_root_uses_its_own_config(self, tmp_path):
        """A --add-dir file must not be judged by another project's rules."""
        project = tmp_path / "project"
        other = tmp_path / "other"
        config = config_for(project, "scan:\n  - '.'\n")

        write(other, constants.CONFIG_FILENAME, "scan:\n  - client-data\n")
        outside = write(other, "client-data/notes.md")

        assert scan_decision(outside, config).should_scan is True

    def test_file_outside_any_config_is_not_scanned(self, tmp_path):
        project = tmp_path / "project"
        config = config_for(project, "scan:\n  - '.'\n")

        decision = scan_decision(write(tmp_path / "other", "notes.md"), config)
        assert decision.should_scan is False
        assert "outside config root" in decision.reason

    def test_inert_project_does_not_veto_an_outside_project(self, tmp_path):
        """A project protecting nothing has no say over a --add-dir file that is."""
        project = tmp_path / "project"
        other = tmp_path / "other"
        config = config_for(project, "")
        assert config.protects_nothing is True

        write(other, constants.CONFIG_FILENAME, "scan:\n  - client-data\n")
        outside = write(other, "client-data/notes.md")

        assert scan_decision(outside, config).should_scan is True

    def test_two_projects_obey_their_own_configs(self, tmp_path):
        strict = tmp_path / "strict"
        loose = tmp_path / "loose"
        write(strict, constants.CONFIG_FILENAME, "scan:\n  - '.'\n")
        write(loose, constants.CONFIG_FILENAME, "scan:\n  - vault\n")

        assert should_scan(write(strict, "notes.md")) is True
        assert should_scan(write(loose, "notes.md")) is False

    def test_decision_always_reports_a_reason(self, tmp_path):
        config = config_for(tmp_path, "scan:\n  - clients\n")
        assert scan_decision(write(tmp_path, "clients/a.md"), config).reason
        assert scan_decision(write(tmp_path, "src/a.py"), config).reason


class TestResolveConfig:
    def test_resolve_returns_inert_default_without_any_config(self, tmp_path):
        config = resolve_config(write(tmp_path, "notes.md"))
        assert config.source is None
        assert config.protects_nothing is True

    def test_resolve_loads_project_config(self, tmp_path):
        write(tmp_path, constants.CONFIG_FILENAME, "scan:\n  - clients\n")
        target = write(tmp_path, "notes.md")
        assert resolve_config(target).source == tmp_path / constants.CONFIG_FILENAME


class TestGlobalFallback:
    """The `~/.rezunate/config.yaml` covering files that belong to no project.

    It is rooted at the filesystem root, since a global config has no project folder to
    anchor to, so its folders are absolute. That re-rooting is the subtlest part of the
    module.
    """

    def global_config(self, home: Path, body: str) -> Path:
        home.mkdir(parents=True, exist_ok=True)
        path = home / "config.yaml"
        path.write_text(body, encoding="utf-8")
        return path

    def test_a_file_in_no_project_falls_back_to_the_global_config(self, tmp_path, isolated_home):
        self.global_config(isolated_home, f"scan:\n  - {tmp_path}/secrets\n")
        target = write(tmp_path, "secrets/key.pem")

        assert resolve_config(target).source == constants.user_config_path()
        assert should_scan(target) is True

    def test_an_absolute_folder_covers_everything_under_it(self, tmp_path, isolated_home):
        self.global_config(isolated_home, f"scan:\n  - {tmp_path}/secrets\n")
        assert should_scan(write(tmp_path, "secrets/a/b/c/key.pem")) is True
        assert should_scan(write(tmp_path, "notes.md")) is False

    def test_the_global_config_is_rooted_at_the_filesystem_root(self, tmp_path, isolated_home):
        """Get this wrong and every global folder silently misses."""
        self.global_config(isolated_home, f"scan:\n  - {tmp_path}/secrets\n")
        config = resolve_config(write(tmp_path, "secrets/key.pem"))
        assert config.root == Path(tmp_path.anchor)

    def test_a_project_config_beats_the_global_one(self, tmp_path, isolated_home):
        self.global_config(isolated_home, f"scan:\n  - {tmp_path}/secrets\n")
        write(tmp_path, constants.CONFIG_FILENAME, "scan:\n  - clients\n")

        assert should_scan(write(tmp_path, "secrets/key.pem")) is False
        assert should_scan(write(tmp_path, "clients/a.md")) is True

    def test_no_global_config_leaves_the_guard_inert(self, tmp_path, isolated_home):
        config = resolve_config(write(tmp_path, "key.pem"))
        assert config.source is None
        assert config.protects_nothing is True
