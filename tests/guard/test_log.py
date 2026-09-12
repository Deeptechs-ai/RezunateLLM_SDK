"""Tests for the problem log."""

from pathlib import Path

from rezunate_guard import constants, log

SOURCE = Path("/project/.rezunate-guard.yaml")


def lines() -> list[str]:
    path = constants.log_path()
    return path.read_text(encoding="utf-8").splitlines() if path.is_file() else []


class TestWriting:
    def test_a_problem_is_written(self):
        log.problem(SOURCE, "folder not found: /project/clients")
        assert len(lines()) == 1
        assert str(SOURCE) in lines()[0]
        assert "folder not found: /project/clients" in lines()[0]

    def test_each_line_is_stamped(self):
        log.problem(SOURCE, "something went wrong")
        assert lines()[0].startswith("20")

    def test_different_problems_all_appear(self):
        log.problem(SOURCE, "first")
        log.problem(SOURCE, "second")
        assert len(lines()) == 2

    def test_a_multi_line_message_becomes_one_line(self):
        """A YAML error arrives with newlines, which would break both the log and the
        repeat check."""
        log.problem(SOURCE, "could not read config:\nline 2\ncolumn 1")
        assert len(lines()) == 1

    def test_a_long_message_is_trimmed(self):
        log.problem(SOURCE, "x" * 500)
        assert len(lines()[0]) < 200


class TestNotRepeatingItself:
    """The hook runs once per tool call, so the same problem arrives again and again."""

    def test_the_same_problem_is_written_once(self):
        for _ in range(50):
            log.problem(SOURCE, "folder not found: /project/clients")
        assert len(lines()) == 1

    def test_several_live_problems_stay_quiet_together(self):
        """Comparing against only the last line would let three problems take turns."""
        for _ in range(20):
            for name in ("clients", "exports", "records"):
                log.problem(SOURCE, f"folder not found: /project/{name}")
        assert len(lines()) == 3


class TestStayingSmall:
    def test_the_log_stops_growing(self):
        for index in range(log.MAX_LINES * 3):
            log.problem(Path(f"/project-{index}/.rezunate-guard.yaml"), "unknown setting")
        assert len(lines()) == log.MAX_LINES

    def test_the_newest_problems_are_the_ones_kept(self):
        for index in range(log.MAX_LINES * 2):
            log.problem(Path(f"/project-{index}/.rezunate-guard.yaml"), "unknown setting")
        assert "/project-0/" not in "\n".join(lines())
        assert f"/project-{log.MAX_LINES * 2 - 1}/" in lines()[-1]


class TestNeverFailing:
    def test_an_unwritable_home_is_not_an_error(self, isolated_home):
        """Logging must never be the reason a tool call fails."""
        isolated_home.mkdir(parents=True, exist_ok=True)
        isolated_home.chmod(0o500)
        try:
            log.problem(SOURCE, "cannot be written")
        finally:
            isolated_home.chmod(0o700)
