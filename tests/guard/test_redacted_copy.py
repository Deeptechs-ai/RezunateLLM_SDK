"""Tests for the redacted copies served in place of unreadable files."""

from __future__ import annotations

import stat

from rezunate_guard import constants, redacted_copy


class TestWritingAStandIn:
    def test_the_text_is_there_under_a_header(self, tmp_path):
        path = redacted_copy.write(tmp_path / "intake.pdf", "Patient [PERSON_NAME_a786]")
        body = path.read_text(encoding="utf-8")
        assert body.endswith("Patient [PERSON_NAME_a786]")
        assert "intake.pdf" in body

    def test_the_header_says_this_is_not_the_file(self, tmp_path):
        """Without it the model reports extracted text as the original document."""
        body = redacted_copy.write(tmp_path / "intake.pdf", "text").read_text(encoding="utf-8")
        assert "not the original file" in body

    def test_it_lands_outside_any_project(self, tmp_path):
        path = redacted_copy.write(tmp_path / "intake.pdf", "text")
        assert path.parent == constants.redacted_copies_dir()
        assert tmp_path not in path.parents

    def test_only_the_owner_can_read_it(self, tmp_path):
        """It still holds the shape of a protected document, so it is not for sharing."""
        path = redacted_copy.write(tmp_path / "intake.pdf", "text")
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700

    def test_the_same_file_reuses_one_copy(self, tmp_path):
        first = redacted_copy.write(tmp_path / "intake.pdf", "text")
        second = redacted_copy.write(tmp_path / "intake.pdf", "text")
        assert first == second
        assert len(list(constants.redacted_copies_dir().iterdir())) == 1

    def test_changed_text_gets_its_own_copy(self, tmp_path):
        first = redacted_copy.write(tmp_path / "intake.pdf", "text")
        second = redacted_copy.write(tmp_path / "intake.pdf", "different")
        assert first != second

    def test_two_files_with_one_name_do_not_collide(self, tmp_path):
        first = redacted_copy.write(tmp_path / "a/intake.pdf", "text")
        second = redacted_copy.write(tmp_path / "b/intake.pdf", "text")
        assert first != second

    def test_no_temporary_file_is_left_behind(self, tmp_path):
        redacted_copy.write(tmp_path / "intake.pdf", "text")
        assert not [p for p in constants.redacted_copies_dir().iterdir() if p.suffix == ".tmp"]
