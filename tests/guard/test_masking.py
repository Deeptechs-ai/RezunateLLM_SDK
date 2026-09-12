"""Tests for placeholder derivation and span replacement."""

from __future__ import annotations

import re
import subprocess
import sys

import pytest

from rezunate_guard import constants
from rezunate_guard.masking import (
    PLACEHOLDER_RE,
    MaskingError,
    mask,
    placeholder,
    placeholder_key,
)
from rezunate_guard.scanner import Entity


def entity(start, end, label="PERSON", score=0.9):
    return Entity(start=start, end=end, label=label, score=score)


@pytest.fixture
def key():
    return b"a" * 32


class TestPlaceholders:
    def test_shape_is_recognisable(self, key):
        assert PLACEHOLDER_RE.fullmatch(placeholder("PERSON", "Ali Hassan", key))

    def test_same_value_same_placeholder(self, key):
        assert placeholder("PERSON", "Ali", key) == placeholder("PERSON", "Ali", key)

    def test_different_values_differ(self, key):
        assert placeholder("PERSON", "Ali", key) != placeholder("PERSON", "Sara", key)

    def test_label_is_visible_in_the_tag(self, key):
        assert placeholder("email address", "a@b.com", key).startswith("[EMAIL_ADDRESS_")

    def test_unlabelled_entities_still_get_a_tag(self, key):
        assert placeholder("!!!", "x", key).startswith("[PII_")

    def test_placeholder_does_not_contain_the_value(self, key):
        assert "Ali" not in placeholder("PERSON", "Ali", key)

    def test_a_different_key_gives_a_different_placeholder(self):
        """Why the key exists: without it, a placeholder is a guessable hash of a
        value with a small search space, like a phone number."""
        assert placeholder("PHONE", "+971501234567", b"k1" * 16) != placeholder(
            "PHONE", "+971501234567", b"k2" * 16
        )


class TestMasking:
    def test_pii_is_gone(self, key):
        text = "Ali Hassan called from +971 50 123 4567"
        masked = mask(
            text,
            [entity(0, 10, "PERSON"), entity(23, 39, "PHONE")],
            key=key,
        )
        assert "Ali Hassan" not in masked
        assert "+971 50 123 4567" not in masked

    def test_surrounding_text_survives(self, key):
        masked = mask("Ali Hassan called the clinic", [entity(0, 10)], key=key)
        assert masked.endswith(" called the clinic")

    def test_repeated_person_gets_one_placeholder(self, key):
        text = "Ali emailed Ali"
        masked = mask(text, [entity(0, 3), entity(12, 15)], key=key)
        assert masked == f"{placeholder('PERSON', 'Ali', key)} emailed " + placeholder(
            "PERSON", "Ali", key
        )

    def test_two_people_stay_distinguishable(self, key):
        masked = mask("Ali met Sara", [entity(0, 3), entity(8, 12)], key=key)
        found = PLACEHOLDER_RE.findall(masked)
        assert len(found) == 2
        assert found[0] != found[1]

    def test_offsets_stay_correct_across_many_spans(self, key):
        """Masking right-to-left is what keeps this true; left-to-right would shift
        every span after the first."""
        text = "A1 B2 C3 D4 E5"
        spans = [entity(i, i + 2) for i in (0, 3, 6, 9, 12)]
        masked = mask(text, spans, key=key)
        assert len(PLACEHOLDER_RE.findall(masked)) == 5
        assert not re.search(r"[A-E]\d", masked)

    def test_no_entities_leaves_text_alone(self, key):
        assert mask("nothing here", [], key=key) == "nothing here"

    def test_out_of_range_spans_are_ignored(self, key):
        assert mask("short", [entity(0, 999)], key=key) == "short"
        assert mask("short", [entity(-5, 2)], key=key) == "short"
        assert mask("short", [entity(3, 3)], key=key) == "short"

    def test_overlapping_spans_raise_rather_than_corrupt(self, key):
        """scanner.drop_overlaps is supposed to prevent this; if it ever fails, the
        masker must not quietly produce mangled text."""
        with pytest.raises(MaskingError, match="overlapping"):
            mask("Ali Hassan", [entity(0, 10), entity(4, 8)], key=key)

    def test_unicode_offsets_are_respected(self, key):
        text = "المريض علي حسن يتصل"
        masked = mask(text, [entity(7, 14, "PERSON")], key=key)
        assert "علي حسن" not in masked
        assert masked.startswith("المريض ")


class TestKey:
    def test_key_is_created_on_first_use(self, isolated_home):
        assert not constants.placeholder_key_path().exists()
        assert len(placeholder_key()) == 32
        assert constants.placeholder_key_path().exists()

    def test_key_is_stable_across_calls(self, isolated_home):
        assert placeholder_key() == placeholder_key()

    def test_key_is_private(self, isolated_home):
        placeholder_key()
        assert (
            constants.placeholder_key_path().stat().st_mode & 0o077 == 0
        ), "the key must not be group or world readable"

    def test_key_survives_a_new_process(self, isolated_home):
        """The whole point: the hook is a fresh process per tool call, so a person has
        to get the same placeholder in the next one."""
        first = placeholder_key()

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "from rezunate_guard.masking import placeholder_key;"
                "import sys; sys.stdout.write(placeholder_key().hex())",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert result.stdout == first.hex()
