"""Tests for reusing what a scan already returned."""

from __future__ import annotations

import time

import pytest

from rezunate_guard import cache, constants, scanner


@pytest.fixture
def sent(monkeypatch):
    """Record every window sent to the API, and answer without touching the network."""
    groups: list[list[str]] = []

    def send(texts, workspace=""):
        groups.append(list(texts))
        return [{"entities": [], "blocked": False} for _ in texts]

    monkeypatch.setattr(scanner, "send_batch", send)
    return groups


class TestReuse:
    def test_the_same_text_is_scanned_once(self, sent):
        scanner.scan_many(["Ali Hassan works here"], "acme")
        scanner.scan_many(["Ali Hassan works here"], "acme")
        assert len(sent) == 1, "the second scan must come from the cache"

    def test_an_edit_only_rescans_the_windows_it_touched(self, sent):
        paragraphs = ["x" * 800, "y" * 800, "z" * 800]
        scanner.scan_many(["\n".join(paragraphs)], "acme")
        first = len(sent[0])

        paragraphs[1] = "Y" * 800
        scanner.scan_many(["\n".join(paragraphs)], "acme")

        assert first > 1, "the sample must span several windows for this to mean anything"
        assert len(sent[1]) < first, "unchanged windows must not be sent again"

    def test_another_workspace_does_not_reuse_the_answer(self, sent):
        scanner.scan_many(["Ali Hassan"], "acme")
        scanner.scan_many(["Ali Hassan"], "hospital")
        assert len(sent) == 2, "workspaces have different rules, so answers cannot be shared"

    def test_a_stale_answer_is_scanned_again(self, sent, monkeypatch):
        scanner.scan_many(["Ali Hassan"], "acme")
        monkeypatch.setattr(cache, "TTL_SECONDS", -1)
        scanner.scan_many(["Ali Hassan"], "acme")
        assert len(sent) == 2


class TestFailsToAMiss:
    def test_an_unusable_cache_costs_a_scan_rather_than_an_error(self, sent):
        constants.cache_path().mkdir(parents=True)  # a directory where the database goes
        results = scanner.scan_many(["Ali Hassan"], "acme")
        assert len(results) == 1
        assert len(sent) == 1

    def test_a_corrupt_cache_costs_a_scan_rather_than_an_error(self, sent):
        path = constants.cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"not a database")
        assert scanner.scan_many(["Ali Hassan"], "acme")[0].blocked is False


class TestWhatIsWritten:
    def test_the_text_itself_is_never_stored(self):
        secret = "Ali Hassan lives at 12 Oxford Road"
        cache.store("acme", {secret: {"entities": [], "blocked": False}})
        assert secret not in constants.cache_path().read_bytes().decode("utf-8", "replace")

    def test_an_entry_expires(self, monkeypatch):
        cache.store("acme", {"text": {"entities": [], "blocked": False}})
        assert cache.lookup("acme", ["text"]) != {}

        later = time.time() + cache.TTL_SECONDS + 1
        monkeypatch.setattr(time, "time", lambda: later)
        cache.store("acme", {"other": {"entities": [], "blocked": False}})
        assert cache.lookup("acme", ["text"]) == {}

    def test_clearing_removes_everything(self):
        cache.store("acme", {"text": {"entities": [], "blocked": False}})
        cache.clear()
        assert cache.lookup("acme", ["text"]) == {}
