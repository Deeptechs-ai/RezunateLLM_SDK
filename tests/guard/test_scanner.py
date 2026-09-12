"""Tests for windowing, offset arithmetic, batching, and the API client."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError

import pytest

from rezunate_guard import constants, scanner
from rezunate_guard.scanner import (
    CHUNK_CHARS,
    MAX_BATCH_TEXTS,
    OVERLAP_CHARS,
    Entity,
    ScanError,
    chunks,
    drop_overlaps,
    scan,
    scan_many,
    send_batch,
)


@pytest.fixture
def sending(monkeypatch):
    """Put a stand-in where the network call goes, so no test touches the wire."""

    def use(sender):
        monkeypatch.setattr(scanner, "send_batch", sender)

    return use


def entity(start, end, label="PERSON", score=0.9):
    return Entity(start=start, end=end, label=label, score=score)


def responder(*, entities=(), blocked=False):
    """A stand-in for the API that returns fixed entities for every text in the batch."""

    def sender(texts):
        return [{"entities": list(entities), "blocked": blocked, "action": "redact"} for _ in texts]

    return sender


def finder(needle, label="PHONE"):
    """A stand-in that actually looks for `needle` inside each text it is given."""

    def scan_one(text):
        found = []
        start = text.find(needle)
        while start != -1:
            found.append(
                {
                    "start": start,
                    "end": start + len(needle),
                    "label": label,
                    "text": needle,
                    "score": 0.99,
                }
            )
            start = text.find(needle, start + 1)
        return {"entities": found, "blocked": False}

    def sender(texts):
        return [scan_one(text) for text in texts]

    return sender


class TestChunking:
    def test_short_text_is_one_chunk(self):
        assert list(chunks("hello")) == [(0, "hello")]

    def test_every_character_is_covered(self):
        text = "".join(f"line {i} of the document\n" for i in range(300))
        covered = bytearray(len(text))
        for offset, chunk in chunks(text):
            for i in range(offset, offset + len(chunk)):
                covered[i] = 1
        assert all(covered), "a gap between chunks is PII nobody scans"

    def test_chunks_report_their_true_offset(self):
        text = "".join(f"line {i} of the document\n" for i in range(300))
        for offset, chunk in chunks(text):
            assert text[offset : offset + len(chunk)] == chunk

    def test_neighbours_overlap(self):
        text = "x" * (CHUNK_CHARS * 3)
        bounds = [(o, o + len(c)) for o, c in chunks(text)]
        for (_, prev_end), (next_start, _) in zip(bounds, bounds[1:], strict=False):
            assert prev_end - next_start == OVERLAP_CHARS

    def test_a_line_longer_than_a_window_is_cut(self):
        text = "y" * (CHUNK_CHARS * 2)
        produced = list(chunks(text))
        assert len(produced) > 1
        assert all(len(c) <= CHUNK_CHARS for _, c in produced)

    def test_overlap_must_be_smaller_than_the_window(self):
        with pytest.raises(ValueError):
            list(chunks("x" * 5000, size=100, overlap=100))


class TestBoundaries:
    def test_entity_on_a_boundary_is_still_found(self, sending):
        needle = "+971 50 123 4567"
        # Drop the needle right where the first chunk ends.
        text = ("a" * (CHUNK_CHARS - 8)) + needle + ("b" * CHUNK_CHARS)
        sending(finder(needle))
        result = scan(text)

        assert len(result.entities) == 1
        found = result.entities[0]
        assert text[found.start : found.end] == needle

    def test_a_duplicate_from_the_overlap_is_collapsed(self, sending):
        needle = "ali@example.com"
        # Sitting inside the overlap, so both chunks report it.
        text = ("a" * (CHUNK_CHARS - OVERLAP_CHARS + 10)) + needle + ("b" * CHUNK_CHARS)
        sending(finder(needle, label="EMAIL"))
        result = scan(text)
        assert len(result.entities) == 1

    def test_offsets_index_the_original_text(self, sending):
        needle = "Ali Hassan"
        text = ("filler line\n" * 200) + needle + ("\nmore filler" * 200)
        sending(finder(needle, label="PERSON"))
        result = scan(text)
        assert [text[e.start : e.end] for e in result.entities] == [needle]


class TestMerge:
    def test_identical_spans_collapse(self):
        assert len(drop_overlaps([entity(0, 5), entity(0, 5)])) == 1

    def test_longer_span_wins(self):
        kept = drop_overlaps([entity(0, 5), entity(0, 12)])
        assert [(e.start, e.end) for e in kept] == [(0, 12)]

    def test_higher_score_breaks_a_tie(self):
        kept = drop_overlaps([entity(0, 5, score=0.4), entity(0, 5, score=0.95)])
        assert kept[0].score == 0.95

    def test_contained_span_is_dropped(self):
        kept = drop_overlaps([entity(0, 20), entity(5, 10)])
        assert [(e.start, e.end) for e in kept] == [(0, 20)]

    def test_distinct_spans_are_kept_in_order(self):
        kept = drop_overlaps([entity(30, 40), entity(0, 10)])
        assert [e.start for e in kept] == [0, 30]

    def test_one_span_survives_two_labels(self):
        """A hospital name can be both ORG and LOCATION. Masking it twice would shift
        every offset after it, so only one detection can survive."""
        kept = drop_overlaps([entity(0, 10, label="PERSON"), entity(0, 10, label="ORG")])
        assert len(kept) == 1

    def test_partial_overlap_is_resolved(self):
        kept = drop_overlaps([entity(0, 10, label="PERSON"), entity(5, 15, label="EMAIL")])
        assert [(e.start, e.end) for e in kept] == [(0, 10)]

    def test_result_is_always_disjoint_and_sorted(self):
        """The invariant the masker depends on."""
        crowded = [
            entity(0, 10),
            entity(5, 15),
            entity(5, 8),
            entity(12, 30),
            entity(30, 40),
            entity(35, 36),
            entity(0, 3),
        ]
        kept = drop_overlaps(crowded)
        assert list(kept) == sorted(kept, key=lambda e: e.start)
        for previous, following in zip(kept, kept[1:], strict=False):
            assert previous.end <= following.start


class TestScan:
    def test_empty_text_costs_nothing(self, sending):
        def explode(texts):
            raise AssertionError("scanned an empty document")

        sending(explode)
        assert scan("   \n  ").entities == ()

    def test_blocked_propagates(self, sending):
        sending(responder(blocked=True))
        assert scan("some text").blocked is True

    def test_blocked_from_any_chunk_wins(self, sending):
        def sender(texts):
            return [{"entities": [], "blocked": index == 1} for index, _ in enumerate(texts)]

        sending(sender)
        assert scan("x" * (CHUNK_CHARS * 3)).blocked is True

    def test_missing_entity_list_is_an_error(self, sending):
        with pytest.raises(ScanError, match="entity list"):
            sending(lambda texts: [{"blocked": False}])
            scan("text")

    def test_malformed_entity_is_an_error(self, sending):
        with pytest.raises(ScanError, match="malformed"):
            sending(lambda texts: [{"entities": [{"start": "nope"}]}])
            scan("text")

    def test_non_object_response_is_an_error(self, sending):
        with pytest.raises(ScanError, match="not an object"):
            sending(lambda texts: ["unexpected"])
            scan("text")


class TestBatching:
    """One tool result is many strings. Scanning them one at a time meant one HTTPS
    round trip each, which is what this batching exists to collapse."""

    def sending_recorder(self):
        sent = []

        def send(texts):
            sent.append(list(texts))
            return [{"entities": [], "blocked": False} for _ in texts]

        return sent, send

    def test_results_come_back_one_per_input_in_order(self, sending):
        sent, send = self.sending_recorder()
        sending(send)
        results = scan_many(["alpha", "beta", "gamma"])
        assert len(results) == 3
        assert sent == [["alpha", "beta", "gamma"]]

    def test_many_strings_cost_one_request(self, sending):
        sent, send = self.sending_recorder()
        sending(send)
        scan_many([f"line {i}" for i in range(50)])
        assert len(sent) == 1, "50 strings must not cost 50 round trips"

    def test_a_repeated_string_is_sent_once(self, sending):
        sent, send = self.sending_recorder()
        sending(send)
        scan_many(["Ali Hassan", "other", "Ali Hassan"])
        assert sent[0].count("Ali Hassan") == 1

    def test_a_repeated_string_still_gets_its_own_result(self, sending):
        sending(finder("Ali Hassan"))
        results = scan_many(["Ali Hassan", "x", "Ali Hassan"])
        assert len(results) == 3
        assert [len(r.entities) for r in results] == [1, 0, 1]

    def test_a_batch_larger_than_the_cap_is_split(self, sending):
        sent, send = self.sending_recorder()
        sending(send)
        scan_many([f"line {i}" for i in range(MAX_BATCH_TEXTS + 40)])
        assert [len(group) for group in sent] == [MAX_BATCH_TEXTS, 40]

    def test_offsets_are_relative_to_their_own_text(self, sending):
        needle = "ali@example.com"
        texts = ["x " + needle, "no match here", "yyyy " + needle]
        sending(finder(needle, label="EMAIL"))
        results = scan_many(texts)
        for text, result in zip(texts, results, strict=True):
            for found in result.entities:
                assert text[found.start : found.end] == needle

    def test_blocked_is_reported_against_the_text_that_caused_it(self, sending):
        def send(texts):
            return [{"entities": [], "blocked": text == "bad"} for text in texts]

        sending(send)
        results = scan_many(["fine", "bad", "also fine"])
        assert [r.blocked for r in results] == [False, True, False]

    def test_empty_texts_cost_nothing_and_still_line_up(self, sending):
        sent, send = self.sending_recorder()
        sending(send)
        results = scan_many(["", "   ", "real"])
        assert sent == [["real"]]
        assert len(results) == 3 and results[0].entities == ()

    def test_a_short_response_is_an_error_rather_than_a_mismatch(self, sending):
        with pytest.raises(ScanError, match="one result per text"):
            sending(lambda texts: [{"entities": [], "blocked": False}])
            scan_many(["a", "b"])


class TestTransport:
    def test_missing_api_key_is_an_error(self, monkeypatch):
        monkeypatch.delenv(constants.API_KEY_ENV, raising=False)
        with pytest.raises(ScanError, match="no API key"):
            send_batch(["text"])

    def test_request_carries_the_key_and_url(self, monkeypatch):
        monkeypatch.setenv(constants.API_KEY_ENV, "secret-key")
        monkeypatch.setenv(constants.BASE_URL_ENV, "https://example.test")
        seen = {}

        class FakeResponse:
            def read(self):
                return json.dumps({"results": [{"entities": [], "blocked": False}]}).encode()

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout=None):
            seen["url"] = request.full_url
            seen["key"] = request.get_header("X-api-key")
            seen["body"] = json.loads(request.data)
            return FakeResponse()

        monkeypatch.setattr("rezunate_guard.scanner.urlopen", fake_urlopen)
        send_batch(["scan me"])

        assert seen["url"] == "https://example.test/api/v1/guardrails/scan-batch"
        assert seen["key"] == "secret-key"
        assert seen["body"] == {"texts": ["scan me"]}

    @pytest.mark.parametrize(
        ("error", "match"),
        [
            (HTTPError("u", 401, "Unauthorized", {}, None), "HTTP 401"),
            (URLError("connection refused"), "could not reach"),
        ],
    )
    def test_network_failures_become_scan_errors(self, monkeypatch, error, match):
        monkeypatch.setenv(constants.API_KEY_ENV, "k")

        def fail(request, timeout=None):
            raise error

        monkeypatch.setattr("rezunate_guard.scanner.urlopen", fail)
        with pytest.raises(ScanError, match=match):
            send_batch(["text"])
