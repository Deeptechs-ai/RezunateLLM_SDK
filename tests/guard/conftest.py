"""Keep the test run off the developer's machine and off the network."""

from __future__ import annotations

import pytest

from rezunate_guard import constants


@pytest.fixture(autouse=True)
def isolated_home(tmp_path_factory, monkeypatch):
    home = tmp_path_factory.mktemp("rezunate-home")
    monkeypatch.setenv("REZUNATE_HOME", str(home))
    monkeypatch.delenv(constants.API_KEY_ENV, raising=False)
    return home
