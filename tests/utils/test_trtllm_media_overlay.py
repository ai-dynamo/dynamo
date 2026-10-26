# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source overlay edits preserve unrelated release definitions."""

import sys
from types import ModuleType

import pytest

from tests.utils.trtllm_media_overlay import _overlay_server, _replace_definition

pytestmark = [pytest.mark.unit, pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.mark.parametrize("existing", [False, True])
def test_replace_processor_method_preserves_release_class(tmp_path, existing):
    destination = tmp_path / "installed.py"
    source = tmp_path / "source.py"
    destination.write_text(
        "class Processor:\n    release_value = 7\n"
        + ("    def feature(self):\n        return 0\n" if existing else "")
        + "\ndef launcher():\n    return 'release'\n"
    )
    source.write_text(
        "class Processor:\n    @classmethod\n"
        "    def feature(cls):\n        return cls.release_value\n"
    )
    _replace_definition(destination, source, "Processor", ["feature"])
    namespace = {}
    exec(destination.read_text(), namespace)
    assert namespace["Processor"].feature() == 7
    assert namespace["launcher"]() == "release"


def test_replace_server_class_preserves_release_launcher(tmp_path):
    destination = tmp_path / "installed.py"
    source = tmp_path / "source.py"
    destination.write_text(
        "class Server:\n    workers = 0\n\ndef launcher():\n    return 'release'\n"
    )
    source.write_text(
        "class Server:\n    workers = 4\n\ndef launcher():\n    return 'new-main'\n"
    )
    _replace_definition(destination, source, None, ["Server"])
    namespace = {}
    exec(destination.read_text(), namespace)
    assert namespace["Server"].workers == 4
    assert namespace["launcher"]() == "release"


def test_overlay_server_installs_startup_dependencies(tmp_path, monkeypatch):
    destination = tmp_path / "installed.py"
    source = tmp_path / "source.py"
    destination.write_text(
        "from typing import Any\n"
        "class OpenEngineServer:\n    workers = 0\n"
        "\ndef launcher():\n    return 'release'\n"
    )
    source.write_text("class OpenEngineServer:\n    workers = 4\n")
    coordinator = ModuleType("overlay_test.coordinator")
    coordinator.FrontendClient = object()
    events = ModuleType("overlay_test.kv_events")
    events.KvEventsUnavailableError = type(
        "KvEventsUnavailableError", (RuntimeError,), {}
    )
    events.events_config = object()
    monkeypatch.setitem(sys.modules, coordinator.__name__, coordinator)
    monkeypatch.setitem(sys.modules, events.__name__, events)

    _overlay_server(destination, source)
    namespace = {"__package__": "overlay_test"}
    exec(destination.read_text(), namespace)
    assert namespace["events_config"] is events.events_config
    assert namespace["KvEventsUnavailableError"] is events.KvEventsUnavailableError
    assert namespace["FrontendClient"] is coordinator.FrontendClient
    assert namespace["OpenEngineServer"].workers == 4
    assert namespace["launcher"]() == "release"
