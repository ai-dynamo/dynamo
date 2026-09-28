#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit tests for the CPython GC freeze helper and pause logger.

Everything here is pure stdlib (``gc``, ``logging``) so it runs without the
Rust extension or an inference backend installed.
"""

import gc
import logging
import weakref

import pytest

from dynamo.common.utils import gc_freeze

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.fixture(autouse=True)
def _restore_gc_state():
    """Undo the process-global side effects each test leaves behind.

    gc.freeze() is per-interpreter and would otherwise leak into every
    later test in the session (harmless, but it makes freeze-count
    assertions order-dependent). The pause logger is a module singleton.
    """
    enabled = gc.isenabled()
    yield
    gc_freeze.uninstall_gc_pause_logger()
    gc.unfreeze()
    if enabled:
        gc.enable()
    else:
        gc.disable()


class _Cycle:
    def __init__(self) -> None:
        self.me = self


def test_freeze_gc_heap_collects_then_freezes(caplog):
    """A dead cycle created before the freeze is reclaimed, not pinned.

    That ordering (collect first) is the whole difference between this helper
    and a bare gc.freeze(), and the reason the fpm gc_policy has to unfreeze
    before its maintenance collect. Proven with a weakref: a pinned cycle
    would keep the referent alive (and unreachable) forever.
    """
    gc.unfreeze()
    gc.collect()
    cycle = _Cycle()
    dead = weakref.ref(cycle)
    del cycle  # unreachable now; only the collector can free it
    assert dead() is not None
    before = gc.get_freeze_count()

    with caplog.at_level(logging.INFO, logger="dynamo.common.utils.gc_freeze"):
        frozen = gc_freeze.freeze_gc_heap(context="unit test")

    assert dead() is None, "cycle was frozen instead of collected"
    # Frozen objects that later die drop out of the permanent generation, so
    # the live count can only be <= what was reported at freeze time.
    assert before < gc.get_freeze_count() <= frozen
    record = next(r for r in caplog.records if "GC heap frozen" in r.getMessage())
    assert "unit test" in record.getMessage()
    assert f"gc.get_freeze_count()={frozen}" in record.getMessage()


def test_freeze_is_repeatable_and_pins_new_static_state():
    """A second freeze after more long-lived state exists pins that too.

    The frontend freezes at startup and again per model registration; the
    later call must extend the permanent generation, not be a no-op.
    """
    gc.unfreeze()
    first = gc_freeze.freeze_gc_heap(context="first")
    # Lists, not object(): a bare object() holds no references and is never
    # GC-tracked, so it would not show up in the freeze count.
    keep_alive = [[] for _ in range(1000)]
    second = gc_freeze.freeze_gc_heap(context="second")
    assert second >= first + len(keep_alive)
    del keep_alive


def test_maybe_freeze_gc_heap_disabled_is_noop(caplog):
    gc.unfreeze()
    before = gc.get_freeze_count()
    with caplog.at_level(logging.DEBUG, logger="dynamo.common.utils.gc_freeze"):
        assert gc_freeze.maybe_freeze_gc_heap(False, context="off") is None
    assert gc.get_freeze_count() == before
    assert not any("GC heap frozen" in r.getMessage() for r in caplog.records)
    assert any("disabled" in r.getMessage() for r in caplog.records)


def test_maybe_freeze_gc_heap_enabled_freezes():
    gc.unfreeze()
    frozen = gc_freeze.maybe_freeze_gc_heap(True, context="on")
    assert frozen is not None and frozen > 0
    assert 0 < gc.get_freeze_count() <= frozen


def test_frozen_objects_are_skipped_by_full_collection():
    """The property the freeze exists for: gen2 no longer walks the frozen set.

    Measured indirectly and deterministically via gc.get_objects(), which
    only enumerates non-permanent generations.
    """
    gc.unfreeze()
    sentinel = _Cycle()  # kept alive by the local name
    gc_freeze.freeze_gc_heap(context="sentinel")
    assert not any(o is sentinel for o in gc.get_objects())
    gc.unfreeze()
    assert any(o is sentinel for o in gc.get_objects())


class TestGcPauseLogger:
    def _run(self, pause_logger, generation, pause_ms):
        """Drive the callback as CPython would, with a fake clock."""
        clock = {"t": 100.0}
        original = gc_freeze.time.perf_counter
        gc_freeze.time.perf_counter = lambda: clock["t"]
        try:
            pause_logger("start", {"generation": generation})
            clock["t"] += pause_ms / 1000.0
            pause_logger(
                "stop", {"generation": generation, "collected": 3, "uncollectable": 0}
            )
        finally:
            gc_freeze.time.perf_counter = original

    def test_logs_only_at_or_above_threshold(self, caplog):
        log = logging.getLogger("test.gc_pause")
        pause_logger = gc_freeze.GcPauseLogger(threshold_ms=50.0, log=log)
        with caplog.at_level(logging.WARNING, logger="test.gc_pause"):
            self._run(pause_logger, generation=0, pause_ms=5.0)
            # Comfortably either side of the threshold: the fake clock adds
            # pause_ms / 1000 to a float, so an exact 50.0 is not reproducible.
            self._run(pause_logger, generation=2, pause_ms=49.0)
            self._run(pause_logger, generation=2, pause_ms=64.0)
            self._run(pause_logger, generation=2, pause_ms=400.0)
        messages = [r.getMessage() for r in caplog.records]
        assert len(messages) == 2
        assert "GC gen2 pause 64.0 ms" in messages[0]
        assert "GC gen2 pause 400.0 ms" in messages[1]
        assert "collected=3" in messages[1]
        assert pause_logger.pauses_logged == 2
        assert pause_logger.max_pause_ms == pytest.approx(400.0)

    def test_stop_without_start_is_ignored(self, caplog):
        log = logging.getLogger("test.gc_pause")
        pause_logger = gc_freeze.GcPauseLogger(threshold_ms=0.0, log=log)
        with caplog.at_level(logging.WARNING, logger="test.gc_pause"):
            pause_logger("stop", {"generation": 1, "collected": 0, "uncollectable": 0})
            pause_logger("bogus", {"generation": 1})
        assert not caplog.records

    def test_real_collection_reaches_the_hook(self, caplog):
        """End to end through gc.callbacks with a zero threshold."""
        pause_logger = gc_freeze.install_gc_pause_logger(threshold_ms=0.001)
        assert pause_logger is not None
        assert pause_logger in gc.callbacks
        with caplog.at_level(logging.WARNING, logger="dynamo.common.utils.gc_freeze"):
            gc.collect()
        assert any("GC gen2 pause" in r.getMessage() for r in caplog.records)

    def test_install_is_idempotent_and_gated(self):
        assert gc_freeze.install_gc_pause_logger(0) is None
        assert gc_freeze.install_gc_pause_logger(-5) is None
        first = gc_freeze.install_gc_pause_logger(10.0)
        second = gc_freeze.install_gc_pause_logger(99.0)
        assert first is second
        assert first.threshold_ms == 10.0
        assert gc.callbacks.count(first) == 1
        gc_freeze.uninstall_gc_pause_logger()
        assert first not in gc.callbacks
        gc_freeze.uninstall_gc_pause_logger()  # second call is a no-op
