# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for dynamo.common.utils.nvtx_utils.

``nvtx_utils`` decides everything at import time: it reads ``DYN_NVTX`` once and
then defines either the real NVTX wrappers or pure-Python no-ops. So every test
here reloads the module from source under a patched environment.

The module is loaded **by file path** rather than as
``dynamo.common.utils.nvtx_utils``: importing it through the package would run
``dynamo/common/utils/__init__.py``, which eagerly imports modules that need the
compiled ``dynamo._core`` extension. Loading the single file keeps these tests
runnable without a built wheel, and it is also what makes reloading under
different environments cheap and side-effect free.

The optional ``nvtx`` package is never required: the enabled-path tests inject a
stub into ``sys.modules``, and the missing-package test forces the import to
fail regardless of whether the real package happens to be installed.
"""

import asyncio
import functools
import importlib.abc
import importlib.util
import sys
import types
from pathlib import Path
from typing import ClassVar

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]

NVTX_UTILS_PATH = Path(__file__).resolve().parents[1] / "nvtx_utils.py"

# Load under a private name so a real ``dynamo.common.utils.nvtx_utils`` entry in
# ``sys.modules`` (if the package is importable in this environment) is untouched.
MODULE_NAME = "_test_nvtx_utils_under_test"


def _load_nvtx_utils():
    spec = importlib.util.spec_from_file_location(MODULE_NAME, NVTX_UTILS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE_NAME] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        # A failed import must not leave a half-initialised module behind.
        sys.modules.pop(MODULE_NAME, None)
        raise
    return module


@pytest.fixture
def load_module(monkeypatch):
    """Reload nvtx_utils from source, cleaning up sys.modules afterwards."""
    yield _load_nvtx_utils
    sys.modules.pop(MODULE_NAME, None)


class _NvtxImportRecorder(importlib.abc.MetaPathFinder):
    def __init__(self, block: bool):
        self.block = block
        self.attempts = 0

    def find_spec(self, fullname, path=None, target=None):
        if fullname != "nvtx" and not fullname.startswith("nvtx."):
            return None
        self.attempts += 1
        if self.block:
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        # Fall through to the normal import machinery.
        return None


@pytest.fixture
def nvtx_import_watch(monkeypatch):
    """Install a finder that observes imports of ``nvtx``.

    Also drops any cached ``nvtx`` module so the import actually reaches the
    finder. monkeypatch restores both on teardown.
    """

    def _install(block: bool = False):
        for name in [m for m in sys.modules if m == "nvtx" or m.startswith("nvtx.")]:
            monkeypatch.delitem(sys.modules, name, raising=False)
        recorder = _NvtxImportRecorder(block=block)
        monkeypatch.setattr(sys, "meta_path", [recorder, *sys.meta_path])
        return recorder

    return _install


# --------------------------------------------------------------------------- #
# Disabled (default) path
# --------------------------------------------------------------------------- #


class TestDisabled:
    @pytest.mark.parametrize("dyn_nvtx", [None, "0"])
    def test_disabled_never_imports_nvtx(
        self, monkeypatch, load_module, nvtx_import_watch, dyn_nvtx
    ):
        if dyn_nvtx is None:
            monkeypatch.delenv("DYN_NVTX", raising=False)
        else:
            monkeypatch.setenv("DYN_NVTX", dyn_nvtx)
        recorder = nvtx_import_watch()

        mod = load_module()

        assert mod.ENABLED is False
        assert recorder.attempts == 0, "nvtx must not be imported when DYN_NVTX is off"
        assert not hasattr(mod, "_nvtx_lib")

    @pytest.fixture
    def disabled(self, monkeypatch, load_module):
        monkeypatch.setenv("DYN_NVTX", "0")
        return load_module()

    def test_every_entry_point_is_a_transparent_noop(self, disabled):
        """Off must mean pass-through, not "wrapped in something cheap".

        Asserted as identity (`is target`) rather than behaviour: a wrapper that
        merely forwards would still pass a call-through check while costing a
        frame per call on paths that run per token.
        """

        def sync_target(a, b):
            return a + b

        assert disabled.start_range("my:range", color="blue") is None
        # The handle is ignored entirely, so an arbitrary one is also accepted.
        assert disabled.end_range(None) is None
        assert disabled.end_range(object()) is None

        assert disabled.annotate("my:func", color="green")(sync_target) is sync_target
        # Defaults exist on the no-op path; call sites elsewhere rely on them.
        assert disabled.annotate()(len) is len
        assert (
            disabled.range_decorator("my:sync", color="green")(sync_target)
            is sync_target
        )

        with disabled.annotate("my:block", color="cyan") as ctx:
            pass
        assert ctx is not None

    def test_annotate_context_manager_does_not_swallow_exceptions(self, disabled):
        with pytest.raises(ValueError, match="boom"):
            with disabled.annotate("my:block"):
                raise ValueError("boom")


# --------------------------------------------------------------------------- #
# Enabled without the optional package
# --------------------------------------------------------------------------- #


class TestEnabledWithoutPackage:
    """A missing package degrades to no-ops: launch presets set DYN_NVTX for
    every process, and an image without the package must still serve."""

    def test_missing_package_warns_and_disables(
        self, monkeypatch, load_module, nvtx_import_watch, caplog
    ):
        monkeypatch.setenv("DYN_NVTX", "1")
        recorder = nvtx_import_watch(block=True)

        with caplog.at_level("WARNING"):
            module = load_module()

        assert recorder.attempts == 1
        assert module.ENABLED is False
        assert "DYN_NVTX" in caplog.text
        assert "not installed" in caplog.text
        assert module.start_range("x") is None
        module.end_range(None)
        with module.annotate("x"):
            pass

    def test_request_scoped_markers_do_not_need_the_package(
        self, monkeypatch, load_module, nvtx_import_watch, fake_core
    ):
        # Request-scoped markers go through dynamo._core, so a missing nvtx
        # package must disable only the package-backed helpers.
        monkeypatch.setenv("DYN_NVTX", "1")
        nvtx_import_watch(block=True)
        module = load_module()

        assert module.ENABLED is False
        assert module.REQUEST_ENABLED is True
        module.request_mark("frontend.first_token", "rid")
        with module.request_range("stage", "rid"):
            pass
        assert [entry[0] for entry in fake_core.log] == ["mark", "start", "end"]

    def test_broken_package_is_not_swallowed(
        self, monkeypatch, load_module, nvtx_import_watch
    ):
        """Only a *missing* nvtx degrades; a broken install stays visible."""
        monkeypatch.setenv("DYN_NVTX", "1")
        broken = types.ModuleType("nvtx")

        class _Finder(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "nvtx":
                    raise ModuleNotFoundError("missing dep", name="some_dep")
                return None

        finder = _Finder()
        monkeypatch.delitem(sys.modules, "nvtx", raising=False)
        monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
        del broken

        with pytest.raises(ModuleNotFoundError, match="missing dep"):
            load_module()

        assert MODULE_NAME not in sys.modules


# --------------------------------------------------------------------------- #
# DYN_NVTX parsing
# --------------------------------------------------------------------------- #


class TestEnvParsing:
    """DYN_NVTX parses the same way in Python and in the Rust runtime.

    Both accept 1/true/on/yes and 0/false/off/no, trimmed and case-insensitive
    (``dynamo_truthy::is_truthy``).
    """

    @pytest.mark.parametrize("value", ["1", "true", "yes", "on", "TRUE", " on "])
    def test_truthy_values_enable(self, monkeypatch, load_module, stub_nvtx, value):
        monkeypatch.setenv("DYN_NVTX", value)
        assert load_module().ENABLED is True

    @pytest.mark.parametrize(
        "value", ["0", "false", "no", "off", "FALSE", " off ", "", "  "]
    )
    def test_falsey_values_disable(self, monkeypatch, load_module, value):
        monkeypatch.setenv("DYN_NVTX", value)
        assert load_module().ENABLED is False

    @pytest.mark.parametrize("value", ["2", "maybe", "enabled", "-1"])
    def test_unrecognized_values_raise(self, monkeypatch, load_module, value):
        """Fail loudly rather than defaulting to off.

        Silently disabling on an unrecognized value is the failure this module
        exists to prevent: the run completes and the timeline is empty.
        """
        monkeypatch.setenv("DYN_NVTX", value)
        with pytest.raises(ValueError, match="not a recognized boolean"):
            load_module()


# --------------------------------------------------------------------------- #
# Enabled with the optional package (simulated)
# --------------------------------------------------------------------------- #


class _StubDomain:
    def __init__(self, name):
        self.name = name
        self.created_attributes = []
        self.started = []
        self.ended = []

    def get_event_attributes(self, message, color):
        attr = (message, color)
        self.created_attributes.append(attr)
        return attr

    def start_range(self, attr):
        handle = len(self.started)
        self.started.append(attr)
        return handle

    def end_range(self, handle):
        self.ended.append(handle)


class _StubAnnotate:
    instances: ClassVar[list] = []

    def __init__(self, message="", color="white", domain=None):
        self.message = message
        self.color = color
        self.domain = domain
        _StubAnnotate.instances.append(self)

    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper.__wrapped_by_nvtx__ = self  # type: ignore[attr-defined]
        return wrapper

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


@pytest.fixture
def stub_nvtx(monkeypatch):
    """Inject a fake ``nvtx`` module and return (module, domains-by-name)."""
    domains: dict = {}

    def get_domain(name):
        return domains.setdefault(name, _StubDomain(name))

    stub = types.ModuleType("nvtx")
    stub.__path__ = []  # type: ignore[attr-defined]
    stub.get_domain = get_domain  # type: ignore[attr-defined]
    stub.annotate = _StubAnnotate  # type: ignore[attr-defined]
    # Identity resolver, so the attribute cache is still observable by color name.
    colors = types.ModuleType("nvtx.colors")
    colors.color_to_hex = lambda color: color  # type: ignore[attr-defined]
    stub.colors = colors  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nvtx", stub)
    monkeypatch.setitem(sys.modules, "nvtx.colors", colors)
    _StubAnnotate.instances = []
    return stub, domains


@pytest.fixture
def enabled(monkeypatch, load_module, stub_nvtx):
    monkeypatch.setenv("DYN_NVTX", "1")
    module = load_module()
    _stub, domains = stub_nvtx
    return module, domains["dynamo"]


class TestEnabledWithPackage:
    def test_ranges_go_to_the_dynamo_domain_with_cached_attributes(self, enabled):
        """The three properties the enabled path has to get right: markers land in
        the `dynamo` domain, one EventAttributes object is built per
        (message, color) and reused, and a handle round-trips to `end_range`."""
        module, domain = enabled
        assert module.ENABLED is True
        assert domain.name == "dynamo"

        first = module.start_range("my:range", color="blue")
        second = module.start_range("my:range", color="blue")
        module.start_range("my:range", color="red")
        module.start_range("other:range", color="blue")
        module.start_range("plain:range")

        assert first != second
        assert domain.created_attributes == [
            ("my:range", "blue"),
            ("my:range", "red"),
            ("other:range", "blue"),
            ("plain:range", "white"),
        ]
        assert len(domain.started) == 5

        module.end_range(first)
        assert domain.ended == [first]

    def test_annotate_is_bound_to_the_dynamo_domain(self, enabled):
        module, _domain = enabled
        with module.annotate("my:block", color="cyan") as ctx:
            pass
        assert ctx.domain == "dynamo"
        assert ctx.message == "my:block"

    def test_range_decorator_wraps_sync_function(self, enabled):
        module, domain = enabled

        @module.range_decorator("my:sync", color="green")
        def target(x):
            return x * 2

        assert target(4) == 8
        assert domain.created_attributes == [("my:sync", "green")]
        assert len(domain.started) == 1
        assert domain.ended == [0]
        assert target.__name__ == "target"

    def test_range_decorator_ends_range_on_exception(self, enabled):
        module, domain = enabled

        @module.range_decorator("my:sync")
        def target():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            target()
        assert domain.ended == [0]

    def test_range_decorator_spans_full_async_generator(self, enabled):
        module, domain = enabled

        @module.range_decorator("my:async_gen", color="green")
        async def target():
            # The range must already be open before the first item is produced,
            # and must stay open until the generator is exhausted.
            assert len(domain.started) == 1
            assert domain.ended == []
            yield 1
            yield 2

        async def collect():
            return [item async for item in target()]

        assert asyncio.run(collect()) == [1, 2]
        assert len(domain.started) == 1
        assert domain.ended == [0]

    def test_range_decorator_ends_range_when_async_generator_is_abandoned(
        self, enabled
    ):
        module, domain = enabled

        @module.range_decorator("my:async_gen")
        async def target():
            yield 1
            yield 2

        async def take_one():
            agen = target()
            async for item in agen:
                assert item == 1
                break
            await agen.aclose()

        asyncio.run(take_one())
        assert domain.ended == [0]


# --------------------------------------------------------------------------- #
# Request-scoped markers and ranges (backed by a fake dynamo._core)
# --------------------------------------------------------------------------- #


class _FakeRange:
    def __init__(self, log, stage, rid, attrs):
        self.log = log
        self.stage = stage
        self.rid = rid
        self.attrs = attrs
        self.ends = 0

    def end(self):
        self.ends += 1
        self.log.append(("end", self.stage, self.rid))


class _FakeCore:
    """Records every call made through the dynamo._core NVTX binding."""

    def __init__(self):
        self.log: list = []
        self.ranges: list[_FakeRange] = []
        self.rust_enabled = True

    def nvtx_request_mark(self, stage, request_id, attrs=None):
        self.log.append(("mark", stage, request_id, attrs))

    def nvtx_request_range(self, stage, request_id, attrs=None):
        rng = _FakeRange(self.log, stage, request_id, attrs)
        self.ranges.append(rng)
        self.log.append(("start", stage, request_id, attrs))
        return rng


@pytest.fixture
def fake_core(monkeypatch):
    core = types.ModuleType("dynamo._core")
    fake = _FakeCore()
    core.nvtx_request_mark = fake.nvtx_request_mark  # type: ignore[attr-defined]
    core.nvtx_request_range = fake.nvtx_request_range  # type: ignore[attr-defined]
    core.nvtx_enabled = lambda: fake.rust_enabled  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "dynamo._core", core)
    return fake


class TestRequestScopedDisabled:
    @pytest.fixture
    def disabled(self, monkeypatch, load_module):
        monkeypatch.setenv("DYN_NVTX", "0")
        return load_module()

    def test_everything_is_a_noop_and_never_touches_core(
        self, disabled, fake_core, monkeypatch
    ):
        class Exploding:
            def __str__(self):
                raise AssertionError("string built while disabled")

        disabled.request_mark("x", "rid", k=Exploding())
        assert disabled.start_request_range("x", "rid", k=Exploding()) is None
        disabled.end_request_range(None)
        with disabled.request_range("x", "rid") as handle:
            assert handle is None
        stages = disabled.handler_stages("decode", "rid")
        assert stages is disabled.NO_STAGES
        stages.end_prepare()
        stages.engine_start("e")
        assert stages.first_output_pending is False
        stages.first_output()
        stages.close()
        assert fake_core.log == []
        assert disabled._core_api is None, "dynamo._core must not even be imported"


class TestRequestScopedEnabled:
    def test_mark_and_range_delegate_with_attrs(self, enabled, fake_core):
        module, _domain = enabled
        module.request_mark("frontend.x", "rid-1", xrid="abc")
        handle = module.start_request_range("a:b", "rid-1", engine_rid="e1", none=None)
        module.end_request_range(handle)
        module.end_request_range(handle)  # idempotent at the wrapper level

        assert fake_core.log[0] == ("mark", "frontend.x", "rid-1", {"xrid": "abc"})
        # None-valued attrs are dropped; no attrs at all becomes None.
        assert fake_core.log[1] == ("start", "a:b", "rid-1", {"engine_rid": "e1"})
        module.request_mark("plain", "rid-2")
        assert fake_core.log[-1] == ("mark", "plain", "rid-2", None)

    def test_does_not_use_the_nvtx_package(self, enabled, fake_core):
        """Per-request strings must not reach nvtx's unbounded string cache."""
        module, domain = enabled
        module.request_mark("m", "r")
        with module.request_range("r", "r"):
            pass
        module.handler_stages("prefill", "r").close()
        assert domain.created_attributes == []
        assert domain.started == []
        assert module._attr_cache == {}

    def test_context_manager_ends_once_on_exception(self, enabled, fake_core):
        module, _domain = enabled
        with pytest.raises(ValueError, match="boom"):
            with module.request_range("s", "rid"):
                raise ValueError("boom")
        assert [r.ends for r in fake_core.ranges] == [1]

    def test_missing_core_degrades_with_one_warning(self, enabled, monkeypatch, caplog):
        module, _domain = enabled
        monkeypatch.setitem(sys.modules, "dynamo._core", None)  # import fails
        with caplog.at_level("WARNING"):
            module.request_mark("x", "rid")
            assert module.start_request_range("x", "rid") is None
            assert module.handler_stages("decode", "rid") is module.NO_STAGES
        warnings = [r for r in caplog.records if "dynamo._core" in r.getMessage()]
        assert len(warnings) == 1

    def test_core_without_functions_degrades(self, enabled, monkeypatch, caplog):
        module, _domain = enabled
        monkeypatch.setitem(
            sys.modules, "dynamo._core", types.ModuleType("dynamo._core")
        )
        with caplog.at_level("WARNING"):
            module.request_mark("x", "rid")
            assert module.start_request_range("x", "rid") is None
        assert len([r for r in caplog.records if "dynamo._core" in r.getMessage()]) == 1


class TestHandlerStages:
    def test_full_lifecycle_orders_ranges(self, enabled, fake_core):
        module, _domain = enabled
        stages = module.handler_stages("prefill", "rid-7")
        assert fake_core.log == [
            ("start", "worker.handler.prepare:prefill", "rid-7", None)
        ]
        stages.engine_start("engine-9")
        assert fake_core.log[1:] == [
            ("end", "worker.handler.prepare:prefill", "rid-7"),
            (
                "start",
                "worker.engine.first_output:prefill",
                "rid-7",
                {"engine_rid": "engine-9"},
            ),
        ]
        assert stages.first_output_pending is True
        stages.first_output()
        assert stages.first_output_pending is False
        stages.first_output()  # later chunks are harmless
        stages.close()
        assert [r.ends for r in fake_core.ranges] == [1, 1]

    def test_engine_rid_equal_to_rid_is_omitted(self, enabled, fake_core):
        module, _domain = enabled
        stages = module.handler_stages("decode", "same")
        stages.engine_start("same")
        assert fake_core.ranges[1].attrs is None
        stages.close()

    def test_split_api_for_late_engine_id(self, enabled, fake_core):
        module, _domain = enabled
        stages = module.handler_stages("aggregated", "rid")
        stages.end_prepare()
        assert fake_core.ranges[0].ends == 1
        stages.start_first_output("42")
        assert fake_core.ranges[1].attrs == {"engine_rid": "42"}
        stages.close()
        stages.close()
        assert [r.ends for r in fake_core.ranges] == [1, 1]

    def test_engine_rid_from_end_prepare_carries_to_a_lazy_first_output(
        self, enabled, fake_core
    ):
        # SGLang prefill: the engine id is known at the generate call, but the lazy
        # stream is consumed (and first_output opened) only after the bootstrap yield.
        module, _domain = enabled
        stages = module.handler_stages("prefill", "rid")
        stages.end_prepare("sgl-7")
        assert len(fake_core.ranges) == 1, "first_output must not open yet"
        stages.start_first_output()
        assert fake_core.ranges[1].attrs == {"engine_rid": "sgl-7"}
        stages.close()

    def test_close_before_engine_ends_prepare_only(self, enabled, fake_core):
        module, _domain = enabled
        stages = module.handler_stages("decode", "rid")
        stages.close()
        assert len(fake_core.ranges) == 1
        assert fake_core.ranges[0].ends == 1

    def test_close_without_output_ends_first_output(self, enabled, fake_core):
        module, _domain = enabled
        stages = module.handler_stages("decode", "rid")
        stages.engine_start()
        stages.close()
        assert [r.ends for r in fake_core.ranges] == [1, 1]

    def test_ends_on_async_generator_close_and_cancellation(self, enabled, fake_core):
        module, _domain = enabled

        async def handler(stages):
            try:
                stages.engine_start()
                yield 1
                await asyncio.sleep(3600)
                yield 2
            finally:
                stages.close()

        async def early_close():
            stages = module.handler_stages("decode", "a")
            agen = handler(stages)
            assert await agen.__anext__() == 1
            await agen.aclose()

        async def cancelled():
            stages = module.handler_stages("decode", "b")

            async def consume():
                async for _ in handler(stages):
                    pass

            task = asyncio.ensure_future(consume())
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

        asyncio.run(early_close())
        asyncio.run(cancelled())
        assert [r.ends for r in fake_core.ranges] == [1, 1, 1, 1]


def test_rust_side_off_skips_markers_without_caching_the_answer(
    monkeypatch, load_module, fake_core, caplog
):
    # DYN_NVTX=1 on a wheel built without the nvtx feature (or before runtime init):
    # warn once, emit nothing, and pick up a later "on" answer.
    monkeypatch.setenv("DYN_NVTX", "1")
    module = load_module()
    fake_core.rust_enabled = False
    with caplog.at_level("WARNING"):
        assert module.handler_stages("decode", "rid") is module.NO_STAGES
        module.request_mark("frontend.first_token", "rid")
    assert fake_core.log == []
    assert caplog.text.count("Rust runtime reports NVTX off") == 1

    fake_core.rust_enabled = True
    module.request_mark("frontend.first_token", "rid")
    assert [entry[0] for entry in fake_core.log] == ["mark"]
