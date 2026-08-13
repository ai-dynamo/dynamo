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
    """Execute nvtx_utils.py fresh and return the resulting module."""
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
    """Records (and optionally blocks) attempts to import ``nvtx``."""

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

    def test_start_and_end_range_are_noops(self, disabled):
        rng = disabled.start_range("my:range", color="blue")
        assert rng is None
        assert disabled.end_range(rng) is None
        # The handle is ignored entirely, so an arbitrary one is also accepted.
        assert disabled.end_range(object()) is None

    def test_annotate_as_decorator_returns_function_unchanged(self, disabled):
        def target(a, b):
            return a + b

        decorated = disabled.annotate("my:func", color="green")(target)

        assert decorated is target
        assert decorated(1, 2) == 3

    def test_annotate_as_context_manager(self, disabled):
        with disabled.annotate("my:block", color="cyan") as ctx:
            value = 42
        assert ctx is not None
        assert value == 42

    def test_annotate_context_manager_does_not_swallow_exceptions(self, disabled):
        with pytest.raises(ValueError, match="boom"):
            with disabled.annotate("my:block"):
                raise ValueError("boom")

    def test_annotate_accepts_no_arguments(self, disabled):
        # Defaults exist on the no-op path; call sites elsewhere rely on them.
        assert disabled.annotate()(len) is len

    def test_range_decorator_returns_sync_function_unchanged(self, disabled):
        def target(x):
            return x * 2

        decorated = disabled.range_decorator("my:sync", color="green")(target)

        assert decorated is target
        assert decorated(3) == 6

    def test_range_decorator_returns_async_gen_unchanged(self, disabled):
        async def target():
            yield 1
            yield 2

        decorated = disabled.range_decorator("my:async_gen")(target)

        assert decorated is target

        async def collect():
            return [item async for item in decorated()]

        assert asyncio.run(collect()) == [1, 2]


# --------------------------------------------------------------------------- #
# Enabled without the optional package
# --------------------------------------------------------------------------- #


class TestEnabledWithoutPackage:
    def test_import_raises_with_actionable_message(
        self, monkeypatch, load_module, nvtx_import_watch
    ):
        monkeypatch.setenv("DYN_NVTX", "1")
        recorder = nvtx_import_watch(block=True)

        with pytest.raises(ImportError) as excinfo:
            load_module()

        assert recorder.attempts == 1
        message = str(excinfo.value)
        assert "ai-dynamo[profiling]" in message
        assert "DYN_NVTX" in message
        # The original failure is chained, not swallowed.
        assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)
        assert excinfo.value.__cause__.name == "nvtx"

    def test_failed_import_leaves_sys_modules_clean(
        self, monkeypatch, load_module, nvtx_import_watch
    ):
        monkeypatch.setenv("DYN_NVTX", "1")
        nvtx_import_watch(block=True)

        with pytest.raises(ImportError):
            load_module()

        assert MODULE_NAME not in sys.modules


# --------------------------------------------------------------------------- #
# DYN_NVTX parsing
# --------------------------------------------------------------------------- #


class TestEnvParsing:
    """DYN_NVTX parses exactly like its Rust twin, DYN_ENABLE_RUST_NVTX.

    Both accept 1/true/on/yes and 0/false/off/no, trimmed and case-insensitive
    (``dynamo_truthy::is_truthy``). Two regressions are guarded here: the
    original ``bool(int(...))``, under which ``true`` raised a bare
    ``ValueError``; and a later ``env_bool`` form, under which ``on`` silently
    recorded nothing while the Rust half of the same capture was enabled.
    """

    @pytest.mark.parametrize(
        "value", ["1", "true", "TRUE", "True", "yes", "YES", "on", "ON", " 1 ", " on "]
    )
    def test_truthy_values_enable(self, monkeypatch, load_module, stub_nvtx, value):
        monkeypatch.setenv("DYN_NVTX", value)
        assert load_module().ENABLED is True

    @pytest.mark.parametrize(
        "value", ["0", "false", "FALSE", "no", "off", "OFF", "", "  "]
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
    """Minimal stand-in for ``nvtx.Domain``."""

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
    """Stand-in for ``nvtx.annotate``: decorator and context manager."""

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
    stub.get_domain = get_domain  # type: ignore[attr-defined]
    stub.annotate = _StubAnnotate  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "nvtx", stub)
    _StubAnnotate.instances = []
    return stub, domains


@pytest.fixture
def enabled(monkeypatch, load_module, stub_nvtx):
    monkeypatch.setenv("DYN_NVTX", "1")
    module = load_module()
    _stub, domains = stub_nvtx
    return module, domains["dynamo"]


class TestEnabledWithPackage:
    def test_enabled_flag_and_domain(self, enabled):
        module, domain = enabled
        assert module.ENABLED is True
        assert domain.name == "dynamo"

    def test_start_range_uses_cached_event_attributes(self, enabled):
        module, domain = enabled

        first = module.start_range("my:range", color="blue")
        second = module.start_range("my:range", color="blue")
        module.start_range("my:range", color="red")
        module.start_range("other:range", color="blue")

        assert first != second
        # One EventAttributes object per (message, color), reused thereafter.
        assert domain.created_attributes == [
            ("my:range", "blue"),
            ("my:range", "red"),
            ("other:range", "blue"),
        ]
        assert len(domain.started) == 4

    def test_start_range_defaults_to_white(self, enabled):
        module, domain = enabled
        module.start_range("my:range")
        assert domain.created_attributes == [("my:range", "white")]

    def test_end_range_forwards_the_handle(self, enabled):
        module, domain = enabled
        rng = module.start_range("my:range")
        module.end_range(rng)
        assert domain.ended == [rng]

    def test_annotate_is_bound_to_the_dynamo_domain(self, enabled):
        module, _domain = enabled
        assert isinstance(module.annotate, functools.partial)
        assert module.annotate.keywords == {"domain": "dynamo"}

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
