# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for BaseWorkerHandler generic tokenizer_manager passthrough.

These tests mock out heavy dependencies (sglang, dynamo._core) so they run in
a lightweight venv with only pytest + pytest-asyncio.
"""

import dataclasses
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.parallel,
]


# ---------------------------------------------------------------------------
# Stub out native / heavy imports, then load handler_base directly from file
# ---------------------------------------------------------------------------


class _StubModule(ModuleType):
    """Module stub that returns MagicMock for any unknown attribute."""

    def __getattr__(self, attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        return MagicMock()


def _ensure_mock_module(name):
    if name not in sys.modules:
        mod = _StubModule(name)
        mod.__path__ = []
        mod.__package__ = name
        sys.modules[name] = mod


_ensure_mock_module("dynamo._core")
_ensure_mock_module("dynamo.common")
_ensure_mock_module("dynamo.common.utils")
_ensure_mock_module("dynamo.common.utils.input_params")
_ensure_mock_module("dynamo.sglang.args")
_ensure_mock_module("dynamo.sglang.publisher")
_ensure_mock_module("yaml")

for _mod in [
    "sglang",
    "sglang.srt",
    "sglang.srt.utils",
    "sglang.srt.managers",
    "sglang.srt.managers.io_struct",
]:
    _ensure_mock_module(_mod)

# Load handler_base.py directly — bypasses request_handlers/__init__.py
# which re-exports every handler and triggers a huge import chain.
_REPO = Path(__file__).resolve().parent.parent
_handler_base_path = (
    _REPO
    / "components"
    / "src"
    / "dynamo"
    / "sglang"
    / "request_handlers"
    / "handler_base.py"
)
_spec = importlib.util.spec_from_file_location(
    "dynamo.sglang.request_handlers.handler_base", _handler_base_path
)
_handler_base = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _handler_base
_spec.loader.exec_module(_handler_base)
BaseWorkerHandler = _handler_base.BaseWorkerHandler


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _make_handler():
    """Create a concrete BaseWorkerHandler with mocked engine/config."""

    class _ConcreteHandler(BaseWorkerHandler):
        async def generate(self, request, context):
            yield {}

    engine = MagicMock()
    engine.tokenizer_manager = MagicMock()
    engine.tokenizer_manager.tokenizer = MagicMock()
    engine.async_generate = MagicMock()

    config = MagicMock()
    config.serving_mode = "prefill_decode"
    config.server_args.skip_tokenizer_init = True
    config.server_args.enable_trace = False

    handler = _ConcreteHandler(engine=engine, config=config)
    return handler


# ---------------------------------------------------------------------------
# _resolve_arg
# ---------------------------------------------------------------------------


class TestResolveArg:
    def setup_method(self):
        self.handler = _make_handler()

    def test_plain_string(self):
        assert self.handler._resolve_arg("hello") == "hello"

    def test_plain_int(self):
        assert self.handler._resolve_arg(42) == 42

    def test_plain_none(self):
        assert self.handler._resolve_arg(None) is None

    def test_plain_list(self):
        assert self.handler._resolve_arg([1, 2, 3]) == [1, 2, 3]

    def test_plain_dict_multiple_keys(self):
        d = {"a": 1, "b": 2}
        assert self.handler._resolve_arg(d) == d

    def test_plain_dict_single_key_no_prefix(self):
        d = {"some_key": {"x": 1}}
        assert self.handler._resolve_arg(d) == d

    def test_io_struct_constructor(self):
        """A dict with one key starting with 'io_struct.' constructs the class."""
        mock_cls = MagicMock()
        mock_cls.return_value = "constructed_instance"
        mock_module = MagicMock()
        mock_module.MyReqInput = mock_cls

        with patch("importlib.import_module", return_value=mock_module) as imp:
            result = self.handler._resolve_arg(
                {"io_struct.MyReqInput": {"addr": "1.2.3.4", "port": 1234}}
            )
            imp.assert_called_once_with("sglang.srt.managers.io_struct")
            mock_cls.assert_called_once_with(addr="1.2.3.4", port=1234)
            assert result == "constructed_instance"

    def test_io_struct_empty_kwargs(self):
        """Constructor with empty kwargs."""
        mock_cls = MagicMock()
        mock_cls.return_value = "empty_instance"
        mock_module = MagicMock()
        mock_module.PauseGenerationReqInput = mock_cls

        with patch("importlib.import_module", return_value=mock_module):
            result = self.handler._resolve_arg(
                {"io_struct.PauseGenerationReqInput": {}}
            )
            mock_cls.assert_called_once_with()
            assert result == "empty_instance"


# ---------------------------------------------------------------------------
# _normalize_result
# ---------------------------------------------------------------------------


class TestNormalizeResult:
    def setup_method(self):
        self.handler = _make_handler()

    def test_none(self):
        assert self.handler._normalize_result(None) == {"status": "ok"}

    def test_tuple_2(self):
        assert self.handler._normalize_result((True, "done")) == {
            "success": True,
            "message": "done",
        }

    def test_tuple_2_failure(self):
        assert self.handler._normalize_result((False, "error msg")) == {
            "success": False,
            "message": "error msg",
        }

    def test_tuple_3(self):
        assert self.handler._normalize_result((True, "ok", 5)) == {
            "success": True,
            "message": "ok",
            "num_paused_requests": 5,
        }

    def test_dict_passthrough(self):
        d = {"foo": "bar", "count": 3}
        assert self.handler._normalize_result(d) is d

    def test_dataclass(self):
        @dataclasses.dataclass
        class FakeResult:
            success: bool
            nodes_pinned: int

        result = FakeResult(success=True, nodes_pinned=10)
        assert self.handler._normalize_result(result) == {
            "success": True,
            "nodes_pinned": 10,
        }

    def test_list_of_dataclasses(self):
        @dataclasses.dataclass
        class LoadInfo:
            dp_rank: int
            num_reqs: int

        items = [LoadInfo(dp_rank=0, num_reqs=5), LoadInfo(dp_rank=1, num_reqs=3)]
        assert self.handler._normalize_result(items) == {
            "result": [
                {"dp_rank": 0, "num_reqs": 5},
                {"dp_rank": 1, "num_reqs": 3},
            ]
        }

    def test_list_of_plain_values(self):
        assert self.handler._normalize_result([1, "two", 3]) == {
            "result": [1, "two", 3]
        }

    def test_list_mixed(self):
        @dataclasses.dataclass
        class Info:
            val: int

        items = [Info(val=1), "plain", 42]
        assert self.handler._normalize_result(items) == {
            "result": [{"val": 1}, "plain", 42]
        }

    def test_other_value(self):
        assert self.handler._normalize_result(42) == {"result": 42}
        assert self.handler._normalize_result("text") == {"result": "text"}

    def test_non_serializable_falls_back_to_str(self):
        obj = object()
        result = self.handler._normalize_result(obj)
        assert result == {"result": str(obj)}


# ---------------------------------------------------------------------------
# call_tokenizer_manager
# ---------------------------------------------------------------------------


class TestCallTokenizerManager:
    def setup_method(self):
        self.handler = _make_handler()

    @pytest.mark.asyncio
    async def test_method_only(self):
        """Calling with just 'method', no args/kwargs."""
        self.handler.engine.tokenizer_manager.flush_cache = AsyncMock(return_value=None)

        result = await self.handler.call_tokenizer_manager({"method": "flush_cache"})

        self.handler.engine.tokenizer_manager.flush_cache.assert_awaited_once_with()
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_with_plain_args(self):
        """Plain value args are passed through."""
        self.handler.engine.tokenizer_manager.some_method = AsyncMock(
            return_value=(True, "ok")
        )

        result = await self.handler.call_tokenizer_manager(
            {"method": "some_method", "args": ["arg1", 42]}
        )

        self.handler.engine.tokenizer_manager.some_method.assert_awaited_once_with(
            "arg1", 42
        )
        assert result == {"success": True, "message": "ok"}

    @pytest.mark.asyncio
    async def test_with_kwargs(self):
        """kwargs including null are passed through."""
        self.handler.engine.tokenizer_manager.some_method = AsyncMock(
            return_value=(True, "done")
        )

        result = await self.handler.call_tokenizer_manager(
            {
                "method": "some_method",
                "args": ["positional"],
                "kwargs": {"request": None},
            }
        )

        self.handler.engine.tokenizer_manager.some_method.assert_awaited_once_with(
            "positional", request=None
        )
        assert result == {"success": True, "message": "done"}

    @pytest.mark.asyncio
    async def test_with_io_struct_arg(self):
        """io_struct constructor args are resolved before calling."""
        mock_cls = MagicMock()
        constructed = MagicMock()
        mock_cls.return_value = constructed
        mock_module = MagicMock()
        mock_module.InitWeightsUpdateGroupReqInput = mock_cls

        self.handler.engine.tokenizer_manager.init_weights_update_group = AsyncMock(
            return_value=(True, "group initialized")
        )

        with patch("importlib.import_module", return_value=mock_module):
            result = await self.handler.call_tokenizer_manager(
                {
                    "method": "init_weights_update_group",
                    "args": [
                        {
                            "io_struct.InitWeightsUpdateGroupReqInput": {
                                "master_address": "1.2.3.4",
                                "master_port": 1234,
                                "rank_offset": 0,
                                "world_size": 4,
                            }
                        }
                    ],
                    "kwargs": {"request": None},
                }
            )

        mock_cls.assert_called_once_with(
            master_address="1.2.3.4", master_port=1234, rank_offset=0, world_size=4
        )
        self.handler.engine.tokenizer_manager.init_weights_update_group.assert_awaited_once_with(
            constructed, request=None
        )
        assert result == {"success": True, "message": "group initialized"}

    @pytest.mark.asyncio
    async def test_tuple_3_result(self):
        """3-tuple results include num_paused_requests."""
        self.handler.engine.tokenizer_manager.update_weights_from_disk = AsyncMock(
            return_value=(True, "updated", 3)
        )

        result = await self.handler.call_tokenizer_manager(
            {
                "method": "update_weights_from_disk",
                "args": ["req_obj"],
                "kwargs": {"request": None},
            }
        )

        assert result == {
            "success": True,
            "message": "updated",
            "num_paused_requests": 3,
        }
