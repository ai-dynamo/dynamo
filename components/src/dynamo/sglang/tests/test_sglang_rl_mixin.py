# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for RLMixin RL training support.

Covers the generic tokenizer_manager passthrough as well as the explicit
RL engine routes (generate_raw, pause/continue/flush/post_process/abort).
"""

import dataclasses
import sys
import types
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dynamo.sglang.request_handlers.handler_base import BaseWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _stub_sglang_io_struct(monkeypatch):
    """Keep unit tests independent from CUDA-only sglang imports."""
    io_struct = types.ModuleType("sglang.srt.managers.io_struct")
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io_struct)
    yield io_struct


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


class _TestWorkerHandler(BaseWorkerHandler):
    async def generate(self, request, context):
        yield {}


def _make_handler() -> _TestWorkerHandler:
    handler = _TestWorkerHandler.__new__(_TestWorkerHandler)
    handler.engine = SimpleNamespace(
        tokenizer_manager=SimpleNamespace(
            auto_create_handle_loop=MagicMock(),
        )
    )
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


# ---------------------------------------------------------------------------
# generate_raw
# ---------------------------------------------------------------------------


class TestGenerateRaw:
    def setup_method(self):
        self.handler = _make_handler()

    @pytest.mark.asyncio
    async def test_basic_generate(self, _stub_sglang_io_struct):
        """generate_raw calls generate_request and returns the response."""
        mock_generate_req_cls = MagicMock()
        mock_req = MagicMock()
        mock_generate_req_cls.return_value = mock_req
        _stub_sglang_io_struct.GenerateReqInput = mock_generate_req_cls

        expected_response = {
            "text": "Hello world",
            "meta_info": {
                "id": "req-123",
                "output_token_logprobs": [[-0.5, 101], [-0.3, 102]],
                "weight_version": "v1",
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 5,
                "completion_tokens": 2,
                "cached_tokens": 0,
            },
        }
        self.handler.engine.tokenizer_manager.generate_request = AsyncMock(
            return_value=expected_response
        )

        result = await self.handler.generate_raw(
            {
                "input_ids": [1, 2, 3, 4, 5],
                "sampling_params": {"temperature": 0.7, "max_new_tokens": 256},
                "return_logprob": True,
            }
        )

        # Verify stream was forced to False
        call_kwargs = mock_generate_req_cls.call_args
        assert call_kwargs[1]["stream"] is False or call_kwargs[0] == ()

        # Verify the full response is returned
        assert result["text"] == "Hello world"
        assert result["meta_info"]["output_token_logprobs"] == [
            [-0.5, 101],
            [-0.3, 102],
        ]
        assert result["meta_info"]["weight_version"] == "v1"
        assert result["meta_info"]["finish_reason"] == {"type": "stop"}

    @pytest.mark.asyncio
    async def test_generate_batched_response(self, _stub_sglang_io_struct):
        """generate_raw wraps list responses in a 'responses' key."""
        mock_generate_req_cls = MagicMock()
        _stub_sglang_io_struct.GenerateReqInput = mock_generate_req_cls

        batched = [
            {"text": "a", "meta_info": {"id": "1"}},
            {"text": "b", "meta_info": {"id": "2"}},
        ]
        self.handler.engine.tokenizer_manager.generate_request = AsyncMock(
            return_value=batched
        )

        result = await self.handler.generate_raw({"input_ids": [1]})

        assert "responses" in result
        assert len(result["responses"]) == 2
        assert result["responses"][0]["text"] == "a"
        assert result["responses"][1]["text"] == "b"


# ---------------------------------------------------------------------------
# _sanitize_generate_response
# ---------------------------------------------------------------------------


class TestSanitizeGenerateResponse:
    def test_plain_dict_unchanged(self):
        resp = {"text": "hi", "meta_info": {"weight_version": "v1"}}
        assert BaseWorkerHandler._sanitize_generate_response(resp) is resp

    def test_routed_experts_tensor_converted(self):
        """Tensor-like routed_experts is base64-encoded."""
        import numpy as np

        class FakeTensor:
            def numpy(self):
                return np.array([1, 2, 3], dtype=np.int32)

        resp = {"text": "hi", "meta_info": {"routed_experts": FakeTensor()}}
        result = BaseWorkerHandler._sanitize_generate_response(resp)
        # Should be a base64 string now
        assert isinstance(result["meta_info"]["routed_experts"], str)

    def test_no_meta_info(self):
        resp = {"text": "hi"}
        assert BaseWorkerHandler._sanitize_generate_response(resp) == {"text": "hi"}

    def test_non_dict_passthrough(self):
        assert BaseWorkerHandler._sanitize_generate_response("string") == "string"


# ---------------------------------------------------------------------------
# pause_generation
# ---------------------------------------------------------------------------


class TestPauseGeneration:
    def setup_method(self):
        self.handler = _make_handler()

    @pytest.mark.asyncio
    async def test_pause_default(self, _stub_sglang_io_struct):
        """pause_generation constructs req and calls tokenizer_manager."""
        mock_cls = MagicMock()
        mock_req = MagicMock()
        mock_cls.return_value = mock_req
        _stub_sglang_io_struct.PauseGenerationReqInput = mock_cls

        self.handler.engine.tokenizer_manager.pause_generation = AsyncMock(
            return_value=None
        )

        result = await self.handler.pause_generation({})

        mock_cls.assert_called_once_with()
        self.handler.engine.tokenizer_manager.pause_generation.assert_awaited_once_with(
            mock_req
        )
        assert result == {"status": "ok"}

    @pytest.mark.asyncio
    async def test_pause_with_mode(self, _stub_sglang_io_struct):
        """pause_generation passes body kwargs to the request input."""
        mock_cls = MagicMock()
        _stub_sglang_io_struct.PauseGenerationReqInput = mock_cls
        self.handler.engine.tokenizer_manager.pause_generation = AsyncMock(
            return_value=None
        )

        await self.handler.pause_generation({"mode": "retract"})

        mock_cls.assert_called_once_with(mode="retract")


# ---------------------------------------------------------------------------
# continue_generation
# ---------------------------------------------------------------------------


class TestContinueGeneration:
    def setup_method(self):
        self.handler = _make_handler()

    @pytest.mark.asyncio
    async def test_continue(self, _stub_sglang_io_struct):
        mock_cls = MagicMock()
        _stub_sglang_io_struct.ContinueGenerationReqInput = mock_cls
        self.handler.engine.tokenizer_manager.continue_generation = AsyncMock(
            return_value=None
        )

        result = await self.handler.continue_generation({})

        mock_cls.assert_called_once_with()
        assert result == {"status": "ok"}


# ---------------------------------------------------------------------------
# flush_cache
# ---------------------------------------------------------------------------


class TestFlushCache:
    def setup_method(self):
        self.handler = _make_handler()

    @pytest.mark.asyncio
    async def test_flush(self):
        self.handler.engine.tokenizer_manager.flush_cache = AsyncMock(
            return_value=None
        )

        result = await self.handler.flush_cache({})

        self.handler.engine.tokenizer_manager.flush_cache.assert_awaited_once()
        assert result == {"status": "ok"}


# ---------------------------------------------------------------------------
# post_process_weights
# ---------------------------------------------------------------------------


class TestPostProcessWeights:
    def setup_method(self):
        self.handler = _make_handler()

    @pytest.mark.asyncio
    async def test_post_process(self, _stub_sglang_io_struct):
        """post_process_weights delegates to call_tokenizer_manager."""
        mock_cls = MagicMock()
        constructed = MagicMock()
        mock_cls.return_value = constructed
        mock_module = MagicMock()
        mock_module.PostProcessWeightsReqInput = mock_cls

        self.handler.engine.tokenizer_manager.post_process_weights = AsyncMock(
            return_value=(True, "done")
        )

        with patch("importlib.import_module", return_value=mock_module):
            result = await self.handler.post_process_weights(
                {
                    "post_process_quantization": True,
                    "restore_weights_before_load": False,
                }
            )

        mock_cls.assert_called_once_with(
            post_process_quantization=True, restore_weights_before_load=False
        )
        assert result == {"success": True, "message": "done"}


# ---------------------------------------------------------------------------
# abort_request
# ---------------------------------------------------------------------------


class TestAbortRequest:
    def setup_method(self):
        self.handler = _make_handler()

    @pytest.mark.asyncio
    async def test_abort_all(self):
        self.handler.engine.tokenizer_manager.abort_request = MagicMock()

        result = await self.handler.abort_request({})

        self.handler.engine.tokenizer_manager.abort_request.assert_called_once_with(
            abort_all=True
        )
        assert result == {"status": "ok", "message": "Requests aborted"}

    @pytest.mark.asyncio
    async def test_abort_specific(self):
        self.handler.engine.tokenizer_manager.abort_request = MagicMock()

        result = await self.handler.abort_request({"abort_all": False})

        self.handler.engine.tokenizer_manager.abort_request.assert_called_once_with(
            abort_all=False
        )
        assert result["status"] == "ok"


# ---------------------------------------------------------------------------
# register_rl_engine_routes
# ---------------------------------------------------------------------------


class TestRegisterRLEngineRoutes:
    def test_all_routes_registered(self):
        handler = _make_handler()
        runtime = MagicMock()

        handler.register_rl_engine_routes(runtime)

        registered = {
            call.args[0] for call in runtime.register_engine_route.call_args_list
        }
        assert registered == {
            "call_tokenizer_manager",
            "generate_raw",
            "pause_generation",
            "continue_generation",
            "flush_cache",
            "post_process_weights",
            "abort_request",
        }
