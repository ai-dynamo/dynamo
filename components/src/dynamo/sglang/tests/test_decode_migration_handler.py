# SPDX-License-Identifier: Apache-2.0

import asyncio
from types import SimpleNamespace
from unittest import IsolatedAsyncioTestCase
from unittest.mock import AsyncMock, Mock

import pytest
from sglang.srt.managers.io_struct import (
    BindDecodeMigrationReqOutput,
    FinalizeDecodeMigrationReqOutput,
    PrepareDecodeMigrationReqOutput,
)

from dynamo.sglang.request_handlers.llm.decode_handler import DecodeWorkerHandler

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


async def collect(generator):
    return [item async for item in generator]


class DecodeMigrationHandlerTests(IsolatedAsyncioTestCase):
    def handler(self):
        handler = object.__new__(DecodeWorkerHandler)
        handler._decode_migration_reservations = {}
        handler._decode_migration_lock = asyncio.Lock()
        handler.engine = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(
                abort_request=Mock(),
                bind_decode_migration_destination=AsyncMock(
                    return_value=BindDecodeMigrationReqOutput(
                        rid="request",
                        migration_id="migration",
                        success=True,
                        status="ready",
                        source_dp_rank=1,
                        pending_token_suppressed=True,
                    )
                ),
                prepare_decode_migration=AsyncMock(),
                finalize_decode_migration=AsyncMock(),
            )
        )
        handler.config = SimpleNamespace(
            server_args=SimpleNamespace(enable_decode_migration=True)
        )
        return handler

    async def test_repeated_arm_returns_existing_ready_session(self):
        handler = self.handler()
        handler.engine.async_generate = Mock(
            side_effect=AssertionError("ready session must not be armed twice")
        )
        handler._decode_migration_reservations["migration"] = {
            "migration_id": "migration",
            "rid": "request",
            "bootstrap_room": 17,
            "source": {
                "bootstrap_host": "127.0.0.1",
                "bootstrap_port": 8998,
            },
            "reserve_tokens": 128,
            "destination_dp_rank": 1,
            "status": "ready",
        }

        responses = await collect(
            handler.prepare_decode_migration(
                {
                    "migration_id": "migration",
                    "rid": "request",
                    "source_state": {"committed_len": 32},
                    "destination_request": {},
                }
            )
        )

        self.assertEqual(responses[0]["status"], "ready")
        self.assertEqual(responses[0]["bootstrap_room"], 17)
        handler.engine.async_generate.assert_not_called()

    async def test_source_generate_uses_framework_migration_rid(self):
        handler = self.handler()
        handler.use_sglang_tokenizer = False
        handler.enable_trace = False
        handler.serving_mode = "aggregated"
        handler._routed_experts_kwargs = {}
        handler._enable_frontend_decoding = False
        handler._mm_hashes_supported = False
        handler._build_sampling_params = lambda request: {}
        handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
        handler._build_logprob_kwargs = lambda request: {}
        handler._resolve_lora = lambda request: None

        async def engine_stream():
            yield {"token_ids": [9]}

        handler.engine.async_generate = AsyncMock(return_value=engine_stream())

        async def process_token_stream(stream, context, *args, **kwargs):
            del context, args, kwargs
            async for item in stream:
                yield item

        handler._process_token_stream = process_token_stream
        context = SimpleNamespace(
            id=lambda: "destination-transport-id",
            trace_id="frontend-trace-id",
        )

        responses = await collect(
            handler.generate(
                {
                    "model": "model",
                    "token_ids": [1, 2, 3],
                    "stop_conditions": {},
                    "sampling_options": {},
                    "output_options": {},
                    "decode_migration_state": {
                        "rid": "source-engine-rid",
                        "migration_id": "migration",
                        "source_dp_rank": 0,
                    },
                },
                context,
            )
        )

        self.assertEqual(responses, [{"token_ids": [9]}])
        self.assertEqual(
            handler.engine.async_generate.await_args.kwargs["rid"],
            "source-engine-rid",
        )

    async def test_prepared_stream_attaches_in_aggregated_mode_with_nested_state(self):
        handler = self.handler()
        handler.use_sglang_tokenizer = False
        handler.enable_trace = False
        handler.serving_mode = "aggregated"
        handler._routed_experts_kwargs = {}
        handler._build_sampling_params = lambda request: {}
        handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
        handler._build_logprob_kwargs = lambda request: {}
        handler._resolve_lora = lambda request: None
        handler.engine.async_generate = AsyncMock(
            side_effect=AssertionError("prepared stream must be reused")
        )

        async def prepared_stream():
            yield {"token_ids": [9]}

        async def process_token_stream(stream, context, *args, **kwargs):
            del context, args, kwargs
            async for item in stream:
                yield item

        handler._process_token_stream = process_token_stream
        handler._decode_migration_reservations["migration"] = {
            "migration_id": "migration",
            "rid": "request",
            "status": "ready",
            "decode_stream": prepared_stream(),
            "destination_dp_rank": 1,
        }
        context = SimpleNamespace(
            id=lambda: "frontend-request",
            trace_id="frontend-trace",
        )

        responses = await collect(
            handler.generate(
                {
                    "model": "model",
                    "token_ids": [1, 2, 3],
                    "stop_conditions": {},
                    "sampling_options": {},
                    "output_options": {},
                    "decode_migration_state": {
                        "rid": "request",
                        "migration_id": "migration",
                        "source_dp_rank": 0,
                    },
                    "bootstrap_info": {
                        "bootstrap_host": "127.0.0.1",
                        "bootstrap_port": 8998,
                        "bootstrap_room": 17,
                    },
                },
                context,
            )
        )

        self.assertEqual(responses, [{"token_ids": [9]}])
        self.assertNotIn("migration", handler._decode_migration_reservations)
        handler.engine.async_generate.assert_not_called()

    async def test_prepare_rejects_worker_without_migration_capability(self):
        handler = self.handler()
        handler.config.server_args.enable_decode_migration = False
        responses = await collect(
            handler.prepare_decode_migration(
                {"migration_id": "migration", "rid": "request"}
            )
        )
        self.assertFalse(responses[0]["success"])
        self.assertIn("--enable-decode-migration", responses[0]["error"])

    async def test_source_controls_route_to_selected_dp_rank(self):
        handler = self.handler()
        handler._get_bootstrap_info = lambda engine: ("127.0.0.1", 8998)
        handler.engine.tokenizer_manager.prepare_decode_migration.return_value = (
            PrepareDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                success=True,
                status="prepared",
                source_dp_rank=3,
            )
        )
        handler.engine.tokenizer_manager.finalize_decode_migration.return_value = (
            FinalizeDecodeMigrationReqOutput(
                rid="request",
                migration_id="migration",
                action="commit",
                success=True,
                source_dp_rank=3,
            )
        )

        described = await collect(
            handler.sync_decode_migration(
                {
                    "rid": "request",
                    "migration_id": "migration",
                    "phase": "describe",
                    "source_dp_rank": 3,
                }
            )
        )
        prepared = await collect(
            handler.sync_decode_migration(
                {
                    "rid": "request",
                    "migration_id": "migration",
                    "phase": "quiesce",
                    "bootstrap_room": 17,
                    "source_dp_rank": 3,
                }
            )
        )
        finalized = await collect(
            handler.finalize_decode_migration(
                {
                    "rid": "request",
                    "migration_id": "migration",
                    "action": "commit",
                    "source_dp_rank": 3,
                }
            )
        )

        self.assertEqual(described[0]["source_dp_rank"], 3)
        self.assertEqual(prepared[0]["source_dp_rank"], 3)
        self.assertEqual(finalized[0]["source_dp_rank"], 3)
        prepare_req = (
            handler.engine.tokenizer_manager.prepare_decode_migration.await_args.args[0]
        )
        finalize_req = (
            handler.engine.tokenizer_manager.finalize_decode_migration.await_args.args[
                0
            ]
        )
        self.assertEqual(prepare_req.routed_dp_rank, 3)
        self.assertEqual(finalize_req.routed_dp_rank, 3)

    async def test_destination_reservation_preserves_selected_dp_rank(self):
        handler = self.handler()

        responses = await collect(
            handler.prepare_decode_migration(
                {
                    "migration_id": "migration",
                    "rid": "request",
                    "destination_dp_rank": 1,
                    "source": {
                        "bootstrap_host": "127.0.0.1",
                        "bootstrap_port": 8998,
                        "dp_rank": 3,
                    },
                    "reserve_tokens": 128,
                }
            )
        )

        self.assertEqual(responses[0]["destination_dp_rank"], 1)
        self.assertEqual(
            handler._decode_migration_reservations["migration"]["destination_dp_rank"],
            1,
        )

    async def test_destination_abort_releases_session_and_engine_request(self):
        handler = self.handler()
        handler._decode_migration_reservations["migration"] = {
            "migration_id": "migration",
            "rid": "request",
            "status": "active",
            "destination_dp_rank": 1,
        }

        responses = await collect(
            handler.finalize_decode_migration(
                {
                    "side": "destination",
                    "action": "abort",
                    "migration_id": "migration",
                }
            )
        )

        self.assertTrue(responses[0]["success"])
        self.assertEqual(responses[0]["status"], "aborted")
        self.assertNotIn("migration", handler._decode_migration_reservations)
        handler.engine.tokenizer_manager.abort_request.assert_called_once_with(
            rid="request", abort_all=False
        )

    async def test_repeated_destination_abort_is_idempotent(self):
        handler = self.handler()

        responses = await collect(
            handler.finalize_decode_migration(
                {
                    "side": "destination",
                    "action": "abort",
                    "migration_id": "missing",
                }
            )
        )

        self.assertTrue(responses[0]["success"])
        self.assertEqual(responses[0]["status"], "unknown")
        handler.engine.tokenizer_manager.abort_request.assert_not_called()

    def configure_destination_handler(self, handler):
        handler.use_sglang_tokenizer = False
        handler.enable_trace = False
        handler.serving_mode = "aggregated"
        handler._routed_experts_kwargs = {}
        handler._enable_frontend_decoding = False
        handler._mm_hashes_supported = False
        handler._build_sampling_params = lambda request: {}
        handler._get_input_param = lambda request: {"input_ids": request["token_ids"]}
        handler._build_logprob_kwargs = lambda request: {}
        handler._resolve_lora = lambda request: None
        handler._session_kwargs = lambda request: {}
        handler._priority_kwargs = lambda priority: {}

    def arm_request(self):
        return {
            "migration_id": "migration",
            "rid": "request",
            "source_state": {
                "committed_input_ids": [1, 2, 3],
                "pending_input_ids": [4],
                "committed_len": 3,
                "logical_len": 4,
            },
            "destination_dp_rank": 1,
            "destination_request": {
                "model": "model",
                "token_ids": [1, 2, 3],
                "stop_conditions": {},
                "sampling_options": {},
                "output_options": {},
                "routing": {"dp_rank": 1},
                "decode_migration_state": {
                    "rid": "request",
                    "migration_id": "migration",
                    "source_dp_rank": 3,
                    "is_destination": True,
                },
                "bootstrap_info": {
                    "bootstrap_host": "127.0.0.1",
                    "bootstrap_port": 8998,
                    "bootstrap_room": 17,
                },
            },
        }

    async def test_concurrent_arm_is_idempotent(self):
        handler = self.handler()
        self.configure_destination_handler(handler)

        async def prepared_stream():
            yield {"token_ids": [9]}

        async def arm(**kwargs):
            del kwargs
            await asyncio.sleep(0)
            return prepared_stream()

        handler.engine.async_generate = AsyncMock(side_effect=arm)
        handler._decode_migration_reservations["migration"] = {
            "migration_id": "migration",
            "rid": "request",
            "bootstrap_room": 17,
            "source": {"bootstrap_host": "127.0.0.1", "bootstrap_port": 8998},
            "reserve_tokens": 128,
            "destination_dp_rank": 1,
            "status": "reserved",
        }

        first, second = await asyncio.gather(
            collect(handler.prepare_decode_migration(self.arm_request())),
            collect(handler.prepare_decode_migration(self.arm_request())),
        )

        self.assertEqual(first[0]["status"], "ready")
        self.assertEqual(second[0]["status"], "ready")
        handler.engine.async_generate.assert_awaited_once()
        self.assertEqual(
            handler.engine.async_generate.await_args.kwargs["data_parallel_rank"], 1
        )
        self.assertEqual(
            handler.engine.async_generate.await_args.kwargs["disagg_prefill_dp_rank"],
            3,
        )

    async def test_destination_bootstrap_starts_before_exact_source_state(self):
        handler = self.handler()
        self.configure_destination_handler(handler)
        started = asyncio.Event()

        async def prepared_stream():
            started.set()
            await asyncio.Event().wait()
            yield {"token_ids": [9]}

        handler.engine.async_generate = AsyncMock(return_value=prepared_stream())
        handler._decode_migration_reservations["migration"] = {
            "migration_id": "migration",
            "rid": "request",
            "bootstrap_room": 17,
            "source": {"bootstrap_host": "127.0.0.1", "bootstrap_port": 8998},
            "reserve_tokens": 128,
            "destination_dp_rank": 1,
            "status": "reserved",
        }
        request = self.arm_request()
        request.pop("source_state")

        responses = await collect(handler.prepare_decode_migration(request))

        await asyncio.wait_for(started.wait(), timeout=1)
        self.assertEqual(responses[0]["status"], "bootstrapping")
        self.assertEqual(
            handler.engine.async_generate.await_args.kwargs["decode_migration_id"],
            "migration",
        )
        await collect(
            handler.finalize_decode_migration(
                {
                    "side": "destination",
                    "action": "abort",
                    "migration_id": "migration",
                }
            )
        )

    async def test_abort_waits_for_inflight_arm_then_cancels_engine_request(self):
        handler = self.handler()
        self.configure_destination_handler(handler)
        arm_started = asyncio.Event()
        allow_arm = asyncio.Event()

        async def prepared_stream():
            yield {"token_ids": [9]}

        async def arm(**kwargs):
            del kwargs
            arm_started.set()
            await allow_arm.wait()
            return prepared_stream()

        handler.engine.async_generate = AsyncMock(side_effect=arm)
        handler._decode_migration_reservations["migration"] = {
            "migration_id": "migration",
            "rid": "request",
            "bootstrap_room": 17,
            "source": {"bootstrap_host": "127.0.0.1", "bootstrap_port": 8998},
            "reserve_tokens": 128,
            "destination_dp_rank": 1,
            "status": "reserved",
        }

        arm_task = asyncio.create_task(
            collect(handler.prepare_decode_migration(self.arm_request()))
        )
        await arm_started.wait()
        abort_task = asyncio.create_task(
            collect(
                handler.finalize_decode_migration(
                    {
                        "side": "destination",
                        "action": "abort",
                        "migration_id": "migration",
                    }
                )
            )
        )
        await asyncio.sleep(0)
        self.assertFalse(abort_task.done())

        allow_arm.set()
        arm_response, abort_response = await asyncio.gather(arm_task, abort_task)

        self.assertEqual(arm_response[0]["status"], "ready")
        self.assertEqual(abort_response[0]["status"], "aborted")
        self.assertNotIn("migration", handler._decode_migration_reservations)
        handler.engine.tokenizer_manager.abort_request.assert_called_once_with(
            rid="request", abort_all=False
        )

    async def test_destination_generate_without_prepared_stream_fails_closed(self):
        handler = self.handler()
        self.configure_destination_handler(handler)
        handler.engine.async_generate = AsyncMock(
            side_effect=AssertionError("must not start a fresh destination request")
        )
        context = SimpleNamespace(
            id=lambda: "destination-transport-id",
            trace_id="frontend-trace-id",
        )
        request = self.arm_request()["destination_request"]

        with self.assertRaisesRegex(RuntimeError, "Prepared decode migration stream"):
            await collect(handler.generate(request, context))
        handler.engine.async_generate.assert_not_called()


if __name__ == "__main__":
    import unittest

    unittest.main()
