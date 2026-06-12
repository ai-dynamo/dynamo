# SPDX-License-Identifier: Apache-2.0

import asyncio
import importlib
import sys
import types
import unittest


def _install_import_stubs():
    names = ("uvloop", "dynamo.runtime", "dynamo.runtime.logging")
    previous = {name: sys.modules.get(name) for name in names}

    if previous["uvloop"] is None:
        uvloop = types.ModuleType("uvloop")
        uvloop.install = lambda: None
        sys.modules["uvloop"] = uvloop

    runtime = types.ModuleType("dynamo.runtime")
    runtime.DistributedRuntime = object

    def dynamo_worker(*args, **kwargs):
        del args, kwargs

        def decorate(fn):
            return fn

        return decorate

    runtime.dynamo_worker = dynamo_worker
    sys.modules["dynamo.runtime"] = runtime

    runtime_logging = types.ModuleType("dynamo.runtime.logging")
    runtime_logging.configure_dynamo_logging = lambda **kwargs: None
    sys.modules["dynamo.runtime.logging"] = runtime_logging
    return previous


def _restore_imports(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


_previous_imports = _install_import_stubs()
try:
    migration = importlib.import_module("dynamo.sglang.decode_migration_frontend")
finally:
    _restore_imports(_previous_imports)


class FakeResponse:
    def __init__(self, data):
        self._data = data

    def data(self):
        return self._data


class FakeStream:
    def __init__(self, events, on_event=None):
        self.events = list(events)
        self.on_event = on_event

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.events:
            raise StopAsyncIteration
        event = self.events.pop(0)
        if self.on_event is not None:
            self.on_event(event)
        if isinstance(event, BaseException):
            raise event
        await asyncio.sleep(0)
        return FakeResponse(event)


class FakeClient:
    def __init__(self, streams):
        self.streams = list(streams)
        self.requests = []

    async def generate(self, request, context=None):
        self.requests.append((request, context))
        if not self.streams:
            raise AssertionError("Unexpected client call")
        stream = self.streams.pop(0)
        return stream if hasattr(stream, "__aiter__") else FakeStream(stream)


class FakeContext:
    trace_id = "request-1"

    def __init__(self):
        self.stopped = False

    def is_stopped(self):
        return self.stopped


def base_request(max_tokens=16):
    return {
        "model": "qwen",
        "token_ids": [101, 102],
        "stop_conditions": {"max_tokens": max_tokens},
        "sampling_options": {"temperature": 0.0},
        "output_options": {},
    }


def prepared(
    *,
    committed_output_ids,
    pending_id,
    seen,
    unforwarded,
):
    prompt = [101, 102]
    committed = prompt + list(committed_output_ids)
    return {
        "rid": "request-1",
        "migration_id": "migration",
        "success": True,
        "status": "prepared",
        "bootstrap_host": "127.0.0.1",
        "bootstrap_port": 8998,
        "bootstrap_room": 7,
        "committed_input_ids": committed,
        "pending_input_ids": [pending_id],
        "unforwarded_committed_output_ids": list(unforwarded),
        "prompt_len": len(prompt),
        "committed_len": len(committed),
        "logical_len": len(committed) + 1,
        "output_tokens_seen": seen,
        "source_dp_rank": 0,
    }


def described():
    return {
        "success": True,
        "status": "described",
        "bootstrap_host": "127.0.0.1",
        "bootstrap_port": 8998,
        "source_dp_rank": 0,
    }


def reserved(status="reserved"):
    return {
        "success": True,
        "status": status,
        "bootstrap_host": "127.0.0.1",
        "bootstrap_port": 8998,
        "bootstrap_room": 7,
        "destination_dp_rank": 0,
    }


class DecodeMigrationCoordinatorTests(unittest.IsolatedAsyncioTestCase):
    def coordinator(self, after=1, on_tokens=None):
        return migration.DecodeMigrationCoordinator(
            object(), after, migrate_on_token_ids=set(on_tokens or ())
        )

    async def collect(self, coordinator, context=None, request=None):
        context = context or FakeContext()
        chunks = []
        async for chunk in coordinator.generate(request or base_request(), context):
            chunks.append(chunk)
        return chunks

    def configure_control(
        self, coordinator, snapshot, source_finalizes=(), destination_finalizes=()
    ):
        coordinator.source_sync_client = FakeClient([[described()], [snapshot]])
        coordinator.destination_prepare_client = FakeClient(
            [[reserved()], [reserved("ready")]]
        )
        coordinator.source_finalize_client = FakeClient(
            [[result] for result in source_finalizes]
        )
        coordinator.destination_finalize_client = FakeClient(
            [[result] for result in destination_finalizes]
        )

    async def test_success_deduplicates_pending_token_in_coalesced_chunk(self):
        coordinator = self.coordinator(after=4)
        coordinator.fast_client = FakeClient(
            [[{"token_ids": [1, 2, 3, 4], "index": 0}]]
        )
        snapshot = prepared(
            committed_output_ids=[1, 2, 3],
            pending_id=4,
            seen=4,
            unforwarded=[],
        )
        coordinator.slow_client = FakeClient(
            [
                [
                    {"token_ids": [4, 5, 6], "index": 0},
                    {"token_ids": [7], "index": 0, "finish_reason": "length"},
                ]
            ]
        )
        self.configure_control(
            coordinator,
            snapshot,
            source_finalizes=[{"success": True, "transfer_status": "transferred"}],
            destination_finalizes=[{"success": True, "status": "active"}],
        )

        chunks = await self.collect(coordinator)
        self.assertEqual(
            [token for chunk in chunks for token in chunk.get("token_ids", [])],
            [1, 2, 3, 4, 5, 6, 7],
        )
        self.assertEqual(
            coordinator.source_finalize_client.requests[0][0]["action"], "commit"
        )

    async def test_unforwarded_committed_tail_is_emitted_before_destination(self):
        coordinator = self.coordinator(after=1)
        coordinator.fast_client = FakeClient([[{"token_ids": [1], "index": 0}]])
        snapshot = prepared(
            committed_output_ids=[1, 2, 3],
            pending_id=4,
            seen=1,
            unforwarded=[2, 3],
        )
        coordinator.slow_client = FakeClient(
            [[{"token_ids": [4, 5], "index": 0, "finish_reason": "length"}]]
        )
        self.configure_control(
            coordinator,
            snapshot,
            source_finalizes=[{"success": True, "transfer_status": "transferred"}],
            destination_finalizes=[{"success": True, "status": "active"}],
        )

        chunks = await self.collect(coordinator)
        self.assertEqual(
            [token for chunk in chunks for token in chunk.get("token_ids", [])],
            [1, 2, 3, 4, 5],
        )

    async def test_destination_failure_resumes_source_and_skips_synthetic_tail(self):
        coordinator = self.coordinator(after=1)
        coordinator.fast_client = FakeClient(
            [
                [
                    {"token_ids": [1], "index": 0},
                    {"token_ids": [2, 3], "index": 0},
                    {"token_ids": [4], "index": 0, "finish_reason": "length"},
                ]
            ]
        )
        snapshot = prepared(
            committed_output_ids=[1, 2, 3],
            pending_id=4,
            seen=1,
            unforwarded=[2, 3],
        )
        coordinator.slow_client = FakeClient([[RuntimeError("receiver failed")]])
        self.configure_control(
            coordinator,
            snapshot,
            source_finalizes=[{"success": True, "transfer_status": "failed"}],
            destination_finalizes=[{"success": True, "status": "aborted"}],
        )

        chunks = await self.collect(coordinator)
        self.assertEqual(
            [token for chunk in chunks for token in chunk.get("token_ids", [])],
            [1, 2, 3, 4],
        )
        self.assertEqual(
            coordinator.source_finalize_client.requests[0][0]["action"], "resume"
        )

    async def test_request_finishing_before_threshold_is_not_migrated(self):
        coordinator = self.coordinator(after=8)
        coordinator.fast_client = FakeClient(
            [[{"token_ids": [1, 2], "index": 0, "finish_reason": "stop"}]]
        )
        coordinator.destination_prepare_client = FakeClient([])
        coordinator.source_sync_client = FakeClient([])
        coordinator.slow_client = FakeClient([])
        coordinator.source_finalize_client = FakeClient([])
        coordinator.destination_finalize_client = FakeClient([])

        chunks = await self.collect(coordinator)
        self.assertEqual(chunks[0]["token_ids"], [1, 2])
        self.assertEqual(coordinator.destination_prepare_client.requests, [])

    async def test_token_boundary_takes_precedence_over_count_threshold(self):
        coordinator = self.coordinator(after=1, on_tokens={9})
        coordinator.fast_client = FakeClient(
            [
                [
                    {"token_ids": [1, 2], "index": 0},
                    {"token_ids": [9], "index": 0},
                ]
            ]
        )
        snapshot = prepared(
            committed_output_ids=[1, 2, 9],
            pending_id=10,
            seen=3,
            unforwarded=[],
        )
        coordinator.slow_client = FakeClient(
            [[{"token_ids": [10, 11], "index": 0, "finish_reason": "stop"}]]
        )
        self.configure_control(
            coordinator,
            snapshot,
            source_finalizes=[{"success": True, "transfer_status": "transferred"}],
            destination_finalizes=[{"success": True, "status": "active"}],
        )

        chunks = await self.collect(coordinator)
        self.assertEqual(
            [token for chunk in chunks for token in chunk.get("token_ids", [])],
            [1, 2, 9, 10, 11],
        )
        quiesce = coordinator.source_sync_client.requests[1][0]
        self.assertEqual(quiesce["output_tokens_seen"], 3)

    async def test_request_finishing_before_token_boundary_is_not_migrated(self):
        coordinator = self.coordinator(after=1, on_tokens={9})
        coordinator.fast_client = FakeClient(
            [[{"token_ids": [1, 2], "index": 0, "finish_reason": "stop"}]]
        )
        coordinator.destination_prepare_client = FakeClient([])
        coordinator.source_sync_client = FakeClient([])
        coordinator.slow_client = FakeClient([])
        coordinator.source_finalize_client = FakeClient([])
        coordinator.destination_finalize_client = FakeClient([])

        chunks = await self.collect(coordinator)
        self.assertEqual(chunks[0]["token_ids"], [1, 2])
        self.assertEqual(coordinator.destination_prepare_client.requests, [])

    async def test_zero_threshold_disables_migration_for_baseline_component(self):
        coordinator = self.coordinator(after=0)
        coordinator.fast_client = FakeClient(
            [[{"token_ids": [1, 2]}, {"token_ids": [3, 4]}]]
        )
        coordinator.destination_prepare_client = FakeClient([])
        coordinator.source_sync_client = FakeClient([])
        coordinator.slow_client = FakeClient([])
        coordinator.source_finalize_client = FakeClient([])
        coordinator.destination_finalize_client = FakeClient([])
        chunks = await self.collect(coordinator)

        self.assertEqual(
            [token for chunk in chunks for token in chunk.get("token_ids", [])],
            [1, 2, 3, 4],
        )
        self.assertEqual(coordinator.destination_prepare_client.requests, [])

    async def test_cancellation_before_destination_ready_cancels_source(self):
        context = FakeContext()
        coordinator = self.coordinator(after=1)
        coordinator.fast_client = FakeClient([[{"token_ids": [1], "index": 0}]])
        snapshot = prepared(
            committed_output_ids=[1],
            pending_id=2,
            seen=1,
            unforwarded=[],
        )

        def stop_context(_):
            context.stopped = True

        coordinator.slow_client = FakeClient(
            [FakeStream([{"token_ids": [2]}], on_event=stop_context)]
        )
        self.configure_control(
            coordinator,
            snapshot,
            source_finalizes=[{"success": True, "transfer_status": "transferring"}],
            destination_finalizes=[{"success": True, "status": "aborted"}],
        )

        await self.collect(coordinator, context)
        self.assertEqual(
            coordinator.source_finalize_client.requests[0][0]["action"], "cancel"
        )


if __name__ == "__main__":
    unittest.main()
