# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, Mock, call, patch

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - local fallback when pytest is unavailable
    pytest = None

if pytest is not None:
    pytestmark = [
        pytest.mark.unit,
        pytest.mark.none,
        pytest.mark.gpu_0,
        pytest.mark.pre_merge,
    ]


class StandaloneRouterTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        modules = {
            "dynamo": ModuleType("dynamo"),
            "dynamo.router": ModuleType("dynamo.router"),
            "dynamo.llm": ModuleType("dynamo.llm"),
            "dynamo.router.args": ModuleType("dynamo.router.args"),
            "dynamo.runtime": ModuleType("dynamo.runtime"),
            "dynamo.runtime.logging": ModuleType("dynamo.runtime.logging"),
            "uvloop": ModuleType("uvloop"),
        }
        modules["dynamo"].__path__ = []
        modules["dynamo.router"].__path__ = []
        modules["dynamo.llm"].AicPerfConfig = type("AicPerfConfig", (), {})
        modules["dynamo.llm"].KvRouterConfig = type("KvRouterConfig", (), {})
        modules["dynamo.llm"].KvRouter = lambda **kwargs: kwargs
        modules["dynamo.router.args"].DynamoRouterConfig = object
        modules["dynamo.router.args"].build_aic_perf_config = lambda config: None
        modules["dynamo.router.args"].build_kv_router_config = lambda config: None
        modules["dynamo.router.args"].parse_args = lambda argv=None: None
        modules["dynamo.runtime"].Client = object
        modules["dynamo.runtime"].DistributedRuntime = object
        modules["dynamo.runtime"].dynamo_worker = lambda enable_nats=None: (
            lambda fn: fn
        )
        modules["dynamo.runtime.logging"].configure_dynamo_logging = lambda: None
        modules["uvloop"].run = lambda coro: coro

        self._module_patch = patch.dict(sys.modules, modules, clear=False)
        self._module_patch.start()
        self.addCleanup(self._module_patch.stop)

        module_path = Path(__file__).resolve().parents[1] / "__main__.py"
        spec = importlib.util.spec_from_file_location(
            f"test_router_main_{id(self)}", module_path
        )
        self.router_main = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(self.router_main)

    async def test_initialize_registers_router_discovery_before_creating_kv_router(self):
        runtime = SimpleNamespace()
        worker_client = SimpleNamespace()
        worker_endpoint = SimpleNamespace(client=AsyncMock(return_value=worker_client))
        router_discovery_endpoint = SimpleNamespace(
            register_endpoint_instance=AsyncMock(),
            unregister_endpoint_instance=AsyncMock(),
        )
        runtime.endpoint = Mock(
            side_effect=[worker_endpoint, router_discovery_endpoint],
        )

        kv_router = object()

        def build_router(**kwargs):
            router_discovery_endpoint.register_endpoint_instance.assert_awaited_once()
            router_discovery_endpoint.unregister_endpoint_instance.assert_not_awaited()
            self.assertIs(kwargs["endpoint"], worker_endpoint)
            self.assertEqual(kwargs["block_size"], 64)
            return kv_router

        self.router_main.KvRouter = build_router
        handler = self.router_main.StandaloneRouterHandler(
            runtime,
            "test-ns.prefill.generate",
            64,
            kv_router_config=object(),
            aic_perf_config=object(),
        )

        await handler.initialize()

        self.assertIs(handler.kv_router, kv_router)
        self.assertIs(handler.worker_client, worker_client)
        self.assertTrue(handler._router_discovery_registered)
        self.assertEqual(
            runtime.endpoint.mock_calls,
            [
                call("test-ns.prefill.generate"),
                call("test-ns.prefill.router-discovery"),
            ],
        )

    async def test_initialize_unregisters_router_discovery_when_kv_router_creation_fails(self):
        runtime = SimpleNamespace()
        worker_endpoint = SimpleNamespace(client=AsyncMock(return_value=SimpleNamespace()))
        router_discovery_endpoint = SimpleNamespace(
            register_endpoint_instance=AsyncMock(),
            unregister_endpoint_instance=AsyncMock(),
        )
        runtime.endpoint = Mock(
            side_effect=[worker_endpoint, router_discovery_endpoint],
        )

        def fail_router(**kwargs):
            raise RuntimeError("router init failed")

        self.router_main.KvRouter = fail_router
        handler = self.router_main.StandaloneRouterHandler(
            runtime,
            "test-ns.prefill.generate",
            64,
            kv_router_config=object(),
            aic_perf_config=object(),
        )

        with self.assertRaisesRegex(RuntimeError, "router init failed"):
            await handler.initialize()

        router_discovery_endpoint.register_endpoint_instance.assert_awaited_once()
        router_discovery_endpoint.unregister_endpoint_instance.assert_awaited_once()
        self.assertFalse(handler._router_discovery_registered)

    async def test_worker_shuts_down_router_discovery_registration(self):
        config = SimpleNamespace(
            endpoint="test-ns.prefill.generate",
            namespace="test-ns",
            router_block_size=64,
            overlap_score_weight=1.0,
            router_temperature=1.0,
            use_kv_events=True,
            durable_kv_events=True,
            router_replica_sync=False,
            router_reset_states=False,
            router_track_active_blocks=False,
            router_track_output_blocks=False,
            router_assume_kv_reuse=False,
            router_track_prefill_tokens=False,
            router_ttl_secs=60,
            router_max_tree_size=1024,
            router_prune_target_ratio=0.75,
        )
        self.router_main.parse_args = Mock(return_value=config)
        self.router_main.build_kv_router_config = Mock(return_value=object())
        self.router_main.build_aic_perf_config = Mock(return_value=object())

        handler = SimpleNamespace(
            initialize=AsyncMock(),
            shutdown=AsyncMock(),
            generate=AsyncMock(),
            best_worker_id=AsyncMock(),
        )
        self.router_main.StandaloneRouterHandler = Mock(return_value=handler)

        generate_endpoint = SimpleNamespace(
            serve_endpoint=AsyncMock(side_effect=RuntimeError("serve failed")),
        )
        best_worker_endpoint = SimpleNamespace(serve_endpoint=AsyncMock(return_value=None))
        runtime = SimpleNamespace(
            endpoint=Mock(side_effect=[generate_endpoint, best_worker_endpoint]),
        )

        with self.assertRaisesRegex(RuntimeError, "serve failed"):
            await self.router_main.worker(runtime)

        handler.initialize.assert_awaited_once()
        handler.shutdown.assert_awaited_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
