# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def stub_module(name: str, **attributes: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for attribute, value in attributes.items():
        setattr(module, attribute, value)
    return module


def load_standalone_router_handler():
    placeholder_type = type("Placeholder", (), {})
    stubs = {
        "uvloop": stub_module("uvloop", run=lambda coroutine: coroutine),
        "dynamo": stub_module("dynamo"),
        "dynamo.llm": stub_module(
            "dynamo.llm",
            AicPerfConfig=placeholder_type,
            KvRouter=placeholder_type,
            KvRouterConfig=placeholder_type,
        ),
        "dynamo.router": stub_module("dynamo.router"),
        "dynamo.router.args": stub_module(
            "dynamo.router.args",
            DynamoRouterConfig=placeholder_type,
            build_aic_perf_config=lambda config: config,
            build_kv_router_config=lambda config: config,
            parse_args=lambda argv=None: argv,
        ),
        "dynamo.runtime": stub_module(
            "dynamo.runtime",
            Client=placeholder_type,
            DistributedRuntime=placeholder_type,
            dynamo_worker=lambda: lambda function: function,
        ),
        "dynamo.runtime.logging": stub_module(
            "dynamo.runtime.logging", configure_dynamo_logging=lambda: None
        ),
    }
    previous = {name: sys.modules.get(name) for name in stubs}
    sys.modules.update(stubs)
    try:
        module_path = Path(__file__).parents[1] / "__main__.py"
        spec = importlib.util.spec_from_file_location(
            "standalone_router_main", module_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.StandaloneRouterHandler
    finally:
        for name, previous_module in previous.items():
            if previous_module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous_module


StandaloneRouterHandler = load_standalone_router_handler()


def handler_with_router():
    handler = StandaloneRouterHandler.__new__(StandaloneRouterHandler)
    router = AsyncMock()
    handler.kv_router = router
    return handler, router


@pytest.mark.asyncio
async def test_best_worker_id_forwards_cache_namespace() -> None:
    handler, router = handler_with_router()
    router.best_worker.return_value = (7, 0, 3)

    results = [
        worker_id
        async for worker_id in handler.best_worker_id(
            [1, 2, 3, 4],
            {"temperature": 0.0},
            cache_namespace="tenant-a",
        )
    ]

    assert results == [7]
    router.best_worker.assert_awaited_once_with(
        [1, 2, 3, 4],
        {"temperature": 0.0},
        cache_namespace="tenant-a",
    )


@pytest.mark.asyncio
async def test_get_overlap_scores_forwards_cache_namespace() -> None:
    handler, router = handler_with_router()
    router.get_overlap_scores.return_value = {"workers": []}
    request = {
        "token_ids": [1, 2, 3, 4],
        "router_config_override": {"temperature": 0.0},
        "block_mm_infos": None,
        "lora_name": "adapter-a",
        "include_shared": False,
        "cache_namespace": "tenant-a",
    }

    results = [scores async for scores in handler.get_overlap_scores(request)]

    assert results == [{"workers": []}]
    router.get_overlap_scores.assert_awaited_once_with(
        [1, 2, 3, 4],
        {"temperature": 0.0},
        None,
        "adapter-a",
        False,
        "tenant-a",
    )


class TestStandaloneRouterRefusesDisaggFlags:
    """Flags the standalone router cannot honour must fail, not be ignored.

    `dynamo.router` builds only a `KvRouter`, never a `PrefillRouter`, so a
    disaggregation flag it accepted would start a plain router that reads as
    the treatment arm.

    These drive `parse_args` rather than mutating a parsed config, so they
    cover the wiring too: a flag that was never registered, or registered to
    the wrong `dest`, would otherwise pass a validator-only test.

    `ThunderAgentRouterConfig` subclasses this config and calls `super()`, so
    it inherits both guards. The messages therefore describe the router rather
    than naming a binary the operator may not have run.
    """

    def _parse(self, *flags: str):
        # Imported here, not at module scope: the stubbing above exists so this
        # file collects without the compiled bindings, and a top-level import
        # would make every test in it uncollectable.
        from dynamo.router.args import parse_args

        # --endpoint first: validate() checks it before it reaches either guard.
        return parse_args(["--endpoint", "test-ns.test-comp.generate", *flags])

    def test_prefill_continue_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no prefill router"):
            self._parse("--router-prefill-continue")

    def test_prefill_continue_config_alone_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no prefill router"):
            self._parse("--router-prefill-continue-config", '{"max_concurrent": 2}')

    def test_conditional_disagg_is_refused(self) -> None:
        """The sibling guard this one mirrors."""
        with pytest.raises(ValueError, match="does not run conditional disaggregation"):
            self._parse("--router-conditional-disagg")

    def test_no_disagg_flags_parse_cleanly(self) -> None:
        """A config that asks for neither must not trip either guard."""
        config = self._parse()

        assert config.prefill_continue_enabled is False
        assert config.conditional_disagg_enabled is False
