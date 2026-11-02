# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed publication and ordinary startup's optional-native import boundary."""

import subprocess
import sys
import textwrap
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import dynamo._core as active_core
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.vllm import decision_native_registry as registry
from dynamo.vllm import decision_runtime_policy as policy

pytestmark = [
    pytest.mark.unit,
    pytest.mark.pre_merge,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.timeout(60),
]


@pytest.mark.parametrize("installed", [None, 1])
def test_nonstring_release_never_calls_the_version_parser(monkeypatch, installed):
    parser = Mock(side_effect=AssertionError("nonstring reached the parser"))
    monkeypatch.setattr(policy, "Version", parser)
    assert policy.is_exact_vllm_release(installed, "0.33.0") is False
    parser.assert_not_called()


@pytest.mark.parametrize("variant", ["NativeReady", "NativeFailed"])
def test_native_publication_returns_the_written_variant(variant):
    publication = getattr(registry, variant)(registry.PROFILES[registry.D3_PROFILE])
    runtime = SimpleNamespace(set_engine_specific=Mock())
    written = registry.publish_native_decision_capability(
        runtime,
        publication,
        ModelInput.Text,
        ModelType.Classify | ModelType.Pooling,
        WorkerType.Aggregated,
    )
    assert written is publication
    assert runtime.set_engine_specific.call_count == (
        2 if variant == "NativeReady" else 1
    )


@pytest.mark.parametrize("phase", ["construct", "initialize"])
def test_qwen_handler_and_factory_do_not_import_optional_native_owners(phase):
    script = textwrap.dedent(
        """
        import asyncio
        import builtins
        import importlib.util
        import sys
        from types import SimpleNamespace

        core_path = sys.argv[2]
        core_spec = importlib.util.spec_from_file_location('dynamo._core', core_path)
        core = importlib.util.module_from_spec(core_spec)
        sys.modules['dynamo._core'] = core
        core_spec.loader.exec_module(core)
        assert core.__file__ == core_path

        blocked = {
            'dynamo.vllm.decision_surface',
            'dynamo.vllm.decision_heads',
            'dynamo.vllm.decision_reads',
            'dynamo.vllm.decision_native_errors',
            'dynamo.vllm.decision_native_lifecycle',
            'dynamo.vllm.decision_native_loader',
            'dynamo.vllm.decision_native_wire',
        }
        original_import = builtins.__import__

        def guarded(name, globals=None, locals=None, fromlist=(), level=0):
            absolute = (
                importlib.util.resolve_name('.' * level + name, globals['__package__'])
                if level else name
            )
            candidates = {absolute} | {
                absolute + '.' + item for item in (fromlist or ())
            }
            assert not candidates & blocked, candidates & blocked
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = guarded
        from dynamo.llm.exceptions import HttpError
        from dynamo.vllm import handlers, worker_factory
        from dynamo.vllm.decision_dispatch import DECISION_DISPATCH_METADATA_KEY

        def base(self, runtime, config, engine, *args, **kwargs):
            self.config = config
            self.engine_client = engine
            self.model_config = kwargs['model_config']

        handlers.BaseWorkerHandler.__init__ = base
        config = SimpleNamespace()
        model = SimpleNamespace(architecture='Qwen3ForCausalLM')
        handler = handlers.DecodeWorkerHandler(
            None, config, object(), {}, model_config=model
        )
        assert handler.decision.publication is None
        assert handler.decision.maybe_handle(
            {'token_ids': [1]}, SimpleNamespace(metadata={})
        ) is None

        async def check():
            if sys.argv[1] == 'initialize':
                handler.decision = await worker_factory.initialize_native_decisions(
                    'read', handler.engine_client, model, config, abort_monitor=None
                )
                assert handler.decision.publication is None
            marked = handler.decision.maybe_handle(
                {'decision_read': {}},
                SimpleNamespace(metadata={DECISION_DISPATCH_METADATA_KEY: 'read'}),
            )
            assert marked is not None
            try:
                await anext(marked)
            except HttpError as cause:
                assert cause.code == 503
            else:
                raise AssertionError('unready private RPC reached ordinary generation')

        asyncio.run(check())
        assert not blocked & sys.modules.keys(), blocked & sys.modules.keys()
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, phase, active_core.__file__],
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
