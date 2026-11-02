# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the narrowly qualified causal vLLM decision profile."""

import builtins
import importlib
import json
from importlib.metadata import PackageNotFoundError
from inspect import signature
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.vllm.decision_execution import (
    VLLM_CAUSAL_DECISION_CAPABILITY,
    publish_vllm_decision_capability,
)
from dynamo.vllm.decision_runtime_policy import is_exact_vllm_release, qualified_runtime
from dynamo.vllm.engine_generate import publish_engine_generate_capability

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


@pytest.mark.parametrize(
    ("installed", "release", "qualified"),
    [
        ("0.32.0", "0.32.0", True),
        ("0.32.0+cu130", "0.32.0", True),
        ("0.32.0.0", "0.32.0", False),
        ("0.33.0rc1", "0.33.0rc1", True),
        ("0.33.0rc1+cu130", "0.33.0rc1", True),
        ("0.33.0rc2", "0.33.0rc1", False),
        ("0.33.0rc1.dev5+g1234567", "0.33.0rc1", False),
        ("0.33.0rc1.post1", "0.33.0rc1", False),
        ("1!0.33.0rc1", "0.33.0rc1", False),
        ("0.33.0", "0.33.0rc1", False),
        ("0.32.0", "0.33.0rc1", False),
        ("0.33.0rc1", "0.32.0", False),
        ("0.32.0.dev1", "0.32.0", False),
        ("0.32.0.post1", "0.32.0", False),
        ("1!0.32.0", "0.32.0", False),
        ("not-a-version", "0.33.0rc1", False),
    ],
)
def test_exact_release_ignores_only_local_build_suffix(installed, release, qualified):
    assert is_exact_vllm_release(installed, release) is qualified


class SamplingParamsSchema:
    def __init__(self, *, logprob_token_ids=None):
        self.logprob_token_ids = logprob_token_ids


class RequestOutputSchema:
    def __init__(self, *, num_cached_tokens=None, num_cache_creation_tokens=None):
        self.num_cached_tokens = num_cached_tokens
        self.num_cache_creation_tokens = num_cache_creation_tokens


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.setattr(
        "dynamo.vllm.decision_execution.qualified_cache_runtime", lambda _: True
    )
    monkeypatch.setattr("dynamo.vllm.decision_execution.version", lambda _: "0.32.0")
    schemas = {
        "vllm.sampling_params": SimpleNamespace(SamplingParams=SamplingParamsSchema),
        "vllm.outputs": SimpleNamespace(RequestOutput=RequestOutputSchema),
    }
    monkeypatch.setattr(
        "dynamo.vllm.decision_execution.import_module", schemas.__getitem__
    )
    return {
        "runtime_config": SimpleNamespace(set_engine_specific=Mock()),
        "model_input": ModelInput.Tokens,
        "worker_type": WorkerType.Aggregated,
        "config": SimpleNamespace(
            disaggregation_mode=DisaggregationMode.AGGREGATED,
            enable_rl=False,
            route_to_encoder=False,
            custom_encoder_class=None,
            enable_multimodal=False,
            realtime=False,
        ),
        "vllm_config": SimpleNamespace(
            model_config=SimpleNamespace(
                architecture="Qwen3ForCausalLM",
                runner_type="generate",
                logprobs_mode="raw_logprobs",
                is_encoder_decoder=False,
                is_diffusion=False,
                model_impl="auto",
                model_class_overrides={},
                convert_type="none",
                max_model_len=2048,
            ),
            scheduler_config=SimpleNamespace(
                max_num_seqs=1, scheduler_cls=None, async_scheduling=False
            ),
            parallel_config=SimpleNamespace(data_parallel_size=1),
            lora_config=None,
            speculative_config=None,
            kv_transfer_config=None,
            ec_transfer_config=None,
        ),
        "native_generate_supported": True,
    }


@pytest.mark.parametrize(
    "field",
    [
        "model_config",
        "parallel_config",
        "lora_config",
        "speculative_config",
        "kv_transfer_config",
        "ec_transfer_config",
    ],
)
def test_shared_runtime_requires_complete_engine_metadata(profile, field):
    runtime = profile["vllm_config"]
    delattr(runtime, field)
    assert not qualified_runtime(
        runtime, runner_type="generate", diffusion=False, min_model_len=2
    )


def test_shared_runtime_uses_the_selected_runner_context_bound(profile):
    runtime = profile["vllm_config"]
    runtime.model_config.runner_type = "pooling"
    runtime.model_config.max_model_len = 1
    assert qualified_runtime(
        runtime, runner_type="pooling", diffusion=False, min_model_len=1
    )
    assert not qualified_runtime(
        runtime, runner_type="generate", diffusion=False, min_model_len=2
    )


@pytest.mark.parametrize("max_num_seqs", [None, 0, 2, 16, True])
def test_causal_scoring_rejects_unqualified_scheduling(profile, max_num_seqs):
    profile["vllm_config"].scheduler_config.max_num_seqs = max_num_seqs
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize("release", ["0.32.0", "0.33.0"])
def test_modified_cache_source_never_publishes_causal_capability(
    profile, monkeypatch, release
):
    monkeypatch.setattr("dynamo.vllm.decision_execution.version", lambda _: release)
    monkeypatch.setattr(
        "dynamo.vllm.decision_execution.qualified_cache_runtime", lambda _: False
    )
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize("data_parallel_size", [0, 2, True])
def test_causal_scoring_requires_single_rank(profile, data_parallel_size):
    profile["vllm_config"].parallel_config.data_parallel_size = data_parallel_size
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize("scheduler_cls", ["custom.Scheduler", object])
def test_causal_scoring_rejects_custom_schedulers(profile, scheduler_cls):
    profile["vllm_config"].scheduler_config.scheduler_cls = scheduler_cls
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model_impl", "transformers"),
        ("model_class_overrides", {"forward": "custom"}),
        ("convert_type", "embed"),
        ("is_diffusion", True),
    ],
)
def test_causal_scoring_rejects_replacement_implementations(profile, field, value):
    setattr(profile["vllm_config"].model_config, field, value)
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
    ],
)
@pytest.mark.parametrize(
    "model_type",
    [
        ModelType.Chat,
        ModelType.Completions,
        ModelType.Chat | ModelType.Completions,
        ModelType.Chat | ModelType.Pooling,
    ],
)
@pytest.mark.parametrize("installed", ["0.32.0", "0.33.0"])
def test_publishes_only_boolean_capability(
    profile, monkeypatch, architecture, model_type, installed
):
    monkeypatch.setattr("dynamo.vllm.decision_execution.version", lambda _: installed)
    profile["vllm_config"].model_config.architecture = architecture
    supported = publish_engine_generate_capability(
        profile["runtime_config"],
        profile["model_input"],
        model_type,
        profile["worker_type"],
        False,
    )
    profile["runtime_config"].set_engine_specific.reset_mock()
    assert supported
    assert publish_vllm_decision_capability(
        **{**profile, "native_generate_supported": supported}
    )
    profile["runtime_config"].set_engine_specific.assert_called_once_with(
        VLLM_CAUSAL_DECISION_CAPABILITY, json.dumps(True)
    )
    assert VLLM_CAUSAL_DECISION_CAPABILITY == "vllm_causal_decision_v1"


@pytest.mark.parametrize(
    "model_type",
    [
        ModelType.Completions | ModelType.Pooling,
        ModelType.Completions | ModelType.Embedding,
        ModelType.Embedding,
        ModelType.Empty,
    ],
)
def test_causal_model_eligibility_comes_from_real_generate_publisher(
    profile, model_type
):
    assert "model_type" not in signature(publish_vllm_decision_capability).parameters
    supported = publish_engine_generate_capability(
        profile["runtime_config"],
        profile["model_input"],
        model_type,
        profile["worker_type"],
        False,
    )
    assert not supported
    assert not publish_vllm_decision_capability(
        **{**profile, "native_generate_supported": supported}
    )
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "vllm_version",
    [
        "0.22.0",
        "0.31.9",
        "0.33.0.dev1",
        "0.33.0.post1",
        "1!0.33.0",
        "0.33.1",
        "0.32.0rc1",
        "0.32.0.dev1",
        "0.32.1rc1.dev172+g9da5f2d31",
        "0.33.0rc1",
        "0.32.1",
        "0.32.0.post1",
        "1!0.32.0",
        "invalid",
    ],
)
def test_unqualified_versions_do_not_import_engine_apis(
    profile, monkeypatch, vllm_version
):
    engine_import = Mock(
        side_effect=AssertionError("Engine import before version gate")
    )
    monkeypatch.setattr("dynamo.vllm.decision_execution.import_module", engine_import)
    monkeypatch.setattr(
        "dynamo.vllm.decision_execution.version", lambda _: vllm_version
    )
    assert not publish_vllm_decision_capability(**profile)
    engine_import.assert_not_called()
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "vllm_version", ["0.32.0", "0.32.0+cu130", "0.33.0", "0.33.0+cu130"]
)
def test_qualified_version_family(profile, monkeypatch, vllm_version):
    monkeypatch.setattr(
        "dynamo.vllm.decision_execution.version", lambda _: vllm_version
    )
    assert publish_vllm_decision_capability(**profile)


@pytest.mark.parametrize(
    "key,value",
    [
        ("native_generate_supported", False),
        ("model_input", ModelInput.Text),
        ("worker_type", WorkerType.Decode),
        ("worker_type", WorkerType.Prefill),
        ("worker_type", WorkerType.Encode),
        ("needs", [[WorkerType.Encode]]),
    ],
)
def test_rejects_unqualified_transport_and_role(profile, key, value):
    assert not publish_vllm_decision_capability(**{**profile, key: value})
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "key,value",
    [
        ("architecture", "LlamaForCausalLM"),
        ("architecture", "UnverifiedQwenForCausalLM"),
        ("runner_type", "pooling"),
        ("logprobs_mode", "processed_logprobs"),
        ("logprobs_mode", "raw_logits"),
        ("is_encoder_decoder", True),
        ("max_model_len", None),
        ("max_model_len", 0),
        ("max_model_len", 1),
        ("max_model_len", True),
    ],
)
def test_rejects_unqualified_effective_model_config(profile, key, value):
    setattr(profile["vllm_config"].model_config, key, value)
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "key",
    ["lora_config", "speculative_config", "kv_transfer_config", "ec_transfer_config"],
)
def test_rejects_engine_extensions(profile, key):
    setattr(profile["vllm_config"], key, SimpleNamespace())
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "key,value",
    [
        ("enable_rl", True),
        ("route_to_encoder", True),
        ("custom_encoder_class", "custom.Encoder"),
        ("disaggregation_mode", DisaggregationMode.DECODE),
    ],
)
def test_rejects_dynamo_extensions(profile, key, value):
    setattr(profile["config"], key, value)
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize("key", ["vllm.sampling_params", "vllm.outputs"])
def test_missing_selected_logprob_or_usage_schema_does_not_publish(
    profile, monkeypatch, key
):
    schemas = {
        "vllm.sampling_params": SimpleNamespace(SamplingParams=SamplingParamsSchema),
        "vllm.outputs": SimpleNamespace(RequestOutput=RequestOutputSchema),
    }
    schemas[key] = SimpleNamespace(
        SamplingParams=SimpleNamespace, RequestOutput=SimpleNamespace
    )
    monkeypatch.setattr(
        "dynamo.vllm.decision_execution.import_module", schemas.__getitem__
    )
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


def test_optional_cache_schema_does_not_claim_measurements(profile):
    output = RequestOutputSchema()
    assert output.num_cached_tokens is None
    assert output.num_cache_creation_tokens is None
    assert publish_vllm_decision_capability(**profile)
    assert output.num_cached_tokens is None
    assert output.num_cache_creation_tokens is None


@pytest.mark.parametrize("field", ["enable_multimodal", "realtime"])
def test_decision_worker_rejects_non_text_modes(profile, field):
    setattr(profile["config"], field, True)
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


def test_uninspectable_causal_schema_logs_disabled_reason(profile, monkeypatch, caplog):
    def unavailable_signature(value):
        raise ValueError("opaque native signature")

    monkeypatch.setattr(
        "dynamo.vllm.decision_execution.signature", unavailable_signature
    )
    assert not publish_vllm_decision_capability(**profile)
    assert "signature" in caplog.text


def test_qualified_engine_apis_are_probed(profile, monkeypatch):
    schemas = {
        "vllm.sampling_params": SimpleNamespace(SamplingParams=SamplingParamsSchema),
        "vllm.outputs": SimpleNamespace(RequestOutput=RequestOutputSchema),
    }
    engine_import = Mock(side_effect=schemas.__getitem__)
    monkeypatch.setattr("dynamo.vllm.decision_execution.import_module", engine_import)
    assert publish_vllm_decision_capability(**profile)
    assert engine_import.call_count == 2


def test_import_does_not_load_vllm(monkeypatch):
    module = importlib.import_module("dynamo.vllm.decision_execution")
    original_import = builtins.__import__

    def reject_engine_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise AssertionError("vLLM must remain optional")
        if name.startswith("dynamo.vllm.decision_native") or name in {
            "dynamo.vllm.decision_heads",
            "dynamo.vllm.decision_reads",
        }:
            raise AssertionError("Causal readiness must not require native adapters")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_engine_import)
    importlib.reload(module)


def test_neutral_runtime_policy_import_does_not_load_an_optional_backend(monkeypatch):
    original_import = builtins.__import__

    def reject_backend_import(name, *args, **kwargs):
        if (
            name == "vllm"
            or name.startswith("vllm.")
            or name.startswith("dynamo.vllm.decision_native")
        ):
            raise AssertionError("Shared policy must not import an optional backend")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_backend_import)
    policy = importlib.import_module("dynamo.vllm.decision_runtime_policy")
    importlib.reload(policy)


def test_runtime_version_is_used_when_not_injected(profile, monkeypatch):
    runtime_version = Mock(return_value="0.22.0")
    monkeypatch.setattr("dynamo.vllm.decision_execution.version", runtime_version)
    assert not publish_vllm_decision_capability(**profile)
    runtime_version.assert_called_once_with("vllm")
    profile["runtime_config"].set_engine_specific.assert_not_called()


def test_missing_backend_distribution_does_not_publish(profile, monkeypatch):
    monkeypatch.setattr(
        "dynamo.vllm.decision_execution.version",
        Mock(side_effect=PackageNotFoundError("vllm")),
    )
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "exception",
    [ModuleNotFoundError("vllm.sampling_params"), AttributeError("SamplingParams")],
)
def test_unavailable_optional_engine_api_does_not_publish(
    profile, monkeypatch, exception
):
    monkeypatch.setattr(
        "dynamo.vllm.decision_execution.import_module", Mock(side_effect=exception)
    )
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "architecture",
    [
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
    ],
)
def test_older_release_gemma_remains_unqualified(profile, architecture):
    profile["vllm_config"].model_config.architecture = architecture
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize("installed", ["0.33.0", "0.33.0+cu130"])
@pytest.mark.parametrize(
    "architecture", ["Gemma4ForCausalLM", "Gemma4ForConditionalGeneration"]
)
def test_stable_gemma_publishes_causal_capability(
    profile, monkeypatch, installed, architecture
):
    monkeypatch.setattr("dynamo.vllm.decision_execution.version", lambda _: installed)
    profile["vllm_config"].model_config.architecture = architecture
    assert publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_called_once_with(
        VLLM_CAUSAL_DECISION_CAPABILITY, json.dumps(True)
    )


def test_stable_unified_gemma_remains_unqualified(profile, monkeypatch):
    monkeypatch.setattr("dynamo.vllm.decision_execution.version", lambda _: "0.33.0")
    profile[
        "vllm_config"
    ].model_config.architecture = "Gemma4UnifiedForConditionalGeneration"
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("worker", "enable_multimodal", True),
        ("worker", "realtime", True),
        ("worker", "enable_rl", True),
        ("scheduler_config", "max_num_seqs", 2),
        ("scheduler_config", "scheduler_cls", "custom.Scheduler"),
        ("parallel_config", "data_parallel_size", 2),
        ("model_config", "model_impl", "transformers"),
    ],
)
def test_stable_gemma_preserves_runtime_restrictions(
    profile, monkeypatch, section, field, value
):
    monkeypatch.setattr("dynamo.vllm.decision_execution.version", lambda _: "0.33.0")
    profile["vllm_config"].model_config.architecture = "Gemma4ForConditionalGeneration"
    target = (
        profile["config"]
        if section == "worker"
        else getattr(profile["vllm_config"], section)
    )
    setattr(target, field, value)
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize("field", ["is_diffusion", "model_class_overrides"])
def test_missing_model_metadata_disables_only_decisions(profile, caplog, field):
    delattr(profile["vllm_config"].model_config, field)
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()
    assert "metadata unavailable" in caplog.text


@pytest.mark.parametrize("field", ["enable_multimodal", "realtime"])
def test_missing_worker_mode_never_advertises_decisions(profile, caplog, field):
    delattr(profile["config"], field)
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()
    assert "metadata unavailable" in caplog.text
