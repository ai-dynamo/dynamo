# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for the narrowly qualified causal vLLM decision profile."""

import builtins
import hashlib
import importlib
import json
from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput, ModelType, WorkerType
from dynamo.vllm import decision_execution as execution
from dynamo.vllm.decision_execution import (
    VLLM_CAUSAL_DECISION_CAPABILITY,
    publish_vllm_decision_capability,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


class SamplingParamsSchema:
    def __init__(self, *, logprob_token_ids=None):
        self.logprob_token_ids = logprob_token_ids


class RequestOutputSchema:
    def __init__(self, *, num_cached_tokens=None, num_cache_creation_tokens=None):
        self.num_cached_tokens = num_cached_tokens
        self.num_cache_creation_tokens = num_cache_creation_tokens


@pytest.fixture
def profile():
    return {
        "runtime_config": SimpleNamespace(set_engine_specific=Mock()),
        "model_input": ModelInput.Tokens,
        "model_type": ModelType.Chat,
        "worker_type": WorkerType.Aggregated,
        "config": SimpleNamespace(
            disaggregation_mode=DisaggregationMode.AGGREGATED,
            enable_rl=False,
            route_to_encoder=False,
            custom_encoder_class=None,
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
            lora_config=None,
            speculative_config=None,
            kv_transfer_config=None,
            ec_transfer_config=None,
        ),
        "native_generate_supported": True,
        "vllm_version": "0.32.0",
        "sampling_params_type": SamplingParamsSchema,
        "request_output_type": RequestOutputSchema,
    }


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
    [ModelType.Chat, ModelType.Completions, ModelType.Chat | ModelType.Completions],
)
def test_publishes_only_boolean_capability(profile, architecture, model_type):
    profile["vllm_config"].model_config.architecture = architecture
    assert publish_vllm_decision_capability(**{**profile, "model_type": model_type})
    profile["runtime_config"].set_engine_specific.assert_called_once_with(
        VLLM_CAUSAL_DECISION_CAPABILITY, json.dumps(True)
    )
    assert VLLM_CAUSAL_DECISION_CAPABILITY == "vllm_causal_decision_v1"


@pytest.mark.parametrize(
    "vllm_version",
    ["0.22.0", "0.31.9", "0.33.0", "0.32.0rc1", "0.32.0.dev1", "invalid"],
)
def test_unqualified_versions_do_not_import_engine_apis(
    profile, monkeypatch, vllm_version
):
    engine_import = Mock(
        side_effect=AssertionError("Engine import before version gate")
    )
    monkeypatch.setattr("dynamo.vllm.decision_execution.import_module", engine_import)
    assert not publish_vllm_decision_capability(
        **{
            **profile,
            "vllm_version": vllm_version,
            "sampling_params_type": None,
            "request_output_type": None,
        }
    )
    engine_import.assert_not_called()
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize("vllm_version", ["0.32.0", "0.32.1", "0.32.0+cu130"])
def test_qualified_version_family(profile, vllm_version):
    assert publish_vllm_decision_capability(**{**profile, "vllm_version": vllm_version})


@pytest.mark.parametrize(
    "key,value",
    [
        ("native_generate_supported", False),
        ("model_input", ModelInput.Text),
        ("worker_type", WorkerType.Decode),
        ("worker_type", WorkerType.Prefill),
        ("worker_type", WorkerType.Encode),
        ("model_type", ModelType.Embedding),
        ("model_type", ModelType.Empty),
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


@pytest.mark.parametrize("key", ["sampling_params_type", "request_output_type"])
def test_missing_selected_logprob_or_usage_schema_does_not_publish(profile, key):
    assert not publish_vllm_decision_capability(**{**profile, key: SimpleNamespace})
    profile["runtime_config"].set_engine_specific.assert_not_called()


def test_optional_cache_schema_does_not_claim_measurements(profile):
    output = RequestOutputSchema()
    assert output.num_cached_tokens is None
    assert output.num_cache_creation_tokens is None
    assert publish_vllm_decision_capability(**profile)
    assert output.num_cached_tokens is None
    assert output.num_cache_creation_tokens is None


def test_qualified_engine_apis_are_probed(profile, monkeypatch):
    schemas = {
        "vllm.sampling_params": SimpleNamespace(SamplingParams=SamplingParamsSchema),
        "vllm.outputs": SimpleNamespace(RequestOutput=RequestOutputSchema),
    }
    engine_import = Mock(side_effect=schemas.__getitem__)
    monkeypatch.setattr("dynamo.vllm.decision_execution.import_module", engine_import)
    assert publish_vllm_decision_capability(
        **{**profile, "sampling_params_type": None, "request_output_type": None}
    )
    assert engine_import.call_count == 2


def test_import_does_not_load_vllm(monkeypatch):
    module = importlib.import_module("dynamo.vllm.decision_execution")
    original_import = builtins.__import__

    def reject_engine_import(name, *args, **kwargs):
        if name == "vllm" or name.startswith("vllm."):
            raise AssertionError("vLLM must remain optional")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_engine_import)
    importlib.reload(module)


def test_runtime_version_is_used_when_not_injected(profile, monkeypatch):
    runtime_version = Mock(return_value="0.22.0")
    monkeypatch.setattr("dynamo.vllm.decision_execution.version", runtime_version)
    assert not publish_vllm_decision_capability(**{**profile, "vllm_version": None})
    runtime_version.assert_called_once_with("vllm")
    profile["runtime_config"].set_engine_specific.assert_not_called()


def test_missing_backend_distribution_does_not_publish(profile, monkeypatch):
    monkeypatch.setattr(
        "dynamo.vllm.decision_execution.version",
        Mock(side_effect=PackageNotFoundError("vllm")),
    )
    assert not publish_vllm_decision_capability(**{**profile, "vllm_version": None})
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
    assert not publish_vllm_decision_capability(
        **{**profile, "sampling_params_type": None, "request_output_type": None}
    )
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.fixture
def pinned_profile(profile, monkeypatch):
    monkeypatch.setattr(
        execution, "_pinned_main_available", Mock(return_value=True), raising=False
    )
    return {**profile, "vllm_version": "0.32.1rc1.dev172+g9da5f2d31"}


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
    ],
)
def test_exact_main_publishes_only_pinned_causal_architectures(
    pinned_profile, architecture
):
    pinned_profile["vllm_config"].model_config.architecture = architecture
    assert publish_vllm_decision_capability(**pinned_profile)
    execution._pinned_main_available.assert_called_once_with(architecture)
    pinned_profile["runtime_config"].set_engine_specific.assert_called_once_with(
        VLLM_CAUSAL_DECISION_CAPABILITY, json.dumps(True)
    )


@pytest.mark.parametrize(
    "architecture",
    [
        "Gemma4ForCausalLM",
        "UnverifiedGemma4ForConditionalGeneration",
        "Lfm2ForCausalLM",
        "Lfm2VlForConditionalGeneration",
        "DiffusionGemmaForBlockDiffusion",
        "DiffusionGemmaForConditionalGeneration",
        "D3ForSequenceClassification",
    ],
)
def test_main_does_not_alias_unqualified_models_to_causal(pinned_profile, architecture):
    pinned_profile["vllm_config"].model_config.architecture = architecture
    assert not publish_vllm_decision_capability(**pinned_profile)
    execution._pinned_main_available.assert_not_called()
    pinned_profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "architecture",
    ["Gemma4ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration"],
)
def test_stable_version_does_not_gain_main_gemma_architectures(profile, architecture):
    profile["vllm_config"].model_config.architecture = architecture
    assert not publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "installed",
    [
        "0.32.1rc1.dev172",
        "0.32.1rc1.dev172+g0000000000",
        "0.32.1rc1.dev173+g9da5f2d31",
        "0.32.1rc1.dev172+g9da5f2d31.local",
        "0.32.1.dev172+g9da5f2d31",
    ],
)
def test_other_main_versions_fail_before_optional_backend_import(
    profile, monkeypatch, installed
):
    backend_import = Mock(side_effect=AssertionError("Unqualified main import"))
    monkeypatch.setattr(execution, "import_module", backend_import)
    assert not publish_vllm_decision_capability(
        **{**profile, "vllm_version": installed}
    )
    backend_import.assert_not_called()
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "key,value",
    [
        ("runner_type", "pooling"),
        ("logprobs_mode", "processed_logprobs"),
        ("logprobs_mode", "raw_logits"),
        ("is_encoder_decoder", True),
        ("is_diffusion", True),
        ("model_impl", "transformers"),
        ("model_class_overrides", {"Gemma4ForConditionalGeneration": "custom.Model"}),
        ("convert_type", "classify"),
    ],
)
def test_main_rejects_noncausal_or_processed_model_configs(pinned_profile, key, value):
    setattr(pinned_profile["vllm_config"].model_config, key, value)
    assert not publish_vllm_decision_capability(**pinned_profile)
    pinned_profile["runtime_config"].set_engine_specific.assert_not_called()


def test_main_with_unverified_source_never_publishes(pinned_profile):
    execution._pinned_main_available.return_value = False
    assert not publish_vllm_decision_capability(**pinned_profile)
    pinned_profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize("key", ["sampling_params_type", "request_output_type"])
def test_verified_main_still_requires_requested_id_and_cache_apis(pinned_profile, key):
    assert not publish_vllm_decision_capability(
        **{**pinned_profile, key: SimpleNamespace}
    )
    pinned_profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.fixture
def pinned_sources(tmp_path, monkeypatch):
    package = tmp_path / "vllm"
    package.mkdir()
    entry = package / "__init__.py"
    entry.write_text("")

    def source(relative):
        content = relative.encode()
        target = package / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return relative, hashlib.sha256(content).hexdigest()

    common = tuple(source(relative) for relative, _ in execution._PINNED_CAUSAL_SOURCES)
    models = {
        architecture: tuple(source(relative) for relative, _ in sources)
        for architecture, sources in execution._PINNED_CAUSAL_MODELS.items()
    }
    monkeypatch.setattr(execution, "_PINNED_CAUSAL_SOURCES", common)
    monkeypatch.setattr(execution, "_PINNED_CAUSAL_MODELS", models)
    modules = {
        "vllm": SimpleNamespace(
            __file__=str(entry), __version__=execution.PINNED_VLLM_VERSION
        ),
        "vllm._version": SimpleNamespace(
            __version__=execution.PINNED_VLLM_VERSION, __commit_id__="g9da5f2d31"
        ),
    }
    monkeypatch.setattr(
        execution, "import_module", Mock(side_effect=modules.__getitem__)
    )
    metadata_version = Mock(return_value=execution.PINNED_VLLM_VERSION)
    monkeypatch.setattr(execution, "version", metadata_version)
    return SimpleNamespace(package=package, modules=modules, version=metadata_version)


def test_source_gate_accepts_only_the_exact_verified_install(pinned_sources):
    assert execution._pinned_main_available("Gemma4ForConditionalGeneration")


@pytest.mark.parametrize(
    "relative",
    [
        "sampling_params.py",
        "outputs.py",
        "config/model.py",
        "model_executor/models/registry.py",
        "model_executor/models/gemma4.py",
        "model_executor/models/gemma4_mm.py",
        "v1/engine/async_llm.py",
        "v1/engine/logprobs.py",
        "v1/sample/sampler.py",
    ],
)
def test_changed_scoring_or_model_sources_fail_closed(pinned_sources, relative):
    (pinned_sources.package / relative).write_bytes(b"changed")
    assert not execution._pinned_main_available("Gemma4ForConditionalGeneration")


def test_unified_gemma_requires_its_own_source_hash(pinned_sources):
    (pinned_sources.package / "model_executor/models/gemma4_unified.py").unlink()
    assert not execution._pinned_main_available("Gemma4UnifiedForConditionalGeneration")
    assert execution._pinned_main_available("Gemma4ForConditionalGeneration")


@pytest.mark.parametrize(
    "name,attribute,value",
    [
        ("vllm", "__version__", "0.32.0"),
        ("vllm._version", "__version__", "0.32.0"),
        ("vllm._version", "__commit_id__", "g0000000000"),
        ("vllm", "__file__", None),
    ],
)
def test_source_gate_rejects_wrong_package_provenance(
    pinned_sources, name, attribute, value
):
    setattr(pinned_sources.modules[name], attribute, value)
    assert not execution._pinned_main_available("Gemma4ForConditionalGeneration")


def test_injected_main_version_cannot_override_installed_distribution(pinned_sources):
    pinned_sources.version.return_value = "0.32.0"
    assert not execution._pinned_main_available("Gemma4ForConditionalGeneration")


@pytest.mark.parametrize(
    "error",
    [
        ImportError("ABI mismatch"),
        OSError("native load"),
        RuntimeError("native schema"),
    ],
)
def test_main_backend_load_errors_disable_advertisement(
    pinned_profile, monkeypatch, error
):
    monkeypatch.setattr(execution, "import_module", Mock(side_effect=error))
    assert not publish_vllm_decision_capability(
        **{**pinned_profile, "sampling_params_type": None, "request_output_type": None}
    )
    pinned_profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "field", ["is_diffusion", "model_impl", "model_class_overrides", "convert_type"]
)
def test_main_with_incomplete_model_schema_fails_closed(pinned_profile, field):
    delattr(pinned_profile["vllm_config"].model_config, field)
    assert not publish_vllm_decision_capability(**pinned_profile)
    pinned_profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Gemma4ForConditionalGeneration",
        "Gemma4UnifiedForConditionalGeneration",
    ],
)
def test_installed_exact_main_sampling_output_and_model_source_apis(architecture):
    try:
        installed = execution.version("vllm")
    except PackageNotFoundError:
        pytest.skip("Optional vLLM runtime is not installed")
    if installed != execution.PINNED_VLLM_VERSION:
        pytest.skip("Exact pinned-main native runtime is not installed")
    assert execution._pinned_main_available(architecture)
    assert execution._decision_api_available(None, None)
    params_type = importlib.import_module("vllm.sampling_params").SamplingParams
    params = params_type(
        max_tokens=1, logprob_token_ids=list(range(26)), detokenize=False
    )
    assert params.logprob_token_ids == list(range(26))
    assert params.max_tokens == 1
    output_type = importlib.import_module("vllm.outputs").RequestOutput
    output = output_type(
        request_id="cpu-schema",
        prompt=None,
        prompt_token_ids=[1],
        prompt_logprobs=None,
        outputs=[],
        finished=True,
    )
    assert output.num_cached_tokens is None
    assert output.num_cache_creation_tokens is None
