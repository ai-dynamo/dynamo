# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qualification guards for causal decision execution; no engine is loaded."""

from importlib.metadata import PackageNotFoundError
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput, WorkerType
from dynamo.vllm import decision_execution as decisions

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.pre_merge,
]


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.setattr(decisions, "version", lambda _: "0.33.0")
    monkeypatch.setattr(decisions, "verify_cache_sources", lambda: True)
    monkeypatch.setattr(decisions, "_scoring_api_available", lambda: True)
    return {
        "runtime_config": SimpleNamespace(set_engine_specific=Mock()),
        "model_input": ModelInput.Tokens,
        "worker_type": WorkerType.Aggregated,
        "native_generate_supported": True,
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
    }


@pytest.mark.parametrize(
    "architecture",
    [
        "Qwen3ForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Gemma4ForCausalLM",
        "Gemma4ForConditionalGeneration",
    ],
)
def test_supported_models_publish_only_the_causal_capability(profile, architecture):
    profile["vllm_config"].model_config.architecture = architecture
    assert decisions.publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_called_once_with(
        decisions.VLLM_CAUSAL_DECISION_CAPABILITY, "true"
    )


@pytest.mark.parametrize(
    "release", ["0.32.0", "0.33.0rc1", "0.33.0.dev1", "0.33.0.post1", "bad", None]
)
def test_unqualified_versions_do_not_modify_runtime_metadata(
    profile, monkeypatch, release
):
    monkeypatch.setattr(decisions, "version", lambda _: release)
    assert not decisions.publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


def test_local_wheel_build_metadata_is_allowed(profile, monkeypatch):
    monkeypatch.setattr(decisions, "version", lambda _: "0.33.0+cu130")
    assert decisions.publish_vllm_decision_capability(**profile)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("model_config", "architecture", "UnknownForCausalLM"),
        ("model_config", "runner_type", "pooling"),
        ("model_config", "logprobs_mode", "processed_logprobs"),
        ("model_config", "is_encoder_decoder", True),
        ("model_config", "is_diffusion", True),
        ("model_config", "model_impl", "transformers"),
        ("model_config", "model_class_overrides", {"forward": "custom"}),
        ("model_config", "convert_type", "embed"),
        ("model_config", "max_model_len", 1),
        ("model_config", "max_model_len", True),
        ("scheduler_config", "max_num_seqs", 2),
        ("scheduler_config", "max_num_seqs", True),
        ("scheduler_config", "scheduler_cls", "custom.Scheduler"),
        ("scheduler_config", "async_scheduling", None),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "data_parallel_size", True),
    ],
)
def test_execution_configuration_is_qualified_before_advertising(
    profile, section, field, value
):
    setattr(getattr(profile["vllm_config"], section), field, value)
    assert not decisions.publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


@pytest.mark.parametrize(
    "field",
    ["lora_config", "speculative_config", "kv_transfer_config", "ec_transfer_config"],
)
def test_unqualified_engine_extensions_are_rejected(profile, field):
    setattr(profile["vllm_config"], field, object())
    assert not decisions.publish_vllm_decision_capability(**profile)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("disaggregation_mode", DisaggregationMode.DECODE),
        ("enable_rl", True),
        ("route_to_encoder", True),
        ("custom_encoder_class", "CustomEncoder"),
        ("enable_multimodal", True),
        ("realtime", True),
    ],
)
def test_unsupported_worker_modes_are_not_advertised(profile, field, value):
    setattr(profile["config"], field, value)
    assert not decisions.publish_vllm_decision_capability(**profile)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("native_generate_supported", False),
        ("model_input", ModelInput.Text),
        ("worker_type", WorkerType.Decode),
        ("needs", [[WorkerType.Prefill]]),
    ],
)
def test_existing_native_generate_contract_remains_a_prerequisite(
    profile, field, value
):
    profile[field] = value
    assert not decisions.publish_vllm_decision_capability(**profile)


@pytest.mark.parametrize(
    "field", ["model_config", "scheduler_config", "parallel_config", "lora_config"]
)
def test_missing_runtime_metadata_fails_closed(profile, field):
    delattr(profile["vllm_config"], field)
    assert not decisions.publish_vllm_decision_capability(**profile)


@pytest.mark.parametrize("gate", ["verify_cache_sources", "_scoring_api_available"])
def test_missing_source_provenance_or_scoring_schema_disables_decisions(
    profile, monkeypatch, gate
):
    monkeypatch.setattr(decisions, gate, lambda: False)
    assert not decisions.publish_vllm_decision_capability(**profile)
    profile["runtime_config"].set_engine_specific.assert_not_called()


def test_absent_package_disables_only_decision_advertisement(profile, monkeypatch):
    def missing(_):
        raise PackageNotFoundError("vllm")

    monkeypatch.setattr(decisions, "version", missing)
    assert not decisions.publish_vllm_decision_capability(**profile)


def test_matching_sync_and_async_schedulers_are_supported(profile):
    for async_mode, scheduler in [
        (False, "vllm.v1.core.sched.scheduler.Scheduler"),
        (True, "vllm.v1.core.sched.async_scheduler.AsyncScheduler"),
    ]:
        profile["vllm_config"].scheduler_config.async_scheduling = async_mode
        profile["vllm_config"].scheduler_config.scheduler_cls = scheduler
        assert decisions.publish_vllm_decision_capability(**profile)


def test_cache_source_verifier_checks_installed_bytes_and_missing_files(tmp_path):
    from dynamo.vllm import decision_cache_provenance as cache

    content = b"cache accounting owner\n"
    source = tmp_path / "vllm" / "outputs.py"
    source.parent.mkdir()
    source.write_bytes(content)
    expected = cache.git_blob_id(content)
    assert cache.verify_installed_sources(
        (("vllm/outputs.py", expected),), root=tmp_path
    )
    source.write_bytes(content + b"modified\n")
    assert not cache.verify_installed_sources(
        (("vllm/outputs.py", expected),), root=tmp_path
    )
    source.unlink()
    assert not cache.verify_installed_sources(
        (("vllm/outputs.py", expected),), root=tmp_path
    )


@pytest.mark.parametrize("missing", [False, True])
def test_source_verification_resolves_the_installed_package(
    tmp_path, monkeypatch, missing
):
    from dynamo.vllm import decision_cache_provenance as cache

    package = tmp_path / "vllm"
    package.mkdir()
    output = package / "outputs.py"
    output.write_bytes(b"measured counter\n")
    monkeypatch.setattr(
        cache,
        "import_module",
        lambda _: SimpleNamespace(
            __file__=None if missing else str(package / "__init__.py")
        ),
    )
    blobs = (("vllm/outputs.py", cache.git_blob_id(output.read_bytes())),)
    assert cache.verify_installed_sources(blobs) is (not missing)


def test_cache_verifier_uses_the_qualified_manifest(monkeypatch):
    from dynamo.vllm import decision_cache_provenance as cache

    verifier = Mock(return_value=True)
    monkeypatch.setattr(cache, "verify_installed_sources", verifier)
    assert cache.verify_cache_sources()
    verifier.assert_called_once_with(cache.CACHE_SOURCE_BLOBS)


@pytest.mark.parametrize("complete", [False, True])
def test_scoring_api_requires_requested_ids_and_measured_cache_fields(
    monkeypatch, complete
):
    class SamplingParams:
        def __init__(self, *, logprob_token_ids=None):
            pass

    class RequestOutput:
        def __init__(self, *, num_cached_tokens=None, num_cache_creation_tokens=None):
            pass

    schemas = {
        "vllm.sampling_params": SimpleNamespace(SamplingParams=SamplingParams),
        "vllm.outputs": SimpleNamespace(
            RequestOutput=RequestOutput if complete else object
        ),
    }
    monkeypatch.setattr(decisions, "import_module", schemas.__getitem__)
    assert decisions._scoring_api_available() is complete


def test_unavailable_scoring_api_disables_advertisement(monkeypatch):
    def unavailable(_):
        raise ImportError("optional backend is absent")

    monkeypatch.setattr(decisions, "import_module", unavailable)
    assert not decisions._scoring_api_available()
