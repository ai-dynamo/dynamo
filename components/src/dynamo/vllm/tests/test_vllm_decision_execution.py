# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup and installed-runtime contracts for causal decision execution."""

import argparse
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import vllm
from transformers import Qwen3Config
from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import MAX_LOGPROB_TOKEN_IDS

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput, ModelRuntimeConfig, WorkerType
from dynamo.vllm import decision_cache_provenance as cache
from dynamo.vllm import decision_execution as decisions
from dynamo.vllm.args import _parse_cli_args
from dynamo.vllm.backend_args import DynamoVllmArgGroup, DynamoVllmConfig
from dynamo.vllm.handlers import build_sampling_params

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.pre_merge,
]


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.setattr(decisions, "version", lambda _: "0.33.0")
    monkeypatch.setattr(decisions, "verify_cache_sources", lambda: True)
    return {
        "runtime_config": ModelRuntimeConfig(),
        "model_input": ModelInput.Tokens,
        "worker_type": WorkerType.Aggregated,
        "native_generate_supported": True,
        "config": SimpleNamespace(
            systemone_serial=True,
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


def test_disabled_worker_preserves_metadata_and_skips_qualification(
    profile, monkeypatch
):
    profile["config"].systemone_serial = False
    qualification = Mock(side_effect=AssertionError("disabled worker must not qualify"))
    monkeypatch.setattr(decisions, "_released_runtime", qualification)
    original = dict(profile["runtime_config"].runtime_data)

    assert not decisions.publish_vllm_decision_capability(**profile)
    assert profile["runtime_config"].runtime_data == original


@pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator")
def test_systemone_serial_accepts_environment_and_cli_override(monkeypatch):
    monkeypatch.setenv("DYN_VLLM_SYSTEMONE_SERIAL", "true")
    parser = argparse.ArgumentParser()
    DynamoVllmArgGroup().add_arguments(parser)
    config = DynamoVllmConfig.from_cli_args(parser.parse_args([]))
    assert config.systemone_serial is True
    disabled, _ = _parse_cli_args(["--no-systemone-serial"])
    assert disabled.systemone_serial is False
    enabled, engine_args = _parse_cli_args(["--systemone-serial"])
    assert enabled.systemone_serial is True
    assert engine_args.max_num_seqs is None


@pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator")
def test_explicit_opt_in_validates_before_engine_start(monkeypatch):
    monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
    monkeypatch.setenv("DYN_FPM_TRACE", "0")
    config, engine_args = _parse_cli_args(["--systemone-serial", "--max-num-seqs", "1"])
    config.engine_args = AsyncEngineArgs.from_cli_args(engine_args)
    config.validate()
    assert config.systemone_serial is True
    assert config.engine_args.max_num_seqs == 1


@pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator")
@pytest.mark.parametrize("setting", ["DYN_FORWARDPASS_METRIC_PORT", "--fpm-trace"])
def test_forward_pass_metrics_rejected_before_engine_start(
    monkeypatch, unused_tcp_port, setting
):
    monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
    monkeypatch.setenv("DYN_FPM_TRACE", "0")
    arguments = ["--systemone-serial", "--max-num-seqs", "1"]
    if setting == "DYN_FORWARDPASS_METRIC_PORT":
        monkeypatch.setenv(setting, str(unused_tcp_port))
    else:
        arguments.append(setting)
    config, engine_args = _parse_cli_args(arguments)
    config.engine_args = AsyncEngineArgs.from_cli_args(engine_args)
    with pytest.raises(ValueError, match=setting):
        config.validate()


def test_wrapper_validates_opt_in_without_owning_engine_arguments():
    parser = argparse.ArgumentParser()
    DynamoVllmArgGroup().add_arguments(parser)
    config = DynamoVllmConfig.from_cli_args(parser.parse_args(["--systemone-serial"]))
    config.validate()
    config.use_vllm_tokenizer = True
    with pytest.raises(ValueError, match="--use-vllm-tokenizer"):
        config.validate()


@pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator")
@pytest.mark.parametrize(
    ("arguments", "setting"),
    [
        (["--max-num-seqs", "4"], "--max-num-seqs=1"),
        (["--logprobs-mode", "processed_logprobs"], "--logprobs-mode=raw_logprobs"),
        (["--disaggregation-mode", "decode"], "--disaggregation-mode=agg"),
        (["--data-parallel-size", "2"], "--data-parallel-size=1"),
        (["--scheduler-cls", "custom.Scheduler"], "--scheduler-cls"),
        (["--enable-lora"], "--enable-lora"),
    ],
)
def test_invalid_explicit_opt_in_fails_before_engine_start(
    arguments, setting, monkeypatch
):
    monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
    monkeypatch.setenv("DYN_FPM_TRACE", "0")
    config, engine_args = _parse_cli_args(
        ["--systemone-serial", "--max-num-seqs", "1", *arguments]
    )
    config.engine_args = AsyncEngineArgs.from_cli_args(engine_args)
    with pytest.raises(ValueError, match=setting):
        config.validate()


def test_qualified_worker_publishes_capability_and_logs_success(profile, caplog):
    with caplog.at_level(logging.INFO, logger=decisions.__name__):
        assert decisions.publish_vllm_decision_capability(**profile)
    assert (
        profile["runtime_config"].runtime_data[
            decisions.VLLM_CAUSAL_DECISION_CAPABILITY
        ]
        == "true"
    )
    assert "Published vLLM causal decision capability" in caplog.text


@pytest.mark.parametrize("release", ["0.32.0", "0.33.0rc1", "0.33.0.dev1"])
def test_explicit_opt_in_rejects_unqualified_runtime(profile, monkeypatch, release):
    monkeypatch.setattr(decisions, "version", lambda _: release)
    with pytest.raises(ValueError, match="vLLM 0.33.0"):
        decisions.publish_vllm_decision_capability(**profile)


def test_explicit_opt_in_names_unsupported_architecture(profile):
    profile["vllm_config"].model_config.architecture = "UnknownForCausalLM"
    original = dict(profile["runtime_config"].runtime_data)
    with pytest.raises(
        ValueError, match="unsupported model architecture 'UnknownForCausalLM'"
    ):
        decisions.publish_vllm_decision_capability(**profile)
    assert profile["runtime_config"].runtime_data == original


def test_explicit_opt_in_rejects_unqualified_runtime_settings(profile):
    profile["vllm_config"].model_config.logprobs_mode = "processed_logprobs"
    with pytest.raises(ValueError, match="qualified causal model"):
        decisions.publish_vllm_decision_capability(**profile)


def test_explicit_opt_in_rejects_modified_cache_source(profile, monkeypatch):
    monkeypatch.setattr(decisions, "verify_cache_sources", lambda: False)
    with pytest.raises(ValueError, match="cache source"):
        decisions.publish_vllm_decision_capability(**profile)


def test_installed_cache_sources_match_qualified_release():
    """Run against the actual pinned vLLM container, not the mocked profile."""
    assert decisions._released_runtime(), "Requalify Decisions when upgrading vLLM"
    assert cache.verify_cache_sources()


@pytest.fixture
def installed_model_config(tmp_path, monkeypatch) -> ModelConfig:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    Qwen3Config(
        architectures=["Qwen3ForCausalLM"],
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=128,
    ).save_pretrained(tmp_path)
    return ModelConfig(model=str(tmp_path), max_model_len=128, skip_tokenizer_init=True)


@pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator")
@pytest.mark.parametrize("async_scheduling", [None, False])
def test_installed_configuration_qualifies_without_model_load(
    installed_model_config, async_scheduling, monkeypatch
):
    monkeypatch.delenv("DYN_FORWARDPASS_METRIC_PORT", raising=False)
    monkeypatch.setenv("DYN_FPM_TRACE", "0")
    scheduler = SchedulerConfig.default_factory(
        max_num_seqs=1,
        max_model_len=128,
        max_num_batched_tokens=128,
        async_scheduling=async_scheduling,
    )
    assert scheduler.async_scheduling is async_scheduling
    runtime = VllmConfig(
        model_config=installed_model_config,
        scheduler_config=scheduler,
        parallel_config=ParallelConfig(),
    )
    config, engine_args = _parse_cli_args(["--systemone-serial", "--max-num-seqs", "1"])
    config.engine_args = AsyncEngineArgs.from_cli_args(engine_args)
    config.validate()
    assert type(runtime.scheduler_config.async_scheduling) is bool
    assert decisions.publish_vllm_decision_capability(
        ModelRuntimeConfig(),
        ModelInput.Tokens,
        WorkerType.Aggregated,
        runtime,
        config,
        native_generate_supported=True,
    )


@pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator")
def test_explicit_candidate_logprobs_exceed_default_top_k_limit(installed_model_config):
    assert decisions._released_runtime(), "Requalify Decisions when upgrading vLLM"
    candidate_ids = list(range(26))
    params = build_sampling_params(
        {"sampling_options": {"logprob_token_ids": candidate_ids}}, {}
    )
    assert installed_model_config.max_logprobs == 20
    assert params.logprobs is None
    assert params.num_logprobs == len(candidate_ids)
    params._validate_logprobs(installed_model_config)
    with pytest.raises(VLLMValidationError, match="greater than max allowed: 20"):
        build_sampling_params(
            {"output_options": {"logprobs": len(candidate_ids)}}, {}
        )._validate_logprobs(installed_model_config)
    with pytest.raises(VLLMValidationError, match="greater than max allowed: 128"):
        build_sampling_params(
            {
                "sampling_options": {
                    "logprob_token_ids": [0] * (MAX_LOGPROB_TOKEN_IDS + 1)
                }
            },
            {},
        )._validate_logprobs(installed_model_config)


def test_cache_source_verifier_reports_missing_module_file(monkeypatch, caplog):
    monkeypatch.setattr(vllm, "__file__", None)
    assert not cache.verify_cache_sources()
    assert "cache source unavailable: vllm/__init__.py" in caplog.text
    assert "disabled" not in caplog.text


def test_cache_source_verifier_rejects_modified_and_missing_files(tmp_path, caplog):
    content = b"cache accounting owner\n"
    source = tmp_path / "vllm" / "outputs.py"
    source.parent.mkdir()
    source.write_bytes(content)
    expected = (("vllm/outputs.py", cache.git_blob_id(content)),)
    assert cache.verify_installed_sources(expected, root=tmp_path)
    source.write_bytes(content + b"modified\n")
    assert not cache.verify_installed_sources(expected, root=tmp_path)
    assert "cache source differs from qualified release: vllm/outputs.py" in caplog.text
    assert "disabled" not in caplog.text
    assert str(tmp_path) not in caplog.text
    caplog.clear()
    source.unlink()
    assert not cache.verify_installed_sources(expected, root=tmp_path)
    assert "cache source unavailable: vllm/outputs.py" in caplog.text
    assert "disabled" not in caplog.text
    assert str(tmp_path) not in caplog.text
