# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup and installed-runtime contracts for causal decision execution."""

import argparse
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from dynamo.common.constants import DisaggregationMode
from dynamo.llm import ModelInput, ModelRuntimeConfig, WorkerType
from dynamo.vllm import decision_cache_provenance as cache
from dynamo.vllm import decision_execution as decisions
from dynamo.vllm.args import _parse_cli_args
from dynamo.vllm.backend_args import DynamoVllmArgGroup, DynamoVllmConfig

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
    disabled, engine_args = _parse_cli_args(["--no-systemone-serial"])
    assert disabled.systemone_serial is False
    assert engine_args.max_num_seqs is None
    assert engine_args.logprobs_mode == "raw_logprobs"


def test_systemone_serial_rejects_invalid_environment(monkeypatch):
    monkeypatch.setenv("DYN_VLLM_SYSTEMONE_SERIAL", "perhaps")
    with pytest.raises(argparse.ArgumentTypeError, match="true/false"):
        DynamoVllmArgGroup().add_arguments(argparse.ArgumentParser())


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
def test_invalid_explicit_opt_in_fails_before_engine_start(arguments, setting):
    config, engine_args = _parse_cli_args(
        ["--systemone-serial", "--max-num-seqs", "1", *arguments]
    )
    config.engine_args = engine_args
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


def test_explicit_opt_in_rejects_unqualified_runtime_settings(profile):
    profile["vllm_config"].model_config.architecture = "UnknownForCausalLM"
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


def test_cache_source_verifier_rejects_modified_and_missing_files(tmp_path):
    content = b"cache accounting owner\n"
    source = tmp_path / "vllm" / "outputs.py"
    source.parent.mkdir()
    source.write_bytes(content)
    expected = (("vllm/outputs.py", cache.git_blob_id(content)),)
    assert cache.verify_installed_sources(expected, root=tmp_path)
    source.write_bytes(content + b"modified\n")
    assert not cache.verify_installed_sources(expected, root=tmp_path)
    source.unlink()
    assert not cache.verify_installed_sources(expected, root=tmp_path)
