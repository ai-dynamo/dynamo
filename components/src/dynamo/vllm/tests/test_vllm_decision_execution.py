# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Startup and installed-runtime contracts for causal decision execution."""

import argparse
import json
import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from transformers import Qwen3Config
from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig
from vllm.engine.arg_utils import AsyncEngineArgs

from dynamo.llm import ModelInput, ModelRuntimeConfig, WorkerType
from dynamo.vllm import decision_execution as decisions
from dynamo.vllm.args import _parse_cli_args
from dynamo.vllm.backend_args import DynamoVllmArgGroup, DynamoVllmConfig
from dynamo.vllm.engine_generate import adapt_engine_generate_request

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.vllm,
    pytest.mark.core,
    pytest.mark.pre_merge,
]


@pytest.fixture
def profile(monkeypatch):
    return {
        "runtime_config": ModelRuntimeConfig(),
        "model_input": ModelInput.Tokens,
        "worker_type": WorkerType.Aggregated,
        "native_generate_supported": True,
        "config": SimpleNamespace(systemone_serial=True),
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
    profile["vllm_config"] = None
    qualification = Mock(side_effect=AssertionError("disabled worker must not qualify"))
    monkeypatch.setattr(decisions, "_unqualified_runtime_condition", qualification)
    original = dict(profile["runtime_config"].runtime_data)

    decisions.qualify_vllm_decision_runtime(
        profile.pop("vllm_config"), profile["config"]
    )
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
    decisions.qualify_vllm_decision_runtime(
        profile.pop("vllm_config"), profile["config"]
    )
    with caplog.at_level(logging.INFO, logger=decisions.__name__):
        assert decisions.publish_vllm_decision_capability(**profile)
    assert (
        profile["runtime_config"].runtime_data[
            decisions.VLLM_CAUSAL_DECISION_CAPABILITY
        ]
        == "true"
    )
    assert "Published vLLM causal decision capability" in caplog.text


def test_capability_publication_does_not_repeat_runtime_qualification(
    profile, monkeypatch
):
    profile.pop("vllm_config")
    qualification = Mock(side_effect=AssertionError("runtime is already qualified"))
    monkeypatch.setattr(decisions, "_unqualified_runtime_condition", qualification)
    assert decisions.publish_vllm_decision_capability(**profile)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("model_input", ModelInput.Text),
        ("worker_type", WorkerType.Decode),
        ("native_generate_supported", False),
        ("needs", [[WorkerType.Prefill]]),
    ],
)
def test_capability_publication_rejects_unqualified_registration(
    profile, setting, value
):
    profile.pop("vllm_config")
    profile[setting] = value
    with pytest.raises(ValueError, match="aggregated token-input worker"):
        decisions.publish_vllm_decision_capability(**profile)
    assert not profile["runtime_config"].runtime_data


@pytest.mark.parametrize(
    "architecture",
    [
        "UnknownForCausalLM",
        "Qwen3_5ForConditionalGeneration",
        "Qwen3_5MoeForConditionalGeneration",
        "Gemma4ForCausalLM",
    ],
)
def test_explicit_opt_in_names_unsupported_architecture(profile, architecture):
    profile["vllm_config"].model_config.architecture = architecture
    original = dict(profile["runtime_config"].runtime_data)
    with pytest.raises(
        ValueError, match=f"unsupported model architecture '{architecture}'"
    ):
        decisions.qualify_vllm_decision_runtime(
            profile["vllm_config"], profile["config"]
        )
    assert profile["runtime_config"].runtime_data == original


@pytest.mark.parametrize(
    ("section", "name", "value"),
    [
        ("model_config", "runner_type", "pooling"),
        ("model_config", "logprobs_mode", "processed_logprobs"),
        ("model_config", "is_encoder_decoder", True),
        ("model_config", "is_diffusion", True),
        ("model_config", "model_impl", "transformers"),
        ("model_config", "model_class_overrides", {"custom": "Model"}),
        ("model_config", "convert_type", "embed"),
        ("model_config", "max_model_len", 1),
        ("model_config", "max_model_len", True),
        ("scheduler_config", "max_num_seqs", 2),
        ("scheduler_config", "max_num_seqs", True),
        ("scheduler_config", "async_scheduling", None),
        ("scheduler_config", "scheduler_cls", "custom.Scheduler"),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "data_parallel_size", True),
        (None, "lora_config", object()),
        (None, "speculative_config", object()),
        (None, "kv_transfer_config", object()),
        (None, "ec_transfer_config", object()),
    ],
)
def test_explicit_opt_in_names_unqualified_runtime_setting(
    profile, section, name, value
):
    runtime = profile["vllm_config"]
    setattr(getattr(runtime, section) if section else runtime, name, value)
    original = dict(profile["runtime_config"].runtime_data)
    with pytest.raises(
        ValueError, match=f"runtime qualification failed: requires .*{name}"
    ):
        decisions.qualify_vllm_decision_runtime(
            profile["vllm_config"], profile["config"]
        )
    assert profile["runtime_config"].runtime_data == original


def test_explicit_opt_in_rejects_missing_runtime_metadata(profile):
    del profile["vllm_config"].model_config.runner_type
    with pytest.raises(ValueError, match="missing runtime metadata"):
        decisions.qualify_vllm_decision_runtime(
            profile["vllm_config"], profile["config"]
        )


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
    decisions.qualify_vllm_decision_runtime(runtime, config)
    assert decisions.publish_vllm_decision_capability(
        ModelRuntimeConfig(),
        ModelInput.Tokens,
        WorkerType.Aggregated,
        config,
        native_generate_supported=True,
    )


@pytest.fixture
def causal_sampling_params():
    """Golden sampling controls from Rust's causal_scoring_request."""
    return {
        "max_tokens": 1,
        "min_tokens": 0,
        "n": 1,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "min_p": 0.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
        "repetition_penalty": 1.0,
        "prompt_logprobs": None,
        "detokenize": False,
        "stop": [],
        "stop_token_ids": [],
        "ignore_eos": False,
        "bad_words": [],
        "logit_bias": {},
        "allowed_token_ids": None,
        "structured_outputs": None,
    }


@pytest.mark.usefixtures("vllm_cpu_platform_when_no_accelerator")
def test_causal_adapter_preserves_explicit_candidate_scoring(
    installed_model_config, causal_sampling_params
):
    candidate_ids = list(range(26))
    envelope = {
        "token_ids": [1, 2, 3],
        "request_id": "causal-scoring",
        "stream": False,
        "cache_salt": "request-cache-scope",
        "sampling_params": {
            **causal_sampling_params,
            "logprob_token_ids": candidate_ids,
            "logprobs": None,
        },
    }
    adapted = adapt_engine_generate_request(
        {"token_ids": [1, 2, 3], "extra_args": {"vllm_tito": envelope}},
        enable_multimodal=False,
        decode_capable=True,
        vllm_config=VllmConfig(
            model_config=installed_model_config,
            scheduler_config=SchedulerConfig.default_factory(
                max_num_seqs=1, max_model_len=128, max_num_batched_tokens=128
            ),
        ),
        default_sampling_params={"temperature": 0.6, "top_k": 20, "top_p": 0.95},
    )
    assert adapted is not None
    params = adapted.sampling_params
    assert params.logprobs is None
    assert params.logprob_token_ids == candidate_ids
    assert params.detokenize is False
    # vLLM canonicalizes greedy top-k to zero.
    assert (params.max_tokens, params.temperature, params.top_k, params.top_p) == (
        1,
        0.0,
        0,
        1.0,
    )
    assert installed_model_config.max_logprobs == 20
    params._validate_logprobs(installed_model_config)


def _write_pooling_metadata(
    model_dir, *, temperature=1.38, codes=None, joint_overrides=None
):
    (model_dir / "decision_config.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "attention_mode": "noncausal_full_attention",
                "temperature": temperature,
                "codes": codes or ["A", "B", "C", "D"],
            }
        ),
        encoding="utf-8",
    )
    joint = {
        "version": 3,
        "template": "joint-question-inline-readout-v1",
        "marker": "<|fim_suffix|>",
        "readout_offset": 0,
        "max_questions": None,
        "max_length": None,
        **(joint_overrides or {}),
    }
    (model_dir / "joint_config.json").write_text(json.dumps(joint), encoding="utf-8")


def _mock_pooling_vllm_config(
    *, use_activation=False, pooling_type=None, model=None, revision=None
):
    return SimpleNamespace(
        model_config=SimpleNamespace(
            runner_type="pooling",
            convert_type="classify",
            model=model,
            revision=revision,
            pooler_config=SimpleNamespace(
                use_activation=use_activation, pooling_type=pooling_type
            ),
        ),
        parallel_config=SimpleNamespace(data_parallel_size=1),
        lora_config=None,
    )


def _publish_pooling(
    model,
    vllm_cfg,
    *,
    model_type=decisions.ModelType.Classify | decisions.ModelType.Pooling,
    engine_args=None,
):
    recorded = {}
    runtime_config = SimpleNamespace(
        set_engine_specific=lambda k, v: recorded.__setitem__(k, json.loads(v))
    )
    ok = decisions.publish_vllm_pooling_decision_capability(
        runtime_config,
        model_type,
        decisions.WorkerType.Aggregated,
        SimpleNamespace(classify_worker=True, model=model, engine_args=engine_args),
        vllm_cfg,
        needs=[],
    )
    return ok, recorded


def test_publish_vllm_pooling_decision_capability_publishes_v3_and_v4(tmp_path):
    _write_pooling_metadata(tmp_path)
    ok, rec = _publish_pooling(str(tmp_path), _mock_pooling_vllm_config())
    assert ok is True
    desc = rec[decisions.VLLM_POOLING_DECISION_CAPABILITY]
    assert desc["template"] == "joint-question-inline-readout-v1"
    assert desc["marker"] == "<|fim_suffix|>"
    assert desc["codes"] == ["A", "B", "C", "D"]
    assert desc["temperature"] == 1.38

    _write_pooling_metadata(
        tmp_path,
        joint_overrides={
            "version": 4,
            "template": "joint-chunked-readout-v1",
            "separator": "<|fim_middle|>",
        },
    )
    ok, rec = _publish_pooling(
        str(tmp_path), _mock_pooling_vllm_config(pooling_type="LAST")
    )
    assert ok is True
    v4 = rec[decisions.VLLM_POOLING_DECISION_CAPABILITY]
    assert v4["template"] == "joint-chunked-readout-v1"
    assert v4["separator"] == "<|fim_middle|>"

    ok, rec = _publish_pooling(
        str(tmp_path), _mock_pooling_vllm_config(pooling_type="MEAN")
    )
    assert not ok and not rec


def test_publish_vllm_pooling_decision_capability_rejects_unqualified(tmp_path):
    # No metadata files -> silent no-op for ordinary classify workers
    ok, rec = _publish_pooling(str(tmp_path), _mock_pooling_vllm_config())
    assert not ok and not rec

    _write_pooling_metadata(tmp_path)
    for kwargs in (
        {"vllm_cfg": _mock_pooling_vllm_config(use_activation=True)},
        {
            "vllm_cfg": _mock_pooling_vllm_config(),
            "model_type": decisions.ModelType.Chat,
        },
    ):
        ok, rec = _publish_pooling(str(tmp_path), **kwargs)
        assert not ok and not rec

    for meta_kwargs in (
        {"temperature": -1.0},
        {"joint_overrides": {"readout_offset": 2**31}},
    ):
        _write_pooling_metadata(tmp_path, **meta_kwargs)
        ok, rec = _publish_pooling(str(tmp_path), _mock_pooling_vllm_config())
        assert not ok and not rec


def test_publish_vllm_pooling_decision_capability_resolves_local_and_hf_cache(
    tmp_path, monkeypatch
):
    import huggingface_hub

    _write_pooling_metadata(tmp_path)
    calls = []
    monkeypatch.setattr(
        huggingface_hub,
        "try_to_load_from_cache",
        lambda *, repo_id, filename, revision=None: (
            calls.append((repo_id, filename, revision)) or str(tmp_path / filename)
        ),
    )

    # HF cache resolution uses vllm_config.model_config.revision
    ok, rec = _publish_pooling(
        "org/custom-pooling-decider",
        _mock_pooling_vllm_config(
            model="org/custom-pooling-decider", revision="rev-123"
        ),
    )
    assert ok is True
    assert calls == [
        ("org/custom-pooling-decider", "decision_config.json", "rev-123"),
        ("org/custom-pooling-decider", "joint_config.json", "rev-123"),
    ]
    assert rec[decisions.VLLM_POOLING_DECISION_CAPABILITY]["codes"] == [
        "A",
        "B",
        "C",
        "D",
    ]

    # Local path in vllm_config.model_config.model or config.engine_args.model beats HF cache
    calls.clear()
    ok, _ = _publish_pooling(
        "s3://bucket/decider", _mock_pooling_vllm_config(model=str(tmp_path))
    )
    assert ok is True and calls == []

    ok, _ = _publish_pooling(
        "s3://bucket/decider",
        _mock_pooling_vllm_config(),
        engine_args=SimpleNamespace(model=str(tmp_path)),
    )
    assert ok is True and calls == []
