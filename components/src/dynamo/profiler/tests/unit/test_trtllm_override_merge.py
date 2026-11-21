# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for TRT-LLM --override-engine-args / --trtllm.* conflict resolution (GitHub #8659)."""

import json
import shlex

import pytest

from dynamo.profiler.utils.config_modifiers.trtllm import (
    _merge_overrides_into_args,
    enable_trtllm_chunked_prefill,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
    pytest.mark.planner,
    pytest.mark.parallel,
]


def _component(name: str, component_type: str, args: list[str], **fields) -> dict:
    return {
        "name": name,
        "type": component_type,
        "podTemplate": {
            "spec": {
                "containers": [{"name": "main", "args": args, **fields}],
            }
        },
    }


def _main_containers_by_name(config: dict) -> dict[str, dict]:
    return {
        component["name"]: component["podTemplate"]["spec"]["containers"][0]
        for component in config["spec"]["components"]
    }


def test_merge_overrides_into_existing_override_engine_args():
    """When --override-engine-args is already present, overrides merge into the
    JSON blob instead of appending mutually-exclusive --trtllm.* flags."""
    existing_json = json.dumps(
        {
            "cache_transceiver_config": {"backend": "DEFAULT"},
            "disable_overlap_scheduler": True,
            "kv_cache_config": {"tokens_per_block": 32},
        }
    )
    args = [
        "--model-path",
        "RedHatAI/Llama-3.3-70B-Instruct-FP8-dynamic",
        "--override-engine-args",
        existing_json,
    ]

    result = _merge_overrides_into_args(
        args,
        {
            "kv_cache_config.enable_block_reuse": False,
            "disable_overlap_scheduler": False,
            "cache_transceiver_config": None,
        },
    )

    assert not any(a.startswith("--trtllm.") for a in result), (
        "Both --override-engine-args and --trtllm.* flags present; "
        "TRT-LLM will reject this combination"
    )

    idx = result.index("--override-engine-args")
    merged = json.loads(result[idx + 1])

    assert merged["disable_overlap_scheduler"] is False
    assert merged["cache_transceiver_config"] is None
    assert merged["kv_cache_config"]["enable_block_reuse"] is False
    assert merged["kv_cache_config"]["tokens_per_block"] == 32


def test_merge_overrides_into_equals_spelled_override_engine_args():
    """A DGD can carry the flag as one list element, `--override-engine-args={...}`.

    Treating that as absent produces exactly the mutually exclusive pair this
    module exists to avoid, and silently drops the user's engine args.
    """
    user_args = {"kv_cache_config": {"free_gpu_memory_fraction": 0.5}}
    args = ["--model-path", "m", f"--override-engine-args={json.dumps(user_args)}"]

    merged_args = _merge_overrides_into_args(args, {"max_batch_size": 64})

    assert not [arg for arg in merged_args if arg.startswith("--trtllm.")]
    assert merged_args.count("--override-engine-args") == 1
    blob = json.loads(merged_args[merged_args.index("--override-engine-args") + 1])
    assert blob["max_batch_size"] == 64
    assert blob["kv_cache_config"] == {"free_gpu_memory_fraction": 0.5}


def test_enable_chunked_prefill_with_equals_spelled_override_engine_args():
    user_args = {"kv_cache_config": {"free_gpu_memory_fraction": 0.5}}
    config = {
        "spec": {
            "components": [
                _component(
                    "worker",
                    "worker",
                    [
                        "--model-path",
                        "m",
                        f"--override-engine-args={json.dumps(user_args)}",
                    ],
                )
            ]
        }
    }

    updated_args = enable_trtllm_chunked_prefill(config)["spec"]["components"][0][
        "podTemplate"
    ]["spec"]["containers"][0]["args"]

    assert not [arg for arg in updated_args if arg.startswith("--trtllm.")]
    assert updated_args.count("--override-engine-args") == 1
    blob = json.loads(updated_args[updated_args.index("--override-engine-args") + 1])
    assert blob["enable_chunked_prefill"] is True
    assert blob["kv_cache_config"] == {"free_gpu_memory_fraction": 0.5}


def test_enable_chunked_prefill_updates_generated_trtllm_workers():
    prefill_override = json.dumps(
        {
            "enable_chunked_prefill": False,
            "kv_cache_config": {"tokens_per_block": 32},
        }
    )
    config = {
        "spec": {
            "components": [
                _component("Frontend", "frontend", []),
                _component(
                    "prefill",
                    "prefill",
                    ["--override-engine-args", prefill_override],
                ),
                _component(
                    "empty_override",
                    "prefill",
                    ["--override-engine-args", "{}"],
                ),
                _component(
                    "decode",
                    "decode",
                    ["--trtllm.enable_chunked_prefill", "false"],
                ),
                _component(
                    "dangling",
                    "decode",
                    ["--trtllm.enable_chunked_prefill"],
                ),
                _component("encode", "encode", []),
            ]
        }
    }

    result = enable_trtllm_chunked_prefill(config)
    result = enable_trtllm_chunked_prefill(result)

    containers = _main_containers_by_name(result)
    prefill_args = containers["prefill"]["args"]
    assert not any(arg.startswith("--trtllm.") for arg in prefill_args)
    override_idx = prefill_args.index("--override-engine-args")
    override = json.loads(prefill_args[override_idx + 1])
    assert override["enable_chunked_prefill"] is True
    assert override["kv_cache_config"]["tokens_per_block"] == 32

    empty_override_args = containers["empty_override"]["args"]
    assert not any(arg.startswith("--trtllm.") for arg in empty_override_args)
    override_idx = empty_override_args.index("--override-engine-args")
    assert json.loads(empty_override_args[override_idx + 1]) == {
        "enable_chunked_prefill": True
    }

    decode_args = containers["decode"]["args"]
    assert decode_args.count("--trtllm.enable_chunked_prefill") == 1
    flag_idx = decode_args.index("--trtllm.enable_chunked_prefill")
    assert decode_args[flag_idx + 1] == "true"

    dangling_args = containers["dangling"]["args"]
    assert dangling_args == ["--trtllm.enable_chunked_prefill", "true"]

    encode_args = containers["encode"]["args"]
    assert encode_args == []


def test_enable_chunked_prefill_preserves_shell_form_workers():
    dynamic_command = (
        "export READY=1 && python3 -m dynamo.trtllm "
        '--model-path "${MODEL_PATH}" '
        "--trtllm.enable_chunked_prefill false && echo ready"
    )
    override_command = (
        "python3 -m dynamo.trtllm "
        '--model-path "${MODEL_PATH}" '
        "--override-engine-args "
        '\'{"kv_cache_config": {"tokens_per_block": 32}}\''
    )
    config = {
        "spec": {
            "components": [
                _component(
                    "dynamic",
                    "decode",
                    [dynamic_command],
                    command=["/bin/sh", "-c"],
                ),
                _component(
                    "override",
                    "prefill",
                    [override_command],
                    command=["sh", "-c"],
                ),
            ]
        }
    }

    result = enable_trtllm_chunked_prefill(config)
    result = enable_trtllm_chunked_prefill(result)

    containers = _main_containers_by_name(result)
    dynamic_args = containers["dynamic"]["args"]
    assert dynamic_args == [
        "export READY=1 && python3 -m dynamo.trtllm "
        '--model-path "${MODEL_PATH}" '
        "--trtllm.enable_chunked_prefill true && echo ready"
    ]

    override_args = containers["override"]["args"]
    assert len(override_args) == 1
    assert '--model-path "${MODEL_PATH}"' in override_args[0]
    override_tokens = shlex.split(override_args[0])
    assert not any(token.startswith("--trtllm.") for token in override_tokens)
    override_index = override_tokens.index("--override-engine-args")
    override = json.loads(override_tokens[override_index + 1])
    assert override["enable_chunked_prefill"] is True
    assert override["kv_cache_config"]["tokens_per_block"] == 32


def test_malformed_override_engine_args_raises_instead_of_being_dropped():
    """A typo must not start the engine with the user's settings missing.

    Every occurrence of the flag is stripped from the returned args, so
    swallowing an unparseable value hands the engine only the profiler's own
    overrides and loses the user's on the way.
    """
    args = ["--model-path", "m", "--override-engine-args={bad}"]

    with pytest.raises(ValueError, match="not valid JSON"):
        _merge_overrides_into_args(args, {"max_batch_size": 64})


def test_non_object_override_engine_args_raises():
    args = ["--model-path", "m", "--override-engine-args=[1, 2]"]

    with pytest.raises(ValueError, match="must be a JSON object"):
        _merge_overrides_into_args(args, {"max_batch_size": 64})
