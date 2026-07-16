# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit test for TRT-LLM --override-engine-args / --trtllm.* conflict resolution (GitHub #8659)."""

import json

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


def test_enable_chunked_prefill_updates_generated_trtllm_workers():
    prefill_override = json.dumps(
        {
            "enable_chunked_prefill": False,
            "kv_cache_config": {"tokens_per_block": 32},
        }
    )
    config = {
        "spec": {
            "services": {
                "Frontend": {"componentType": "frontend"},
                "prefill": {
                    "componentType": "worker",
                    "subComponentType": "prefill",
                    "extraPodSpec": {
                        "mainContainer": {
                            "args": ["--override-engine-args", prefill_override]
                        }
                    },
                },
                "empty_override": {
                    "componentType": "worker",
                    "subComponentType": "prefill",
                    "extraPodSpec": {
                        "mainContainer": {"args": ["--override-engine-args", "{}"]}
                    },
                },
                "decode": {
                    "componentType": "worker",
                    "subComponentType": "decode",
                    "extraPodSpec": {
                        "mainContainer": {
                            "args": [
                                "--trtllm.enable_chunked_prefill",
                                "false",
                            ]
                        }
                    },
                },
                "dangling": {
                    "componentType": "worker",
                    "subComponentType": "decode",
                    "extraPodSpec": {
                        "mainContainer": {"args": ["--trtllm.enable_chunked_prefill"]}
                    },
                },
                "encode": {
                    "componentType": "worker",
                    "subComponentType": "encode",
                    "extraPodSpec": {"mainContainer": {"args": []}},
                },
            }
        }
    }

    result = enable_trtllm_chunked_prefill(config)
    result = enable_trtllm_chunked_prefill(result)

    prefill_args = result["spec"]["services"]["prefill"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert not any(arg.startswith("--trtllm.") for arg in prefill_args)
    override_idx = prefill_args.index("--override-engine-args")
    override = json.loads(prefill_args[override_idx + 1])
    assert override["enable_chunked_prefill"] is True
    assert override["kv_cache_config"]["tokens_per_block"] == 32

    empty_override_args = result["spec"]["services"]["empty_override"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert not any(arg.startswith("--trtllm.") for arg in empty_override_args)
    override_idx = empty_override_args.index("--override-engine-args")
    assert json.loads(empty_override_args[override_idx + 1]) == {
        "enable_chunked_prefill": True
    }

    decode_args = result["spec"]["services"]["decode"]["extraPodSpec"]["mainContainer"][
        "args"
    ]
    assert decode_args.count("--trtllm.enable_chunked_prefill") == 1
    flag_idx = decode_args.index("--trtllm.enable_chunked_prefill")
    assert decode_args[flag_idx + 1] == "true"

    dangling_args = result["spec"]["services"]["dangling"]["extraPodSpec"][
        "mainContainer"
    ]["args"]
    assert dangling_args == ["--trtllm.enable_chunked_prefill", "true"]

    encode_args = result["spec"]["services"]["encode"]["extraPodSpec"]["mainContainer"][
        "args"
    ]
    assert encode_args == []
