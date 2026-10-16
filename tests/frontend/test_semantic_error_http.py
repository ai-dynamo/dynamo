# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise semantic engine failures through real HTTP/TCP with no GPU inference."""

import importlib.util
import json
import os
import sys

import pytest

from tests.utils.constants import QWEN
from tests.utils.managed_process import ManagedProcess

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.model(QWEN),
]


@pytest.mark.parametrize(
    "processor",
    [
        "dynamo",
        pytest.param(
            "sglang",
            marks=[
                pytest.mark.sglang,
                pytest.mark.skipif(
                    importlib.util.find_spec("sglang") is None,
                    reason="SGLang is optional",
                ),
            ],
        ),
    ],
)
@pytest.mark.parametrize(
    "default_policy", [False, True], ids=["overrides", "legacy-defaults"]
)
@pytest.mark.timeout(120)
def test_semantic_error_http(
    processor, default_policy, tmp_path, predownload_tokenizers
):
    # Only tokenizer/config assets are loaded; no model weights or GPU allocation.
    output = tmp_path / "results.json"
    command = [
        sys.executable,
        "-m",
        "tests.frontend.semantic_error_http_worker",
        "--processor",
        processor,
        "--model-path",
        QWEN,
        "--output",
        str(output),
    ]
    env = {
        **os.environ,
        "DYN_SYSTEM_PORT": "-1",
        "DYN_SGLANG_STREAM_INTERVAL": "1",
        "DYN_HTTP_PRE_COMMIT_ERROR_PEEK_MS": "30",
    }
    if default_policy:
        env = {
            key: value
            for key, value in env.items()
            if key != "DYN_HTTP_ERROR_STATUS_OVERRIDES"
        }
        command = [*command, "--legacy-controls-only", "--default-policy"]
    else:
        env = {
            **env,
            "DYN_HTTP_ERROR_STATUS_OVERRIDES": "CapacityExhausted=429,InvalidRequest=422,UnsupportedMedia=409,Internal=502,RateLimited=429",
        }
    with ManagedProcess(
        command=command,
        env=env,
        log_dir=str(tmp_path / "worker"),
        terminate_all_matching_process_names=False,
        display_output=False,
        display_name="semantic-error-worker",
    ) as process:
        assert process.proc.wait(timeout=100) == 0, process.read_logs()
    result = json.loads(output.read_text())
    assert len(result["errors"]) == (0 if default_policy else 21)
    assert len(result["controls"]) == 3
