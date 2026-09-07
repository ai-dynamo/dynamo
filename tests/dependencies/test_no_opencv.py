# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression guard: opencv-python-headless is not installed in TRT-LLM images.

Upstream TRT-LLM made cv2 an optional video extra
(https://github.com/NVIDIA/TensorRT-LLM/pull/16206) and its 1.3.x imports are
already function-local, so nothing needs it at import time. Dynamo's
trtllm_runtime.Dockerfile drops the preinstall (including the vendored media
libraries under opencv_python_headless.libs/) in both the runtime_full stage and
the pre_runtime whiteout, slimming the image. This test keeps a base-image bump
or Dockerfile refactor from silently reintroducing it.
"""

import importlib.util
import pathlib

import pytest

pytestmark = [
    pytest.mark.trtllm,
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.gpu_0,
    pytest.mark.unit,
    pytest.mark.skipif(
        importlib.util.find_spec("tensorrt_llm") is None,
        reason="TRT-LLM images only (other frameworks may ship opencv)",
    ),
]


def test_opencv_not_installed():
    """cv2 must not be importable anywhere on the image's python path."""
    spec = importlib.util.find_spec("cv2")
    assert spec is None, (
        f"opencv-python-headless is installed (cv2 at {spec.origin}); "
        "it must be removed from TRT-LLM images — see "
        "trtllm_runtime.Dockerfile (uninstall RUN + pre_runtime whiteout)"
    )


def test_no_vendored_opencv_libs():
    """The wheel's vendored shared libraries must be gone from dist-packages."""
    leftovers = list(
        pathlib.Path("/usr/local/lib/python3.12/dist-packages").glob(
            "opencv_python_headless*"
        )
    )
    assert not leftovers, (
        f"vendored opencv artifacts still on disk: {leftovers}; "
        "the pre_runtime whiteout in trtllm_runtime.Dockerfile must remove them"
    )
