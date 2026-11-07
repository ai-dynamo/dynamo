# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end `/v1/classify` through the Dynamo frontend, dynamo-triton-sidecar,
and a real tritonserver subprocess loading the shipped `classifier` example."""

import logging

import pytest
import requests

from tests.utils.triton_sidecar import sidecar_stack

pytest.importorskip("tritonserver")

logger = logging.getLogger(__name__)

_MODEL_NAME = "classifier"
_NUM_CLASSES = 3


@pytest.fixture(scope="function")
def sidecar_classify_stack(request, tmp_path, start_services_with_http):
    yield from sidecar_stack(
        request, tmp_path, start_services_with_http, "--classify", _MODEL_NAME
    )


@pytest.mark.e2e
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.triton
@pytest.mark.parallel
@pytest.mark.timeout(360)
def test_classify_returns_deterministic_probabilities(sidecar_classify_stack):
    """Guards the `/v1/classify` → sidecar → Triton KServe v2 request/response mapping."""
    frontend_port, model = sidecar_classify_stack
    prompt = "hello"

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/classify",
        json={"model": model, "input": prompt},
        timeout=30,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "list"
    assert body["id"].startswith("classify-")
    assert body["model"] == model

    assert len(body["data"]) == 1
    row = body["data"][0]
    expected_class = sum(ord(c) for c in prompt) % _NUM_CLASSES
    expected_probs = [0.0] * _NUM_CLASSES
    expected_probs[expected_class] = 1.0
    assert row["index"] == 0
    assert row["num_classes"] == _NUM_CLASSES
    assert row["probs"] == expected_probs
    # The shipped model has no `label_filename`, so the sidecar reports no label.
    assert row["label"] is None
