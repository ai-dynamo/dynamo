# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the Dynamo frontend, dynamo-triton-sidecar, and a real
tritonserver subprocess, covering the OpenAI `/v1/classify` and `/v1/embeddings`
paths against the shipped `classifier` and `embedder` example models."""

import base64
import hashlib
import logging
import struct

import pytest
import requests

from tests.utils.triton_sidecar import sidecar_stack

pytest.importorskip("tritonserver")

logger = logging.getLogger(__name__)

_CLASSIFIER_MODEL = "classifier"
_NUM_CLASSES = 3

_EMBEDDER_MODEL = "embedder"
_EMBEDDING_DIM = 8


def _expected_vector(prompt: str) -> list[float]:
    digest = hashlib.sha256(prompt.encode("utf-8")).digest()
    return [byte / 255.0 for byte in digest[:_EMBEDDING_DIM]]


@pytest.fixture(scope="function")
def sidecar_classify_stack(request, tmp_path, start_services_with_http):
    yield from sidecar_stack(
        request, tmp_path, start_services_with_http, "--classify", _CLASSIFIER_MODEL
    )


@pytest.fixture(scope="function")
def sidecar_embeddings_stack(request, tmp_path, start_services_with_http):
    yield from sidecar_stack(
        request, tmp_path, start_services_with_http, "--embeddings", _EMBEDDER_MODEL
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


@pytest.mark.e2e
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.triton
@pytest.mark.parallel
@pytest.mark.timeout(360)
def test_embeddings_returns_deterministic_float_vector(sidecar_embeddings_stack):
    """Guards the `/v1/embeddings` happy path when the frontend decodes base64 to floats."""
    frontend_port, model = sidecar_embeddings_stack
    prompt = "hello"

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/embeddings",
        json={"model": model, "input": prompt, "encoding_format": "float"},
        timeout=30,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["object"] == "list"
    assert body["model"] == model

    assert len(body["data"]) == 1
    row = body["data"][0]
    assert row["index"] == 0
    assert row["object"] == "embedding"
    assert row["embedding"] == pytest.approx(_expected_vector(prompt), rel=1e-6)


@pytest.mark.e2e
@pytest.mark.pre_merge
@pytest.mark.gpu_0
@pytest.mark.triton
@pytest.mark.parallel
@pytest.mark.timeout(360)
def test_embeddings_base64_round_trips_to_the_same_vector(sidecar_embeddings_stack):
    """Guards the base64 wire format so a byte-order or alphabet change is caught."""
    frontend_port, model = sidecar_embeddings_stack
    prompt = "hello"

    response = requests.post(
        f"http://localhost:{frontend_port}/v1/embeddings",
        json={"model": model, "input": prompt, "encoding_format": "base64"},
        timeout=30,
    )

    assert response.status_code == 200, response.text
    body = response.json()
    row = body["data"][0]
    # Base64 is little-endian FP32 bytes, standard alphabet; see
    # lib/sidecar/triton/src/pooling/embeddings.rs::encode_base64.
    decoded = base64.standard_b64decode(row["embedding"])
    floats = list(struct.unpack(f"<{_EMBEDDING_DIM}f", decoded))
    assert floats == pytest.approx(_expected_vector(prompt), rel=1e-6)
