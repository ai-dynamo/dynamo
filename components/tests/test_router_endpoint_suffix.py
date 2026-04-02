# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dynamo.router.endpoint_suffix import apply_worker_suffix_to_endpoint


def test_apply_worker_suffix_to_endpoint_without_suffix(monkeypatch):
    monkeypatch.delenv("DYN_NAMESPACE_WORKER_SUFFIX", raising=False)
    endpoint = "dynamo-test-gp-prefill-0.prefill.generate"
    assert apply_worker_suffix_to_endpoint(endpoint) == endpoint


def test_apply_worker_suffix_to_endpoint_with_suffix(monkeypatch):
    monkeypatch.setenv("DYN_NAMESPACE_WORKER_SUFFIX", "7b81fa27")
    endpoint = "dynamo-test-gp-prefill-0.prefill.generate"
    assert (
        apply_worker_suffix_to_endpoint(endpoint)
        == "dynamo-test-gp-prefill-0-7b81fa27.prefill.generate"
    )


def test_apply_worker_suffix_to_endpoint_invalid_format(monkeypatch):
    monkeypatch.setenv("DYN_NAMESPACE_WORKER_SUFFIX", "7b81fa27")
    endpoint = "invalid-endpoint-format"
    assert apply_worker_suffix_to_endpoint(endpoint) == endpoint


def test_apply_worker_suffix_to_endpoint_ignores_invalid_suffix(monkeypatch):
    monkeypatch.setenv("DYN_NAMESPACE_WORKER_SUFFIX", "bad.suffix")
    endpoint = "dynamo-test-gp-prefill-0.prefill.generate"
    assert apply_worker_suffix_to_endpoint(endpoint) == endpoint
