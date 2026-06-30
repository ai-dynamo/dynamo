# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for worker-self-hosted metadata files (gh-8749).

Single-worker check: opt the mocker into self-host via
`DYN_SELF_HOST_METADATA=true`, then `requests.get` the
`/v1/metadata/{namespace}/{component}/{endpoint}/{slug}/{suffix}/{filename}`
route on the worker's own
`system_status_server`. No frontend resolution involved — this only
exercises the worker producer + HTTP route shipped in PR1 (#8855).

Scope grows as later PRs land — PR2 will add frontend
verify-and-cache assertions; PR4 will add multi-replica + LoRA cases.
"""

from __future__ import annotations

import json
import logging

import pytest
import requests

from tests.frontend.conftest import MockerWorkerProcess, wait_for_http_completions_ready
from tests.utils.constants import QWEN

logger = logging.getLogger(__name__)

TEST_MODEL = QWEN

pytestmark = [
    pytest.mark.gpu_0,  # mocker is CPU-only
    pytest.mark.e2e,
    pytest.mark.post_merge,
    pytest.mark.model(TEST_MODEL),
]


_SLUG_KEEP = set("abcdefghijklmnopqrstuvwxyz0123456789-_")


def _slugify(name: str) -> str:
    """Mirror of `dynamo_runtime::slug::Slug::slugify` for URL construction.

    Lowercase, then replace any char outside `[a-z0-9_-]` with `_`, then
    strip leading underscores.
    """
    out = "".join(c if c in _SLUG_KEEP else "_" for c in name.lower())
    return out.lstrip("_")


@pytest.mark.timeout(120)
def test_worker_serves_metadata_via_http(
    request,
    start_services_with_http,
    predownload_tokenizers,
) -> None:
    """Worker advertises `/v1/metadata/...`; assert curl returns the file."""
    frontend_port, system_port = start_services_with_http

    with MockerWorkerProcess(
        request,
        TEST_MODEL,
        frontend_port,
        system_port,
        worker_id="self-host-mocker",
        extra_env={"DYN_SELF_HOST_METADATA": "true"},
    ):
        # Wait until the model registration has fully propagated through
        # discovery — proxy for the worker having opted into self-host.
        wait_for_http_completions_ready(frontend_port=frontend_port, model=TEST_MODEL)

        slug = _slugify(TEST_MODEL)
        suffix = "_base"  # BASE_SUFFIX in lib/runtime/src/metadata_registry.rs
        # MockerWorkerProcess defaults: --component=sample, the rest pick up
        # WorkerConfig defaults from lib/runtime — namespace=dynamo,
        # endpoint=generate. The route is now keyed on the endpoint triple
        # so the URL includes (namespace, component, endpoint).
        base = f"http://localhost:{system_port}/v1/metadata/dynamo/sample/generate"

        url = f"{base}/{slug}/{suffix}/config.json"
        response = requests.get(url, timeout=10)
        assert (
            response.status_code == 200
        ), f"GET {url} returned {response.status_code}: {response.text}"
        # config.json is JSON — body should parse and look like a HF config.
        parsed = json.loads(response.content)
        assert isinstance(parsed, dict), f"expected dict, got {type(parsed)}"
        assert (
            "model_type" in parsed or "architectures" in parsed
        ), f"config.json missing expected HF keys: {sorted(parsed)}"

        # Unknown filename under the same (slug, suffix) → 404.
        miss = requests.get(
            f"{base}/{slug}/{suffix}/missing.json",
            timeout=5,
        )
        assert miss.status_code == 404, miss.text

        # extra_files harvest: a non-typed sibling that exists in the
        # Qwen3 HF snapshot (`vocab.json`) must also be served. Doubles
        # as a symlink-follow check — HF snapshot entries are symlinks
        # into `blobs/`, so a non-following stat would miss this file.
        extra_url = f"{base}/{slug}/{suffix}/vocab.json"
        extra_response = requests.get(extra_url, timeout=10)
        assert (
            extra_response.status_code == 200
        ), f"GET {extra_url} returned {extra_response.status_code}: harvest did not advertise vocab.json"
        assert extra_response.content, "vocab.json served but body was empty"

        logger.info(
            "self-host metadata verified: %s/%s/{config,vocab}.json served from worker on port %d",
            slug,
            suffix,
            system_port,
        )
