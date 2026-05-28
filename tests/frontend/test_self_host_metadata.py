# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for worker-self-hosted metadata files (gh-8749).

Single-worker check: opt the mocker into self-host via
`DYN_SELF_HOST_METADATA=true`, then `requests.get` the
`/v1/metadata/{slug}/{suffix}/{filename}` route on the worker's own
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

        url = f"http://localhost:{system_port}/v1/metadata/{slug}/{suffix}/config.json"
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
            f"http://localhost:{system_port}/v1/metadata/{slug}/{suffix}/missing.json",
            timeout=5,
        )
        assert miss.status_code == 404, miss.text

        logger.info(
            "self-host metadata verified: %s/%s/config.json served from worker on port %d",
            slug,
            suffix,
            system_port,
        )
