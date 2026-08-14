# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared fixtures for recipe tests run against an already-deployed frontend."""

import pytest


@pytest.fixture(scope="module")
def endpoint_client(attached_endpoint):
    """An OpenAI client addressed at the deployed frontend.

    ``max_retries=0`` on purpose: a retry would paper over exactly the protocol
    failures these tests exist to catch.
    """
    openai = pytest.importorskip("openai")
    return openai.OpenAI(
        base_url=f"{attached_endpoint.base_url}/v1",
        api_key="not-used",  # Dynamo does not authenticate by default
        timeout=300.0,
        max_retries=0,
        default_headers=dict(attached_endpoint.headers or {}),
    )
