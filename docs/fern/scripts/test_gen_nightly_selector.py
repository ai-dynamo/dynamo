# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the nightly selector generator's ledger package guard.

Run: pytest -c docs/fern/scripts/pytest.ini docs/fern/scripts/test_gen_nightly_selector.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import gen_nightly_selector as gen  # noqa: E402

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


class TestLedgerPackageGuard:
    def test_ledger_requires_every_advertised_package(self):
        version = "1.5.0.dev20260914"
        missing_companion = {
            "ai-dynamo": {version},
            "ai-dynamo-runtime": set(),
            "kvbm": {version},
        }
        assert not gen.ledger_version_published(version, missing_companion)

    def test_ledger_accepts_a_version_published_by_every_package(self):
        version = "1.5.0.dev20260914"
        published = {package: {version} for package in gen.NIGHTLY_PACKAGES}
        assert gen.ledger_version_published(version, published)
