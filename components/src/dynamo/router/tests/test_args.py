# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from dynamo.router.args import parse_args

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]

_ENDPOINT = "dynamo.backend.generate"


def test_standalone_router_rejects_local_and_remote_indexers(monkeypatch) -> None:
    monkeypatch.delenv("DYN_SERVE_INDEXER", raising=False)
    monkeypatch.delenv("DYN_USE_REMOTE_INDEXER", raising=False)

    with pytest.raises(
        ValueError,
        match="--serve-indexer and --use-remote-indexer are mutually exclusive",
    ):
        parse_args(
            [
                "--endpoint",
                _ENDPOINT,
                "--serve-indexer",
                "--use-remote-indexer",
            ]
        )


def test_standalone_router_rejects_conflicting_indexer_environment(monkeypatch) -> None:
    monkeypatch.setenv("DYN_SERVE_INDEXER", "true")
    monkeypatch.setenv("DYN_USE_REMOTE_INDEXER", "true")

    with pytest.raises(
        ValueError,
        match="--serve-indexer and --use-remote-indexer are mutually exclusive",
    ):
        parse_args(["--endpoint", _ENDPOINT])
