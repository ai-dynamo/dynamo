# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the input-mode rule that guards frontend route extensions.

Route extensions are mounted on the HTTP frontend only, so pairing
``--frontend-route-extension`` with a non-HTTP input mode is a
misconfiguration. ``FrontendConfig.validate()`` runs before the distributed
runtime and the engine exist, so rejecting the pairing there is what makes the
frontend fail at startup rather than after it has taken runtime resources.
"""

from __future__ import annotations

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.unit,
    pytest.mark.gpu_0,
]

# Defaults are read from the environment when the parser is built, so anything
# set in the ambient environment would otherwise decide the outcome here.
_ENV_VARS_READ_BY_THESE_TESTS = (
    "DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD",
    "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD",
    "DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC",
    "DYN_FRONTEND_ROUTE_EXTENSIONS",
    "DYN_INTERACTIVE",
    "DYN_KSERVE_GRPC_SERVER",
)


def _config_from(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> FrontendConfig:
    for name in _ENV_VARS_READ_BY_THESE_TESTS:
        monkeypatch.delenv(name, raising=False)
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    return FrontendConfig.from_cli_args(parser.parse_args(argv))


def test_extension_http_only_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config_from(
        monkeypatch,
        ["-i", "--frontend-route-extension", "some.module:provider"],
    )

    with pytest.raises(ValueError, match="--frontend-route-extension"):
        config.validate()


def test_extension_http_only_guard_rejects_grpc_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_from(
        monkeypatch,
        [
            "--kserve-grpc-server",
            "--frontend-route-extension",
            "some.module:provider",
        ],
    )

    with pytest.raises(ValueError, match="--frontend-route-extension"):
        config.validate()


def test_extension_http_only_guard_allows_default_http_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_from(
        monkeypatch,
        ["--frontend-route-extension", "some.module:provider"],
    )

    config.validate()


def test_extension_http_only_guard_allows_interactive_without_extension(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config_from(monkeypatch, ["-i"])

    config.validate()
