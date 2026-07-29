# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [
    pytest.mark.unit,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def _frontend_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    return parser


def test_basetenkenizer_cli_backend_is_accepted(monkeypatch):
    monkeypatch.delenv("DYN_TOKENIZER", raising=False)

    args = _frontend_parser().parse_args(["--tokenizer", "basetenkenizer"])

    assert args.tokenizer_backend == "basetenkenizer"
    assert "basetenkenizer" in FrontendConfig._VALID_TOKENIZER_BACKENDS


def test_basetenkenizer_env_backend_is_accepted(monkeypatch):
    monkeypatch.setenv("DYN_TOKENIZER", "basetenkenizer")

    args = _frontend_parser().parse_args([])

    assert args.tokenizer_backend == "basetenkenizer"


def test_unknown_tokenizer_backend_is_rejected(monkeypatch):
    monkeypatch.delenv("DYN_TOKENIZER", raising=False)

    with pytest.raises(SystemExit):
        _frontend_parser().parse_args(["--tokenizer", "baseten"])
