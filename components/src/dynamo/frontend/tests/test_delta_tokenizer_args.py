# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import argparse

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def parse(args):
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(args))
    config.validate()
    return config


def test_disabled_by_default():
    assert parse([]).experimental_delta_tokenizer is False


def test_cli_overrides_environment(monkeypatch):
    monkeypatch.setenv("DYN_EXPERIMENTAL_DELTA_TOKENIZER", "true")
    assert (
        parse(["--no-experimental-delta-tokenizer"]).experimental_delta_tokenizer
        is False
    )


def test_enabled_single_process(monkeypatch):
    monkeypatch.setenv("DYN_DELTA_TOKENIZER_MAX_ENTRIES", "4")
    config = parse(
        [
            "--dyn-chat-processor",
            "sglang",
            "--dyn-preprocess-workers",
            "0",
            "--experimental-delta-tokenizer",
        ]
    )
    assert config.delta_tokenizer_max_entries == 4


@pytest.mark.parametrize(
    "args",
    [
        ["--delta-tokenizer-max-entries", "0"],
        ["--delta-tokenizer-max-bytes", "-1"],
        ["--delta-tokenizer-ttl-secs", "nan"],
        ["--delta-tokenizer-verify-rate", "1.1"],
        ["--delta-tokenizer-verify-rate", "nan"],
    ],
)
def test_invalid_limits(args):
    with pytest.raises(ValueError):
        parse(args)


def test_other_processor_is_rejected():
    with pytest.raises(ValueError, match="SGLang"):
        parse(["--experimental-delta-tokenizer"])


def test_worker_pool_is_rejected():
    with pytest.raises(ValueError, match="workers"):
        parse(
            [
                "--experimental-delta-tokenizer",
                "--dyn-chat-processor",
                "sglang",
                "--dyn-preprocess-workers",
                "2",
            ]
        )
