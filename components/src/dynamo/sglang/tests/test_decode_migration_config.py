# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.sglang.backend_args import DynamoSGLangArgGroup, DynamoSGLangConfig

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.pre_merge,
]


def test_worker_taints_are_repeatable():
    parser = argparse.ArgumentParser()
    DynamoSGLangArgGroup().add_arguments(parser)
    args = parser.parse_args(
        [
            "--worker-taint",
            "decode/fast",
            "--worker-taint",
            "rack/a",
        ]
    )
    config = DynamoSGLangConfig.from_cli_args(args)
    assert config.worker_taint == ["decode/fast", "rack/a"]
