# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse

import pytest

from dynamo.common.configuration.groups.runtime_args import DynamoRuntimeArgGroup
from dynamo.frontend.frontend_args import FrontendArgGroup

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


@pytest.mark.parametrize("arg_group", [DynamoRuntimeArgGroup, FrontendArgGroup])
def test_runtime_clis_reject_valkey_as_generic_event_plane(arg_group) -> None:
    parser = argparse.ArgumentParser()
    arg_group().add_arguments(parser)

    with pytest.raises(SystemExit):
        parser.parse_args(["--event-plane", "valkey"])


@pytest.mark.parametrize("arg_group", [DynamoRuntimeArgGroup, FrontendArgGroup])
def test_runtime_clis_reject_valkey_event_plane_from_env(
    monkeypatch, arg_group
) -> None:
    monkeypatch.setenv("DYN_EVENT_PLANE", "valkey")
    parser = argparse.ArgumentParser()
    with pytest.raises(argparse.ArgumentTypeError, match="nats.*zmq"):
        arg_group().add_arguments(parser)
