# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
from argparse import Namespace

from sglang.srt.server_args import ServerArgs

from dynamo.sdk.cli.utils import reserve_free_port
from dynamo.sdk.lib.config import ServiceConfig


def parse_sglang_args(service_name, prefix) -> ServerArgs:
    config = ServiceConfig.get_instance()
    sglang_args = config.as_args(service_name, prefix=prefix)
    parser = argparse.ArgumentParser()
    bootstrap_port = _reserve_disaggregation_bootstrap_port()

    # add future dynamo arguments here

    ServerArgs.add_cli_args(parser)
    args = parser.parse_args(sglang_args)
    args_dict = vars(args)
    args_dict["disaggregation_bootstrap_port"] = bootstrap_port
    args = Namespace(**args_dict)
    return ServerArgs.from_cli_args(args)


def _reserve_disaggregation_bootstrap_port():
    """
    Each worker requires a unique port for disaggregation_bootstrap_port.
    We use an existing utility function that reserves a free port on your
    machine to avoid collisions.
    """
    with reserve_free_port() as port:
        return port
