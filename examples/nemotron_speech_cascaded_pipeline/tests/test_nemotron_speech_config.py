# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
from nemotron_speech.config import (
    DEFAULT_NIM_STARTUP_TIMEOUT_S,
    NimConnectionConfig,
    add_nim_connection_args,
    nim_connection_config_from_namespace,
    resolve_dynamo_endpoint,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def test_connection_defaults():
    parser = argparse.ArgumentParser()
    add_nim_connection_args(parser)
    args = parser.parse_args([])

    assert nim_connection_config_from_namespace(args) == NimConnectionConfig()
    assert args.nim_startup_timeout_s == DEFAULT_NIM_STARTUP_TIMEOUT_S


def test_connection_overrides():
    parser = argparse.ArgumentParser()
    add_nim_connection_args(parser)
    args = parser.parse_args(
        [
            "--nim-server=speech.example:443",
            "--nim-use-ssl",
            "--nim-api-key=test-key",
            "--nim-function-id=test-function",
            "--nim-ssl-root-cert=ca.pem",
            "--nim-startup-timeout-s=12.5",
        ]
    )

    assert nim_connection_config_from_namespace(args) == NimConnectionConfig(
        server="speech.example:443",
        use_ssl=True,
        api_key="test-key",
        function_id="test-function",
        ssl_root_cert="ca.pem",
    )
    assert args.nim_startup_timeout_s == 12.5


@pytest.mark.parametrize(
    "namespace,suffix,expected",
    [
        (None, None, "dynamo"),
        ("speech", None, "speech"),
        ("speech", "blue", "speech-blue"),
    ],
)
def test_default_endpoint(monkeypatch, namespace, suffix, expected):
    for name, value in (
        ("DYN_NAMESPACE", namespace),
        ("DYN_NAMESPACE_WORKER_SUFFIX", suffix),
    ):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    assert resolve_dynamo_endpoint(None, "asr") == f"{expected}.asr.generate"


def test_explicit_endpoint(monkeypatch):
    monkeypatch.setenv("DYN_NAMESPACE", "speech")
    monkeypatch.setenv("DYN_NAMESPACE_WORKER_SUFFIX", "blue")
    endpoint = "custom.tts.generate"

    assert resolve_dynamo_endpoint(endpoint, "tts") == endpoint
