# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
from pathlib import Path

import pytest

from dynamo.frontend.frontend_args import FrontendArgGroup, FrontendConfig

pytestmark = [pytest.mark.pre_merge, pytest.mark.unit, pytest.mark.gpu_0]


def parse_frontend_config(args: list[str]) -> FrontendConfig:
    parser = argparse.ArgumentParser()
    FrontendArgGroup().add_arguments(parser)
    config = FrontendConfig.from_cli_args(parser.parse_args(args))
    config.validate()
    return config


def test_http_mtls_cli_configuration() -> None:
    config = parse_frontend_config(
        [
            "--tls-cert-path",
            "server.crt",
            "--tls-key-path",
            "server.key",
            "--tls-client-ca-cert-path",
            "client-ca.crt",
        ]
    )

    assert config.tls_cert_path == Path("server.crt")
    assert config.tls_key_path == Path("server.key")
    assert config.tls_client_ca_cert_path == Path("client-ca.crt")


def test_http_mtls_environment_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DYN_TLS_CERT_PATH", "server.crt")
    monkeypatch.setenv("DYN_TLS_KEY_PATH", "server.key")
    monkeypatch.setenv("DYN_TLS_CLIENT_CA_CERT_PATH", "client-ca.crt")

    config = parse_frontend_config([])

    assert config.tls_cert_path == Path("server.crt")
    assert config.tls_key_path == Path("server.key")
    assert config.tls_client_ca_cert_path == Path("client-ca.crt")


def test_http_mtls_requires_server_identity() -> None:
    with pytest.raises(
        ValueError,
        match=("--tls-client-ca-cert-path requires --tls-cert-path and --tls-key-path"),
    ):
        parse_frontend_config(["--tls-client-ca-cert-path", "client-ca.crt"])
