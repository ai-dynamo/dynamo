# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise HTTP client CA forwarding through the Python frontend and Rust bindings."""

import http.client
import os
import re
import ssl
import subprocess
import sys
import time

import pytest

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.core,
    pytest.mark.parallel,
    pytest.mark.gpu_0,
    pytest.mark.timeout(60),
]


def make_certificates(directory):
    def openssl(*args):
        subprocess.run(
            ["openssl", *args],
            cwd=directory,
            check=True,
            capture_output=True,
            timeout=10,
        )

    openssl(
        "req",
        "-x509",
        "-newkey",
        "rsa:2048",
        "-nodes",
        "-days",
        "1",
        "-subj",
        "/CN=Test CA",
        "-keyout",
        "ca.key",
        "-out",
        "ca.pem",
        "-addext",
        "basicConstraints=critical,CA:TRUE",
    )
    for name, usage in (("server", "serverAuth"), ("client", "clientAuth")):
        openssl(
            "req",
            "-new",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-subj",
            f"/CN={name}",
            "-keyout",
            f"{name}.key",
            "-out",
            f"{name}.csr",
        )
        (directory / f"{name}.ext").write_text(
            f"extendedKeyUsage={usage}\nsubjectAltName=IP:127.0.0.1\n"
        )
        openssl(
            "x509",
            "-req",
            "-in",
            f"{name}.csr",
            "-CA",
            "ca.pem",
            "-CAkey",
            "ca.key",
            "-CAcreateserial",
            "-days",
            "1",
            "-extfile",
            f"{name}.ext",
            "-out",
            f"{name}.pem",
        )


def test_frontend_forwards_http_client_ca(tmp_path):
    make_certificates(tmp_path)
    env = os.environ.copy()
    for name in (
        "DYN_TLS_CERT_PATH",
        "DYN_TLS_KEY_PATH",
        "DYN_TLS_CLIENT_CA_CERT_PATH",
    ):
        env.pop(name, None)
    env["DYN_LOG"] = "info"
    env["DYN_LOGGING_CONSOLE_FORMAT"] = "readable"
    command = [
        sys.executable,
        "-m",
        "dynamo.frontend",
        "--discovery-backend",
        "mem",
        "--request-plane",
        "tcp",
        "--event-plane",
        "zmq",
        "--http-host",
        "127.0.0.1",
        "--http-port",
        "0",
        "--tls-cert-path",
        str(tmp_path / "server.pem"),
        "--tls-key-path",
        str(tmp_path / "server.key"),
        "--tls-client-ca-cert-path",
        str(tmp_path / "ca.pem"),
    ]
    log_path = tmp_path / "frontend.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                output = re.sub(r"\x1b\[[0-9;]*m", "", log_path.read_text())
                match = re.search(
                    r"HTTPS server listening[^\n]*address[=:]\s*127\.0\.0\.1:(\d+)",
                    output,
                )
                if match:
                    port = int(match.group(1))
                    break
                assert process.poll() is None, f"frontend exited:\n{output}"
                assert (
                    time.monotonic() < deadline
                ), f"frontend did not listen:\n{output}"
                time.sleep(0.05)

            trusted = ssl.create_default_context(cafile=str(tmp_path / "ca.pem"))
            trusted.load_cert_chain(tmp_path / "client.pem", tmp_path / "client.key")
            connection = http.client.HTTPSConnection(
                "127.0.0.1", port, context=trusted, timeout=5
            )
            try:
                connection.request("GET", "/live")
                response = connection.getresponse()
                assert response.status == 200
                response.read()
            finally:
                connection.close()

            no_identity = ssl.create_default_context(cafile=str(tmp_path / "ca.pem"))
            connection = http.client.HTTPSConnection(
                "127.0.0.1", port, context=no_identity, timeout=5
            )
            try:
                # A timeout or connection refusal must not count as certificate rejection.
                with pytest.raises(ssl.SSLError):
                    connection.request("GET", "/live")
                    connection.getresponse()
            finally:
                connection.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
