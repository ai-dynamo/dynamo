# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Multipart -> production Rust frontend -> TCP -> Python boundary, without a GPU."""

import base64
import hashlib
import io
import json
import os
import socket
import time

import pytest
import requests
from PIL import Image

from tests.frontend.image_edit_worker import MODEL_NAME
from tests.utils.managed_process import DynamoFrontendProcess, ManagedProcess

pytestmark = [
    pytest.mark.pre_merge,
    pytest.mark.integration,
    pytest.mark.gpu_0,
    pytest.mark.timeout(120),
]


class ImageWorkerProcess(ManagedProcess):
    def __init__(self, request, port, *, limit=None, legacy=False, cancel_file=None):
        """Launch one isolated worker with explicit capability and reader versions."""
        env = os.environ.copy()
        env.pop("IMAGE_EDIT_CAPABILITY", None)
        if limit is not None:
            env["IMAGE_EDIT_CAPABILITY"] = str(limit)
        env["IMAGE_EDIT_LEGACY_READER"] = "1" if legacy else "0"
        env.pop("IMAGE_EDIT_CANCEL_FILE", None)
        if cancel_file is not None:
            env["IMAGE_EDIT_CANCEL_FILE"] = str(cancel_file)
        super().__init__(
            command=["python3", "-m", "tests.frontend.image_edit_worker"],
            health_check_urls=[
                (
                    f"http://localhost:{port}/v1/models",
                    lambda response: response.ok
                    and any(
                        m["id"] == MODEL_NAME for m in response.json().get("data", [])
                    ),
                )
            ],
            timeout=60,
            env=env,
            display_output=True,
            terminate_all_matching_process_names=False,
            log_dir=f"{request.node.name}_worker",
        )


@pytest.fixture
def frontend(request, file_storage_backend, dynamo_dynamic_ports):
    """Run the production HTTP frontend over TCP with file-backed discovery."""
    _ = file_storage_backend
    port = dynamo_dynamic_ports.frontend_port
    with DynamoFrontendProcess(
        request,
        frontend_port=port,
        extra_args=[
            "--discovery-backend",
            "file",
            "--request-plane",
            "tcp",
            "--event-plane",
            "zmq",
        ],
        terminate_all_matching_process_names=False,
    ):
        yield port


def references():
    """Use distinct PNG bytes so hash comparisons detect input reordering."""
    result = []
    for color in ["red", "blue"]:
        with io.BytesIO() as buffer:
            Image.new("RGB", (2, 2), color).save(buffer, format="PNG")
            result.append(buffer.getvalue())
    return result


def upload(port, images, alias="image"):
    """Send real multipart parts in list order through the public edits route."""
    return requests.post(
        f"http://localhost:{port}/v1/images/edits",
        data={"model": MODEL_NAME, "prompt": "edit"},
        files=[(alias, ("input.png", data, "image/png")) for data in images],
        timeout=20,
    )


def test_image_edits_order_json_parity_and_legacy_wire(frontend, request):
    """Compare worker-observed order across multipart aliases and JSON endpoints."""
    with ImageWorkerProcess(request, frontend, limit=4):
        images = references()
        uris = [
            "data:image/png;base64," + base64.b64encode(data).decode()
            for data in images
        ]
        expected = [hashlib.sha256(uri.encode()).hexdigest() for uri in uris]
        for alias in ["image", "image[]"]:
            response = upload(frontend, images, alias)
            assert response.status_code == 200, response.text
            report = json.loads(response.json()["data"][0]["revised_prompt"])
            assert report == {"wire_type": "list", "digests": expected}
        for endpoint in ["edits", "generations"]:
            response = requests.post(
                f"http://localhost:{frontend}/v1/images/{endpoint}",
                json={"model": MODEL_NAME, "prompt": "edit", "input_reference": uris},
                timeout=20,
            )
            assert response.status_code == 200, response.text
            assert (
                json.loads(response.json()["data"][0]["revised_prompt"])["digests"]
                == expected
            )
        response = upload(frontend, images[:1])
        assert response.status_code == 200, response.text
        assert (
            json.loads(response.json()["data"][0]["revised_prompt"])["wire_type"]
            == "str"
        )


def test_image_edits_disconnect_cancels_worker(frontend, request, tmp_path):
    """An HTTP disconnect must cancel the in-flight worker request over TCP."""
    cancelled = tmp_path / "cancelled"
    started = tmp_path / "cancelled.started"

    def wait_for_marker(path, expected):
        """Wait for the worker's explicit state signal, not a log or fixed sleep."""
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if path.exists() and path.read_text() == expected:
                return
            time.sleep(0.05)
        pytest.fail(f"worker did not report {expected}")

    with ImageWorkerProcess(request, frontend, limit=4, cancel_file=cancelled):
        prepared = requests.Request(
            "POST",
            f"http://localhost:{frontend}/v1/images/edits",
            data={"model": MODEL_NAME, "prompt": "wait-for-cancel"},
            files=[("image", ("input.png", references()[0], "image/png"))],
        ).prepare()
        headers = (
            f"POST {prepared.path_url} HTTP/1.1\r\n"
            f"Host: localhost:{frontend}\r\n"
            + "".join(f"{key}: {value}\r\n" for key, value in prepared.headers.items())
            + "\r\n"
        ).encode("ascii")
        with socket.create_connection(("localhost", frontend), timeout=20) as client:
            client.sendall(headers + prepared.body)
            wait_for_marker(started, "started")
            assert not cancelled.exists(), "request was cancelled before disconnect"

        # Assert before worker teardown, which can also cancel pending requests.
        wait_for_marker(cancelled, "cancelled")
        response = upload(frontend, references()[:1])
        assert response.status_code == 200, response.text


@pytest.mark.parametrize("limit,legacy", [(None, True), (0, False), (4, True)])
def test_image_edits_unknown_unsupported_and_stale_worker(
    frontend, request, limit, legacy
):
    """Exercise unknown metadata, explicit zero, and an inaccurately advertised reader."""
    with ImageWorkerProcess(request, frontend, limit=limit, legacy=legacy):
        response = upload(frontend, references())
        assert response.status_code == 400, response.text
        assert "data" not in response.json()
        response = upload(frontend, references()[:1])
        assert response.status_code == (400 if limit == 0 else 200), response.text


def test_image_edits_mixed_worker_join_and_removal(frontend, request):
    """A legacy member lowers the set-wide limit until discovery removes it."""

    def wait_for_status(expected):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            response = upload(frontend, references())
            if response.status_code == expected:
                return
            time.sleep(0.1)
        pytest.fail(f"image edit did not reach HTTP {expected}: {response.text}")

    with ImageWorkerProcess(request, frontend, limit=4):
        wait_for_status(200)
        with ImageWorkerProcess(request, frontend, legacy=True):
            wait_for_status(400)
            for _ in range(10):
                assert upload(frontend, references()).status_code == 400
            assert upload(frontend, references()[:1]).status_code == 200
        wait_for_status(200)
