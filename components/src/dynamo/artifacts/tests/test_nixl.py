# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lease, budget and receipt behavior of NIXL artifact delivery, with a fake NIXL."""

import asyncio
import hashlib
import json
import os
import subprocess
import sys
import textwrap

import pytest

torch = pytest.importorskip("torch")

from dynamo.artifacts.nixl import (  # noqa: E402 -- optional dependency guard
    NixlArtifactCapacityError,
    NixlArtifactServer,
    read_nixl_artifact,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


class _FakeMetadata:
    def __init__(self, token):
        self.token = token

    def model_dump(self, mode=None):
        return {"token": self.token}


class _FakeReadable:
    def __init__(self, fake, descriptor):
        self.fake = fake
        self.token = f"op-{len(fake.readables)}"
        self.tensor = descriptor.tensor
        self.read = asyncio.Event()
        self.released = False

    def metadata(self):
        return _FakeMetadata(self.token)

    async def wait_for_completion(self):
        await self.read.wait()

    def __exit__(self, *_):
        self.released = True


class _FakeRead:
    def __init__(self, source, target):
        self.source = source
        self.target = target

    async def wait_for_completion(self):
        self.target.copy_(self.source.tensor)
        self.source.read.set()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _FakeNixl:
    """Stands in for dynamo.nixl_connect; reads copy between registered tensors."""

    def __init__(self):
        self.readables = {}
        fake = self

        class Descriptor:
            def __init__(self, tensor):
                self.tensor = tensor

        class Connector:
            async def create_readable(self, descriptor):
                readable = _FakeReadable(fake, descriptor)
                fake.readables[readable.token] = readable
                return readable

            async def begin_read(self, metadata, descriptor):
                return _FakeRead(fake.readables[metadata["token"]], descriptor.tensor)

        class RdmaMetadata:
            @staticmethod
            def model_validate(value):
                return value

        self.Descriptor = Descriptor
        self.Connector = Connector
        self.RdmaMetadata = RdmaMetadata


DEST = {"kind": "nixl_read", "lease_seconds": 30}


def _server(fake, **kwargs):
    kwargs.setdefault("max_held_bytes", 1 << 20)
    return NixlArtifactServer(nixl_connect=fake, **kwargs)


@pytest.mark.asyncio
async def test_read_returns_exact_bytes_and_releases_the_buffer():
    fake = _FakeNixl()
    server = _server(fake)
    data = bytes(range(256)) * 4

    receipt = await server.submit(data, ["hidden_states"], DEST)

    assert receipt["mode"] == "nixl_read"
    assert receipt["state"] == "ready"
    assert receipt["byte_count"] == len(data)
    assert receipt["sha256"] == hashlib.sha256(data).hexdigest()
    assert receipt["expires_at"].endswith("Z")
    assert server.held_bytes == len(data)
    json.dumps(receipt)

    tensor = await read_nixl_artifact(receipt, nixl_connect=fake)
    assert tensor.numpy().tobytes() == data
    for _ in range(10):
        await asyncio.sleep(0)
    assert server.held_bytes == 0
    assert fake.readables["op-0"].released


@pytest.mark.asyncio
async def test_unread_artifact_is_released_when_its_lease_expires():
    fake = _FakeNixl()
    server = _server(fake)
    await server.submit(b"x" * 64, ["hidden_states"], {**DEST, "lease_seconds": 1})
    assert server.held_bytes == 64
    await asyncio.sleep(1.2)
    assert server.held_bytes == 0
    assert fake.readables["op-0"].released


@pytest.mark.asyncio
async def test_budget_bounds_held_bytes_and_recovers_after_release():
    fake = _FakeNixl()
    server = _server(fake, max_held_bytes=100)
    await server.submit(b"a" * 60, ["hidden_states"], DEST)
    with pytest.raises(NixlArtifactCapacityError):
        await server.submit(b"b" * 60, ["hidden_states"], DEST)
    assert server.held_bytes == 60
    fake.readables["op-0"].read.set()
    for _ in range(10):
        await asyncio.sleep(0)
    await server.submit(b"b" * 60, ["hidden_states"], DEST)
    assert server.held_bytes == 60


@pytest.mark.asyncio
async def test_validate_rejects_bad_destinations_and_a_full_budget():
    fake = _FakeNixl()
    server = _server(fake, max_held_bytes=10, max_lease_seconds=60)
    for destination in (
        {"kind": "nixl_read", "lease_seconds": 0},
        {"kind": "nixl_read", "lease_seconds": 61},
        {"kind": "nixl_read", "lease_seconds": True},
        {"kind": "nixl_read"},
        {"kind": "nixl_read", "lease_seconds": 5, "host": "x"},
        {"kind": "presigned_http_put", "lease_seconds": 5},
    ):
        with pytest.raises(ValueError):
            server.validate(destination, "nixl_read")
    with pytest.raises(ValueError):
        server.validate(DEST, "object_store")
    await server.submit(b"z" * 10, ["hidden_states"], {**DEST, "lease_seconds": 5})
    with pytest.raises(NixlArtifactCapacityError):
        server.validate({**DEST, "lease_seconds": 5}, "nixl_read")


@pytest.mark.asyncio
async def test_close_releases_unread_artifacts_and_refuses_new_ones():
    fake = _FakeNixl()
    server = _server(fake)
    await server.submit(b"q" * 32, ["hidden_states"], DEST)
    await server.close()
    assert server.held_bytes == 0
    assert fake.readables["op-0"].released
    with pytest.raises(NixlArtifactCapacityError):
        await server.submit(b"q", ["hidden_states"], DEST)


@pytest.mark.asyncio
async def test_failed_registration_returns_reserved_bytes():
    fake = _FakeNixl()

    async def broken(descriptor):
        raise RuntimeError("register failed")

    server = _server(fake)
    server._connector = fake.Connector()
    server._connector.create_readable = broken
    with pytest.raises(RuntimeError, match="register failed"):
        await server.submit(b"q" * 32, ["hidden_states"], DEST)
    assert server.held_bytes == 0


@pytest.mark.asyncio
async def test_reader_rejects_other_receipts_and_digest_mismatch():
    fake = _FakeNixl()
    server = _server(fake)
    receipt = await server.submit(b"abc", ["hidden_states"], DEST)
    for bad in (
        {**receipt, "mode": "object_store"},
        {**receipt, "state": "failed"},
        {**receipt, "byte_count": 0},
        {key: value for key, value in receipt.items() if key != "rdma_metadata"},
    ):
        with pytest.raises(ValueError):
            await read_nixl_artifact(bad, nixl_connect=fake)
    with pytest.raises(ValueError, match="digest"):
        await read_nixl_artifact({**receipt, "sha256": "0" * 64}, nixl_connect=fake)


def test_capture_session_accepts_nixl_read_with_a_server():
    pytest.importorskip("msgspec")
    pytest.importorskip("zstandard")
    from dynamo.artifacts.vllm.capture import CaptureBudget, VllmArtifactSession

    request = {
        "extra_args": {
            "nvext": {
                "metadata_upload": {
                    "kind": "generation_artifact",
                    "contents": ["selected_logprobs"],
                    "mode": "nixl_read",
                    "destination": DEST,
                }
            }
        }
    }
    session = VllmArtifactSession.from_request(
        request,
        prompt_token_count=2,
        max_tokens=2,
        budget=CaptureBudget(1 << 20),
        runtime=_server(_FakeNixl()),
    )
    assert session.settings["mode"] == "nixl_read"
    session.close()
    with pytest.raises(ValueError, match="authority"):
        VllmArtifactSession.from_request(
            request,
            prompt_token_count=2,
            max_tokens=2,
            budget=CaptureBudget(1 << 20),
        )


@pytest.mark.asyncio
async def test_finalized_nixl_artifact_is_uncompressed_and_decodes():
    pytest.importorskip("msgspec")
    pytest.importorskip("zstandard")
    from types import SimpleNamespace

    from dynamo.artifacts.vllm.capture import CaptureBudget, VllmArtifactSession
    from dynamo.artifacts.vllm.format import CODEC_NONE, decode_generation_artifact

    fake = _FakeNixl()
    request = {
        "extra_args": {
            "nvext": {
                "metadata_upload": {
                    "kind": "generation_artifact",
                    "contents": ["selected_logprobs"],
                    "mode": "nixl_read",
                    "destination": DEST,
                }
            }
        }
    }
    session = VllmArtifactSession.from_request(
        request,
        prompt_token_count=2,
        max_tokens=2,
        budget=CaptureBudget(1 << 20),
        runtime=_server(fake),
    )
    session.observe(
        SimpleNamespace(
            index=0, token_ids=[12], logprobs=[{12: SimpleNamespace(logprob=-0.5)}]
        ),
        [10, 11],
    )
    receipt = await session.finalize()
    assert receipt["mode"] == "nixl_read"
    tensor = await read_nixl_artifact(receipt, nixl_connect=fake)
    data = tensor.numpy().tobytes()
    # Prelude "<8sHHHIQ": magic, major, minor, then the codec at bytes 12-13.
    assert int.from_bytes(data[12:14], "little") == CODEC_NONE
    choice = decode_generation_artifact(data).choices[0]
    assert choice.sequence_token_ids.as_array().tolist() == [10, 11, 12]


_REAL_TRANSFER = textwrap.dedent(
    """
    import asyncio, json, sys
    from dynamo.artifacts.nixl import NixlArtifactServer, read_nixl_artifact

    async def serve(path):
        server = NixlArtifactServer(max_held_bytes=1 << 24)
        data = bytes(range(256)) * 4096
        receipt = await server.submit(
            data, ["hidden_states"], {"kind": "nixl_read", "lease_seconds": 120}
        )
        with open(path + ".tmp", "w") as f:
            json.dump(receipt, f)
        import os; os.replace(path + ".tmp", path)
        while server.held_bytes:
            await asyncio.sleep(0.05)

    async def read(path):
        import os
        while not os.path.exists(path):
            await asyncio.sleep(0.05)
        receipt = json.load(open(path))
        tensor = await read_nixl_artifact(receipt, timeout_seconds=120)
        assert tensor.numpy().tobytes() == bytes(range(256)) * 4096
        print("READ_OK")

    role, path = sys.argv[1], sys.argv[2]
    asyncio.run(serve(path) if role == "serve" else read(path))
    """
)


@pytest.mark.gpu_1
@pytest.mark.timeout(300)
def test_real_nixl_read_between_two_processes(tmp_path):
    pytest.importorskip("nixl")
    if not torch.cuda.is_available():
        pytest.skip("pinned host staging is exercised on CUDA hosts")
    script = tmp_path / "transfer.py"
    script.write_text(_REAL_TRANSFER)
    receipt = str(tmp_path / "receipt.json")
    env = dict(os.environ)
    server = subprocess.Popen([sys.executable, str(script), "serve", receipt], env=env)
    try:
        reader = subprocess.run(
            [sys.executable, str(script), "read", receipt],
            env=env,
            capture_output=True,
            text=True,
            timeout=280,
        )
        assert "READ_OK" in reader.stdout, reader.stderr[-2000:]
        assert server.wait(timeout=60) == 0
    finally:
        if server.poll() is None:
            server.kill()
