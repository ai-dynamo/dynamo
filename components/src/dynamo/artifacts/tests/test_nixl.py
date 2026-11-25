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
    NixlArtifactReader,
    NixlArtifactServer,
    read_nixl_artifact,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


class _FakeMetadata:
    def __init__(self, token, nixl_metadata="worker-a"):
        self.token = token
        self.nixl_metadata = nixl_metadata

    def model_dump(self, mode=None):
        return {"token": self.token, "nixl_metadata": self.nixl_metadata}


def _tensors(descriptors):
    if isinstance(descriptors, list):
        return [descriptor.tensor for descriptor in descriptors]
    return [descriptors.tensor]


class _FakeReadable:
    def __init__(self, fake, descriptor):
        self.fake = fake
        self.token = f"op-{len(fake.readables)}"
        self.tensors = _tensors(descriptor)
        self.list_form = isinstance(descriptor, list)
        self.read = asyncio.Event()
        self.released = False

    def metadata(self):
        return _FakeMetadata(self.token)

    async def wait_for_completion(self):
        await self.read.wait()

    def __exit__(self, *_):
        self.released = True


class _FakeRead:
    def __init__(self, source, targets):
        assert len(targets) == len(source.tensors)
        self.source = source
        self.targets = targets

    async def wait_for_completion(self):
        for target, tensor in zip(self.targets, self.source.tensors):
            target.copy_(tensor)
        self.source.read.set()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass


class _FakeNixl:
    """Stands in for dynamo.nixl_connect; reads copy between registered tensors."""

    def __init__(self):
        self.readables = {}
        self.connections = 0
        self.remotes_added = []
        self.remotes_removed = []
        fake = self

        class Descriptor:
            def __init__(self, tensor):
                self.tensor = tensor

        class Connector:
            def __init__(self):
                self.connection = None

            async def _create_connection(self):
                if self.connection is None:
                    fake.connections += 1
                    self.connection = object()
                return self.connection

            async def create_readable(self, descriptor):
                readable = _FakeReadable(fake, descriptor)
                fake.readables[readable.token] = readable
                return readable

            async def begin_read(self, metadata, descriptor):
                return _FakeRead(fake.readables[metadata.token], _tensors(descriptor))

        class RdmaMetadata:
            @staticmethod
            def model_validate(value):
                return _FakeMetadata(value["token"], value.get("nixl_metadata"))

        class Remote:
            def __init__(self, connection, nixl_metadata):
                assert connection is not None
                self.name = nixl_metadata
                fake.remotes_added.append(nixl_metadata)

            def __exit__(self, *_):
                fake.remotes_removed.append(self.name)

        self.Descriptor = Descriptor
        self.Connector = Connector
        self.RdmaMetadata = RdmaMetadata
        self.Remote = Remote


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

    tensor = await read_nixl_artifact(
        receipt, reader=NixlArtifactReader(nixl_connect=fake)
    )
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
            await read_nixl_artifact(bad, reader=NixlArtifactReader(nixl_connect=fake))
    with pytest.raises(ValueError, match="digest"):
        await read_nixl_artifact(
            {**receipt, "sha256": "0" * 64},
            reader=NixlArtifactReader(nixl_connect=fake),
        )


@pytest.mark.asyncio
async def test_reader_keeps_each_worker_connected_across_reads():
    fake = _FakeNixl()
    server = _server(fake)
    reader = NixlArtifactReader(nixl_connect=fake)
    for _ in range(3):
        receipt = await server.submit(b"r" * 16, ["hidden_states"], DEST)
        await reader.read(receipt)
    assert fake.remotes_added == ["worker-a"]
    assert fake.remotes_removed == []
    assert reader.connected_remotes == 1
    reader.close()
    assert fake.remotes_removed == ["worker-a"]
    assert reader.connected_remotes == 0


@pytest.mark.asyncio
async def test_reader_evicts_the_least_recently_used_worker():
    fake = _FakeNixl()
    reader = NixlArtifactReader(nixl_connect=fake, max_remotes=2)
    for agent in ("a", "b", "a", "c"):
        await reader._pin_remote(agent)
    assert fake.remotes_added == ["a", "b", "c"]
    assert fake.remotes_removed == ["b"]
    assert reader.connected_remotes == 2


@pytest.mark.asyncio
async def test_warm_starts_one_agent_and_reads_reuse_it():
    fake = _FakeNixl()
    server = _server(fake)
    await server.warm()
    await server.warm()
    assert fake.connections == 1
    reader = NixlArtifactReader(nixl_connect=fake)
    await reader.warm()
    receipt = await server.submit(b"w" * 16, ["hidden_states"], DEST)
    await reader.read(receipt)
    assert fake.connections == 2


def test_reader_rejects_a_nonpositive_remote_limit():
    with pytest.raises(ValueError):
        NixlArtifactReader(nixl_connect=_FakeNixl(), max_remotes=0)


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
    tensor = await read_nixl_artifact(
        receipt, reader=NixlArtifactReader(nixl_connect=fake)
    )
    data = tensor.numpy().tobytes()
    # Prelude "<8sHHHIQ": magic, major, minor, then the codec at bytes 12-13.
    assert int.from_bytes(data[12:14], "little") == CODEC_NONE
    choice = decode_generation_artifact(data).choices[0]
    assert choice.sequence_token_ids.as_array().tolist() == [10, 11, 12]


@pytest.mark.asyncio
async def test_chunk_parts_are_held_and_read_in_one_operation():
    fake = _FakeNixl()
    server = _server(fake)
    chunks = [b"a" * 100, b"b" * 50]
    parts = {hashlib.sha256(chunk).hexdigest(): chunk for chunk in chunks}
    receipt = await server.submit(b"object", ["hidden_states"], DEST, parts=parts)

    assert receipt["byte_count"] == len(b"object")
    assert receipt["parts"] == [
        {"sha256": digest, "byte_count": len(chunk)} for digest, chunk in parts.items()
    ]
    assert server.held_bytes == 156
    assert fake.readables["op-0"].list_form
    json.dumps(receipt)

    reader = NixlArtifactReader(nixl_connect=fake)
    with pytest.raises(ValueError, match="read_parts"):
        await reader.read(receipt)
    artifact, read_parts = await reader.read_parts(receipt)
    assert artifact.numpy().tobytes() == b"object"
    assert {k: v.numpy().tobytes() for k, v in read_parts.items()} == parts
    for _ in range(10):
        await asyncio.sleep(0)
    assert server.held_bytes == 0


@pytest.mark.asyncio
async def test_chunk_parts_count_toward_the_budget():
    fake = _FakeNixl()
    server = _server(fake, max_held_bytes=100)
    parts = {hashlib.sha256(b"x" * 99).hexdigest(): b"x" * 99}
    with pytest.raises(NixlArtifactCapacityError):
        await server.submit(b"object", [], DEST, parts=parts)
    assert server.held_bytes == 0
    assert server.max_artifact_bytes == 100


@pytest.mark.asyncio
async def test_reader_rejects_a_part_that_differs_from_its_digest():
    fake = _FakeNixl()
    server = _server(fake)
    receipt = await server.submit(b"object", [], DEST, parts={"0" * 64: b"rows"})
    with pytest.raises(ValueError, match="digest"):
        await NixlArtifactReader(nixl_connect=fake).read_parts(receipt)


_REAL_TRANSFER = textwrap.dedent(
    """
    import asyncio, hashlib, json, sys
    from dynamo.artifacts.nixl import NixlArtifactReader, NixlArtifactServer

    DATA = bytes(range(256)) * 4096
    CHUNKS = [bytes([i]) * (3 << 20) for i in (1, 2)]

    async def serve(path):
        server = NixlArtifactServer(max_held_bytes=1 << 24)
        destination = {"kind": "nixl_read", "lease_seconds": 120}
        plain = await server.submit(DATA, ["hidden_states"], destination)
        chunked = await server.submit(
            DATA[:4096],
            ["hidden_states"],
            destination,
            parts={hashlib.sha256(c).hexdigest(): c for c in CHUNKS},
        )
        with open(path + ".tmp", "w") as f:
            json.dump([plain, chunked], f)
        import os; os.replace(path + ".tmp", path)
        while server.held_bytes:
            await asyncio.sleep(0.05)

    async def read(path):
        import os
        while not os.path.exists(path):
            await asyncio.sleep(0.05)
        plain, chunked = json.load(open(path))
        reader = NixlArtifactReader()
        tensor = await reader.read(plain, timeout_seconds=120)
        assert tensor.numpy().tobytes() == DATA
        artifact, parts = await reader.read_parts(chunked, timeout_seconds=120)
        assert artifact.numpy().tobytes() == DATA[:4096]
        assert [parts[hashlib.sha256(c).hexdigest()].numpy().tobytes() for c in CHUNKS] == CHUNKS
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
