# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke of the SGLang encoder path with the real frontend and sidecars.

The Dynamo frontend and dynamo-sglang-sidecar run as real processes. SGLang is
replaced by fakes in this file: two HTTP encoders (/health, /encode) and one
aggregated language worker (sglang.api.v1 gRPC plus native HTTP /generate).
No GPU, model weights or SGLang installation is used. --model is a local
Qwen-VL tokenizer/config directory for the frontend.

The language worker's gRPC stubs are generated at run time from the vendored
protos in ../proto, so the driver needs grpcio and grpcio-tools.
"""

import argparse
import asyncio
import base64
import importlib
import json
import os
import socket
import struct
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import zlib
from contextlib import AsyncExitStack
from pathlib import Path

import grpc
import grpc_tools


def png(rgb):
    def chunk(name, data):
        return (
            struct.pack(">I", len(data))
            + name
            + data
            + struct.pack(">I", zlib.crc32(name + data))
        )

    header = struct.pack(">2I5B", 56, 56, 8, 2, 0, 0, 0)
    pixels = (b"\x00" + bytes(rgb) * 56) * 56
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(data).decode()


IMAGES = [png((255, 0, 0)), png((0, 0, 255))]
# The fakes check delivery and ordering, not video decoding.
VIDEOS = ["data:video/mp4;base64,AA==", "data:video/mp4;base64,AQ=="]
REASONS = {200: "OK", 400: "Bad Request", 500: "Internal Server Error"}


class HttpServer:
    """A minimal HTTP/1.1 server. `handle(method, path, body, reader)` returns
    (status, payload, content type). It may read `reader` to wait for the
    client to disconnect."""

    def __init__(self, handle):
        self.handle = handle
        self.writers = set()
        self.server = None
        self.port = None

    async def start(self):
        self.server = await asyncio.start_server(self.serve, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self):
        self.server.close()
        for writer in list(self.writers):
            writer.close()

    async def serve(self, reader, writer):
        self.writers.add(writer)
        try:
            while True:
                head = (await reader.readuntil(b"\r\n\r\n")).decode("latin-1")
                request_line, *lines = head.split("\r\n")
                method, path, _ = request_line.split(" ", 2)
                headers = {}
                for line in lines:
                    if line:
                        name, value = line.split(":", 1)
                        headers[name.strip().lower()] = value.strip()
                body = await reader.readexactly(int(headers.get("content-length", 0)))
                status, payload, content_type = await self.handle(
                    method, path, body, reader
                )
                if not isinstance(payload, bytes):
                    payload = json.dumps(payload).encode()
                writer.write(
                    f"HTTP/1.1 {status} {REASONS.get(status, 'Error')}\r\n"
                    f"content-type: {content_type}\r\n"
                    f"content-length: {len(payload)}\r\n\r\n".encode() + payload
                )
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            self.writers.discard(writer)
            writer.close()


class Encoder:
    """SGLang's HTTP encoder. `mode` is "ok", "hold", or an error status."""

    def __init__(self):
        self.http = HttpServer(self.handle)
        self.mode = "ok"
        self.encodes = []
        self.disconnects = []

    @property
    def url(self):
        return f"http://127.0.0.1:{self.http.port}"

    async def handle(self, method, path, body, reader):
        if method == "GET":
            return 200, b"", "text/plain"
        assert (method, path) == ("POST", "/encode"), (method, path)
        request = json.loads(body)
        self.encodes.append(request)
        if self.mode == "hold":
            # The real encoder answers only after delivery. EOF is the
            # sidecar dropping the request.
            await reader.read(1)
            self.disconnects.append(request["req_id"])
            raise ConnectionResetError
        if self.mode != "ok":
            error = {
                "status": "error",
                "message": "private encoder detail",
                "req_id": request["req_id"],
            }
            return self.mode, error, "application/json"
        await asyncio.sleep(0.05)
        return 200, {"status": "ok"}, "application/json"


def load_stubs(tmp):
    """Generate sglang.api.v1 stubs from the vendored protos. The generated
    `sglang` package shadows any installed SGLang, which this driver never
    imports."""
    proto = Path(__file__).resolve().parents[1] / "proto"
    out = tmp / "stubs"
    out.mkdir()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{proto}",
            f"-I{Path(grpc_tools.__file__).parent / '_proto'}",
            f"--python_out={out}",
            f"--grpc_python_out={out}",
            *sorted(map(str, (proto / "sglang/api").rglob("*.proto"))),
            str(proto / "sglang/json/v1/options.proto"),
        ],
        check=True,
    )
    for package in ("sglang", "sglang/api", "sglang/api/v1", "sglang/json"):
        (out / package / "__init__.py").touch()
    (out / "sglang/json/v1/__init__.py").touch()
    sys.path.insert(0, str(out))
    return (
        importlib.import_module("sglang.api.v1.generate_pb2"),
        importlib.import_module("sglang.api.v1.service_pb2"),
        importlib.import_module("sglang.api.v1.service_pb2_grpc"),
    )


def passed(message):
    print("PASS " + message, flush=True)


async def run(args, tmp, stack):
    generate_pb2, service_pb2, service_pb2_grpc = load_stubs(tmp)
    model = str(args.model.resolve())
    config = json.loads((args.model / "config.json").read_text())
    image_token = config["image_token_id"]
    video_token = config["video_token_id"]
    start = config.get("vision_start_token_id", 10)
    end = config.get("vision_end_token_id", 11)

    class Language(service_pb2_grpc.SglangServiceServicer):
        """An aggregated SGLang Rust server: gRPC and native HTTP /generate."""

        def __init__(self):
            self.http = HttpServer(self.handle)
            self.requests = []
            self.refuse = False

        async def GetModelInfo(self, req, ctx):
            return service_pb2.GetModelInfoResponse(
                model_path=model, tokenizer_path=model, is_generation=True
            )

        async def GetServerInfo(self, req, ctx):
            return service_pb2.GetServerInfoResponse(
                model_path=model,
                served_model_name="vision",
                incremental_streaming_output=True,
                disaggregation_mode="null",
                host="127.0.0.1",
                port=self.http.port,
                max_context_length=4096,
                max_total_num_tokens=4096,
                deployment_info=service_pb2.DeploymentInfo(
                    nnodes=1, dp_size=1, max_running_requests=4, page_size=16
                ),
            )

        async def ListModels(self, req, ctx):
            return service_pb2.ListModelsResponse(
                models=[
                    service_pb2.ModelCard(id="vision", root=model, max_model_len=4096)
                ]
            )

        async def HealthCheck(self, req, ctx):
            return service_pb2.HealthCheckResponse(healthy=True)

        def record(self, via, rid, ids, handoff, raw_media):
            self.requests.append(
                {
                    "via": via,
                    "rid": rid,
                    "ids": ids,
                    "handoff": handoff,
                    "raw_media": raw_media,
                    "refused": self.refuse,
                }
            )

        async def Generate(self, req, ctx):
            ids = list(req.input_ids.one.ids)
            self.record(
                "grpc",
                req.rid.one,
                ids,
                (
                    json.loads(req.encoder_handoff)
                    if req.HasField("encoder_handoff")
                    else None
                ),
                any(
                    req.HasField(k) for k in ("image_data", "video_data", "audio_data")
                ),
            )
            if self.refuse:
                await ctx.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "language admission refused"
                )
            item = generate_pb2.GenerateStreamItem()
            item.frame.meta_info.id = req.rid.one
            item.frame.meta_info.prompt_tokens = len(ids)
            item.frame.meta_info.completion_tokens = 1
            item.frame.meta_info.finish_reason.length.length = 1
            item.frame.output_ids.ids.append(16)
            yield item

        async def handle(self, method, path, body, reader):
            if method == "GET":
                return 200, b"", "text/plain"
            assert (method, path) == ("POST", "/generate"), (method, path)
            body = json.loads(body)
            assert body["stream"]
            ids = body["input_ids"]
            self.record(
                "http",
                body["rid"],
                ids,
                body.get("encoder_handoff"),
                bool({"image_data", "video_data", "audio_data"} & set(body)),
            )
            if self.refuse:
                error = {"error": {"message": "language admission refused"}}
                return 400, error, "application/json"
            item = {
                "output_ids": [16],
                "text": "1",
                "meta_info": {
                    "id": body["rid"],
                    "prompt_tokens": len(ids),
                    "completion_tokens": 1,
                    "finish_reason": {"type": "length"},
                },
            }
            stream = f"data: {json.dumps(item)}\n\ndata: [DONE]\n\n".encode()
            return 200, stream, "text/event-stream"

    env = {
        **os.environ,
        "DYN_DISCOVERY_BACKEND": "file",
        "DYN_FILE_KV": str(tmp / "kv"),
        "DYN_NAMESPACE": "enc-" + uuid.uuid4().hex,
        "DYN_SYSTEM_PORT": "0",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_EVENT_PLANE": "zmq",
        "DYN_TCP_RPC_PORT": "0",
        "DYN_TCP_RESPONSE_STREAM_PORT": "0",
        "HF_HUB_OFFLINE": "1",
        "DYN_SGLANG_ENABLE_GENERATE": "1",
    }
    # The frontend runs in its own interpreter/environment.
    for name in ("PYTHONPATH", "DYN_DISAGGREGATION_MODE", "DYN_ROUTE_TO_ENCODER"):
        env.pop(name, None)
    # Dynamo resolves the model card under HOME or USERPROFILE. Its cache
    # symlinks need absolute targets. Isolate all workers in one test profile.
    env.pop("HOME", None)
    env["USERPROFILE"] = str(tmp)
    args.logs.mkdir(exist_ok=True, parents=True)

    async def stop(child):
        if child.returncode is None:
            child.terminate()
            try:
                await asyncio.wait_for(child.wait(), 8)
            except asyncio.TimeoutError:
                child.kill()
                await child.wait()

    children = []

    async def spawn(label, *cmd):
        log = stack.enter_context((args.logs / f"{label}.log").open("w"))
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=tmp, env=env, stdout=log, stderr=log
        )
        stack.push_async_callback(stop, proc)
        children.append((label, proc))
        return proc

    sidecar = (
        [str(args.sidecar)]
        if args.sidecar
        else [str(args.frontend_python), "-m", "dynamo.sglang.sidecar"]
    )

    encoders = [Encoder(), Encoder()]
    for encoder in encoders:
        await encoder.http.start()
        stack.push_async_callback(encoder.http.stop)
    language = Language()
    await language.http.start()
    stack.push_async_callback(language.http.stop)
    server = grpc.aio.server()
    grpc_port = server.add_insecure_port("127.0.0.1:0")
    service_pb2_grpc.add_SglangServiceServicer_to_server(language, server)
    await server.start()
    stack.push_async_callback(server.stop, 0)

    await spawn(
        "language",
        *sidecar,
        "--grpc-endpoint",
        f"127.0.0.1:{grpc_port}",
        "--route-to-encoder",
    )
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        http_port = reservation.getsockname()[1]
    await spawn(
        "frontend",
        str(args.frontend_python),
        "-m",
        "dynamo.frontend",
        "--http-port",
        str(http_port),
    )
    url = f"http://127.0.0.1:{http_port}"

    async def exchange(path, payload):
        """(status, parsed JSON or raw bytes) of one POST."""

        def send():
            req = urllib.request.Request(
                url + path,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as response:
                    return response.status, json.load(response)
            except urllib.error.HTTPError as exc:
                return exc.code, exc.read()
            except OSError as exc:
                return 0, str(exc).encode()

        return await asyncio.to_thread(send)

    def payload(media):
        content = [{"type": "text", "text": "Describe in order."}]
        for source in media:
            key = source.split(":", 1)[1].split("/", 1)[0] + "_url"
            content.append({"type": key, key: {"url": source}})
            content.append({"type": "text", "text": "Next."})
        return {
            "model": "vision",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1,
        }

    async def chat(media):
        return await exchange("/v1/chat/completions", payload(media))

    def encode_count():
        return sum(len(e.encodes) for e in encoders)

    async def until(what, probe, seconds=60):
        """Poll `probe` until it returns a value other than None."""
        deadline = time.monotonic() + seconds
        last = None
        while time.monotonic() < deadline:
            for label, proc in children:
                if proc.returncode is not None and not label.startswith("allowed"):
                    raise RuntimeError(f"{label} exited; see {args.logs}")
            done, last = await probe()
            if done:
                return last
            await asyncio.sleep(0.2)
        raise TimeoutError(f"{what}: last saw {last!r}; see {args.logs}")

    # 7. The language worker requires an encoder. Without one its model is
    # unavailable, and no media request reaches it.
    async def unavailable():
        status, body = await chat(IMAGES)
        assert status != 200, "a media request was served without an encoder"
        return status == 503, (status, body)

    await until("503 without an encoder", unavailable)
    assert not language.requests and not encode_count()
    passed("no encode worker: a media request gets 503 before the language worker")

    for index, encoder in enumerate(encoders):
        await spawn(
            # Encoder 0 is taken down at the end, which ends its sidecar.
            ("allowed-" if index == 0 else "") + f"encode-{index}",
            *sidecar,
            "--disaggregation-mode",
            "encode",
            "--grpc-endpoint",
            f"127.0.0.1:{encoder.http.port}",
            "--encoder-model-path",
            model,
            "--encoder-served-model-name",
            "vision",
        )

    async def both_encoders_serving():
        status, body = await chat(IMAGES)
        ready = status == 200 and all(e.encodes for e in encoders)
        return ready, (status, body, [len(e.encodes) for e in encoders])

    await until("both encode workers serving", both_encoders_serving, 90)

    async def served(send, parts, ids=None, via="grpc"):
        """Send one request and check its encodes and its Generate.
        `parts` is [(modality, sources)] in part order. Returns its encoder."""
        marks = [len(e.encodes) for e in encoders]
        generated = len(language.requests)
        status, body = await send()
        assert status == 200, (status, body)
        (request,) = language.requests[generated:]
        assert request["via"] == via, request
        assert not request["raw_media"], "raw media reached the language worker"
        new = [e.encodes[mark:] for e, mark in zip(encoders, marks)]
        (index,) = [i for i, calls in enumerate(new) if calls]
        calls = sorted(new[index], key=lambda call: call["part_idx"])
        assert calls == [
            {
                "mm_items": sources,
                "num_parts": len(parts),
                "part_idx": part,
                "req_id": f"{request['rid']}_local_part_{part}",
                "modality": modality.upper(),
                "prefill_host": None,
                "embedding_port": None,
            }
            for part, (modality, sources) in enumerate(parts)
        ], calls
        assert request["handoff"] == {
            "encoder_url": encoders[index].url,
            "parts": [
                {"modality": modality, "items": len(sources)}
                for modality, sources in parts
            ],
        }, request
        placeholders = [t for t in request["ids"] if t in (image_token, video_token)]
        if ids is None:
            # One placeholder per item: the template does not expand them.
            assert len(placeholders) == sum(len(s) for _, s in parts), request
        else:
            assert request["ids"] == ids, request
        return index, request, placeholders

    # 1. Chat with images over both encoders.
    chosen = [
        (await served(lambda: chat(IMAGES), [("image", IMAGES)]))[0] for _ in range(6)
    ]
    assert set(chosen) == {0, 1}, chosen
    passed(
        f"chat images: encoders {chosen}, one /encode per request under the "
        "Generate rid, handoff names that encoder, no raw media"
    )

    # 3. Text bypasses the encoder.
    before, generated = encode_count(), len(language.requests)
    status, body = await chat([])
    assert status == 200, (status, body)
    assert encode_count() == before
    (request,) = language.requests[generated:]
    assert request["handoff"] is None and not request["raw_media"], request
    passed("text-only chat made no /encode call and carried no handoff")

    # 4. Mixed media: one part per modality, image before video.
    mixed = [VIDEOS[0], IMAGES[0], VIDEOS[1]]
    _, request, placeholders = await served(
        lambda: chat(mixed), [("image", IMAGES[:1]), ("video", VIDEOS)]
    )
    assert placeholders == [video_token, image_token, video_token], request
    passed("chat video+image+video: image part 0, video part 1, prompt order kept")

    # 2. Native /generate through the frontend.
    native_ids = [10, start, image_token, end, 11, start, image_token, end, 12]
    body = {
        "input_ids": native_ids,
        "image_data": IMAGES,
        "sampling_params": {"max_new_tokens": 2},
    }
    await served(
        lambda: exchange("/generate", body), [("image", IMAGES)], native_ids, "http"
    )
    mixed_ids = [start, video_token, end, start, image_token, end]
    mixed_body = {
        **body,
        "input_ids": mixed_ids,
        "image_data": IMAGES[0],
        "video_data": VIDEOS[:1],
    }
    await served(
        lambda: exchange("/generate", mixed_body),
        [("image", IMAGES[:1]), ("video", VIDEOS[:1])],
        mixed_ids,
        "http",
    )
    before, generated = encode_count(), len(language.requests)
    handoff = {"encoder_url": encoders[0].url, "parts": []}
    status, detail = await exchange("/generate", {**body, "encoder_handoff": handoff})
    assert status == 400 and b"encoder_handoff" in detail, (status, detail)
    assert (encode_count(), len(language.requests)) == (before, generated)
    passed("/generate: images and image+video encoded; client encoder_handoff is 400")

    # 5. An early encoder failure ends the request before the language worker.
    routes = [("/v1/chat/completions", payload(IMAGES)), ("/generate", body)]
    seen = {}
    for mode, expected in ((400, {400}), (503, {503}), (500, range(500, 600))):
        for encoder in encoders:
            encoder.mode = mode
        for path, request in routes:
            before, generated = encode_count(), len(language.requests)
            status, detail = await exchange(path, request)
            assert status in expected, (path, mode, status, detail)
            assert b"private encoder detail" not in detail, (path, detail)
            assert encode_count() > before, (path, mode)
            assert len(language.requests) == generated, (path, mode)
            seen[(mode, path)] = (status, encode_count() - before)
    passed(f"encoder /encode error -> (client status, /encode calls): {seen}")

    # 6. A request the language worker refuses releases its open /encode.
    for encoder in encoders:
        encoder.mode = "hold"
    language.refuse = True
    generated = len(language.requests)
    started = time.monotonic()
    status, detail = await chat(IMAGES)
    assert status >= 400, (status, detail)
    language.refuse = False
    refused = language.requests[generated:]
    assert refused, "the refused request never reached the language worker"
    part = refused[0]["rid"] + "_local_part_0"

    async def released():
        dropped = [d for e in encoders for d in e.disconnects]
        return part in dropped, dropped

    await until("release of the refused request's /encode", released, 10)
    elapsed = time.monotonic() - started
    assert elapsed < 15, elapsed
    passed(f"language refusal (HTTP {status}) closed /encode in {elapsed:.1f}s")
    for encoder in encoders:
        encoder.mode = "ok"

    # 5b. A dead encoder: its requests fail as unavailable until its sidecar
    # stops serving, then every request uses the other encoder.
    await encoders[0].http.stop()
    statuses = []

    async def only_survivor():
        mark = len(encoders[1].encodes)
        status, detail = await chat(IMAGES)
        assert status in (200, 503), (status, detail)
        statuses.append(status)
        if status == 200:
            assert len(encoders[1].encodes) == mark + 1
        return statuses[-4:] == [200] * 4, statuses

    await until("requests settle on the surviving encoder", only_survivor, 40)
    passed(f"dead encoder: statuses until settled on the survivor {statuses}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--sidecar", type=Path, help="default: python -m dynamo.sglang.sidecar"
    )
    parser.add_argument("--frontend-python", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    args = parser.parse_args()
    # Child processes run under tmp. Keep venv interpreter symlinks intact.
    if args.sidecar is not None:
        args.sidecar = args.sidecar.absolute()
    args.frontend_python = args.frontend_python.absolute()
    args.logs = args.logs.absolute()
    with tempfile.TemporaryDirectory() as tmp:
        async with AsyncExitStack() as stack:
            await asyncio.wait_for(run(args, Path(tmp), stack), 200)


if __name__ == "__main__":
    asyncio.run(main())
