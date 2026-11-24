# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU smoke with real Dynamo frontend, sidecars and SGLang Stage/Deliver leases.

Run the driver in the patched SGLang runtime with grpcio-tools and pyzmq installed.
--model is a local Qwen-VL or Qwen-Omni tokenizer/config directory.
The fixture replaces model computation with identifiable byte strings. No GPU
or KV data is accessed. --split verifies P/D bootstrap and the metadata-only D.
--native-generate also exercises the patched frontend's public /generate route
and its private native HTTP hop (requires aiohttp).
"""

import argparse
import asyncio
import base64
import hashlib
import importlib
import json
import os
import secrets
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
import sglang
import zmq
import zmq.asyncio
from sglang.encoder.v1 import encoder_pb2, encoder_pb2_grpc
from sglang.srt.disaggregation.encoder.handoff import EncoderHandoff
from sglang.srt.disaggregation.encoder.handoff_auth import (
    EncoderHandoffSigner,
    handoff_key_id,
    verify_handoff,
)
from sglang.test.staged_encoder import StagedEncoderFixture

try:
    from aiohttp import web
    from aiohttp.test_utils import TestServer
except ImportError:
    web = TestServer = None


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


# The fixture tests byte delivery and ordering, independent of video decoding.
VIDEOS = ["data:video/mp4;base64,AA==", "data:video/mp4;base64,AQ=="]


AUDIOS = ["data:audio/wav;base64,AA==", "data:audio/mpeg;base64,AQ=="]
AUDIO_MODEL_TYPE = "qwen3_omni_moe"
LAYOUT_ID = "ab" * 32


class Encoder(StagedEncoderFixture):
    def __init__(self, model, sockets, signer):
        self.sockets = sockets
        self.parts = {}
        self.encodes = []
        self.sends = []
        self.releases = []
        self.release_rpcs = []
        self.revoked_parts = set()
        self.deliver_rpcs = []
        self.fail = False
        self.stage_failure = None
        self.stage_calls = 0
        self.corrupt_stage = None
        super().__init__(
            encoder_pb2.GetEncoderInfoResponse(
                model_path=model,
                served_model_name="vision",
                supports_staged_video=True,
                staged_transfer_backend="zmq_to_scheduler",
                staged_layout_id=LAYOUT_ID,
                staged_audio_model_type=AUDIO_MODEL_TYPE,
            ),
            backend="zmq_to_scheduler",
            capacity=4,
            ttl=30,
            stage_timeout=5,
            signer=signer,
        )

    def _describe_part(self, part):
        return encoder_pb2.DescribeResponse(
            shape=[len(self.parts[part]), 16], dtype="bfloat16"
        )

    async def Stage(self, request, context):
        self.stage_calls += 1
        if self.stage_failure == grpc.StatusCode.DEADLINE_EXCEEDED:
            # Let the sidecar enforce the deadline on a genuinely stalled RPC.
            await asyncio.Future()
        if self.stage_failure is not None:
            await context.abort(self.stage_failure, "private injected Stage error")
        response = await super().Stage(request, context)
        if self.corrupt_stage == "request_id":
            response.transfer.request_id = "wrong-request-id"
        elif self.corrupt_stage == "image_count":
            response.transfer.image_count += 1
        if self.corrupt_stage:
            response.handoff_mac = self._handoff_signer.sign(
                response.transfer, response.instance_id
            )
        return response

    async def Deliver(self, request, context):
        self.deliver_rpcs.append(request.transfer.transfer_id)
        return await super().Deliver(request, context)

    async def Release(self, request, context):
        released_before = len(self.releases)
        response = await super().Release(request, context)
        self.revoked_parts.update(self.releases[released_before:])
        self.release_rpcs.append(request.transfer_id)
        return response

    async def _stage_media(self, part, images, videos, audios):
        assert images == IMAGES[: len(images)]
        assert videos == VIDEOS[: len(videos)]
        assert audios == AUDIOS[: len(audios)]
        self.encodes.append(part)
        if self.fail:
            raise ValueError("injected encode failure")

        def digest(source):
            return hashlib.sha256(source.encode()).digest()

        self.parts[part] = [digest(source) for source in (*images, *videos, *audios)]
        return (
            [
                encoder_pb2.ImageLayout(
                    t=1,
                    h=2,
                    w=2,
                    identity=digest(image).hex(),
                    embedding_rows=int(bool(audios)),
                )
                for image in images
            ],
            [
                encoder_pb2.VideoLayout(
                    t=2,
                    h=2,
                    w=2,
                    identities=[digest(video).hex()],
                    # This 29.97-fps interval needs a correctly rounded JSON parser.
                    second_per_grid_ts=27 / 29.97,
                    embedding_rows=int(bool(audios)),
                )
                for video in videos
            ],
            [
                encoder_pb2.AudioLayout(
                    identity=digest(audio).hex(),
                    feature_length=2,
                    embedding_rows=1,
                    embedding_width=16,
                )
                for audio in audios
            ],
            16,
        )

    async def send(self, state, destination):
        part, url = state.req_id, destination.endpoint
        self.sends.append((part, url))
        with self.sockets.socket(zmq.PUSH) as sock:
            sock.connect("tcp://" + url)
            await sock.send_multipart([part.encode(), *self.parts[part]])

    async def release(self, state):
        part = state.req_id
        self.parts.pop(part, None)
        self.releases.append(part)


class EncoderEndpoint:
    """Allow a backend swap at the same gRPC address between discovery polls."""

    def __init__(self, encoder):
        self.registered = self.serving = encoder

    async def GetEncoderInfo(self, request, context):
        return await self.registered.GetEncoderInfo(request, context)

    def __getattr__(self, method):
        async def dispatch(request, context):
            return await getattr(self.serving, method)(request, context)

        return dispatch


async def run(args, tmp, stack):
    # Generate only the vendored language service: the patched SGLang package
    # supplies the encoder stubs and production lease/receipt code.
    proto = Path(__file__).resolve().parents[1] / "proto"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "grpc_tools.protoc",
            f"-I{proto}",
            f"-I{Path(grpc_tools.__file__).parent / '_proto'}",
            f"--python_out={tmp}",
            f"--grpc_python_out={tmp}",
            *sorted(map(str, (proto / "sglang/api").rglob("*.proto"))),
            str(proto / "sglang/json/v1/options.proto"),
        ],
        check=True,
    )
    sglang.__path__.append(str(tmp / "sglang"))
    generate_pb2 = importlib.import_module("sglang.api.v1.generate_pb2")
    service_pb2 = importlib.import_module("sglang.api.v1.service_pb2")
    service_pb2_grpc = importlib.import_module("sglang.api.v1.service_pb2_grpc")

    model = str(args.model.resolve())
    config = json.loads((args.model / "config.json").read_text())
    config = config.get("thinker_config", config)
    # /generate accepts pre-tokenized requests even on the visual-only test tokenizer.
    audio_token = config.get("audio_token_id", 9876)
    image_token = config["image_token_id"]
    video_token = config["video_token_id"]
    sockets = zmq.asyncio.Context()
    stack.callback(sockets.destroy, 0)
    requests = []
    refuse_handoff = False
    refused = []
    handoff_key = secrets.token_bytes(32)

    class Language(service_pb2_grpc.SglangServiceServicer):
        def __init__(self, role):
            self.role = role
            self.receiver = sockets.socket(zmq.PULL)
            self.receiver_port = self.receiver.bind_to_random_port("tcp://127.0.0.1")
            self.receives = []
            self.http_port = None
            self.http_requests = 0
            self.grpc_requests = 0

        async def GetModelInfo(self, req, ctx):
            return service_pb2.GetModelInfoResponse(
                model_path=model, tokenizer_path=model, is_generation=True
            )

        async def GetServerInfo(self, req, ctx):
            return service_pb2.GetServerInfoResponse(
                model_path=model,
                served_model_name="vision",
                incremental_streaming_output=True,
                disaggregation_mode=self.role,
                accepts_encoder_handoff=True,
                staged_transfer_backend=(
                    "zmq_to_scheduler" if self.role != "decode" else ""
                ),
                staged_layout_id=LAYOUT_ID,
                encoder_handoff_key_id=handoff_key_id(handoff_key),
                staged_audio_model_type=AUDIO_MODEL_TYPE,
                supports_staged_video=True,
                host="127.0.0.1",
                port=self.http_port,
                max_context_length=4096,
                max_total_num_tokens=4096,
                deployment_info=service_pb2.DeploymentInfo(
                    nnodes=1,
                    dp_size=1,
                    max_running_requests=4,
                    page_size=16,
                    disaggregation_bootstrap_port=self.receiver_port,
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

        async def consume(self, ids, handoff, room, host, port):
            images = ids.count(image_token)
            videos = ids.count(video_token)
            audios = ids.count(audio_token)
            if args.split:
                assert port == languages[0].receiver_port
                assert host == "127.0.0.1"
            transfer = None
            if handoff is not None:
                try:
                    transfer = EncoderHandoff.parse(handoff, images, videos, audios)
                except ValueError:
                    # Capture the ticket that reached admission while preserving
                    # the production parser's refusal for the public HTTP caller.
                    receipt = EncoderHandoff.parse(handoff, None, None, None)
                    verify_handoff(receipt, handoff_key)
                    refused.append(receipt)
                    raise
                verify_handoff(transfer, handoff_key)
                assert transfer.version == (4 if audios else 2 if videos else 1)
                if self.role != "decode":
                    await asyncio.sleep(0.05)
                    await transfer.deliver(f"127.0.0.1:{self.receiver_port}", 5)
                    frames = await asyncio.wait_for(self.receiver.recv_multipart(), 5)
                    assert frames == [
                        transfer.part_id.encode(),
                        *(
                            hashlib.sha256(i.encode()).digest()
                            for i in IMAGES[:images] + VIDEOS[:videos] + AUDIOS[:audios]
                        ),
                    ]
                    self.receives.append(transfer.part_id)
            else:
                assert not (images or videos or audios)
            requests.append((self.role, room, transfer, ids))

        async def Generate(self, req, ctx):
            self.grpc_requests += 1
            assert all(
                not req.HasField(k) for k in ("image_data", "video_data", "audio_data")
            )
            ids = list(req.input_ids.one.ids)
            if refuse_handoff and self.role != "decode":
                receipt = EncoderHandoff.parse(
                    req.encoder_handoff,
                    ids.count(image_token),
                    ids.count(video_token),
                    ids.count(audio_token),
                )
                verify_handoff(receipt, handoff_key)
                refused.append(receipt)
                await ctx.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "language admission refused"
                )
            await self.consume(
                ids,
                req.encoder_handoff if req.HasField("encoder_handoff") else None,
                (
                    req.bootstrap_room.one.value
                    if req.HasField("bootstrap_room")
                    else None
                ),
                req.bootstrap_host.one.value,
                req.bootstrap_port.one.value,
            )
            item = generate_pb2.GenerateStreamItem()
            item.frame.meta_info.id = req.rid.one
            item.frame.meta_info.prompt_tokens = len(ids)
            item.frame.meta_info.completion_tokens = 1
            item.frame.meta_info.finish_reason.length.length = 1
            item.frame.output_ids.ids.append(16)
            yield item

        async def serve_http(self):
            if web is None:
                raise RuntimeError("--native-generate requires aiohttp")

            async def health(request):
                return web.Response()

            async def generate(request):
                self.http_requests += 1
                body = await request.json()
                assert not {"image_data", "video_data", "audio_data"}.intersection(body)
                assert body["stream"]
                ids = body["input_ids"]
                try:
                    await self.consume(
                        ids,
                        json.dumps(body["encoder_handoff"]),
                        body.get("bootstrap_room"),
                        body.get("bootstrap_host"),
                        body.get("bootstrap_port"),
                    )
                except ValueError as exc:
                    return web.json_response(
                        {"error": {"message": str(exc)}}, status=400
                    )
                assert body["mm_hashes"] == (
                    ["0xa1", "0xc1", "0xd1", "0xd2"]
                    if audio_token in ids
                    else (
                        ["0xa1", "0xc1", "0xc2"]
                        if video_token in ids
                        else ["0xa1", "0xb2"]
                    )
                )
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
                return web.Response(
                    text=f"data: {json.dumps(item)}\n\ndata: [DONE]\n\n",
                    content_type="text/event-stream",
                )

            app = web.Application()
            app.router.add_get("/health", health)
            app.router.add_post("/generate", generate)
            server = TestServer(app)
            await stack.enter_async_context(server)
            self.http_port = server.port

    def listener():
        server = grpc.aio.server()
        return server, server.add_insecure_port("127.0.0.1:0")

    async def serve(server, add, service):
        add(service, server)
        await server.start()
        stack.push_async_callback(server.stop, 0)

    env = {
        **os.environ,
        "DYN_DISCOVERY_BACKEND": "file",
        "DYN_FILE_KV": str(tmp / "kv"),
        "DYN_NAMESPACE": "img-" + uuid.uuid4().hex,
        "DYN_SYSTEM_PORT": "0",
        "DYN_REQUEST_PLANE": "tcp",
        "DYN_EVENT_PLANE": "zmq",
        "DYN_TCP_RPC_PORT": "0",
        "DYN_TCP_RESPONSE_STREAM_PORT": "0",
        "HF_HUB_OFFLINE": "1",
        "DYN_HTTP_OVERLOAD_STATUS_CODE": "529",
    }
    # The frontend runs in its own interpreter/environment.
    env.pop("PYTHONPATH", None)
    env.pop("DYN_SGLANG_ENABLE_GENERATE", None)
    if args.native_generate:
        env["DYN_SGLANG_ENABLE_GENERATE"] = "1"
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

    async def start(label, *cmd):
        log = stack.enter_context((args.logs / f"{label}.log").open("w"))
        proc = await asyncio.create_subprocess_exec(
            *cmd, cwd=tmp, env=env, stdout=log, stderr=log
        )
        stack.push_async_callback(stop, proc)
        return proc

    sidecar = (
        [args.sidecar]
        if args.sidecar
        else [args.frontend_python, "-m", "dynamo.sglang.sidecar"]
    )

    def create_encoder(port):
        signer = EncoderHandoffSigner(f"grpc://127.0.0.1:{port}", handoff_key)
        return Encoder(model, sockets, signer)

    encoders = []
    endpoints = []
    encoder_workers = []
    encoder_ports = []
    for i in range(2):
        server, port = listener()
        encoder = create_encoder(port)
        endpoint = EncoderEndpoint(encoder)
        await serve(
            server, encoder_pb2_grpc.add_SglangEncoderServicer_to_server, endpoint
        )
        encoders.append(encoder)
        endpoints.append(endpoint)
        encoder_ports.append(port)
        encoder_workers.append(
            await start(
                f"encode-{i}",
                *sidecar,
                "--grpc-endpoint",
                f"127.0.0.1:{port}",
                "--encoder-stage-timeout-secs",
                "2",
            )
        )

    roles = ["prefill", "decode"] if args.split else ["null", "null"]
    languages = [Language(role) for role in roles]
    for i, service in enumerate(languages):
        if args.native_generate:
            await service.serve_http()
        server, port = listener()
        await serve(
            server, service_pb2_grpc.add_SglangServiceServicer_to_server, service
        )
        opts = ["--route-to-encoder"] if service.role != "decode" else []
        if service.role == "prefill":
            opts.extend(["--bootstrap-host", "127.0.0.1"])
        await start(
            f"language-{i}", *sidecar, "--grpc-endpoint", f"127.0.0.1:{port}", *opts
        )

    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        http_port = reservation.getsockname()[1]
    frontend = await start(
        "frontend",
        args.frontend_python,
        "-m",
        "dynamo.frontend",
        "--http-port",
        str(http_port),
    )
    url = f"http://127.0.0.1:{http_port}"

    async def exchange(path, payload=None):
        def send():
            req = urllib.request.Request(
                url + path,
                data=json.dumps(payload).encode() if payload else None,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as response:
                return json.load(response)

        return await asyncio.to_thread(send)

    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            models = await exchange("/v1/models")
            if any(m["id"] == "vision" for m in models["data"]):
                break
        except (OSError, KeyError):
            pass
        if frontend.returncode is not None or any(
            w.returncode is not None for w in encoder_workers
        ):
            raise RuntimeError(f"worker exited during startup; see {args.logs}")
        await asyncio.sleep(0.25)
    else:
        raise TimeoutError(f"model discovery; see {args.logs}")

    def payload(media):
        content = [{"type": "text", "text": "Describe in order."}]
        for source in media:
            if source == AUDIOS[0]:
                content.append(
                    {
                        "type": "input_audio",
                        "input_audio": {"format": "wav", "data": source.split(",")[1]},
                    }
                )
            else:
                key = source.split(":", 1)[1].split("/", 1)[0] + "_url"
                content.append({"type": key, key: {"url": source}})
            content.append({"type": "text", "text": "Next."})
        return {
            "model": "vision",
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 1,
        }

    async def chat(images):
        result = await exchange("/v1/chat/completions", payload(images))
        assert result["choices"][0]["finish_reason"] == "length", result

    for _ in range(8):
        await chat(IMAGES)
    assert all(e.encodes for e in encoders), "Dynamo must select both encoder replicas"
    assert sum(len(e.encodes) for e in encoders) == 8
    assert sum(len(language.receives) for language in languages) == 8
    if args.split:
        for _, room, transfer, ids in [r for r in requests if r[0] == "prefill"]:
            assert room is not None and ("decode", room, transfer, ids) in requests
        assert not languages[1].receives
    else:
        assert all(
            language.receives for language in languages
        ), "exercise both destination schedulers"

    await chat([])
    assert sum(len(e.encodes) for e in encoders) == 8
    # Both producers are live. If P/PD refuses an already-staged request, the
    # frontend must revoke the original producer before its 30-second TTL.
    refuse_handoff = True
    try:
        await exchange("/v1/chat/completions", payload(IMAGES))
    except urllib.error.HTTPError:
        pass
    else:
        raise AssertionError("language admission refusal should fail the request")
    refuse_handoff = False

    async def assert_released(transfer_id, part_id):
        (issuer,) = [e for e in encoders if part_id in e.encodes]

        async def wait_for_release():
            while transfer_id not in issuer.release_rpcs:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(wait_for_release(), 5)
        assert sum(e.release_rpcs.count(transfer_id) for e in encoders) == 1
        assert part_id in issuer.revoked_parts
        assert part_id not in issuer.encoder.req_states and part_id not in issuer.parts
        assert all(transfer_id not in e.deliver_rpcs for e in encoders)

    assert refused and all(r == refused[0] for r in refused)
    receipt = refused[0]
    await assert_released(receipt.transfer_id, receipt.part_id)
    rejected_parts = {receipt.part_id}
    print("PASS Chat admission refusal released the issuing E before Deliver and TTL")

    if args.native_generate:
        # A valid image and a prompt with zero image placeholders: E succeeds
        # and the native SGLang handoff parser rejects it at P/PD admission.
        refused.clear()
        stage_calls = sum(e.stage_calls for e in encoders)
        try:
            await exchange("/generate", {"input_ids": [1], "image_data": IMAGES[0]})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400, (exc.code, exc.read())
        else:
            raise AssertionError("missing image placeholder passed native admission")
        assert sum(e.stage_calls for e in encoders) == stage_calls + 1
        assert refused and all(r == refused[0] for r in refused)
        receipt = refused[0]
        await assert_released(receipt.transfer_id, receipt.part_id)
        rejected_parts.add(receipt.part_id)
        print(
            "PASS /generate input_ids=[1] + image: HTTP 400, issuing E released before Deliver and TTL"
        )

    # A successful Stage RPC carrying an invalid response is an upstream fault.
    chat_request = ("/v1/chat/completions", payload(IMAGES))
    count_request = (
        ("/generate", {"input_ids": [image_token], "image_data": IMAGES[0]})
        if args.native_generate
        else chat_request
    )
    for field, request in (
        ("request_id", chat_request),
        ("image_count", count_request),
    ):
        for encoder in encoders:
            encoder.corrupt_stage = field
        parts = {p for e in encoders for p in e.encodes}
        language_calls = sum(s.grpc_requests + s.http_requests for s in languages)
        try:
            await exchange(*request)
        except urllib.error.HTTPError as exc:
            detail = exc.read()
            assert exc.code == 502, (request[0], field, exc.code, detail)
            assert b"wrong-request-id" not in detail
        else:
            raise AssertionError("malformed Stage response reached language admission")
        assert (
            sum(s.grpc_requests + s.http_requests for s in languages) == language_calls
        )
        (part,) = {p for e in encoders for p in e.encodes} - parts
        await assert_released(part.removesuffix("_local_part_0"), part)
        rejected_parts.add(part)
    for encoder in encoders:
        encoder.corrupt_stage = None
    print(
        "PASS wrong Stage request ID and image count each returned HTTP 502 and released the issuing E before admission"
    )

    await stop(encoder_workers[0])
    removed = len(encoders[0].encodes)
    for _ in range(4):
        await chat(IMAGES)
    assert len(encoders[0].encodes) == removed
    await chat([VIDEOS[0]])
    await chat([VIDEOS[0], IMAGES[0], VIDEOS[1]])
    media_requests = 14
    if "audio_token_id" in config:
        before = len(requests)
        await chat([AUDIOS[0], IMAGES[0], VIDEOS[0], AUDIOS[1]])
        for _, _, transfer, ids in requests[before:]:
            assert transfer.version == 4
            assert [v for v in ids if v in (audio_token, image_token, video_token)] == [
                audio_token,
                image_token,
                video_token,
                audio_token,
            ]
        media_requests += 1
        print(
            "PASS Chat audio, image, video and audio placeholders retained input order"
        )
    if args.native_generate:
        start, end = config["vision_start_token_id"], config["vision_end_token_id"]
        native_ids = [10, start, image_token, end, 11, start, image_token, end, 12]
        body = {
            "input_ids": native_ids,
            "image_data": IMAGES,
            "mm_hashes": ["0xa1", "0xb2"],
            "sampling_params": {"max_new_tokens": 2},
            "rid": "client-native-image",
        }
        before = len(requests)
        try:
            await exchange("/generate", {**body, "encoder_handoff": {"version": 999}})
        except urllib.error.HTTPError as exc:
            assert exc.code == 400, (exc.code, exc.read())
            assert b"encoder_handoff" in exc.read()
        else:
            raise AssertionError("public /generate accepted private encoder_handoff")
        assert len(requests) == before
        assert sum(len(e.encodes) for e in encoders) == media_requests + len(
            rejected_parts
        )
        http_calls = sum(language.http_requests for language in languages)

        result = await exchange("/generate", body)
        assert result["meta_info"]["id"] == body["rid"], result
        assert result["meta_info"]["finish_reason"]["type"] == "length", result
        assert result["output_ids"], result
        hops = requests[before:]
        assert len(hops) == (2 if args.split else 1)
        assert sum(
            language.http_requests for language in languages
        ) == http_calls + len(hops)
        assert all(hop[3] == native_ids for hop in hops)
        assert hops[0][2].part_id == encoders[1].encodes[-1]
        if args.split:
            assert {h[0] for h in hops} == {"prefill", "decode"}
            assert hops[0][1] is not None and hops[0][1:3] == hops[1][1:3]
        mixed = {
            **body,
            "rid": "client-native-mixed",
            "input_ids": [
                10,
                start,
                video_token,
                end,
                start,
                image_token,
                end,
                start,
                video_token,
                end,
            ],
            "image_data": IMAGES[0],
            "video_data": VIDEOS,
            "mm_hashes": ["0xa1", "0xc1", "0xc2"],
        }
        before = len(requests)
        result = await exchange("/generate", mixed)
        assert result["meta_info"]["id"] == mixed["rid"]
        hops = requests[before:]
        assert len(hops) == (2 if args.split else 1)
        assert all(hop[3] == mixed["input_ids"] and hop[2].version == 2 for hop in hops)
        audio_mixed = {
            **body,
            "rid": "client-native-audio",
            "input_ids": [audio_token, image_token, video_token, audio_token],
            "audio_data": AUDIOS,
            "image_data": IMAGES[0],
            "video_data": VIDEOS[0],
            "mm_hashes": ["0xa1", "0xc1", "0xd1", "0xd2"],
        }
        before = len(requests)
        result = await exchange("/generate", audio_mixed)
        assert result["meta_info"]["id"] == audio_mixed["rid"]
        hops = requests[before:]
        assert len(hops) == (2 if args.split else 1)
        assert all(
            hop[3] == audio_mixed["input_ids"] and hop[2].version == 4 for hop in hops
        )
        media_requests += 3
        print(
            "PASS public /generate routed images, video and interleaved audio, and rejected private encoder_handoff"
        )

    assert sum(len(e.encodes) for e in encoders) == media_requests + len(rejected_parts)
    assert sum(len(language.receives) for language in languages) == media_requests
    if args.split:
        assert not languages[1].receives
    for e in encoders:
        assert sorted(e.encodes) == sorted(e.releases)
        assert sorted(
            part for part in e.encodes if part not in rejected_parts
        ) == sorted(part for part, _ in e.sends)
        assert not e.parts
    encoder = encoders[1]
    failure_routes = [("/v1/chat/completions", payload(IMAGES))]
    if args.native_generate:
        failure_routes.append(("/generate", body))
    for status, expected in (
        (grpc.StatusCode.INVALID_ARGUMENT, 400),
        (grpc.StatusCode.UNAVAILABLE, 503),
        (grpc.StatusCode.RESOURCE_EXHAUSTED, 529),
        (grpc.StatusCode.DEADLINE_EXCEEDED, 504),
    ):
        encoder.stage_failure = status
        for path, request in failure_routes:
            stage_calls = encoder.stage_calls
            language_calls = sum(s.grpc_requests + s.http_requests for s in languages)
            started = time.monotonic()
            try:
                await exchange(path, request)
            except urllib.error.HTTPError as exc:
                message = exc.read()
                assert exc.code == expected, (path, status, exc.code, message)
                assert b"private injected Stage error" not in message, (path, message)
            else:
                raise AssertionError(f"{path} accepted Stage failure {status}")
            if status == grpc.StatusCode.DEADLINE_EXCEEDED:
                assert (
                    time.monotonic() - started < 10
                ), f"{path} exceeded the Stage deadline"
            assert encoder.stage_calls == stage_calls + 1, path
            assert (
                sum(s.grpc_requests + s.http_requests for s in languages)
                == language_calls
            ), path
    encoder.stage_failure = None
    print(
        "PASS required Stage preserved client, unavailable, overload and timeout status"
    )

    # Fail after the native forward starts to exercise cleanup of a staged lease.
    encoders[1].fail = True
    before = len(requests)
    try:
        await exchange("/v1/chat/completions", payload(IMAGES))
    except urllib.error.HTTPError:
        pass
    else:
        raise AssertionError("injected Stage failure should fail the image request")
    assert len(requests) == before, "native Generate ran after Stage failed"
    assert (
        sum(len(e.encodes) for e in encoders)
        == media_requests + len(rejected_parts) + 1
    )
    assert len(encoders[1].encodes) == len(encoders[1].releases)
    print(
        f"PASS ({roles}): {media_requests} media requests, {media_requests} transfers, "
        "text bypass, replica withdrawal and failed Stage."
    )
    print(
        "encodes per E (includes rejected admission and injected encoder failure)",
        [len(e.encodes) for e in encoders],
    )
    print("deliveries per receiver", [len(language.receives) for language in languages])

    replacement = create_encoder(encoder_ports[1])
    endpoints[1].serving = replacement
    before_requests = len(requests)
    try:
        await exchange("/v1/chat/completions", payload(IMAGES))
    except urllib.error.HTTPError:
        pass
    else:
        raise AssertionError("Stage from a replacement encoder reached the consumer")
    assert len(requests) == before_requests and not replacement.sends
    assert len(replacement.encodes) == 1
    assert replacement.releases == replacement.encodes and not replacement.parts
    print("PASS rejected and released a replacement encoder's Stage for the same model")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--sidecar", type=Path, help="default: python -m dynamo.sglang.sidecar"
    )
    parser.add_argument("--frontend-python", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--split", action="store_true")
    parser.add_argument("--native-generate", action="store_true")
    args = parser.parse_args()
    # Child processes run under tmp. Keep venv interpreter symlinks intact.
    if args.sidecar is not None:
        args.sidecar = args.sidecar.absolute()
    args.frontend_python = args.frontend_python.absolute()
    with tempfile.TemporaryDirectory() as tmp:
        async with AsyncExitStack() as stack:
            await asyncio.wait_for(run(args, Path(tmp), stack), 250)


if __name__ == "__main__":
    asyncio.run(main())
