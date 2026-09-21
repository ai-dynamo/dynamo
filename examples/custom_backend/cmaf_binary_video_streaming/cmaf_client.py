# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless client for ``POST /v1/videos/stream/cmaf``.

Reads the framed byte stream as it arrives, prints one line per frame with the
wall-clock offset at which it landed (so first-segment latency is visible), and
writes ``init.mp4`` / ``segNNN.m4s`` plus a concatenated ``stream.mp4`` that any
player can open.

Usage:

    python examples/custom_backend/cmaf_binary_video_streaming/cmaf_client.py \
        --url http://localhost:8000 --model Wan-AI/Wan2.1-T2V-1.3B-Diffusers \
        --prompt "a cat playing piano" --out /tmp/cmaf

Wire format -- a sequence of ``[kind:u8][len:u32 big-endian][payload]``:

    0x01 metadata  JSON: mime_type, width, height, fps, target_duration, ...
    0x02 init      fMP4 init segment (ftyp+moov)
    0x03 segment   one media fragment (moof+mdat), starts with an IDR
    0x04 error     UTF-8 message; the stream failed, output is incomplete
    0x05 done      empty; end of stream

Exit code 0 = a ``done`` frame arrived with no preceding ``error`` frame.
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import urllib.request
from pathlib import Path

KIND_METADATA = 0x01
KIND_INIT = 0x02
KIND_SEGMENT = 0x03
KIND_ERROR = 0x04
KIND_DONE = 0x05

KIND_NAMES = {
    KIND_METADATA: "metadata",
    KIND_INIT: "init",
    KIND_SEGMENT: "segment",
    KIND_ERROR: "error",
    KIND_DONE: "done",
}

HEADER = struct.Struct(">BI")

# The frontend is reached directly. An ``http_proxy`` in the environment would
# otherwise send a request for localhost to a corporate proxy, which answers 403.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def _read_exactly(stream, n: int) -> bytes:
    """Read exactly ``n`` bytes or return short/empty at end of stream."""
    buf = bytearray()
    while len(buf) < n:
        part = stream.read(n - len(buf))
        if not part:
            break
        buf += part
    return bytes(buf)


def iter_frames(stream):
    """Yield ``(kind, payload)`` for each framed message on ``stream``."""
    while True:
        header = _read_exactly(stream, HEADER.size)
        if not header:
            return
        if len(header) < HEADER.size:
            raise RuntimeError(f"truncated frame header ({len(header)} bytes)")
        kind, length = HEADER.unpack(header)
        payload = _read_exactly(stream, length)
        if len(payload) < length:
            raise RuntimeError(
                f"truncated {KIND_NAMES.get(kind, kind)} payload "
                f"({len(payload)}/{length} bytes)"
            )
        yield kind, payload


def run(args: argparse.Namespace) -> int:
    body = {
        "model": args.model,
        "prompt": args.prompt,
        "size": args.size,
    }
    if args.seconds is not None:
        body["seconds"] = args.seconds
    nvext = {
        "fps": args.fps,
        "num_frames": args.num_frames,
        "num_inference_steps": args.steps,
    }
    nvext = {k: v for k, v in nvext.items() if v is not None}
    if nvext:
        body["nvext"] = nvext

    url = args.url.rstrip("/") + "/v1/videos/stream/cmaf"
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    concat = (outdir / "stream.mp4").open("wb")

    started = time.monotonic()
    segments = 0
    error: str | None = None
    done = False

    print(f"POST {url}")
    try:
        with _opener.open(request, timeout=args.timeout) as response:
            for kind, payload in iter_frames(response):
                dt = time.monotonic() - started
                name = KIND_NAMES.get(kind)
                if name is None:
                    print(f"  {dt:7.2f}s  unknown kind 0x{kind:02x}, dropping")
                    continue
                if kind == KIND_METADATA:
                    meta = json.loads(payload)
                    print(f"  {dt:7.2f}s  metadata {meta}")
                    (outdir / "metadata.json").write_bytes(payload)
                elif kind == KIND_INIT:
                    print(f"  {dt:7.2f}s  init     {len(payload)} bytes")
                    (outdir / "init.mp4").write_bytes(payload)
                    concat.write(payload)
                elif kind == KIND_SEGMENT:
                    print(f"  {dt:7.2f}s  segment  #{segments} {len(payload)} bytes")
                    (outdir / f"seg{segments:03d}.m4s").write_bytes(payload)
                    concat.write(payload)
                    segments += 1
                elif kind == KIND_ERROR:
                    error = payload.decode("utf-8", "replace")
                    print(f"  {dt:7.2f}s  ERROR    {error}")
                elif kind == KIND_DONE:
                    print(f"  {dt:7.2f}s  done")
                    done = True
    finally:
        concat.close()

    print(f"\n{segments} segment(s) -> {outdir}")
    if error:
        print("FAILED (server sent an error frame)")
        return 1
    if not done:
        print("FAILED (stream ended without a done frame)")
        return 1
    if segments == 0:
        print("FAILED (no media segments)")
        return 1
    print("OK")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", default="http://localhost:8000", help="frontend base URL")
    p.add_argument("--model", required=True)
    p.add_argument("--prompt", default="a cat playing piano")
    p.add_argument("--size", default="832x480")
    p.add_argument("--seconds", type=int, default=None)
    p.add_argument("--fps", type=int, default=None)
    p.add_argument("--num-frames", type=int, default=None)
    p.add_argument("--steps", type=int, default=None, help="num_inference_steps")
    p.add_argument("--out", default="cmaf_out", help="dir for init/segments")
    p.add_argument("--timeout", type=float, default=1800.0)
    return run(p.parse_args())


if __name__ == "__main__":
    sys.exit(main())
