#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fetch the pinned SGLang gRPC contract used by the sidecar.

Normal builds download the immutable file from the SGLang commit below. Set
``SGLANG_PROTO_PATH`` to an exact local proto, or ``SGLANG_SOURCE`` to the root
of an SGLang checkout, when building without network access. Every source is
checked against the pinned digest before it is passed to ``tonic-build``.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

SGLANG_REVISION = "cf2aab82a3c5342f6bc89b7bae47a078ace1149c"
SGLANG_PROTO_RELATIVE_PATH = "proto/sglang/runtime/v1/sglang.proto"
SGLANG_PROTO_SHA256 = "a2e14952ddb2b34b6e22cbbc4e76d76d70c44f2dbf087cb9918aed3399d9ef42"
SGLANG_PROTO_URL = (
    "https://raw.githubusercontent.com/sgl-project/sglang/"
    f"{SGLANG_REVISION}/{SGLANG_PROTO_RELATIVE_PATH}"
)


def _read_proto() -> tuple[bytes, str]:
    if local_proto := os.environ.get("SGLANG_PROTO_PATH"):
        path = Path(local_proto).expanduser().resolve()
        try:
            return path.read_bytes(), str(path)
        except OSError as err:
            raise SystemExit(f"failed to read SGLANG_PROTO_PATH {path}: {err}") from err

    if sglang_source := os.environ.get("SGLANG_SOURCE"):
        path = Path(sglang_source).expanduser().resolve() / SGLANG_PROTO_RELATIVE_PATH
        try:
            return path.read_bytes(), str(path)
        except OSError as err:
            raise SystemExit(
                f"failed to read SGLANG_SOURCE proto {path}: {err}"
            ) from err

    request = Request(SGLANG_PROTO_URL, headers={"User-Agent": "dynamo-build"})
    try:
        with urlopen(request, timeout=30) as response:
            return response.read(), SGLANG_PROTO_URL
    except (OSError, URLError) as err:
        raise SystemExit(
            f"failed to download SGLang proto from {SGLANG_PROTO_URL}: {err}\n"
            "For an offline build, set SGLANG_PROTO_PATH to the proto file or "
            "SGLANG_SOURCE to a checkout at the pinned revision."
        ) from err


def _verify_proto(contents: bytes, source: str) -> None:
    actual = hashlib.sha256(contents).hexdigest()
    if actual != SGLANG_PROTO_SHA256:
        raise SystemExit(
            f"SGLang proto checksum mismatch for {source}: "
            f"expected {SGLANG_PROTO_SHA256}, got {actual}. "
            "Update the pinned revision and checksum together."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path, help="destination for the verified proto")
    args = parser.parse_args()

    contents, source = _read_proto()
    _verify_proto(contents, source)

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() == contents:
        return

    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(contents)
    os.replace(temporary, output)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
