#!/usr/bin/env python3
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
"""Prepare and verify the tracker's pinned tokenizer-only client cache."""
import hashlib
import json
import os
from pathlib import Path

from aiperf.common.tokenizer import Tokenizer

MODEL = "Motif-Technologies/Motif-3-NVFP4"
REVISION = "79f2fad1f8229f5db8a7ad08bbd24a20d6fb0bff"
HASHES = {
    "tokenizer.json": "956d8a693e0ede5283c803aba3fe1d4b46b202d9faac007f947c6f5b61ce7864",
    "tokenizer_config.json": "9cf7345dbacf8514c9ef790f39ed1fdd9b18083af33ece279b4fd913af57fd56",
    "chat_template.jinja": "998a066fd57a07187040a896e7fc455b2aa21dd490d156e82b63518be0a49297",
}


def main() -> None:
    relative = Path("models--" + MODEL.replace("/", "--")) / "snapshots" / REVISION
    source = Path("/shared-model-cache/hub") / relative
    target = Path(os.environ["HF_HOME"]) / "hub" / relative
    if (target / "config.json").exists():
        raise ValueError("Client cache must contain only tokenizer files")
    verified = {}
    for name, expected in HASHES.items():
        data = (source / name).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"Unexpected tokenizer file hash: {source / name}")
        if (target / name).exists() and (target / name).read_bytes() != data:
            raise ValueError(f"Existing client tokenizer file differs: {target / name}")
        verified[name] = data
    target.mkdir(parents=True, exist_ok=True)
    for name, data in verified.items():
        if not (target / name).exists():
            (target / name).write_bytes(data)
    tokenizer = Tokenizer.from_pretrained(
        MODEL, trust_remote_code=False, revision=REVISION, resolve_alias=False
    )
    if tokenizer is None:
        raise RuntimeError("AIPerf did not load the Motif tokenizer")
    print(
        json.dumps(
            {"tokenizer_preflight": "passed", "revision": REVISION, "sha256": HASHES}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
