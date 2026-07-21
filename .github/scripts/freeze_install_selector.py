#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze the Quickstart install selector for a versioned docs snapshot.

Dev docs use the interactive ``<InstallSelector />``, whose data tracks ``main``.
A versioned release snapshot must be static and pinned, so this replaces the
marked selector block with ``docker run ...:<version>`` tabs and drops the
component import. No-op if the marked block is absent (older snapshots).

Usage:
    freeze_install_selector.py <quickstart.mdx> <version>
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

RUNTIMES = [
    ("SGLang", "sglang", "sglang"),
    ("TensorRT-LLM", "tensorrtllm", "trtllm"),
    ("vLLM", "vllm", "vllm"),
]


def tabs(version: str) -> str:
    body = "\n".join(
        f'  <Tab title="{label}" language="{language}">\n'
        f"    ```bash\n"
        f"    docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/{image}-runtime:{version}\n"
        f"    ```\n"
        f"  </Tab>"
        for label, image, language in RUNTIMES
    )
    return (
        "Containers have all dependencies pre-installed. Pick your backend:\n\n<Tabs>\n"
        + body
        + "\n</Tabs>"
    )


def freeze(text: str, version: str) -> str:
    if "BEGIN:install-selector" not in text:
        return text
    text = re.sub(
        r"\{/\* BEGIN:install-selector.*?\*/\}.*?\{/\* END:install-selector \*/\}",
        tabs(version),
        text,
        flags=re.S,
    )
    return re.sub(r"^import \{ InstallSelector \}.*\n\n?", "", text, flags=re.M)


def main() -> int:
    page, version = Path(sys.argv[1]), sys.argv[2]
    if not page.exists():
        return 0
    out = freeze(page.read_text(), version)
    page.write_text(out)
    print(f"froze install selector to {version} in {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
