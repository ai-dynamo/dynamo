#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze the Quickstart install selector for a versioned docs snapshot.

Dev docs use the interactive ``<InstallSelector />``, whose data tracks ``main``.
A versioned release snapshot must be static and pinned, so this replaces the
marked selector block with pinned NVIDIA container commands plus the Intel XPU
source-build commands, then drops the component import. No-op if the marked
block is absent (older snapshots).

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
    nvidia = "\n\n".join(
        f'```bash title="{label}"\n'
        f"docker run --gpus all --network host --rm -it nvcr.io/nvidia/ai-dynamo/{image}-runtime:{version}\n"
        f"```"
        for label, image, _language in RUNTIMES
    )
    xpu = r"""```bash title="vLLM"
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
container/render.py --framework=vllm --device=xpu --target=runtime
docker build -t dynamo:latest-vllm-xpu-runtime \
  -f container/vllm-runtime-xpu-amd64-rendered.Dockerfile .
container/run.sh --image dynamo:latest-vllm-xpu-runtime --device=xpu -it
```

```bash title="SGLang"
git clone https://github.com/ai-dynamo/dynamo.git
cd dynamo
container/render.py --framework=sglang --device=xpu --target=runtime
docker build -t dynamo:latest-sglang-xpu-runtime \
  -f container/sglang-runtime-xpu-amd64-rendered.Dockerfile .
container/run.sh --image dynamo:latest-sglang-xpu-runtime --device=xpu -it
```"""
    return f"""<Tabs>
  <Tab title="NVIDIA GPU">
    Containers have all dependencies pre-installed. Pick your backend:

    <CodeBlocks>
{nvidia}
    </CodeBlocks>
  </Tab>
  <Tab title="Intel XPU">
    Intel XPU images are built from source for vLLM and SGLang:

    <CodeBlocks>
{xpu}
    </CodeBlocks>
  </Tab>
</Tabs>"""


def freeze(text: str, version: str) -> str:
    if "BEGIN:install-selector" not in text:
        return text  # older snapshot that never had the selector
    # Fail loudly rather than shipping a half-frozen page: if the marker is present
    # the block MUST be substituted and the import stripped, or the release would
    # render a literal <InstallSelector /> ("Unsupported JSX tag") that fern check
    # does not catch.
    text, blocks = re.subn(
        r"\{/\* BEGIN:install-selector.*?\*/\}.*?\{/\* END:install-selector \*/\}",
        tabs(version),
        text,
        flags=re.S,
    )
    if blocks != 1:
        raise SystemExit(
            f"freeze_install_selector: BEGIN marker present but block substitution "
            f"count is {blocks} (missing or duplicate END:install-selector marker?)"
        )
    text, imports = re.subn(
        r"^import \{ InstallSelector \}.*\n\n?", "", text, flags=re.M
    )
    if imports != 1:
        raise SystemExit(
            f"freeze_install_selector: expected 1 InstallSelector import to strip, found {imports}"
        )
    return text


def main() -> int:
    page, version = Path(sys.argv[1]), sys.argv[2]
    if not page.exists():
        print(f"warning: {page} not found — skipping selector freeze", file=sys.stderr)
        return 0
    out = freeze(page.read_text(), version)
    page.write_text(out)
    print(f"froze install selector to {version} in {page}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
