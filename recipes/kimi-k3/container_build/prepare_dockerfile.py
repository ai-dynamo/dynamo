# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Add the reference build's ARM64 smoke check and source labels."""

import sys
from pathlib import Path


MARKER = """FROM pre_runtime AS runtime

COPY --from=licenses /legal /legal

# --- TRTLLM Stages
"""

REPLACEMENT = """FROM pre_runtime AS runtime

COPY --from=licenses /legal /legal

# Build-time smoke must reflect the ABI layout of the pinned ARM64 vLLM
# nightly.  It ships the stable-libtorch extension rather than vllm._C.
# Verify that extension is packaged here, and defer loading it to the native
# GPU smoke where libcuda is available.
RUN python3 - <<'PY'
import importlib.metadata as md
import importlib.util

import dynamo
import dynamo._core
import dynamo.frontend
import dynamo.vllm
import vllm

extension = importlib.util.find_spec("vllm._C_stable_libtorch")
assert extension is not None, "ARM64 vLLM stable-libtorch extension is missing"
assert md.version("vllm") == "0.26.1rc1.dev602+g65b7662d3"
assert md.version("ai-dynamo") == "1.4.0"
print("vllm", md.version("vllm"), vllm.__file__)
print("vllm_extension", extension.origin)
print("dynamo", md.version("ai-dynamo"), dynamo._core.__file__)
print("dynamo_frontend", dynamo.frontend.__file__)
print("dynamo_vllm", dynamo.vllm.__file__)
PY

LABEL org.opencontainers.image.revision="bf7542fd26613495cc2a59ded28848861e1fee3c" \\
      ai.oakhaven.dynamo.sha="bf7542fd26613495cc2a59ded28848861e1fee3c" \\
      ai.oakhaven.vllm.base="vllm/vllm-openai:nightly-65b7662d3fcb773afaf751ab29ac6960a0cf011d@sha256:3ae6337cbc8423ce6af3286a38b759df8c218bfdb29e1d0353cabc273a22fb0b" \\
      ai.oakhaven.vllm.commit="65b7662d3fcb773afaf751ab29ac6960a0cf011d"

# --- TRTLLM Stages
"""


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: {sys.argv[0]} <rendered.Dockerfile>")

    dockerfile = Path(sys.argv[1])
    text = dockerfile.read_text(encoding="utf-8")
    occurrences = text.count(MARKER)
    if occurrences != 1:
        raise SystemExit(
            f"expected one runtime marker in {dockerfile}, found {occurrences}"
        )
    dockerfile.write_text(text.replace(MARKER, REPLACEMENT), encoding="utf-8")


if __name__ == "__main__":
    main()
