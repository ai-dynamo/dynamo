#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Build the sidecar and the frontend bindings from the checkout at /workspace
# and install them over the copies in the SGLang image. Needs no GPU.

set -euo pipefail
cd /workspace

cargo build -p dynamo-sglang-sidecar
(cd lib/bindings/python && maturin develop --uv)
uv pip install --python /opt/sglang/bin/python3 --no-deps -e /workspace
# Install last: replacing the image's ai-dynamo wheel removes its sidecar binary.
install -m 0755 "$CARGO_TARGET_DIR/debug/dynamo-sglang-sidecar" /opt/sglang/bin/dynamo-sglang-sidecar

/opt/sglang/bin/python3 -c "import dynamo._core as core; print('bindings:', core.__file__)"
/opt/sglang/bin/dynamo-sglang-sidecar --help >/dev/null && echo "sidecar: /opt/sglang/bin/dynamo-sglang-sidecar"
