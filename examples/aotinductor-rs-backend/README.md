<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AOTInductor Rust Backend

Minimal standalone Dynamo unified backend using
[`pierric/aotinductor-rs`](https://github.com/pierric/aotinductor-rs).

The example takes preprocessed token IDs, pads them into a `[1, 4]`
float tensor, runs an AOTInductor `.pt2` package, and returns the first
output tensor as generated token IDs.

`aotinductor-rs` currently depends on `tch` 0.23, whose `torch-sys`
build expects PyTorch/libtorch 2.10.0. Set `LIBTORCH` to the matching
PyTorch package directory before building.

```bash
python3 -m venv /tmp/aoti-rs-torch210
/tmp/aoti-rs-torch210/bin/python -m pip install --upgrade pip
/tmp/aoti-rs-torch210/bin/python -m pip install \
  torch==2.10.0 \
  --index-url https://download.pytorch.org/whl/cpu

/tmp/aoti-rs-torch210/bin/python \
  examples/aotinductor-rs-backend/scripts/export_tiny.py \
  /tmp/aotinductor-tiny.pt2

export LIBTORCH=/tmp/aoti-rs-torch210/lib/python3.10/site-packages/torch
export LD_LIBRARY_PATH=$LIBTORCH/lib:${LD_LIBRARY_PATH:-}
```

Run the contract tests:

```bash
AOTINDUCTOR_TEST_PACKAGE=/tmp/aotinductor-tiny.pt2 \
  cargo test --manifest-path examples/aotinductor-rs-backend/Cargo.toml
```

Run the backend with local in-memory discovery:

```bash
DYN_DISCOVERY_BACKEND=mem \
DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS=0 \
  cargo run --manifest-path examples/aotinductor-rs-backend/Cargo.toml -- \
  --model-package /tmp/aotinductor-tiny.pt2 \
  --model-name aotinductor-tiny \
  --model-path ""
```
