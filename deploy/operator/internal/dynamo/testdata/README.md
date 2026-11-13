<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Native geometry parity

`enginegroup-native-geometry.json` is shared by the Go resolver test and native
configuration checks. Neither checker rewrites launch arguments or mocks the
native geometry. Run the native checks when changing the resolver or updating
its supported native prerequisites; Go-only tests cannot establish parser parity.

Use the native Elastic EP integration images required by the backend READMEs,
with a local model config. These commands resolve config but launch no engine:

```sh
python check_vllm_geometry.py /path/to/model
CUDA_VISIBLE_DEVICES=0 python check_sglang_geometry.py /path/to/model 0
CUDA_VISIBLE_DEVICES=0,1 python check_sglang_geometry.py /path/to/model 1
```

SGLang derives allocation width from actual visible devices, so each case runs
in a fresh process with the indicated visibility. A CUDA-capable native runtime
is required by both engines' configuration validation. Unsupported abbreviations,
opaque config sources and shell commands remain deliberately outside the parser
contract; this fixture is not an attempt to reproduce native argparse in Go.
