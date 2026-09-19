<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Sidecar testkit

The library owns protocol-independent request controls, persistent observations,
assertions, and bounded server lifetime. Concrete native adapters belong under
`tests/support/`; conformance scenarios are generic over that fixture contract.
Only vLLM is instantiated in this rollout. Existing SGLang, TensorRT-LLM and E2E
coverage retains its current ownership and scheduling.

Run or collect CPU wire tests:

```sh
python3 lib/sidecar/testkit/run.py --framework vllm --level wire
python3 lib/sidecar/testkit/run.py --level wire --list
```

Run the same binaries without engines, CUDA, a model cache or external networking:

```sh
python3 lib/sidecar/testkit/run.py --export /tmp/sidecar-cpu
cp lib/sidecar/testkit/run.py /tmp/sidecar-cpu/run.py
docker build -f lib/sidecar/testkit/CPU.Dockerfile -t sidecar-cpu /tmp/sidecar-cpu
docker run --rm --network none sidecar-cpu
```

The runner rejects empty collection, failed cases, and ignored cases. Wire tests
use loopback sockets and the existing CPU Mocker scheduler; they do not establish
real GPU-work release or actual KV transfer. See [coverage](COVERAGE.md) and the
separate [deviation report](DEVIATIONS.md).
