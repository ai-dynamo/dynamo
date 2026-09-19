<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Sidecar testkit

The library owns protocol-independent request controls, persistent observations,
assertions, and bounded server lifetime. Concrete native adapters belong under
`tests/support/`; conformance scenarios are generic over that fixture contract.
Only vLLM is instantiated in this rollout. Existing SGLang, TensorRT-LLM and E2E
coverage retains its current ownership and scheduling.

Run or collect CPU tests (only vLLM is enabled):

```sh
python3 lib/sidecar/testkit/run.py --framework vllm --level all
python3 lib/sidecar/testkit/run.py --level unit --list
python3 lib/sidecar/testkit/run.py --level pre-merge
python3 lib/sidecar/testkit/run.py --level integration
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

`unit` includes the shared common crate once and vLLM's isolated modules.
`wire` includes shared conformance and retained socket suites. `process` starts
actual sidecar children with local discovery; `integration` combines wire and
process. `pre-merge` selects units and wire; `all` selects all CPU layers.
`--framework all` currently enables the same vLLM profile, without activating
other backends. The runner rejects missing exported layers and mismatched
collection/execution counts, in addition to failures and ignored cases.

Native engine tests have a separate opt-in feature and launcher; see
[NATIVE.md](NATIVE.md). CPU wire/process tests never substitute for its scheduler
and transfer observations. New process/native CI runs after merge and nightly;
existing E2E scheduling remains unchanged. Detailed isolated and process mappings
are in [UNITS.md](UNITS.md) and [PROCESS.md](PROCESS.md). The full refreshed
capability ledger is [SUPPORT.md](SUPPORT.md).
