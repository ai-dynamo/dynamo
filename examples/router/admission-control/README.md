<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Router Overload Protection

Run the [CPU overload-protection test](https://github.com/ai-dynamo/dynamo/blob/main/tests/router/test_overload_protection.py)
to watch router admission control protect workers as backend capacity drops.
The test owns server startup, the client workload, and cleanup. Its
[enabled](https://github.com/ai-dynamo/dynamo/blob/main/tests/router/configs/overload_protection/on.yaml)
and [disabled](https://github.com/ai-dynamo/dynamo/blob/main/tests/router/configs/overload_protection/off.yaml)
policies live alongside the tests.

Use Linux with NVIDIA Dynamo [built from this checkout](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/advanced-customizations/building-from-source.md)
and its [test dependencies](https://github.com/ai-dynamo/dynamo/blob/main/container/deps/requirements.test.txt).
No GPU, model weights, etcd, or NATS service is needed. The first run may download
the tokenizer and configuration for `Qwen/Qwen3-0.6B`.

## Run

From the **repository root**, run with overload protection enabled:

```bash
python3 -m pytest tests/router/test_overload_protection.py -k 'on-3' -q
```

To compare with protection disabled:

```bash
python3 -m pytest tests/router/test_overload_protection.py -k 'off-3' -q
```

Interactive runs show live worker-arrival rates and occupancy bars every
0.5 seconds; no `-s` is needed. Each phase retains a panel with average rates
and occupancy at the end of its measurement window.

## What to Observe

The test starts one frontend with an embedded KV router and three CPU Mocker
workers, using reserved ports and an isolated file-discovery directory with TCP
requests. It sends the same 90 requests/s in both modes, with 512 input tokens
and one output token per request. Prefix caching is disabled.

It measures three healthy workers, stops worker 3, then restarts worker 3 at
half speed with the same capacity settings. Each phase measures five seconds
after three seconds of settling. With protection enabled, expect roughly:

| Phase | Worker 1 | Worker 2 | Worker 3 |
| --- | --- | --- | --- |
| Healthy | 20 requests/s | 20 requests/s | 20 requests/s |
| Worker 3 stopped | 20 requests/s | 20 requests/s | 0 |
| Worker 3 at half speed | 20 requests/s | 20 requests/s | 10 requests/s |

Each bar shows the router's outstanding uncached prefill tokens for that worker.
**100% is 8,704 tokens**, or 17 requests: the 8,192-token batch budget plus one
512-token request allowed by the current admission boundary check. Both modes
and all workers use that same reference, including worker 3 after slowing down.

`#` fills the bar up to 100%; `+` shows excess work to its right. A `>` marks
clipping beyond 200%, while the percentage and token counts remain exact.
The bars include work in transit, waiting, or executing. They do not directly
measure the engine queue: the four-sequence engine limit can cause waiting
below the 17-request reference.

With protection enabled, workers stay near 100% and the separate **Router
waiting** count grows as excess work stays upstream. With protection disabled,
total worker arrivals stay near 90 requests/s and worker backlog grows beyond
100%. The default cost function can change individual workers' arrival shares.
Arrival rates count requests received by workers, not completed responses.

## Run Both Cases in CI

```bash
python3 -m pytest tests/router/test_overload_protection.py -n 2 -q
```

The test asserts dispatch rates, queue growth, worker backlog, and cleanup.
Parallel, CI, and redirected runs capture diagnostics for failures.
