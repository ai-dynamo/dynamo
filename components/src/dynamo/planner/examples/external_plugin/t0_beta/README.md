<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# t0-beta traffic prediction plugin

This external PREDICT plugin uses
[t0-beta](https://huggingface.co/theforecastingcompany/t0-beta) to forecast the
next observation window's request count, average input sequence length, and
average output sequence length. Dynamo's existing proposal, reconciliation,
and constraint stages turn those predictions into compute allocations.

The example runs the open weights locally with
[`tfc-t0`](https://github.com/theforecastingcompany/tfc-t0). It does not require
a Retrocast API account. `tfc-t0>=0.5.0` is required because earlier versions
use a normalization convention incompatible with t0-beta.

## Start the plugin

Use a Dynamo environment built from this checkout, with the planner and its
generated gRPC modules installed. From the repository root:

```bash
uv pip install -r components/src/dynamo/planner/examples/external_plugin/t0_beta/requirements.txt
uv run python -m dynamo.planner.examples.external_plugin.t0_beta.runner \
    --listen 127.0.0.1:9099 \
    --model theforecastingcompany/t0-beta \
    --device cpu \
    --interval-seconds 60 \
    --context-length 512 \
    --min-history 32 \
    --quantile 0.9
```

The model downloads and loads before the server starts listening. Wait for
the listening message before starting the planner. The weights are public;
the first start needs Hugging Face access and space for the model cache.
CPU is the default. Use `--device cuda` only when the plugin has an allocated
GPU; reserve its resources separately from the inference workers.

## Register with the planner

Merge this fragment into an existing, working planner YAML configuration.
Run the plugin and planner on the same host or in the same Kubernetes Pod,
where they share the loopback interface:

```yaml
optimization_target: sla
enable_throughput_scaling: true
throughput_adjustment_interval_seconds: 60
plugin_registration:
  auth:
    trusted_sources: [allow_unauthenticated]
  transport:
    allow_insecure_grpc: true
    request_timeout_seconds: 5
scheduling:
  scale_interval_seconds: 5
  tick_max_duration_seconds: 30
  gateway:
    enabled: false
  external_plugins:
    - plugin_id: t0-beta-predict
      plugin_type: predict
      priority: 10
      endpoint: grpc://127.0.0.1:9099
      protocol_version: "1.0"
      execution_interval_seconds: 60
      observation_window_seconds: 60
      hold_policy: ACCEPT_WHEN_IDLE
      needs: [observations.traffic]
      requires_produced_fields: [observations.traffic]
```

> [!WARNING]
> This is a local development configuration: it explicitly permits
> unauthenticated registration and plaintext gRPC. The current transport does
> not provide TLS. Keep the plugin bound to loopback and the registration
> gateway disabled. An exposed deployment requires an appropriate transport
> security boundary. For authenticated registration, use `static_secret` and
> render mounted Secret values into `auth.static_secrets` and the plugin's
> `auth_token`; never commit credentials in configuration.

Static registration does not require a registration gateway. Authentication
configuration is still required: an empty `trusted_sources` list fails
closed. Confirm startup logs show successful registration; a failed external
registration is logged without stopping the planner.

The three 60-second settings must agree with `--interval-seconds`: throughput
adjustment interval, plugin execution interval, and observation window.
Request count is a count over that window, not requests per second.
`ACCEPT_WHEN_IDLE` avoids reusing a forecast on intervening ticks.

This example requires the built-in predictor's fallback priority of 100
included in this change. PREDICT uses first-writer-wins, so priority 10 must
run before the built-in predictor. An unmodified v1.5.0 checkout gives the
built-in predictor priority 0 and does not support this configuration as
intended.

## Prediction and fallback behavior

The plugin retains up to 512 regularly spaced observations in memory and
begins forecasting after 32 valid observations, approximately 32 minutes at
the default cadence. A restart loses that history. Each forecast batches
the three signals as independent series and selects the configured quantile
for one future interval. The default 0.9 quantile provides a configurable
demand estimate; it is not a capacity or SLO guarantee.
Idle windows retain the last known mean token lengths; initial idle traffic
without a known request shape uses the built-in fallback.

During warmup, invalid observations, or an already busy model, the plugin
returns no prediction and Dynamo uses its built-in predictor. Inference
failures return a gRPC error, which Dynamo logs and counts toward its circuit
breaker while continuing with the built-in predictor. A missed sample or
invalid observation clears history and restarts warmup. Successful responses
leave `final` false so the built-in predictor
continues updating its history and supplying KV-cache and acceptance
metadata. RPC timeout or plugin unavailability also leaves the built-in
fallback available.

Measure inference latency on the chosen device. The example gives each RPC
five seconds; increase `request_timeout_seconds` only with a corresponding
pipeline deadline and cadence budget. Cancelling an RPC does not necessarily
stop an inference already running. Size the plugin's CPU, memory, and any GPU
allocation so forecasting does not contend with serving.
Shutdown waits for active inference. A process supervisor should enforce a
termination grace period if native model execution becomes unresponsive.

## Evaluate before enabling scaling

Replay representative traffic against the built-in predictor and this plugin
with identical SLOs, GPU budgets, and startup delays. Compare request-count
and token-length forecast error, quantile coverage, GPU-seconds, scaling
churn, TTFT, and ITL. Include the forecasting process's resource cost and
exercise warmup, missing observations, inference errors, and timeouts.
The one-window horizon must cover the allocation lead time for the workload;
this example makes no claim of measured cost savings or SLO improvement.
