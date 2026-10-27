---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Planner Configuration (PlannerConfig)
subtitle: Field reference for the PlannerConfig JSON/YAML object consumed by the Dynamo Planner service.
---

`PlannerConfig` is the configuration object for the Dynamo Planner. The Planner service parses it with [Pydantic](https://docs.pydantic.dev/), so every field, type, default, and constraint on this page comes from the [`PlannerConfig` model](https://github.com/ai-dynamo/dynamo/blob/main/components/src/dynamo/planner/config/planner_config.py). For the deployment workflow and scaling-mode concepts, see the [Planner Guide](../../developer-guide/knowledge-base/modular-components/planner/planner-guide.md); for the autoscaler overview, see the [Planner overview](../../developer-guide/knowledge-base/modular-components/planner/overview.md).

## How the config is loaded

The Planner service loads `PlannerConfig` from a single required `--config` argument — a path to a `.json`, `.yaml`, or `.yml` file, or an inline JSON string. The service auto-detects which, loads it, and validates it against the `PlannerConfig` model. A value that is neither a readable file nor valid JSON fails at startup. How you supply it depends on where you run the Planner.

<Card>
<Tabs>
  <Tab title="Kubernetes">
    Set the object under `spec.features.planner` in a [DynamoGraphDeploymentRequest](../kubernetes-api/dynamo-graph-deployment-request.mdx) (DGDR). DGDR passes it through without field-level validation, mounts it as a `planner-config-*` ConfigMap, and launches the Planner with `--config` pointing at the mounted file. The Planner validates it at startup. Every field below can be set here.

    ```yaml
    spec:
      features:
        planner:
          mode: disagg
          backend: vllm
          # optimization_target defaults to "throughput" — works with no further config
    ```
  </Tab>
  <Tab title="Local">
    Run the Planner module directly and pass `--config` yourself, as either a file path or an inline JSON string:

    ```bash
    # From a YAML or JSON file
    python -m dynamo.planner --config planner_config.yaml

    # Or inline JSON
    python -m dynamo.planner --config '{"mode": "disagg", "backend": "vllm"}'
    ```
  </Tab>
</Tabs>
</Card>

<Note>
  Several Prometheus fields default from environment variables and are excluded when the config is serialized back out. Set them either in the config object or through the environment variable noted on each field. See [Environment variables](#environment-variables).
</Note>

## Core settings

<ParamField path="mode" type="string" default="disagg">
  Planner operating mode, matching the deployment topology.

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>disagg</Badge> <Badge intent="note" minimal>prefill</Badge> <Badge intent="note" minimal>decode</Badge> <Badge intent="note" minimal>agg</Badge></span>
</ParamField>

<ParamField path="backend" type="string" default="vllm">
  Inference backend the Planner scales.

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>vllm</Badge> <Badge intent="note" minimal>sglang</Badge> <Badge intent="note" minimal>trtllm</Badge> <Badge intent="note" minimal>mocker</Badge></span>
</ParamField>

<ParamField path="environment" type="string" default="kubernetes">
  Runtime environment that determines how the Planner applies scaling actions.

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>kubernetes</Badge> <Badge intent="note" minimal>virtual</Badge> <Badge intent="note" minimal>global-planner</Badge></span>
</ParamField>

<ParamField path="namespace" type="string" default="env DYN_NAMESPACE, else dynamo">
  Dynamo namespace of the deployment the Planner manages. Defaults from `DYN_NAMESPACE`, which the operator injects as `{k8s_namespace}-{dgd_name}`. Excluded from serialized output.
</ParamField>

<ParamField path="model_name" type="string" default="null">
  Optional model name override. Auto-detected from the deployment when unset.
</ParamField>

<ParamField path="global_planner_namespace" type="string" default="null">
  Namespace where the GlobalPlanner runs. **Required** when `environment` is `global-planner`. See the [Global Planner Guide](../../developer-guide/knowledge-base/modular-components/planner/global-planner-guide.md).
</ParamField>

<ParamField path="log_dir" type="string" default="null">
  Optional directory for Planner log output.
</ParamField>

## Optimization target

<ParamField path="optimization_target" type="string" default="throughput">
  Scaling strategy. `throughput` and `latency` use static thresholds on queue depth and KV cache utilization — no SLA targets or profiling required. `load` uses user-defined prefill queue-token and decode KV-utilization thresholds. `sla` uses the Planner engine-query layer and AIC core performance model to target specific `ttft_ms` / `itl_ms` values.

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>throughput</Badge> <Badge intent="note" minimal>latency</Badge> <Badge intent="note" minimal>load</Badge> <Badge intent="note" minimal>sla</Badge></span>
</ParamField>

When `optimization_target` is `throughput`, `latency`, or `load`, the Planner forces load-based scaling on and throughput-based scaling off, and it ignores `ttft_ms` / `itl_ms`. The two scaling-mode flags below apply only when `optimization_target` is `sla`.

<ParamField path="enable_throughput_scaling" type="boolean" default="true">
  Enable predictive, traffic-based scaling. Only honored when `optimization_target` is `sla`.
</ParamField>

<ParamField path="enable_load_scaling" type="boolean" default="false">
  Enable reactive, load-based scaling. Only honored when `optimization_target` is `sla`. At least one scaling mode must be enabled.
</ParamField>

<ParamField path="ttft_ms" type="number" default="500.0">
  Time To First Token SLA target, in milliseconds. Also accepts the alias `ttft`. Must be greater than 0. Used only under `optimization_target: sla`.
</ParamField>

<ParamField path="itl_ms" type="number" default="50.0">
  Inter-Token Latency SLA target, in milliseconds. Also accepts the alias `itl`. Used only under `optimization_target: sla`.
</ParamField>

## Performance model and pre-deployment sweeping

<ParamField path="pre_deployment_sweeping_mode" type="string" default="rapid">
  How to generate optional bootstrap performance data. `none` skips bootstrap data; `rapid` uses AIConfigurator to simulate engine performance (~30s); `thorough` measures on real GPUs (several hours).

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>none</Badge> <Badge intent="note" minimal>rapid</Badge> <Badge intent="note" minimal>thorough</Badge></span>
</ParamField>

<ParamField path="profile_results_dir" type="string" default="profiling_results">
  Directory holding profiler-generated performance data (npz or JSON), used to bootstrap or tune the performance model when the `get_perf_metrics` endpoint is unavailable.
</ParamField>

<ParamField path="ais_perf_model" type="AISPerfModelSpec" default="null">
  Role-indexed AISimulate forward-pass configurations. Constructs each model through
  `aisimulate_core.sdk.RustForwardPassPerfModel.best_available(config)` without an
  interpolation sweep. Configure model identity and estimator controls through
  `ais_perf_model.roles`; the retired `aic_perf_model` and `hf_id` / pick shape
  are rejected.
</ParamField>

<Indent>
  <ParamField path="roles" type="object" required={true}>
    Maps deployment roles to complete upstream `ForwardPassPerfModelConfig` objects.
    Supply `prefill` and `decode` for `disagg`, the matching single role for `prefill`
    or `decode`, and `aggregated` for `agg`. Each key must match `worker_type`.
  </ParamField>
  <ParamField path="roles.&lt;role&gt;.model / system / backend" type="string" required={true}>
    Model identifier, AISimulate system identifier, and inference backend
    (`vllm`, `sglang`, or `trtllm`). Optional `backend_version`, `tp`, `pp`,
    `attention_dp`, MoE axes, quantization, KV block size, and speculation fields
    use the upstream schema. Published worker capabilities must agree with the
    configured identity.
  </ParamField>
  <ParamField path="roles.&lt;role&gt;.estimation_mode" type="string" default="auto">
    `auto` searches `op_level`, `fpm_interpolation`, then `fpm_regression`.
    Explicit modes select the named estimator.
  </ParamField>
  <ParamField path="roles.&lt;role&gt;.fallback_policy" type="string" default="deny">
    `deny` rejects failure of an explicitly selected mode. `allow` tries the other modes in
    `op_level`, `fpm_interpolation`, `fpm_regression` priority order; `regression` permits a regression fallback.
    Auto selection searches the full priority order even with `deny`.
  </ParamField>
  <ParamField path="roles.&lt;role&gt;.systems_paths" type="list[string]" default="[]">
    Ordered performance-data roots. The SDK resolves its packaged root when unset.
    `database_mode` and `transfer_policy` are also passed through unchanged.
  </ParamField>
  <ParamField path="roles.&lt;role&gt;.estimator_config" type="object" default="{}">
    Complete upstream estimator controls, including independent regression and
    correction sampling, fit settings, feature weights, and correction bounds.
    Explicit settings override Planner sampling defaults. A legacy total bucket
    count of 16 maps to `bins_per_axis: [4, 4]`. Unknown fields are rejected by
    AISimulate. See the [AISimulate core API](https://github.com/ai-dynamo/aisimulate/blob/main/docs/core-api.md).
  </ParamField>
</Indent>

<ParamField path="aic_interpolation" type="AICInterpolationSpec" default="null">
  AIConfigurator interpolation spec. Populated by the profiler in rapid mode and written onto the Planner ConfigMap; you do not normally set this by hand. When present, the Planner runs the AIC sweep in-process at bootstrap to seed the performance model. See [AICInterpolationSpec](#aicinterpolationspec).
</ParamField>

## GPU budget

<ParamField path="max_gpu_budget" type="integer" default="8">
  Maximum total GPUs the Planner may allocate across worker types.
</ParamField>

<ParamField path="min_gpu_budget" type="integer" default="-1">
  Per-DGD GPU floor enforced by the local Planner. `-1` disables it. When set with `max_gpu_budget` such that `min == max`, the Planner pins the per-DGD total and only redistributes replicas between prefill and decode.
</ParamField>

<ParamField path="min_endpoint" type="integer" default="1">
  Minimum engine endpoints (replicas) for aggregated deployments. In disaggregated deployments, this value applies to both prefill and decode unless a role-specific value is set. In prefill-only or decode-only deployments, it supplies the active role when the corresponding role-specific value is unset. Must be nonnegative; use `0` for scale-to-zero compatibility.
</ParamField>

<ParamField path="prefill_min_endpoint" type="integer" default="null">
  Minimum prefill endpoints in `disagg` and `prefill` modes. When set, replaces the prefill value supplied by `min_endpoint`. Must be at least `1`.
</ParamField>

<ParamField path="decode_min_endpoint" type="integer" default="null">
  Minimum decode endpoints in `disagg` and `decode` modes. When set, replaces the decode value supplied by `min_endpoint`. Must be at least `1`.
</ParamField>

The Planner does not have per-component maximum endpoint fields. `max_gpu_budget`, the power budget when enabled, Global Planner allocation, and cluster capacity continue to bound scale-up. At startup and for runtime updates, the Planner rejects minimum endpoint combinations that cannot fit the configured GPU or power budget. Send `min_gpu_budget` and `max_gpu_budget` with the endpoint fields to `PATCH /v1/min-endpoints` to update them atomically at runtime. Set either budget to `-1` to disable that bound. When both budgets are enabled, `min_gpu_budget` must not exceed `max_gpu_budget`. The response from `GET` or `PATCH` includes both budgets.

<ParamField path="decode_engine_num_gpu" type="integer" default="null">
  GPUs per decode engine replica. Auto-detected from the deployment when unset.
</ParamField>

<ParamField path="prefill_engine_num_gpu" type="integer" default="null">
  GPUs per prefill engine replica. Auto-detected from the deployment when unset.
</ParamField>

## Throughput-based scaling

<ParamField path="throughput_adjustment_interval_seconds" type="integer" default="180">
  Seconds between throughput-based scaling decisions. Also accepts the alias `throughput_adjustment_interval`.
</ParamField>

<ParamField path="max_throughput_scaling_replicas" type="integer" default="8">
  Maximum replica-count change per component from one throughput observation. The Planner caps and stores the throughput lower bound, so load-scaling ticks cannot repeatedly apply the same uncapped prediction. GPU and power budgets can reduce the change further. At the nominal minimum GPU budget, scale-down-only requests hold. At a full GPU budget, two scale-up requests hold; an opposing prefill/decode proposal applies only the largest atomic pair that fits the budget. Endpoint and out-of-band budget recovery take precedence over this limit. Must be greater than `0`.
</ParamField>

<ParamField path="throughput_metrics_source" type="string" default="frontend">
  Prometheus traffic source. `frontend` reads `dynamo_frontend_*` metrics from the public Frontend; `router` reads `dynamo_component_router_*` from a LocalRouter (use for a pool-local Planner in GlobalPlanner deployments).

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>frontend</Badge> <Badge intent="note" minimal>router</Badge></span>
</ParamField>

## Load-based scaling

<ParamField path="load_adjustment_interval_seconds" type="integer" default="5">
  Seconds between FPM tuning updates and load-based scaling decisions. Even when only throughput scaling is enabled, live FPM observations feed the performance model at this interval. Also accepts the alias `load_adjustment_interval`. Must be greater than 0 and, when both scaling modes are on, shorter than `throughput_adjustment_interval_seconds`.
</ParamField>

<ParamField path="max_num_fpm_samples" type="integer" default="64">
  Maximum retained ForwardPassMetrics observations for online tuning and regression fallback.
</ParamField>

<ParamField path="fpm_sample_bucket_size" type="integer" default="16">
  Number of buckets for observation retirement. Must be a perfect square.
</ParamField>

<ParamField path="load_scaling_down_sensitivity" type="integer" default="80">
  Scale-down sensitivity from 0 to 100 (0 = never scale down, 100 = aggressive).
</ParamField>

<ParamField path="load_min_observations" type="integer" default="5">
  Minimum observations before load-based scaling decisions begin (cold-start threshold).
</ParamField>

<ParamField path="prefill_scale_up_queue_tokens" type="integer" default="null">
  Prefill queue-token count that triggers scale-up. Required (with the scale-down field) when `optimization_target` is `load` and `mode` includes prefill. Must be greater than 0 and greater than `prefill_scale_down_queue_tokens`.
</ParamField>

<ParamField path="prefill_scale_down_queue_tokens" type="integer" default="null">
  Prefill queue-token count that allows scale-down. Required alongside `prefill_scale_up_queue_tokens`.
</ParamField>

<ParamField path="decode_scale_up_kv_rate" type="number" default="null">
  Decode KV-utilization percentage (0–100) that triggers scale-up. Required (with the scale-down field) when `optimization_target` is `load` and `mode` includes decode. Must be greater than `decode_scale_down_kv_rate`.
</ParamField>

<ParamField path="decode_scale_down_kv_rate" type="number" default="null">
  Decode KV-utilization percentage (0–100) that allows scale-down. Required alongside `decode_scale_up_kv_rate`.
</ParamField>

<ParamField path="speculative_nextn" type="integer" default="0">
  Manual fallback speculative-decoding depth. A worker's published `runtime_config.runtime_data.spec_decode.nextn` takes precedence when present. Must be 0 or greater.
</ParamField>

## Load prediction

<ParamField path="load_predictor" type="string" default="arima">
  Prediction method for request count, ISL, and OSL.

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>constant</Badge> <Badge intent="note" minimal>arima</Badge> <Badge intent="note" minimal>kalman</Badge> <Badge intent="note" minimal>prophet</Badge></span>
</ParamField>

<ParamField path="load_predictor_log1p" type="boolean" default="false">
  Apply a log1p transform to predicted request count, ISL, and OSL.
</ParamField>

<ParamField path="prophet_window_size" type="integer" default="50">
  Window size (seconds) for the Prophet predictor.
</ParamField>

<ParamField path="load_predictor_warmup_trace" type="string" default="null">
  Path to a warmup trace file for bootstrapping predictions.
</ParamField>

<ParamField path="kalman_q_level" type="number" default="1.0">
  Kalman process noise for the level component.
</ParamField>

<ParamField path="kalman_q_trend" type="number" default="0.1">
  Kalman process noise for the trend component.
</ParamField>

<ParamField path="kalman_r" type="number" default="10.0">
  Kalman measurement noise.
</ParamField>

<ParamField path="kalman_min_points" type="integer" default="5">
  Minimum data points before Kalman predictions activate.
</ParamField>

## Prometheus metrics

These fields default from environment variables and are excluded from serialized output. Set them in the config object or through the noted environment variable.

<ParamField path="metric_pulling_prometheus_endpoint" type="string" default="env PROMETHEUS_ENDPOINT">
  Prometheus endpoint the Planner queries for traffic metrics. Defaults to `PROMETHEUS_ENDPOINT`, else `http://prometheus-kube-prometheus-prometheus.monitoring.svc.cluster.local:9090`.
</ParamField>

<ParamField path="metric_pulling_prometheus_token" type="string" default="env PROMETHEUS_TOKEN">
  Optional bearer token sent as `Authorization: Bearer <token>` on every PromQL request. Read once at startup.
</ParamField>

<ParamField path="metric_pulling_prometheus_token_file" type="string" default="env PROMETHEUS_TOKEN_FILE">
  Path to a file containing a bearer token, re-read before every request so rotated tokens are picked up without a restart.
</ParamField>

<ParamField path="metric_pulling_prometheus_ssl_verify" type="boolean" default="false">
  Verify the upstream Prometheus TLS certificate. Defaults from `PROMETHEUS_SSL_VERIFY` (`1`/`true`/`yes` enable it). Pair with a CA bundle for a private CA.
</ParamField>

<ParamField path="metric_pulling_prometheus_extra_query_params" type="object" default="env PROMETHEUS_EXTRA_QUERY_PARAMS">
  Fixed key/value pairs appended as URL query parameters on every PromQL request. Set `PROMETHEUS_EXTRA_QUERY_PARAMS` as a URL query string, for example `namespace=my-ns&tenant=foo`.
</ParamField>

<ParamField path="metric_pulling_prometheus_ca_bundle" type="string" default="env PROMETHEUS_CA_BUNDLE">
  Path to a CA bundle for verifying the upstream Prometheus TLS certificate. When set, the bundle is used for verification regardless of `metric_pulling_prometheus_ssl_verify`. Must point to an existing file.
</ParamField>

<ParamField path="metric_pulling_prometheus_request_timeout_seconds" type="number" default="10.0">
  Connection and read inactivity timeout in seconds for each Prometheus API request. Defaults from `DYN_PLANNER_PROMETHEUS_REQUEST_TIMEOUT_SECONDS` and must be greater than `0`. This is not a total wall-clock deadline: a response that continues delivering data can run longer. The Planner does not retry failed requests within the same collection cycle.
</ParamField>

<ParamField path="metric_reporting_prometheus_port" type="integer" default="0">
  Port on which the Planner exposes its own `dynamo_planner_*` metrics. Defaults from `PLANNER_PROMETHEUS_PORT`. `0` disables.
</ParamField>

## Advisory mode

<ParamField path="advisory" type="boolean" default="false">
  Suggestion-only mode. The Planner computes, logs, exports, and reports recommended replica counts and [retune](#retune-experimental-simulation-only) decisions without executing any of them or changing the deployment. Use it to evaluate a new configuration or validate SLA targets against production traffic. The runtime configuration APIs, such as `PATCH /v1/min-endpoints`, still change the Planner's own settings.

  Planner-in-the-loop replay honors it too: an advisory replay keeps the simulated deployment at its starting replicas and router parameters, reports the recommendations in `planner.recommended_scaling_events`, and records a retune decision that would apply with outcome `advisory`. See [Replay in Advisory Mode](../../developer-guide/knowledge-base/modular-components/planner/aisimulate-planner-replay.mdx#replay-in-advisory-mode).
</ParamField>

## Diagnostics and reporting

<ParamField path="report_interval_hours" type="number" default="24.0">
  Generate an HTML diagnostics report every N hours (simulated time). Set to `null` to disable. Must be a positive finite number or `null`.
</ParamField>

<ParamField path="report_output_dir" type="string" default="./planner_reports">
  Directory for HTML diagnostics reports.
</ParamField>

<ParamField path="report_filename" type="string" default="null">
  Fixed report filename written under `report_output_dir`. When unset, a timestamped name is used.
</ParamField>

<ParamField path="report_write_gzip_log" type="boolean" default="true">
  Write a compressed JSONL diagnostics log (`.log.jsonl.gz`) next to each HTML report.
</ParamField>

<ParamField path="live_dashboard_port" type="integer" default="8080">
  Port for the live diagnostics dashboard HTTP server. `0` disables. When enabled, visit `http://host:port/` for a real-time Plotly report.
</ParamField>

## Runtime configuration

<ParamField path="control_api_port" type="integer" default="9086">
  Port for the runtime minimum-endpoint API. The server listens only on `127.0.0.1`; `0` disables it. Updates are process-local, are not persisted to the Planner ConfigMap, and take effect on the next planner tick.
</ParamField>

### Runtime minimum endpoint API

The Planner exposes `GET /v1/min-endpoints` and partial `PATCH /v1/min-endpoints`. The request and response fields depend on the active mode:

| Mode | Runtime fields |
| --- | --- |
| `disagg` | `prefill_min_endpoint`, `decode_min_endpoint` |
| `prefill` | `prefill_min_endpoint` |
| `decode` | `decode_min_endpoint` |
| `agg` | `min_endpoint` |

The API has no authentication and is not exposed by a Kubernetes Service. For Kubernetes deployments, port-forward directly to the Planner pod:

```bash
kubectl port-forward pod/<planner-pod> 9086:9086
curl http://127.0.0.1:9086/v1/min-endpoints
curl --request PATCH http://127.0.0.1:9086/v1/min-endpoints \
  --header 'Content-Type: application/json' \
  --data '{"prefill_min_endpoint": 2}'
```

PATCH is atomic. Malformed JSON, unknown fields, and empty patches return HTTP `400`. `null`, a negative `min_endpoint`, component-specific minimums below `1`, fields that are inactive for the current mode, and minimums that exceed the configured GPU or power budget return HTTP `422` without changing the active configuration.

### System-port reconfiguration API

<ParamField path="enable_runtime_reconfiguration" type="boolean" default="false">
  Enable the Planner control API on the existing `DYN_SYSTEM_PORT` listener.
  The runtime must have a system listener enabled. Kubernetes Planner deployments
  configure this port as `9090`.
</ParamField>

<ParamField path="runtime_warmup_trace_dir" type="string" default="null">
  Directory containing permitted runtime warmup traces. Required when submitting
  `warmup_trace`; paths must remain inside this directory, and symlinks are rejected.
  This setting cannot be changed through the runtime API.
</ParamField>

<Warning>
  This API changes scaling behavior and has no application authentication. Restrict
  access to the system listener to trusted controllers and operators before enabling it.
  The existing localhost `9086` API remains available with its original configuration.
</Warning>

| Method and path | Response |
| --- | --- |
| `GET /engine/planner/config` | Applied settings, `instance_id`, `revision`, and `supported_fields`. |
| `PATCH /engine/planner/config` | `202` with an operation ID after validation and admission. |
| `GET /engine/planner/operations/{request_id}` | Operation state: `preparing`, `warming`, `applied`, or `failed`. |

The supported configuration groups are:

| Group | Fields |
| --- | --- |
| Endpoint floors and GPU budgets | Mode-specific minimum endpoint fields from the table above, `min_gpu_budget`, `max_gpu_budget`. |
| SLA targets | `ttft_ms`, `itl_ms`. |
| Scaling policy | `max_throughput_scaling_replicas`, `load_scaling_down_sensitivity`, `prefill_scale_up_queue_tokens`, `prefill_scale_down_queue_tokens`, `decode_scale_up_kv_rate`, `decode_scale_down_kv_rate`. |
| Load prediction | `load_predictor`, `load_predictor_log1p`, `prophet_window_size`, `kalman_q_level`, `kalman_q_trend`, `kalman_r`, `kalman_min_points`. |

Fields retain their normal mode-specific meaning. For example, the explicit queue/KV
thresholds are used by `optimization_target: load`. Predictor replacement requires
`optimization_target: sla` with throughput scaling enabled. Scheduling intervals,
AIS performance models, deployment topology, and other fields are rejected by this API.

Use the instance ID and revision returned by GET when submitting an update:

```json
{
  "instance_id": "<instance_id from GET>",
  "expected_revision": 0,
  "request_id": "sweep-42",
  "config": {
    "ttft_ms": 800,
    "load_predictor": "kalman",
    "kalman_min_points": 5
  },
  "warmup_trace": "recent-traffic.jsonl"
}
```

`config` is a nonempty partial update. A predictor change requires either `warmup_trace`
or explicit `cold_start: true`. The trace must be a single regular Dynamo/Mooncake JSONL
or JSONL.GZ file inside `runtime_warmup_trace_dir`. Relative paths are resolved from
that directory; absolute paths must also be inside it. Runtime warmup accepts at most
16 MiB of stored data and 16 MiB after decompression, and at most 4,096 aggregation
intervals (including idle gaps). Directories and globs remain supported by the startup
`load_predictor_warmup_trace` option but are rejected by runtime reconfiguration.
Observations use the unchanged throughput adjustment interval. Warmup must provide enough
effective samples for request count, ISL, and OSL and complete a successful forecast.
SLA/budget-only updates do not require warmup. Supplying a trace explicitly also allows rewarming an unchanged predictor.

The active configuration continues serving while a candidate is prepared. Preparation
has a 120-second deadline and only one preparation can run at a time. If a background
fit outlives its deadline, its result cannot activate and further preparations are
rejected until that work exits; updates that do not replace the predictor remain available.
Invalid data, failed fitting, or a changed revision leave the active settings unchanged.
Successful activation occurs between decisions and increments the revision. Updates preserve
the existing prediction and scaling cadence, including changes to SLA targets,
`max_throughput_scaling_replicas`, and predictor settings. The next scheduled throughput
decision uses the active settings and the full trailing traffic window; predictor replacement
does not delay that decision. Performance models, worker state, and already-admitted
scaling operations are retained. `applied` means the
configuration is active; it does not mean the deployment has finished scaling.

A successful write through the legacy minimum-endpoint API also changes the revision
reported here. If it happens during preparation, the pending update fails with
`error_type: conflict`.
Updates are process-local and are not written to the Planner ConfigMap. A restart creates
a new `instance_id` and restores startup settings. Address the intended Planner instance
directly rather than distributing a conditional write across replicas.

Retry the identical request with the same `request_id` to retrieve its retained operation
without applying it twice. The last 32 operations are retained in memory. Reusing an ID
for different content returns `409`; an evicted operation returns `404`, so inspect the applied revision before submitting another update.

Malformed JSON returns `400`; invalid or unsupported updates return `422`; instance,
revision, idempotency, and concurrent-update conflicts return `409`; unavailable control
returns `503`. Requests exceeding 64 KiB return `413`. Preparation failures after `202`
are reported by the operation's `state`, `error`, and `error_type` fields.

## Scheduling and plugin pipeline

The Planner runs the builtin plugin pipeline by default. `scheduling` controls pipeline cadence; `plugin_registration` controls how plugins register, authenticate, and communicate. For the pipeline model, see the [Planner Guide](../../developer-guide/knowledge-base/modular-components/planner/planner-guide.md#scheduling--plugin-pipeline).

<ParamField path="scheduling" type="SchedulingConfig">
  Plugin-pipeline scheduling. Rejects unknown fields.
</ParamField>

<Indent>
  <ParamField path="scale_interval_seconds" type="number" default="gcd of enabled builtin intervals">
    Base pipeline cadence. The pipeline wakes once per interval; each plugin's `execution_interval_seconds` decides whether it fires. When unset, computed as the gcd of `load_adjustment_interval_seconds` and, when throughput scaling is enabled, `throughput_adjustment_interval_seconds`. Must be greater than 0 and evenly divide both adjustment intervals.
  </ParamField>
  <ParamField path="tick_max_duration_seconds" type="number" default="30.0">
    Outer wall-clock deadline wrapping the full 4-stage pipeline. Exceeding it aborts the tick; the next tick runs from a clean state. Replay does not apply it: a replayed tick runs to completion, however long it takes on the host. Must be greater than 0.
  </ParamField>
  <ParamField path="external_plugins" type="[]ExternalPluginEntry" default="[]">
    Static external-plugin registrations applied at startup. Per-entry failures are logged but do not crash the Planner. See [ExternalPluginEntry](#externalpluginentry).
  </ParamField>
  <ParamField path="gateway" type="GatewayConfig">
    gRPC registration gateway for self-registering plugins. Disabled by default.
  </ParamField>
  <Indent>
    <ParamField path="enabled" type="boolean" default="false">
      Open the gRPC gateway at `listen`. Not needed for static `external_plugins`.
    </ParamField>
    <ParamField path="listen" type="string" default="unix:///var/run/dynamo/planner/registry.sock">
      Bind address. A `unix:` socket path for in-Pod registration, or `host:port` for TCP.
    </ParamField>
    <ParamField path="allow_insecure" type="boolean" default="false">
      Permit binding a plaintext (no-TLS) gateway on a TCP listen. Fails closed by default because the gateway receives plugins' shared-secret tokens. `unix:` listens are always allowed.
    </ParamField>
  </Indent>
</Indent>

<ParamField path="plugin_registration" type="PluginRegistrationConfig">
  Plugin registry configuration — auth, transport, heartbeat, and in-process plugins. Rejects unknown fields.
</ParamField>

<Indent>
  <ParamField path="auth" type="AuthConfig">
    Registry authentication. In the Planner process, an empty `trusted_sources` activates a legacy compatibility fallback to unauthenticated access and logs a warning. This fallback is **DEV ONLY**; do not rely on it as an intentional configuration. Select `allow_unauthenticated` explicitly for development. Production deployments must configure `static_secret`.
  </ParamField>
  <Indent>
    <ParamField path="trusted_sources" type="[]string" default="[]">
      Accepted auth sources. Use `['allow_unauthenticated']` for development, or `['static_secret']` with `static_secrets` for production.

      <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>static_secret</Badge> <Badge intent="note" minimal>allow_unauthenticated</Badge></span>
    </ParamField>
    <ParamField path="static_secrets" type="object" default="{}">
      Map of `secret_value` to `subject_label`. Populate from a mounted Kubernetes Secret rather than hard-coding in the ConfigMap.
    </ParamField>
  </Indent>
  <ParamField path="transport" type="TransportConfig">
    Outbound gRPC transport settings for calling plugins.
  </ParamField>
  <Indent>
    <ParamField path="allow_insecure_grpc" type="boolean" default="false">
      Allow plaintext `grpc://` plugin endpoints. Required to use any `grpc://` endpoint; mTLS support is not yet shipped.
    </ParamField>
    <ParamField path="request_timeout_seconds" type="number" default="5.0">
      Per-RPC timeout applied to every plugin call. Replay does not apply it to in-process plugins: a replayed tick runs to completion, however long it takes on the host. Must be greater than 0.
    </ParamField>
    <ParamField path="keepalive_time_ms" type="integer" default="30000">
      gRPC keepalive time, in milliseconds.
    </ParamField>
    <ParamField path="max_message_size_bytes" type="integer" default="10485760">
      Maximum gRPC message size (10 MB).
    </ParamField>
  </Indent>
  <ParamField path="protocol_version_min" type="string" default="1.0">
    Minimum accepted plugin protocol version.
  </ParamField>
  <ParamField path="protocol_version_max" type="string" default="1.0">
    Maximum accepted plugin protocol version.
  </ParamField>
  <ParamField path="heartbeat_timeout_seconds" type="number" default="15.0">
    Reserved liveness setting. Heartbeats update registry timestamps, but no monitor currently applies this timeout or evicts plugins automatically.
  </ParamField>
  <ParamField path="heartbeat_missed_threshold" type="integer" default="2">
    Reserved liveness setting. Automatic missed-heartbeat eviction is not currently wired.
  </ParamField>
  <ParamField path="in_process_plugins" type="[]InProcessPluginSpec" default="[]">
    In-process plugins loaded from Python modules at startup. See [InProcessPluginSpec](#inprocesspluginspec).
  </ParamField>
  <ParamField path="admin" type="AdminAuthConfig">
    Admin (ListPlugins) RBAC config. Parsed but currently inert — the gateway default-denies `ListPlugins` regardless of `mode`.
  </ParamField>
  <Indent>
    <ParamField path="mode" type="string" default="allow_all">
      <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>allow_all</Badge> <Badge intent="note" minimal>k8s_rbac</Badge></span>
    </ParamField>
  </Indent>
</Indent>

## Retune (Experimental, Simulation Only)

`retune` makes the Planner periodically replay the last window of recorded traffic against a grid of router and Planner parameters and apply the best point by SLA goodput per GPU. This section is the field contract. For the concept, see [Periodic Retune](../../developer-guide/knowledge-base/modular-components/planner/planner-guide.md#periodic-retune-experimental); for windows, scoring, and decision timing, see [Planner Design](../../developer-guide/knowledge-base/modular-components/planner/planner-design.md#retune-experimental-simulation-only); to run it, see [Simulate Retune](../../developer-guide/knowledge-base/modular-components/planner/aisimulate-planner-replay.mdx#simulate-retune-experimental).

<Warning>
**Experimental.** In this release, retune runs only inside planner-in-the-loop offline replay: `dynamo.replay.run_trace_replay` with `trace_format="dynamo"`, or `aisimulate predict --stack dynamo` on a Dynamo request trace. A live Planner rejects a config that sets `retune` at startup.
</Warning>

<ParamField path="retune" type="RetuneConfig" default="null">
  Periodic replay-based retune. Rejects unknown fields. Requires `mode` `agg` or `disagg`, and a non-empty `router` or `planner` search domain.

  To retune with fixed replicas (retune-only mode), set **both** `enable_load_scaling: false` and `enable_throughput_scaling: false` explicitly. Only that combination skips the easy-mode forcing and the "at least one scaling mode" rule; every other combination keeps them. A retune-only Planner never scales, not even to fit `min_gpu_budget`/`max_gpu_budget` or a minimum endpoint count, runs no load loop, and pulls no forward-pass metrics. It still ticks every `scheduling.scale_interval_seconds` (by default the 5-second load interval); to tick less often, set `scheduling.scale_interval_seconds` and `load_adjustment_interval_seconds` together.

  With load or throughput scaling enabled, `scheduling.external_plugins` must be empty and `scheduling.gateway.enabled` must be `false`, because every candidate replay runs an inner Planner. That inner Planner always scales and never runs batch scheduling, even when the Planner is advisory.

  With [`advisory: true`](#advisory-mode), retune keeps sweeping and scoring, but a decision that would apply is recorded with outcome `advisory` and the point it recommends, and nothing changes.
</ParamField>

<Indent>
  <ParamField path="interval_seconds" type="number" required={true}>
    Window length and decision cadence `T`. Must be greater than 0 and a multiple of `scheduling.scale_interval_seconds`.
  </ParamField>
  <ParamField path="grace_seconds" type="number" default="60.0">
    Delay after a window ends before its sweep launches. Requests that complete after the launch time are excluded from the window. Must be 0 or greater.
  </ParamField>
  <ParamField path="history_seconds" type="number" default="null">
    Trace history before each window that warms the load predictors of candidate planners and of a newly selected predictor. With throughput scaling, replay reads this history from the replayed trace and, before its first request, from `load_predictor_warmup_trace`, which must then be a Dynamo request trace that ends at or before the first replayed request; without throughput scaling no history is read. `null` resolves to the minimum the predictors need: `throughput_adjustment_interval_seconds` × max(`prophet_window_size`, `kalman_min_points`) over the current predictor and the load-predictor candidates, or 0 without throughput scaling. An explicit value must be 0 or greater, and a value smaller than that minimum is rejected.
  </ParamField>
  <ParamField path="decision_latency_seconds" type="number | measured" default="0.0">
    Delay from sweep launch to decision. A number is a fixed delay, 0 or greater. `measured` uses the sweep's own wall-clock time, capped at `timeout_seconds`, times `decision_latency_scale`; it depends on the replay host, so it is not reproducible across hosts or runs. The decision lands no earlier than one Planner tick after launch. See [Decision Timing](../../developer-guide/knowledge-base/modular-components/planner/planner-design.md#decision-timing) for the mechanics and [Decision Latency](../../developer-guide/knowledge-base/modular-components/planner/aisimulate-planner-replay.mdx#decision-latency) for choosing a value.
  </ParamField>
  <ParamField path="decision_latency_scale" type="number" default="1.0">
    Multiplier on the measured sweep time, for example to model a production sweep host that is slower or faster than the replay host. Must be greater than 0. A value other than 1 requires `decision_latency_seconds: measured`. Above 1, set `timeout_seconds` ≤ `interval_seconds` / scale (the default, `interval_seconds`, is rejected); the modeled deadline is then scale × `timeout_seconds`.
  </ParamField>
  <ParamField path="timeout_seconds" type="number" default="null">
    Wall-clock deadline for one sweep; its workers are killed after it and the decision is `timeout`. `null` means `interval_seconds`. Must be greater than 0. Requires `timeout_seconds` ≤ `interval_seconds`. A numeric `decision_latency_seconds` must also be ≤ `timeout_seconds`; with `measured`, `decision_latency_scale` × `timeout_seconds` must also be ≤ `interval_seconds`. This keeps at most one sweep in flight.
  </ParamField>
  <ParamField path="sla_ttft_ms" type="number" default="null">
    Scoring TTFT SLA. Must be greater than 0. `null` uses the Planner's `ttft_ms`; in `aisimulate predict`, `null` takes `evaluation.sla.ttft_ms` when it is set (see [Requirements](../../developer-guide/knowledge-base/modular-components/planner/aisimulate-planner-replay.mdx#requirements)).
  </ParamField>
  <ParamField path="sla_itl_ms" type="number" default="null">
    Scoring ITL SLA. Must be greater than 0. `null` uses the Planner's `itl_ms`; in `aisimulate predict`, `null` takes `evaluation.sla.itl_ms` when it is set (see [Requirements](../../developer-guide/knowledge-base/modular-components/planner/aisimulate-planner-replay.mdx#requirements)).
  </ParamField>
  <ParamField path="min_improvement_percentage" type="number" default="3.0">
    Apply the best candidate only if its score exceeds the current point's score on the same window by more than this percentage. Must be in [0, 100). Candidate replays share one random seed, so a candidate that changes no routing or scaling decision scores exactly the current point and is kept. With the pinned AISimulate release, an inner Planner with `optimization_target: sla` is not guaranteed to be bitwise reproducible, so such a candidate can miss an exact tie; see [Candidates and the Sweep](../../developer-guide/knowledge-base/modular-components/planner/planner-design.md#candidates-and-the-sweep). To choose a value, see [Measure the Gain](../../developer-guide/knowledge-base/modular-components/planner/aisimulate-planner-replay.mdx#measure-the-gain).
  </ParamField>
  <ParamField path="max_candidates" type="integer" default="512">
    Upper bound on the expanded grid size (the product of all domain sizes), on the values of each knob, and separately on the number of load-predictor candidates. Must be at least 1. A grid that exceeds it is rejected with the size of each dimension. For the candidate set a sweep replays, see [Candidates and the Sweep](../../developer-guide/knowledge-base/modular-components/planner/planner-design.md#candidates-and-the-sweep).
  </ParamField>
  <ParamField path="max_parallel" type="integer" default="2">
    Number of sweep worker processes. Must be at least 1. Each worker replays one candidate at a time, so memory grows with this value and the window size.
  </ParamField>
  <ParamField path="router" type="object | string" default="null">
    Router search domain, written like the `router` section of `aisimulate recommend --stack dynamo` (see [Search domains](#search-domains)). A knob omitted from the map searches its AISimulate default domain; write a scalar to pin it. `default` searches the whole AISimulate default domain. `null` (or no `router` field) keeps every router parameter at its startup value; an empty map `{}` is rejected. A searched router requires `router_mode="kv_router"` (`router.policy: kv_router` in `aisimulate predict`) with the default worker-selection policy, no `overlap_score_weight`, and conditional disaggregation disabled.
  </ParamField>
  <ParamField path="planner" type="object | string" default="null">
    Planner search domain, written like the `planner` section of `aisimulate recommend --stack dynamo` and limited to what a running Planner can change (see [Search domains](#search-domains)). `default` searches the parts of the AISimulate default domain that apply to this Planner. `null` (or no `planner` field) keeps every Planner field unchanged; an empty map `{}` is rejected. A written knob or group requires load or throughput scaling, and the scaling mode that uses it: for example, `load_sensitivity` needs load scaling, and `load_predictor` needs `optimization_target: sla` with throughput scaling. A searched field must have a value in the Planner config. `ttft_ms` and `itl_ms` are never searched.
  </ParamField>
</Indent>

### Search domains

`retune.router` and `retune.planner` use the knob names, value syntax, and default domains of the Dynamo router and Planner adapters of `aisimulate recommend --stack dynamo`:

- **Values.** A knob takes the [domain syntax](dynosim-sweep-reference.mdx#domain-syntax) of `aisimulate recommend`: a scalar, `{choices: [...]}`, `{range: {min, max, step}}`, or, for the `load_sensitivity` and `load_predictor` groups, `{preset: ...}`. Retune adds these rules:
  - A range must be enumerable: a range without `step` and a `scale: log` range are rejected. A range with `min` equal to `max` is one value. Stepped ranges are computed in decimal, so `{min: 0.0, max: 0.3, step: 0.1}` ends at 0.3.
  - Each knob may expand to at most `max_candidates` values.
  - A bare list (`[0.5, 1.0]`) is rejected; write `{choices: [0.5, 1.0]}`.
  - In YAML, write floats with a decimal point and a signed exponent (`1.0e-3`): YAML 1.1 reads `1e-3` as a string, which the schema rejects.
- **Infeasible points.** When the Planner starts, a Planner grid point that fails `PlannerConfig` validation or does not fit the deployment, such as a `min_workers` above what `max_num_gpus` allows, is dropped, and one warning reports how many points were dropped. Startup fails only when some value of a searched Planner knob fits in no grid point, or when a load-predictor candidate is invalid. A decision that would apply Planner fields that no longer validate or fit, for example after a budget change, is `failed`.
- **Omitted knobs.** A knob that has an AISimulate default domain searches it when omitted, as in `aisimulate recommend`. The retune-only knobs in the tables below keep their current value unless written. Knobs fixed at startup always keep their running value.
- **Names.** Write the public names below. A live name, such as `router_temperature`, `min_endpoint`, or `max_gpu_budget`, is rejected with the public name to use. Decisions and `report.planner.retune_events` record the live names: the PATCH `/router_policy` parameter names and the `PlannerConfig` field names.

Router knobs:

| Knob | Live name | Omitted | Written |
|---|---|---|---|
| `policy` | `router_mode` (startup) | `kv_router` | Must be `kv_router` |
| `prefill_load_model.type` | `router_prefill_load_model` (startup) | Startup value | Must equal the startup value (`none` or `ais`) |
| `overlap_score_credit` | `overlap_score_credit` | `{0, 0.5, 1}`; `{0}` when the startup credit is 0 | Domain; a positive value is rejected when the startup credit is 0 |
| `prefill_load_scale` | `prefill_load_scale` | `{0, 0.25, 0.5, 1, 2, 4, 8, 16, 32}` | Domain |
| `temperature` | `router_temperature` | `{0, 0.2, 0.5, 1}` | Domain |
| `overlap_score_credit_decay`, `decode_active_request_weight` (retune only) | Same | Startup value | Domain, non-negative |

`router: default` is 3 × 9 × 4 = 108 points.

The other router policy parameters are not retunable and are rejected:

- `host_cache_hit_weight` and `disk_cache_hit_weight`: the startup router configuration also uses them to count a request's cached tokens, which a runtime update does not change. A candidate replay would start its router with the candidate's weights and score a router that no applied update can produce.
- `shared_cache_multiplier`: replay reports no shared cache hits, so every value would score the same.

Planner knobs:

| Knob | Live name | Omitted | Written |
|---|---|---|---|
| `load_sensitivity` group | `load_scaling_down_sensitivity` | With load scaling: `{70, 80, 90}`, from the `aggressive`, `default`, and `conservative` presets | `{preset: [...]}` with preset names or mappings, or `{preset: false}` with `load_scaling_down_sensitivity` as a domain |
| `load_predictor` group | Predictor fields | With `optimization_target: sla` and throughput scaling: the 11 simulation predictor presets | `{preset: [...]}` with preset names or complete mappings, or `{preset: false}` with `type` and the predictor knobs as domains |
| `min_workers` | `min_endpoint` | Current value | Domain; `mode: agg` only |
| `prefill_min_workers`, `decode_min_workers` | `prefill_min_endpoint`, `decode_min_endpoint` | Current value | Domain; `mode: disagg` only; `null` is rejected |
| `max_num_gpus`, `min_num_gpus` | `max_gpu_budget`, `min_gpu_budget` | Current value | Domain; values follow `max_gpu_budget` and `min_gpu_budget` (`min_num_gpus: -1` disables the floor) |
| `max_throughput_scaling_replicas` | Same | Current value | Domain; requires throughput scaling |
| `prefill_scale_up_queue_tokens`, `prefill_scale_down_queue_tokens` (retune only) | Same | Current value | Domain; requires `optimization_target: load` and `mode: disagg` |
| `decode_scale_up_kv_rate`, `decode_scale_down_kv_rate` (retune only) | Same | Current value | Domain; requires `optimization_target: load` |
| `policy`, `scaling_policy`, `fpm_sampling` and their knobs, `load_min_observations` | — | Running value | Rejected: fixed when the Planner starts |
| `ttft_ms`, `itl_ms` | — | Fixed | Rejected: set `sla_ttft_ms` and `sla_itl_ms` to change the scoring SLA |

The group rules are those of `aisimulate recommend`: to write a knob of a group, set the group to `{preset: false}`; a group that is omitted or `{preset: default}` cannot be combined with its knobs. Within these groups:

- **Load sensitivity.** Retune searches only `load_scaling_down_sensitivity`; `load_min_observations` keeps its running value. A preset name contributes its sensitivity alone, so with a running `load_min_observations` other than 5 the `default` preset is applied only in part. A preset mapping must set `load_min_observations` to the running value.
- **Load predictor.** The predictor is never a grid dimension: a [pre-sweep](../../developer-guide/knowledge-base/modular-components/planner/planner-design.md#load-predictor-pre-sweep) picks one candidate per sweep. Candidates that give the same predictor are evaluated once (`{preset: false}` with the default knob domains is 256 combinations and 40 predictors). Records label a candidate with its preset name, or `custom_<i>` for the `i`-th entry. The number of candidates must not exceed `max_candidates`.
- **Planner defaults.** `planner: default` searches `load_scaling_down_sensitivity` ∈ `{70, 80, 90}` with load scaling and the 11 predictor presets with `optimization_target: sla` and throughput scaling. Without scaling it expands to nothing.

An explicit domain for a disaggregated SLA Planner with load and throughput scaling. The router grid is 3 × 1 × 3 × 2 = 18 points and the Planner grid 3 × 2 = 6, so the sweep replays 108 grid points, after a pre-sweep over three predictors:

```yaml
retune:
  interval_seconds: 21600
  router:
    overlap_score_credit: {choices: [0.5, 1.0, 2.0]}
    prefill_load_scale: 1.0                               # pinned
    temperature: {range: {min: 0.0, max: 0.2, step: 0.1}} # 0, 0.1, 0.2
    decode_active_request_weight: {choices: [0.5, 1.0]}   # retune-only knob
  planner:
    load_sensitivity: {preset: false}
    load_scaling_down_sensitivity: {choices: [70, 80, 90]}
    decode_min_workers: {choices: [1, 2]}
    load_predictor: {preset: [arima_log1p, kalman_default_raw, kalman_reactive_log1p]}
```

## Additional types

Nested object types referenced above, broken out to keep the field lists shallow.

### PickedParallelConfig

A parallelism pick emitted by AIConfigurator. Referenced by `AICInterpolationSpec`.

<ParamField path="tp" type="integer" default="1">
  Tensor-parallel size.
</ParamField>
<ParamField path="pp" type="integer" default="1">
  Pipeline-parallel size.
</ParamField>
<ParamField path="dp" type="integer" default="1">
  Data-parallel (attention-DP) size.
</ParamField>
<ParamField path="moe_tp" type="integer" default="1">
  Mixture-of-Experts tensor-parallel size.
</ParamField>
<ParamField path="moe_ep" type="integer" default="1">
  Mixture-of-Experts expert-parallel size.
</ParamField>

### AICInterpolationSpec

Everything the Planner needs to reproduce a rapid-mode AIC sweep. Written by the profiler onto the Planner ConfigMap; not normally hand-authored.

<ParamField path="hf_id" type="string" required={true}>
  HuggingFace model id.
</ParamField>
<ParamField path="system" type="string" required={true}>
  AIC system identifier.
</ParamField>
<ParamField path="backend" type="string" required={true}>
  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>trtllm</Badge> <Badge intent="note" minimal>vllm</Badge> <Badge intent="note" minimal>sglang</Badge></span>
</ParamField>
<ParamField path="isl" type="integer" required={true}>
  Input sequence length for the sweep. Must be greater than 0.
</ParamField>
<ParamField path="osl" type="integer" required={true}>
  Output sequence length for the sweep. Must be greater than 0.
</ParamField>
<ParamField path="sweep_max_context_length" type="integer" required={true}>
  Maximum context length swept. Must be greater than 0.
</ParamField>
<ParamField path="prefill_interpolation_granularity" type="integer" required={true}>
  Number of prefill interpolation points. Must be greater than 0.
</ParamField>
<ParamField path="decode_interpolation_granularity" type="integer" required={true}>
  Number of decode interpolation points. Must be greater than 0.
</ParamField>
<ParamField path="prefill_pick" type="PickedParallelConfig" required={true}>
  Prefill parallelism pick. See [PickedParallelConfig](#pickedparallelconfig).
</ParamField>
<ParamField path="decode_pick" type="PickedParallelConfig" required={true}>
  Decode parallelism pick. See [PickedParallelConfig](#pickedparallelconfig).
</ParamField>

### ExternalPluginEntry

One static external-plugin registration under `scheduling.external_plugins`. Rejects unknown fields.

<ParamField path="plugin_id" type="string" required={true}>
  Unique identifier. Must not collide with a builtin plugin id.
</ParamField>
<ParamField path="plugin_type" type="string" required={true}>
  Pipeline stage the plugin participates in.

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>predict</Badge> <Badge intent="note" minimal>propose</Badge> <Badge intent="note" minimal>reconcile</Badge> <Badge intent="note" minimal>constrain</Badge></span>
</ParamField>
<ParamField path="priority" type="integer" required={true}>
  Stage priority. Smaller number is more authoritative.
</ParamField>
<ParamField path="endpoint" type="string" required={true}>
  Wire endpoint. Must start with `grpc://host:port`; `inproc://` is rejected here.
</ParamField>
<ParamField path="auth_token" type="string" default="">
  Bearer token validated by the registry. Populate from a mounted Secret.
</ParamField>
<ParamField path="protocol_version" type="string" default="1.0">
  Plugin protocol version. Must fall within the registry's supported range.
</ParamField>
<ParamField path="version" type="string" default="v1">
  Plugin's own version string, surfaced in ListPlugins.
</ParamField>
<ParamField path="execution_interval_seconds" type="number" default="0.0">
  `0.0` runs every tick; a positive value throttles to every N seconds. Must be 0 or greater.
</ParamField>
<ParamField path="hold_policy" type="string" default="HOLD_LAST">
  Behavior when throttled. `HOLD_LAST` reuses the cached result; `ACCEPT_WHEN_IDLE` treats the plugin as no-opinion when not due. Accepts the name (case-insensitive) or its integer value.

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>HOLD_LAST</Badge> <Badge intent="note" minimal>ACCEPT_WHEN_IDLE</Badge></span>
</ParamField>
<ParamField path="needs" type="[]string" default="[]">
  Capability list consumed by the type-aware merge.
</ParamField>
<ParamField path="requires_produced_fields" type="[]string" default="[]">
  Dot-paths into the pipeline context (for example `predictions`, `observations.traffic`) that must be set for the plugin to fire on a tick. Empty means no gating.
</ParamField>
<ParamField path="observation_window_seconds" type="number" default="0.0">
  Aggregation window for windowed observation types in `needs`. `0.0` uses `scale_interval` freshness; a positive value aggregates over the last N seconds and must be an integer multiple of `scale_interval_seconds`.
</ParamField>

### InProcessPluginSpec

One in-process plugin under `plugin_registration.in_process_plugins`. Rejects unknown fields.

<ParamField path="module" type="string" required={true}>
  Python module path containing the plugin class.
</ParamField>
<ParamField path="class" type="string" required={true}>
  Plugin class name within `module`.
</ParamField>
<ParamField path="plugin_id" type="string" required={true}>
  Unique plugin identifier.
</ParamField>
<ParamField path="plugin_type" type="string" required={true}>
  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>predict</Badge> <Badge intent="note" minimal>propose</Badge> <Badge intent="note" minimal>reconcile</Badge> <Badge intent="note" minimal>constrain</Badge></span>
</ParamField>
<ParamField path="priority" type="integer" required={true}>
  Stage priority. Smaller number is more authoritative.
</ParamField>
<ParamField path="execution_interval_seconds" type="number" default="0.0">
  `0.0` runs every tick; a positive value throttles to every N seconds.
</ParamField>
<ParamField path="hold_policy" type="string" default="ACCEPT_WHEN_IDLE">
  Behavior when throttled.

  <span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>ACCEPT_WHEN_IDLE</Badge> <Badge intent="note" minimal>HOLD_LAST</Badge></span>
</ParamField>
<ParamField path="needs" type="[]string" default="[]">
  Capability list consumed by the type-aware merge.
</ParamField>
<ParamField path="requires_produced_fields" type="[]string" default="[]">
  Dot-paths into the pipeline context that must be set for the plugin to fire on a tick.
</ParamField>
<ParamField path="observation_window_seconds" type="number" default="0.0">
  Aggregation window for windowed observation types in `needs`. Must be `0.0` or a positive integer multiple of `scale_interval_seconds`.
</ParamField>
<ParamField path="kwargs" type="object" default="{}">
  Keyword arguments passed to the plugin class constructor.
</ParamField>

## Environment variables

The Planner reads these environment variables. The Prometheus and namespace variables set the defaults for the config fields noted above; the rest control runtime behavior for specific environments.

| Variable | Effect | Default |
|---|---|---|
| `DYN_NAMESPACE` | Default for `namespace`. Operator injects `{k8s_namespace}-{dgd_name}`. | `dynamo` |
| `PROMETHEUS_ENDPOINT` | Default for `metric_pulling_prometheus_endpoint`. | in-cluster URL |
| `PROMETHEUS_TOKEN` | Default for `metric_pulling_prometheus_token`. | unset |
| `PROMETHEUS_TOKEN_FILE` | Default for `metric_pulling_prometheus_token_file`. | unset |
| `PROMETHEUS_SSL_VERIFY` | Default for `metric_pulling_prometheus_ssl_verify` (`1`/`true`/`yes`). | `false` |
| `PROMETHEUS_EXTRA_QUERY_PARAMS` | Default for `metric_pulling_prometheus_extra_query_params` (URL query string). | unset |
| `PROMETHEUS_CA_BUNDLE` | Default for `metric_pulling_prometheus_ca_bundle`. | unset |
| `DYN_PLANNER_PROMETHEUS_REQUEST_TIMEOUT_SECONDS` | Default for `metric_pulling_prometheus_request_timeout_seconds`. | `10.0` |
| `PLANNER_PROMETHEUS_PORT` | Default for `metric_reporting_prometheus_port`. | `0` |
| `DYN_PARENT_DGD_K8S_NAME` | Parent DGD name for the GlobalPlanner connector (`environment: global-planner`). | required |
| `POD_NAMESPACE` | Kubernetes namespace for the GlobalPlanner connector. | required |
| `SCALING_CHECK_INTERVAL` | Poll interval (seconds) for the virtual environment connector (`environment: virtual`). | `10` |
| `SCALING_MAX_WAIT_TIME` | Maximum wait (seconds) for a scaling action in the virtual environment. | `1800` |
| `DYNAMO_PLANNER_TEST` | Test-only. Must be `1` to allow the virtual clock. Not for production. | unset |

## Validation rules

The Planner enforces these cross-field rules at startup and rejects a config that violates any of them:

- `ttft_ms` must be greater than 0.
- `report_interval_hours` must be a positive finite number or `null`.
- `fpm_sample_bucket_size` must be a perfect square.
- `global_planner_namespace` is required when `environment` is `global-planner`.
- Under `optimization_target: load`, prefill modes require `prefill_scale_up_queue_tokens` > `prefill_scale_down_queue_tokens`, and decode modes require `decode_scale_up_kv_rate` > `decode_scale_down_kv_rate`.
- At least one scaling mode must be enabled. Under any `optimization_target` other than `sla`, the Planner forces load scaling on and throughput scaling off and ignores `ttft_ms` / `itl_ms`. The only exception is retune-only mode: `retune` set with both `enable_load_scaling` and `enable_throughput_scaling` explicitly `false` (see [Retune](#retune-experimental-simulation-only)).
- When `ais_perf_model` is set, its `roles` must cover the deployment: `prefill` and `decode` for `disagg`, the matching role for `prefill`/`decode`, or `aggregated` for `agg`. Role keys must agree with `worker_type`. Unknown canonical fields and the retired `aic_perf_model` input are rejected.
- `scheduling.scale_interval_seconds` must evenly divide `load_adjustment_interval_seconds` and, when throughput scaling is enabled, `throughput_adjustment_interval_seconds`.
- When both scaling modes are enabled, `load_adjustment_interval_seconds` must be shorter than `throughput_adjustment_interval_seconds`.
- `retune` has its own cross-field rules, listed with its fields in [Retune](#retune-experimental-simulation-only). A live Planner rejects any config that sets `retune`.

## Related pages

<CardGroup cols={2}>
  <Card title="Planner Guide" icon="book" href="../../developer-guide/knowledge-base/modular-components/planner/planner-guide.md">
    Deployment workflow, scaling modes, and profiler integration.
  </Card>
  <Card title="Planner Overview" icon="circle-info" href="../../developer-guide/knowledge-base/modular-components/planner/overview.md">
    Why LLM inference needs a different autoscaler.
  </Card>
  <Card title="Simulate Planner Decisions with AISimulate" icon="flask" href="../../developer-guide/knowledge-base/modular-components/planner/aisimulate-planner-replay.mdx">
    Replay a trace through the Planner offline, including retune.
  </Card>
  <Card title="DGDR Reference" icon="diagram-project" href="../kubernetes-api/dynamo-graph-deployment-request.mdx">
    The custom resource whose `spec.features.planner` carries this config.
  </Card>
  <Card title="Environment Variables" icon="terminal" href="../observability/environment-variables.mdx">
    Observability environment variables shared across Dynamo processes.
  </Card>
</CardGroup>
