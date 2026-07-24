<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Handoff: Observability Reference section

**Status:** research complete, plan drafted, **no files written yet** (nothing under `docs/` touched).
**Goal:** Add a new **Observability** section under the **Reference** tab that catalogs every observability
*field* (env vars, metrics, metric labels) using Fern's `ParamField` component. Existing guide pages under
User Guide stay as-is (how-to prose); the new Reference pages are the field catalog (Diátaxis split).

Pick up at [Next steps](#next-steps).

---

## Decisions locked in (from the user)

1. **Metric definitions:** include the **full metric catalog** in Reference (not just config fields). Reproduce
   every `dynamo_*` metric as `ParamField` entries — accept the duplication with the existing Metrics guide.
2. **Local content:** **per-field scope labeling + `(Local)` pages.** Env vars live in shared pages, each field
   noting its scope (Shared / Kubernetes / Local). Genuinely local-only tools (Local Resource Monitor) get their
   own page with **"(Local)"** in the title. Do **not** fully split Local vs K8s for identical env vars.

---

## Key finding: labels are dimensions, not statistics (the user's instinct was right)

The "component labels" are **not metrics** — they're the label dimensions attached to metric series. They should
be sorted into their **own reference surface**, separate from the metric catalog. Three groups:

- **Auto-injected by the Dynamo runtime:** `dynamo_namespace`, `dynamo_component`, `dynamo_endpoint`, `worker_id`
- **Added at registration time by backend code:** `model`, `model_name`
- **Metric-specific:** `error_type`, `stage`, `phase`, `worker_type`, `dp_rank`, `migration_type`, `status`, `event_type`
- **Injected by Prometheus/Kubernetes scraper (K8s scope):** `instance`, `pod`, `container`, `namespace`, `job`, `endpoint`

Plus the label *enumerations* that belong with the labels, not the metrics:
- **Component names** (`dynamo_component` values): `router`, `Planner` (capital P), `prefill`, `backend` (= decode
  worker / vLLM agg), `encode`, `diffusion`
- **Endpoint names** (`dynamo_endpoint` values): `generate`, `clear_kv_blocks`, `worker_kv_indexer_query_dp{N}`
- **Error types** (`error_type` values): `deserialization`, `invalid_message`, `response_stream`, `generate`,
  `publish_response`, `publish_final`
- **Collision warnings** worth carrying over verbatim: `dynamo_namespace` vs `namespace`, `dynamo_endpoint` vs `endpoint`.

Best source to port from: `docs/observability/metrics.md` §"Component Labels" (lines ~129-208) — it's already
well written, just needs re-homing into a Reference page as `ParamField`s.

---

## Shared vs Kubernetes vs Local (the scope question)

**Nearly every env var is SHARED** — the same variable, set via shell `export` locally or via `envs:`/`env:` in a
DGD for Kubernetes. The differences are in **defaults/behavior**, not existence:

- `DYN_HEALTH_CHECK_ENABLED` — default `false` locally, `true` in K8s (operator sets it).
- `DYN_SYSTEM_PORT` — operator conventionally sets `9090`; local examples use `8081`. Default `-1` (disabled).

**Kubernetes-only knobs (NOT env vars — CRD annotations / Helm values):** `nvidia.com/metrics-enabled`,
`nvidia.com/enable-metrics`, Helm `dynamo-operator.metricsService.enabled`,
`dynamo-operator.dynamo.metrics.prometheusEndpoint`, PodMonitor/ServiceMonitor. → Open question whether these live
in the Observability reference or stay in the CRD refs. Leaning: cross-link, don't duplicate.

**Local-only:** `dynamo_local_resource_monitor.py` host script + its CLI flags and metrics → the `(Local)` page.

---

## Env var inventory (ready to turn into ParamField)

All defaults verified against the source pages listed under [Source pages](#source-pages).

### System / Metrics
| Var | Default | Scope notes |
|---|---|---|
| `DYN_SYSTEM_PORT` | `-1` (disabled) | Shared. K8s operator ~`9090`, local `8081`. |
| `DYN_HTTP_PORT` | `8000` | Shared. Also `--http-port`. |
| `NIXL_TELEMETRY_ENABLE` | `n` | Shared. `y`/`n`. |
| `NIXL_TELEMETRY_EXPORTER` | — | Shared. e.g. `prometheus`. |
| `NIXL_TELEMETRY_PROMETHEUS_PORT` | `19090` | Shared. Separate port from Dynamo metrics. |

### OpenTelemetry (traces + logs)
| Var | Default | Scope notes |
|---|---|---|
| `OTEL_EXPORT_ENABLED` | `false` | Shared. Gates BOTH traces and logs. |
| `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` | `http://localhost:4317` | Shared. gRPC only. |
| `OTEL_EXPORTER_OTLP_LOGS_ENDPOINT` | = traces endpoint | Shared. gRPC only. |
| `OTEL_SERVICE_NAME` | `dynamo` | Shared. Per-component name. |

### Logging
| Var | Default | Scope notes |
|---|---|---|
| `DYN_LOGGING_JSONL` | `false` | Shared. |
| `DYN_LOGGING_SPAN_EVENTS` | `false` | Shared. |
| `DYN_LOG` | `info` | Shared. Per-target level syntax. |
| `DYN_LOG_USE_LOCAL_TZ` | `false` | Shared. |
| `DYN_LOGGING_CONFIG_PATH` | none | Shared. TOML path. |
| `VLLM_LOGGING_LEVEL` | `INFO` | Shared. Independent of `DYN_LOG`. |
| `TLLM_LOG_LEVEL` | `INFO` | Shared. Read once at import time. |
| `DYN_SKIP_SGLANG_LOG_FORMATTING` | `false` | Shared. |

### Health checks
| Var | Default | Scope notes |
|---|---|---|
| `DYN_SYSTEM_STARTING_HEALTH_STATUS` | `notready` | Shared. |
| `DYN_SYSTEM_HEALTH_PATH` | `/health` | Shared. |
| `DYN_SYSTEM_LIVE_PATH` | `/live` | Shared. |
| `DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS` | none | Shared. e.g. `["generate"]`. |
| `DYN_HEALTH_CHECK_ENABLED` | `false` (**K8s: `true`**) | **Scope differs.** |
| `DYN_CANARY_WAIT_TIME` | `10` | Shared. |
| `DYN_HEALTH_CHECK_REQUEST_TIMEOUT` | `3` | Shared. |

### Request replay tracing (Mooncake)
`DYN_REQUEST_TRACE` (unset), `DYN_REQUEST_TRACE_SINKS` (`jsonl_gz`), `DYN_REQUEST_TRACE_OUTPUT_PATH`
(`/tmp/dynamo-request-trace`), `DYN_REQUEST_TRACE_CAPACITY` (`1024`), `DYN_REQUEST_TRACE_JSONL_BUFFER_BYTES`
(`1048576`), `DYN_REQUEST_TRACE_JSONL_FLUSH_INTERVAL_MS` (`1000`), `DYN_REQUEST_TRACE_JSONL_GZ_ROLL_BYTES`
(`268435456`), `DYN_REQUEST_TRACE_JSONL_GZ_ROLL_LINES` (unset), `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_ENDPOINT`
(unset), `DYN_REQUEST_TRACE_TOOL_EVENTS_ZMQ_TOPIC` (`agent-tool-events`). All shared; mostly local-dev use.

---

## Metric inventory (full catalog — decision #1)

- **Frontend `dynamo_frontend_*`** (~20): `active_requests`, `stage_requests` (labels stage/phase),
  `inflight_requests` (deprecated), `queued_requests` (deprecated), `disconnected_clients`,
  `input_sequence_tokens`, `cached_tokens`, `inter_token_latency_seconds`, `output_sequence_tokens`,
  `output_tokens_total`, `request_duration_seconds`, `requests_total`, `time_to_first_token_seconds`,
  `model_migration_total`, `worker_*` (5 gauges), `model_*` (6 config gauges).
- **Component `dynamo_component_*`** (backend workers): `inflight_requests`, `request_bytes_total`,
  `request_duration_seconds`, `requests_total`, `errors_total` (label `error_type`), `response_bytes_total`,
  `uptime_seconds`; plus comparison-page extras (`cancellation_total`, `gpu_cache_usage_percent`,
  `model_load_time_seconds`, `total_blocks`).
- **Router**: `dynamo_component_router_*` (6), `dynamo_router_overhead_*` (5), `dynamo_frontend_router_queue_*`,
  `dynamo_component_kv_cache_events_applied`. Note the **availability-by-config matrix** in metrics.md — worth
  preserving as a table on the Reference page.
- **Operator `dynamo_operator_*`** (K8s scope): reconcile/webhook/resource-inventory. Source:
  `docs/kubernetes/observability/operator-metrics.md`.
- **Engine pass-through** (`vllm:`, `sglang:`, `trtllm_`): already a big comparison table in
  `docs/observability/metrics-comparison.md`. Don't re-key by hand — link, or transclude the existing table.
- **NIXL telemetry**: separate port, upstream repo owns the full list — link out.

---

## Proposed page structure (Reference tab → new `- section: Observability`)

All pages `.mdx` (ParamField is a component). Precedent: `docs/kubernetes/dgd-reference.mdx` etc. already use
`ParamField` + `<Indent>` under the Reference tab.

1. `reference/observability/environment-variables.mdx` — all env vars as ParamField, grouped by subsystem
   (System/Metrics, OpenTelemetry, Logging, Health, Request Tracing, NIXL, Backend Log Levels). Per-field scope note.
2. `reference/observability/metrics-catalog.mdx` — all `dynamo_*` metrics as ParamField (name/type/description),
   grouped Frontend / Component / Router / Model Config. Include the router availability-by-config matrix.
3. `reference/observability/metric-labels.mdx` — the **dimensions** (the user's core ask): runtime-injected /
   registration-time / metric-specific / scraper-injected (K8s), plus the enumerations (component names, endpoint
   names, error types, stage/phase). Carry the collision warnings.
4. `reference/observability/operator-metrics.mdx` — K8s-only `dynamo_operator_*` catalog. (Or fold into #2 with a
   K8s scope tag — see open questions.)
5. `reference/observability/local-resource-monitor.mdx` — **title: "Local Resource Monitor (Local)"** — host
   script metrics + CLI flags. Genuinely local-only.

Engine pass-through metrics: **link** to existing `metrics-comparison.md` rather than duplicate.

---

## Next steps (Monday, in order)

1. **Settle the two open questions below** (scope-rendering convention + operator-metrics placement). Quick.
2. **Create the nav section**: add `- section: Observability` under `- tab: reference` in `docs/index.yml`
   (pick an `icon:` — `chart-line` or `gauge`), with a `- page:` per file above.
3. **Write page 3 (metric-labels) first** — it's the highest-value/most-novel part and the clearest port from
   existing prose (`metrics.md` §Component Labels). Validates the ParamField-for-labels pattern early.
4. Write pages 1, 2, then 4, 5 using the inventories above.
5. **SPDX + frontmatter** on every `.mdx`: two `#` SPDX lines *inside* `---`, a `title:`, no body `# H1`, body
   starts at `##`. Copyright range `2025-2026`.
6. **MDX gotchas**: blank line after `<div>`/before `</div>`; code fences at column 0; `ParamField`/`Indent`
   spelling matches the DGD reference exactly.
7. **Validate**: `fern check` and `fern docs broken-links` (both mirror CI). No redirects needed — these are new
   pages, no moves/renames.
8. **Commit** with `git commit -s` (per global rules; no Claude co-author). Suggested: `docs: add Observability
   reference section`.

**Optional follow-up (not required):** once Reference exists, trim the duplicated env-var *tables* in the User
Guide guide pages down to a link into Reference. Leave the prose/how-to. Do this as a separate change so the
review stays scoped.

---

## Open questions to resolve Monday

1. **How to render "scope" in a ParamField?** It has no native scope prop. Options: (a) a bold `**Scope:**
   Shared` line in the body; (b) a leading badge/emoji convention; (c) rely on `default=` for the value and note
   K8s differences in prose. Lean (a) for consistency + grep-ability.
2. **Operator metrics: own page (#4) or a K8s-scoped group inside the metrics catalog (#2)?** Own page keeps the
   catalog clean and matches the existing standalone operator-metrics guide.
3. **K8s annotations/Helm knobs** (`nvidia.com/*`, `metricsService.enabled`): include in the Observability
   reference or leave to the CRD refs and cross-link? Lean: cross-link, don't duplicate.

---

## Source pages (already read — don't re-research)

Local (`docs/observability/`): `README.md` (env var summary table + shared-var footnote †), `metrics.md`
(labels, component/endpoint/error enums, router availability matrix), `logging.md`, `tracing.md`,
`health-checks.md`, `request-tracing.md`, `metrics-comparison.md`, `local-resource-monitor.md`,
`prometheus-grafana.md`, `metrics-developer-guide.md`.

Kubernetes (`docs/kubernetes/observability/`): `metrics.md`, `logging.md`, `operator-metrics.md`.

Nav + config: `docs/index.yml` (Reference tab ~line 623; note observability appears in User Guide under
Installation, Operations, and Local Deployment). `fern/docs.yml` redirects ~line 153.

ParamField precedent: `docs/kubernetes/dgd-reference.mdx`, `dgdr-reference.mdx`, `dcd-reference.mdx`.

Env var source of truth (code): `lib/runtime/src/logging.rs` `setup_logging()`;
`lib/runtime/src/metrics/prometheus_names.rs` (metric name constants, component_names).
