---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Frontend Configuration Reference
subtitle: Complete reference for all frontend CLI arguments, environment variables, and HTTP endpoints
---

This page documents all configuration options for the Dynamo Frontend (`python -m dynamo.frontend`).

Every CLI argument has a corresponding environment variable. CLI arguments take precedence over environment variables.

## HTTP & Networking

| CLI Argument | Env Var | Default | Description |
|-------------|---------|---------|-------------|
| `--http-host` | `DYN_HTTP_HOST` | `0.0.0.0` | HTTP listen address |
| `--http-port` | `DYN_HTTP_PORT` | `8000` | HTTP listen port |
| `--tls-cert-path` | `DYN_TLS_CERT_PATH` | — | TLS certificate path (PEM). Must be paired with `--tls-key-path` |
| `--tls-key-path` | `DYN_TLS_KEY_PATH` | — | TLS private key path (PEM). Must be paired with `--tls-cert-path` |

The Rust HTTP server also reads these environment variables (not exposed as CLI args):

| Env Var | Default | Description |
|---------|---------|-------------|
| `DYN_HTTP_BODY_LIMIT_MB` | `192` | Maximum request body size in MB |
| `DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS` | `5` | Graceful shutdown timeout in seconds |

## Router

| CLI Argument | Env Var | Default | Description |
|-------------|---------|---------|-------------|
| `--router-mode` | `DYN_ROUTER_MODE` | `round-robin` | Routing strategy: `round-robin`, `random`, `kv`, `direct` |
| `--router-kv-overlap-score-weight` | `DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT` | `1.0` | Weight for KV cache overlap in worker scoring. Higher = prefer cache reuse |
| `--router-temperature` | `DYN_ROUTER_TEMPERATURE` | `0.0` | Softmax temperature for worker sampling. 0 = deterministic |
| `--router-kv-events` / `--no-router-kv-events` | `DYN_ROUTER_USE_KV_EVENTS` | `true` | Enable KV cache state events from workers. Disable for prediction-based routing |
| `--router-ttl-secs` | `DYN_ROUTER_TTL_SECS` | `120.0` | Block TTL when KV events are disabled |
| `--router-max-tree-size` | `DYN_ROUTER_MAX_TREE_SIZE` | `1048576` | Max radix tree size before pruning (no-events mode) |
| `--router-prune-target-ratio` | `DYN_ROUTER_PRUNE_TARGET_RATIO` | `0.8` | Target size ratio after pruning (no-events mode) |
| `--router-replica-sync` / `--no-router-replica-sync` | `DYN_ROUTER_REPLICA_SYNC` | `false` | Sync state across multiple router instances |
| `--router-snapshot-threshold` | `DYN_ROUTER_SNAPSHOT_THRESHOLD` | `1000000` | Messages before triggering a snapshot |
| `--router-reset-states` / `--no-router-reset-states` | `DYN_ROUTER_RESET_STATES` | `false` | Reset router state on startup. **Warning:** affects existing replicas |
| `--router-track-active-blocks` / `--no-router-track-active-blocks` | `DYN_ROUTER_TRACK_ACTIVE_BLOCKS` | `true` | Track blocks used by in-progress requests for load balancing |
| `--router-assume-kv-reuse` / `--no-router-assume-kv-reuse` | `DYN_ROUTER_ASSUME_KV_REUSE` | `true` | Assume KV cache reuse when tracking active blocks |
| `--router-track-output-blocks` / `--no-router-track-output-blocks` | `DYN_ROUTER_TRACK_OUTPUT_BLOCKS` | `false` | Track output blocks with fractional decay during generation |
| `--router-event-threads` | `DYN_ROUTER_EVENT_THREADS` | `1` | Event processing threads. >1 enables concurrent radix tree |
| `--router-queue-threshold` | `DYN_ROUTER_QUEUE_THRESHOLD` | — | Queue threshold fraction of prefill capacity. Enables priority scheduling |
| `--enable-cache-control` / `--no-enable-cache-control` | `DYN_ENABLE_CACHE_CONTROL` | `false` | Enable TTL-based cache pinning (requires `--router-mode=kv`) |
| `--decode-fallback` / `--no-decode-fallback` | `DYN_DECODE_FALLBACK` | `false` | Fall back to aggregated mode when prefill workers unavailable |

## Fault Tolerance

| CLI Argument | Env Var | Default | Description |
|-------------|---------|---------|-------------|
| `--migration-limit` | `DYN_MIGRATION_LIMIT` | `0` | Max request migrations per worker disconnect. 0 = disabled |
| `--active-decode-blocks-threshold` | `DYN_ACTIVE_DECODE_BLOCKS_THRESHOLD` | — | KV cache utilization fraction (0.0–1.0) for busy detection |
| `--active-prefill-tokens-threshold` | `DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD` | — | Absolute token count for prefill busy detection |
| `--active-prefill-tokens-threshold-frac` | `DYN_ACTIVE_PREFILL_TOKENS_THRESHOLD_FRAC` | — | Fraction of `max_num_batched_tokens` for prefill busy detection. OR logic with absolute threshold |

## Model Discovery

| CLI Argument | Env Var | Default | Description |
|-------------|---------|---------|-------------|
| `--namespace` | `DYN_NAMESPACE` | — | Exact namespace for model discovery scoping |
| `--namespace-prefix` | `DYN_NAMESPACE_PREFIX` | — | Namespace prefix for discovery (e.g., `ns` matches `ns`, `ns-abc123`). Takes precedence over `--namespace` |
| `--model-name` | `DYN_MODEL_NAME` | — | Override model name string |
| `--model-path` | `DYN_MODEL_PATH` | — | Path to local model directory (for private/custom models) |
| `--kv-cache-block-size` | `DYN_KV_CACHE_BLOCK_SIZE` | — | KV cache block size override |

## Infrastructure

| CLI Argument | Env Var | Default | Description |
|-------------|---------|---------|-------------|
| `--discovery-backend` | `DYN_DISCOVERY_BACKEND` | `etcd` | Service discovery: `kubernetes`, `etcd`, `file`, `mem` |
| `--request-plane` | `DYN_REQUEST_PLANE` | `tcp` | Request distribution: `tcp` (fastest), `nats`, `http` |
| `--event-plane` | `DYN_EVENT_PLANE` | `nats` | Event publishing: `nats`, `zmq` |

## KServe gRPC

| CLI Argument | Env Var | Default | Description |
|-------------|---------|---------|-------------|
| `--kserve-grpc-server` / `--no-kserve-grpc-server` | `DYN_KSERVE_GRPC_SERVER` | `false` | Start KServe gRPC v2 server |
| `--grpc-metrics-port` | `DYN_GRPC_METRICS_PORT` | `8788` | HTTP metrics port for gRPC service |

See the [Frontend Guide](frontend-guide.md) for KServe message formats and integration details.

## Monitoring

| CLI Argument | Env Var | Default | Description |
|-------------|---------|---------|-------------|
| `--metrics-prefix` | `DYN_METRICS_PREFIX` | `dynamo_frontend` | Prefix for frontend Prometheus metrics |
| `--dump-config-to` | `DYN_DUMP_CONFIG_TO` | — | Dump resolved config to file path |

## Experimental

| CLI Argument | Env Var | Default | Description |
|-------------|---------|---------|-------------|
| `--enable-anthropic-api` | `DYN_ENABLE_ANTHROPIC_API` | `false` | Enable `/v1/messages` (Anthropic Messages API) |
| `--dyn-chat-processor` | `DYN_CHAT_PROCESSOR` | `dynamo` | Chat processor: `dynamo` or `vllm` |
| `--dyn-debug-perf` | `DYN_DEBUG_PERF` | `false` | Log per-function timing for preprocessing (vllm processor only) |
| `--dyn-preprocess-workers` | `DYN_PREPROCESS_WORKERS` | `0` | Worker processes for CPU-bound preprocessing. 0 = main event loop (vllm processor only) |
| `-i` / `--interactive` | `DYN_INTERACTIVE` | `false` | Interactive text chat mode |

## HTTP Endpoints

The frontend exposes the following HTTP endpoints:

### OpenAI-Compatible

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/chat/completions` | Chat completions (streaming and non-streaming) |
| `POST` | `/v1/completions` | Text completions |
| `POST` | `/v1/embeddings` | Text embeddings |
| `POST` | `/v1/responses` | Responses API |
| `POST` | `/v1/images/generations` | Image generation |
| `POST` | `/v1/videos/generations` | Video generation |
| `POST` | `/v1/videos/generations/stream` | Video generation (streaming) |
| `GET` | `/v1/models` | List available models |

### Anthropic (Experimental)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/messages` | Anthropic Messages API (requires `--enable-anthropic-api`) |
| `POST` | `/v1/messages/count_tokens` | Token counting for Anthropic API |

### Infrastructure

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/live` | Liveness check |
| `GET` | `/metrics` | Prometheus metrics |
| `GET` | `/openapi.json` | OpenAPI specification |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/busy_threshold` | Set busy thresholds |
| `GET` | `/busy_threshold` | Get current busy thresholds |

### Endpoint Path Customization

All endpoint paths can be overridden via environment variables:

| Env Var | Default Path |
|---------|-------------|
| `DYN_HTTP_SVC_CHAT_PATH_ENV` | `/v1/chat/completions` |
| `DYN_HTTP_SVC_CMP_PATH_ENV` | `/v1/completions` |
| `DYN_HTTP_SVC_EMB_PATH_ENV` | `/v1/embeddings` |
| `DYN_HTTP_SVC_RESPONSES_PATH_ENV` | `/v1/responses` |
| `DYN_HTTP_SVC_MODELS_PATH_ENV` | `/v1/models` |
| `DYN_HTTP_SVC_ANTHROPIC_PATH_ENV` | `/v1/messages` |
| `DYN_HTTP_SVC_HEALTH_PATH_ENV` | `/health` |
| `DYN_HTTP_SVC_LIVE_PATH_PATH_ENV` | `/live` |
| `DYN_HTTP_SVC_METRICS_PATH_ENV` | `/metrics` |

## Deprecated

| CLI Argument | Env Var | Description |
|-------------|---------|-------------|
| `--router-durable-kv-events` | `DYN_ROUTER_DURABLE_KV_EVENTS` | Use event-plane local indexer instead |

## See Also

- [Frontend Overview](README.md) — quick start and feature matrix
- [Frontend Guide](frontend-guide.md) — KServe gRPC configuration
- [NVIDIA Request Extensions (nvext)](nvext.md) — custom request fields
- [Router Guide](../router/router-guide.md) — detailed routing configuration
- [Metrics](../../observability/metrics.md) — available Prometheus metrics
- [Fault Tolerance](../../fault-tolerance/README.md) — request migration and rejection
