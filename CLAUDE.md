# CLAUDE.md — ai-dynamo/dynamo

This file orients Claude Code (and other agentic assistants) in the Dynamo monorepo. Read it first; then follow links into the specific area you're touching.

## What Dynamo is

Dynamo is a distributed LLM serving framework: an orchestration-aware frontend (OpenAI-compatible) plus disaggregation-aware workers for TRT-LLM, vLLM, and SGLang. It ships a Kubernetes operator and a Dynamo Graph Deployment Request (DGDR) CRD for profiler-driven sizing.

## Repo map

| Looking for | Path |
|---|---|
| Getting started | [`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md) |
| Architecture overview | [`docs/design-docs/architecture.md`](docs/design-docs/architecture.md) |
| Backend selection matrix | [`docs/kubernetes/README.md`](docs/kubernetes/README.md) |
| TRT-LLM backend | [`docs/backends/trtllm/`](docs/backends/trtllm/README.md) + [`examples/backends/trtllm/`](examples/backends/trtllm/) |
| vLLM backend | [`docs/backends/vllm/`](docs/backends/vllm/README.md) + [`examples/backends/vllm/`](examples/backends/vllm/) |
| SGLang backend | [`docs/backends/sglang/`](docs/backends/sglang/README.md) + [`examples/backends/sglang/`](examples/backends/sglang/) |
| Production recipes (ready-to-deploy K8s YAMLs) | [`recipes/`](recipes/README.md) |
| DGDR CRD / operator | [`docs/kubernetes/dgdr.md`](docs/kubernetes/dgdr.md) + [`deploy/operator/`](deploy/operator/) |
| Observability (metrics, logs, traces) | [`docs/observability/`](docs/observability/README.md) |
| Release artifacts (canonical container / PyPI / Helm / crate tags) | [`docs/reference/release-artifacts.md`](docs/reference/release-artifacts.md) |
| Troubleshooting | [`docs/troubleshooting.md`](docs/troubleshooting.md) |
| Claude Code skills | [`SKILLS.md`](SKILLS.md) |

## Conventions

- Canonical image tag: `:1.0.1`. See `docs/reference/release-artifacts.md`. The Kimi-k2.5 recipes are the sole exception — they require a top-of-tree image by design.
- DGDR `spec.gpuProductName` accepts exactly the CRD enum values in `deploy/operator/api/v1beta1/dynamographdeploymentrequest_types.go`. See the table in `docs/kubernetes/dgdr.md`.
- Observability requires `OTEL_EXPORT_ENABLED=true` on every Dynamo process for traces/logs to reach Tempo/Loki.

## When editing

- Docs under `docs/` are the source of truth for user-facing behavior. Recipes and examples should link to docs, not duplicate them.
- YAML changes in `recipes/**/deploy.yaml` are user-facing. Treat as functional, not cosmetic.
- Examples under `examples/` are for learning and scripting; recipes under `recipes/` are for production.
