---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Request Migration
subtitle: Keep in-flight generations alive across worker failures by retrying them on a healthy worker.
---

<p align="left">
  <a href="./request-migration.zh-CN.md" hreflang="zh-CN"><img src="../assets/img/readme-zh-cn-link.svg" alt="简体中文" height="28" /></a>
</p>

When a worker fails mid-generation, Dynamo can migrate the in-progress request to a healthy worker and continue from the exact point of failure — no tokens lost or duplicated, and no interruption visible to the client. Migration is configured once on the **Frontend** and applies globally to every model it serves.

Migration is **off by default** (`--migration-limit 0`). The steps below turn it on, optionally bound its memory use, and verify it is working.

> **How it works:** the Migrator's position in the pipeline, token-state accumulation, and the two failure scenarios it handles are documented in [Request Migration Architecture](../design-docs/request-migration.md).

<Steps toc={true} tocDepth={2}>

<Step title="Enable migration on the Frontend">

Set `--migration-limit N` — the maximum number of times a single request may be migrated to another worker. Configure it on the **Frontend** component, either as an `args:` entry or via the `DYN_MIGRATION_LIMIT` environment variable in `env:`:

```yaml
  - name: Frontend
    type: frontend
    replicas: 1
    podTemplate:
      spec:
        containers:
        - name: main
          image: ${RUNTIME_IMAGE}
          env:
          - name: DYN_MIGRATION_LIMIT
            value: "3"                 # allow up to 3 migration attempts
```

The limit applies to all models served by this frontend. A value of `0` (the default) disables migration. Start with `3` — enough to survive transient worker loss without retrying indefinitely.

</Step>

<Step title="Bound memory for long sequences">

Optional. Migration caches token state for each in-flight request so it can be replayed on a new worker. For very long sequences this cache grows, so you can cap it with `--migration-max-seq-len` (or `DYN_MIGRATION_MAX_SEQ_LEN`) on the Frontend:

```yaml
          env:
          - name: DYN_MIGRATION_MAX_SEQ_LEN
            value: "32000"
```

Once a request's total sequence length (prompt + generated tokens) **strictly exceeds** this limit, migration is disabled for that request and token tracking stops. Exactly at the limit is still migratable. The check runs both at request start (prompt length) and during generation. Leave it unset for no limit.

</Step>

<Step title="Verify migrations">

Migration counters are exposed on the Frontend's `/metrics` endpoint (default port 8000). Port-forward the Frontend and scrape for the migration counter:

```bash
kubectl port-forward svc/<deployment-name>-frontend 8000:8000 -n ${NAMESPACE}
curl -s localhost:8000/metrics | grep dynamo_frontend_model_migration
```

- `dynamo_frontend_model_migration_total` — total migrations, labeled by `model` and `migration_type` (`new_request` vs `ongoing_request`).
- `dynamo_frontend_model_migration_max_seq_len_exceeded_total` — times migration was disabled because a request exceeded `--migration-max-seq-len`.

A rising `max_seq_len_exceeded` counter means your limit may be too low for the workload. For full field definitions and labels, see [Migration](../reference/observability/metrics-catalog.mdx#migration) in the Metrics Catalog.

</Step>

</Steps>

## Known Limitations

Migration is silently skipped for two request types even when `--migration-limit` is greater than 0. Know these before relying on it:

### Multiple Choices (`n > 1`)

Request migration is **not supported** for OpenAI-compatible requests that ask for multiple generated choices with `n > 1`. Multi-choice generation maintains separate per-choice output state; the current migration path preserves a single continuation state, so retrying an interleaved `n > 1` request could duplicate or drop choice-specific output. This does not affect normal single-choice requests where `n` is omitted or set to 1.

### Guided Decoding (Structured Output)

Request migration is **not supported** for requests that use guided decoding (structured output / JSON schema). When a worker fails mid-stream, the error is propagated to the client instead of migrating. Inference backends initialize the guided-decoding finite state machine (FSM) fresh for every request and advance it only on newly-generated tokens; a migrated worker replays prior tokens as context but starts the FSM from the schema root, and that mismatch produces corrupted output (typically duplicated or nested JSON). This applies equally to all backends (vLLM, SGLang, TRT-LLM).

## Benefits

Migration preserves partial generations rather than restarting them, so individual worker failures become transparent to clients — the system continues operating with no token loss and no visible interruption, tunable per deployment via the migration limit.

## Related Documentation

- [Request Migration Architecture](../design-docs/request-migration.md) - Pipeline position, token-state tracking, and failure scenarios
- [Graceful Shutdown](graceful-shutdown.md) - Draining in-flight requests on planned shutdown
- [Metrics Catalog](../reference/observability/metrics-catalog.mdx#migration) - Migration metrics
