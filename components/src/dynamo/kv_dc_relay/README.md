<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# DC KV Relay

**Experimental.** The DC KV Relay discovers NVIDIA Dynamo inference endpoints and publishes
endpoint-local key-value (KV) pool facts through the universal publisher. Its optional WAN
adapter exposes pool catalogs, Cuckoo-filter (CKF) streams, serving readiness, and load over
Protobuf/gRPC. Relay does not merge independent pools or implement cross-data-center routing policy.

## Usage

Match the existing deployment's discovery and event-plane settings using the
[runtime environment reference](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/reference/components/kv-dc-relay-configuration.md#runtime-environment), then run:

```bash
python -m dynamo.kv_dc_relay \
  --dc-id dc-a \
  --namespaces production-llama \
  --bind 127.0.0.1:5561
```

- Keep `--dc-id` stable across restarts.
- `--namespaces` selects logical Dynamo namespaces. Omit the scope to watch all namespaces
  visible to the configured discovery backend, or use `--watch-all` explicitly.
- `--endpoint-prefix` narrows the scope; repeat it to include multiple endpoint prefixes.
- `--bind` starts the plaintext gRPC listener. Omit it to run only the local producer.
- `DYN_NAMESPACE` names Relay's own runtime endpoints, not the watched worker scope.

The Kubernetes discovery backend currently sees only the Relay pod's Kubernetes namespace;
the logical namespace flags do not expand that watch. Transport security is external: use an
optional TLS/mTLS sidecar across trust boundaries, and do not expose the plaintext listener.

## Documentation

- [Architecture and concepts](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/developer-guide/knowledge-base/modular-components/router/multi-dc-kv-routing.md)
- [Kubernetes deployment](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/kubernetes/kv-aware-routing/kv-dc-relay.md)
- [CLI, environment variables, and diagnostics](https://github.com/ai-dynamo/dynamo/blob/main/docs/fern/pages/reference/components/kv-dc-relay-configuration.md)
- [Rust implementation](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/kv_dc_relay/docs/architecture.md)
- [gRPC contract](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/kv_dc_relay/docs/grpc-contract.md)
- [Protocol helpers and CBI1](https://github.com/ai-dynamo/dynamo/blob/main/lib/llm/src/kv_dc_relay/wan/grpc/protocol/README.md)

## File mode and state

Use `--sources-file /etc/relay/sources.json` instead of discovery filters to
select an explicit, live-reloaded set of logical namespaces:

```json
{"version":1,"sources":[{"namespace":"production-llama"}]}
```

Without this option, discovery remains the default. An empty source list watches
nothing. Invalid initial files fail startup; invalid updates retain the last
working source set. Transport and credentials still come from runtime settings.
An optional `connectionRevision` must match `DYN_RELAY_CONNECTION_REVISION`; omit both for
standalone operation without a connection guard. Kubernetes operators supply both.
Both modes feed a common namespace reconciler. File mode polls the file once per
second; discovery mode follows model-registration events and periodically refreshes
the namespace set through DRT. Explicit `--namespaces` remain a fixed selection.
Endpoint discovery within the selected namespaces is shared by both modes.
Source updates preserve unchanged watches and do not restart the process.

When system HTTP is enabled, `GET /engine/state` reports the mode, listener readiness,
selected namespace count (including namespaces without matching endpoints), and desired/applied revisions with the last reload error. File
revisions are null in discovery mode. The standard runtime health endpoints and
gRPC catalog retain their existing contracts.
