<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Working in KV routing

Read the [route planning and lifetime guide](README.md) before changing preview, planning, dispatch, or cleanup. Use the [selection-service guide](../../../kv-router/src/services/selection/README.md) for admission modes and the [prefill guide](prefill_router/README.md) for topology decisions.

## Trace the full route

- Start with `PreprocessedRequest`, `RoutePreview`, and `RoutePlan`. Follow the selected worker and booking through dispatch, release, drop, or transfer to request-ID-based lease tracking.
- Keep preview observational. It skips queue admission and returns one target; it does not guarantee later admission.
- Treat `RequestPhase` as routing-hop context. It is not evidence that backend prefill has completed.
- Check integrated frontend, conditional-disaggregation, prefill, and runtime-EPP callers when changing selection and booking APIs; they do not currently share one general plan API.
- Preserve one cleanup budget across preview, admission, and dispatch. Preserve request identity checks when consuming a preview.

## Ownership and progress

- Distinguish scheduler booking handles, route cleanup, and request leases. Document any transfer of cleanup responsibility at the declaration.
- Verify cancellation while queued, failure after booking, an unused plan, and detached-lease cleanup when behavior changes.
- Changes to schedule-only paths must account for dispatch-only features: affinity, classifier progression, KV hints, output accounting, and cleanup.
- `query_instance_id` currently enters admission on the KV path despite advisory wording in a comment. Verify the implementation before claiming it is a nonblocking probe.
- Use injected dispatch for tests where appropriate; do not claim `from_client_with_dispatch` is a production call path based only on its public visibility.

## Module map

- `kv_router.rs` estimates how much of a request each worker already has cached, then asks the scheduler to choose a worker. It selects a worker but does not send the request.
- `embedded.rs` connects discovered workers and their current load to the scheduling code in `lib/kv-router`.
- `indexer/` tracks which KV-cache blocks each worker holds; lookups for reusable blocks run in `lib/kv-router` before selection.
- `publisher/` receives KV-cache events and worker metrics from inference engines.
- `routing_host.rs` owns selection, dispatch, and response cleanup. `routing_host/kv_selection.rs` chooses KV-routed workers, `routing_host/request_guard.rs` tracks progress and cleanup, and `routing_host/cancellation.rs` stops unfinished work when the client cancels.
- `prefill_router/` optionally runs a request on a prefill worker before sending it to decode.
- `encoder_router.rs` optionally runs multimodal inputs through an encoder worker before token generation.
