<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Prefill Router Invariants

Read `README.md` in this directory before changing `PrefillRouter::generate`,
activation, or conditional disaggregation. It records why the current order and
fallbacks exist. Keep the operation-order doc comment on `generate` accurate
when you change the steps.

## Invariants

Preserve these unless a DEP or an issue explicitly changes them. Each lists the
tests in this directory that guard it.

- **Fail closed while `Pending`. Pass through only when `Untargeted`.** Never
  forward a request to decode without a prefill hop while a target is set.
  Guarded by `pending_target_fails_closed_instead_of_forwarding_to_decode`,
  `untargeted_router_still_passes_through`, and
  `cleared_target_still_passes_through`.
- **Re-resolve activation at the binding load.** A target replaced after the
  gate must fail closed, not pass through. Guarded by
  `target_replaced_mid_flight_fails_closed_at_binding_load`.
- **Query-only requests go to the decode `RoutingHost` in every state.**
  Guarded by `pending_target_still_serves_query_only_requests` and
  `conditional_disagg_query_only_forwards_to_decode_host`.
- **Clients cannot request a conditional bypass.** Strip
  `x-bypass-remote-prefill` from incoming annotations before any policy runs.
  No test guards this. Add one if you touch that code.
- **The prefill hop never starts the classifier lifecycle.** Do not move
  lifecycle start into a path the Prefill phase reaches.
- **Prefill survives client cancellation**, and decode routing proceeds after
  the client context stops. Guarded by
  `client_cancellation_does_not_cancel_prefill_context` and
  `disconnected_client_does_not_drop_prefill_transfer`.
- **The decode override stays in step with the EPP.** A change to
  `build_decode_router_override` needs the matching change to
  `decode_router_config_override` in
  `deploy/inference-gateway/ext-proc/src/epp.rs`. Guarded by
  `decode_router_override_disables_overlap_and_prefill_tracking` and
  `decode_request_keeps_client_limits_and_applies_prefill_topology`.

## Changing Behavior

- Reordering the activation gate and conditional disaggregation is a behavior
  change, not a refactor. See the README for what it would need to decide.
- When conditional planning gains a new input, decide what a missing value
  means and write it down in the README. Do not let `None` mean several things.
- Keep refactors behavior-preserving and separate from behavior changes, each
  with its guarding test named in the PR.

## Testing

On macOS, run this module's tests without the default features:

```bash
cargo test -p dynamo-llm --no-default-features --lib kv_router::prefill_router
```
