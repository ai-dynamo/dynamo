<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Working in the ext_proc endpoint picker

Read the [runtime EPP lifecycle guide](README.md) before changing selection, response callbacks, or protocol adaptation.

## Identify the path first

- `epp.rs` is runtime EPP. `epp_router.rs` and `selector.rs` implement a different standalone adapter. Do not infer one path's capabilities or gaps from the other.
- `server.rs` owns Envoy protocol events and per-stream context. `picker.rs` defines what the picker receives. `envoy_helpers.rs` owns protocol/header/body transformations.
- D selection and load registration are separate in the current runtime EPP. Do not describe a successful query as an owned route plan or an atomic booking.

## Preserve the booking association

- Retain the EPP-minted reservation ID in each stream's `RequestContext`. Do not key cleanup solely by a potentially reused HTTP request ID.
- Keep ownership continuous from queued selection through registration, endpoint resolution, result adoption, callbacks, and release.
- Check both P and D cleanup on every failure after P selection. P/D admission is sequential; there is no atomic pair reservation.
- Preserve the distinction between routing accounting and backend execution. A detached booking must have an explicit cleanup owner and expiry path.

## Describe callbacks accurately

- First nonempty response bytes are not proof of the first generated token. Response headers alone are not prefill completion.
- The completion hook currently follows processing-loop exit. It does not carry an explicit backend-success or cancellation outcome.
- Usage parsing supplies optional final counts only. Incremental output accounting requires a separate implementation.
- Keep the public picker interface aligned with server call sites. Adding an event branch does not automatically expose a picker callback.
- Do not promise backend cancellation on ext_proc disconnect while retry/disconnect ownership remains unresolved.

## Changes to shared frontend integration

Preserve candidate-subset constraints, destination resolution, authoritative routing headers, body mutation, token-injection rules, and booking callbacks. Check model resolution, conditional disaggregation, affinity, classifier progress, output accounting, KV hints, and migration behavior separately. A schedule-only flag is a contract change, not merely a different response format.
