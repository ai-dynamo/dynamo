<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Working in prefill routing

Read the [prefill routing guide](README.md) before changing activation, conditional disaggregation, or prefill reservation ownership.

- Trace `activation.rs` for endpoint-to-binding construction, `conditional_bypass.rs` for the preview/policy/admission sequence, `query.rs` for reservation-only calls, and `mod.rs` for execution and fallback.
- Distinguish the logical pool endpoint from an individual worker. Worker churn in a pool does not require the same machinery as rebinding to a new pool.
- Preserve activation-state handling. A configured target that is still activating is not the same as no target; execution may fail closed while a query-only path can continue.
- A preview returns one unbooked D target. Keep the policy decision and admitted plan explicit. Do not turn preview into a commitment without updating its ownership and cancellation contract.
- Preserve explicit-P-pin behavior and cancellation/invalid-argument propagation. Do not describe remote-prefill fallback as unconditional.
- Check KV and built-in routing separately: `reserve_prefill_worker` only owns a scheduler booking in the KV case.
- Treat P and D reservations as separate lifetimes. Runtime EPP reserves P before querying D, then registers D separately; a failure after P succeeds must still release P.
- Verify router accounting separately from engine execution. Marking prefill complete on D retains its decode booking and is not an engine RPC.
