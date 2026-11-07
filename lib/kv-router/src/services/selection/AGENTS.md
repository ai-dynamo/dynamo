<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Working in the selection service

Read the [selection and ownership guide](README.md) before changing admission modes, reservation APIs, or lifecycle handling.

## Follow the shared implementation

- Begin at `core/operation.rs` for the mode contract, `core/run.rs` for selection, `core/reservations.rs` for lifecycle, and `server.rs` for HTTP exposure.
- Trace both standalone HTTP and frontend `EmbeddedSelection` callers. A change in `SelectionCore` can affect both.
- Keep admission, selection, and booking distinct in documentation and API names. `Query` admits without booking; `Advisory` bypasses admission; `Lease` and `Book` differ in cleanup ownership.
- Do not describe `/select` as holding capacity or requiring a later reservation. Do not describe replica synchronization as globally atomic admission.

## Preserve ownership and accounting

- Identify the exact booking owner at each return, cancellation, and error path. An HTTP reservation must survive its creation handler; an abandoned armed handle must release its booking.
- Keep prefill completion distinct from full release. Preserve output-block accounting and its local-only replica behavior.
- Use the catalog reconciliation path for discovery-driven membership. Preserve bookings on surviving ranks during valid metadata changes.
- Keep raw overlap observability separate from effective prefill-load accounting.
- Maintain both this folder's explanation and the public standalone-selection API guide when changing their contracts.

The proposed inspection rule in the README is design guidance awaiting an API decision. It is not permission to silently change compatibility behavior.
