<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Working across frontend and routing boundaries

Read the [frontend and shared-routing guide](README.md) for the roles of `ModelManager`, `EmbeddedSelection`, and runtime EPP.

- Follow request data from protocol parsing through preprocessing, selection inputs, dispatch, and response cleanup. Sharing a selector does not establish full pipeline parity.
- Treat `SelectionService` as the shared concrete service and `EmbeddedSelection` as its runtime adapter. Avoid describing them as implementations of one trait.
- Verify which `ModelManager` facilities a caller actually initializes. Runtime EPP uses routing/watch support without building the full frontend model catalog.
- Separate endpoint identity, worker membership, and model-card configuration. Do not claim a one-shot pool discovery task handles namespace rebinding.
- Preserve metadata needed by selection: prompt representation, cache namespace, constraints, priorities, expected output, sessions, LoRA, and multimodal block identity as supported by the path.
- A new EPP protocol adapter can call shared code in-process. Do not introduce a self-HTTP dependency merely to reuse frontend behavior.
- Before introducing schedule-only behavior, define ownership of returned bookings and which layer consumes progress, completion, and cancellation signals. Check built-in routing as well as KV routing.
- Keep design proposals distinct from implemented behavior. `query_instance_id` currently does not provide a universal booked scheduling result.

More specific folder guidance applies in routing, prefill, discovery, and protocol modules. This draft supplements those owners; it does not replace their existing guidance.
