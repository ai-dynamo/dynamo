<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Reconciliation

- Uncached API-server ("quorum") reads are most likely the wrong solution to informer-cache races.
- Build level-driven reconcilers that tolerate stale cached state and converge from every observable state.
- Watch every dependency whose create, update, or delete can unblock reconciliation. Let those events requeue the owning resource after changes become visible in the cache.
- Treat a dependency that has not appeared in the cache yet as pending when its watch can drive progress; do not turn expected informer lag into a terminal failure.
- After a successful write, either continue with the object returned by the client or wait for its watch event. Do not immediately read the write back through an uncached client.
- Use durable status, conditions, or transaction markers when the reconciler must distinguish work that is pending from work that was previously observed and later deleted.
- Use an uncached read only when a concrete correctness requirement cannot be represented with durable state and watches, and document that exception.
