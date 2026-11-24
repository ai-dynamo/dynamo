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

# Pure resource rendering

- Put shared resource renderers in a subpackage for their lowering under
  `internal/controller/`, in files named `*_render.go`.
- A `Render` function is pure: it takes explicit, resolved inputs and returns
  concrete Kubernetes objects without mutating its inputs. It must not accept
  a Kubernetes client, perform reads or writes, or consult process state.
- Resolve cluster-dependent inputs in the controller before calling `Render`.
  Admission and reconciliation must use the same render functions for resource
  construction, generated metadata, and resource-existence decisions. Do not
  maintain a separate admission model of a lowering.
- Keep lowering-specific decisions with their lowering. Do not collect them in
  generic `dynamo`, `names`, or utility packages, or add forwarding wrappers.
- Use import names that describe the lowering, such as `disaggregatedset` or
  `grove`; avoid aliases tied to one role, such as `dsrender`.
- Extract only the rendering needed by the change. Moving a complete lowering's
  reconciliation and lifecycle into its subpackage is a separate refactor.

# Tests

- Use one `t.Log` heading before each block that implements a test step so the test output tells the scenario's story.
- Keep bespoke test execution in the test body. Do not hide it in closures; reserve closures for a very small set of standard control helpers such as `Eventually` and manager setup.
- Do not share DGD, DGDR, or other test fixtures between tests. Every fixture must be owned by one test.
- Keep fixtures in local variables or constants so their ownership remains visible.
- Prefer a shared construction DSL when complex fixtures repeat boilerplate. The DSL should describe the object at a domain level; share the DSL, never the constructed fixture.

# clusterenv golden review

- Before changing clusterenv goldens, their normalization, or hash bindings, read [the golden review rules](testdata/README.md). These rules cover all controller goldens, including DCD, DGD, DGDR, and upgrade scenarios.
- Every change to an existing golden requires a field-by-field semantic review. Never replace expected output, recompute expected hashes, or add wildcards merely to make a failing test pass. Regeneration does not establish correctness.
- Treat changes to rollout-relevant hashes, generated names, selectors, and pod templates as potential production rollouts, even when DGD intent and runtime images are unchanged. Investigate the production change first; preserve compatibility or document and validate an intentional migration. Explain each semantic golden change and its upgrade consequences in the PR.
