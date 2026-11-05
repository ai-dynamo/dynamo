<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Missing-origin legacy takeover

This independent compatibility snapshot exercises the 1.4 → 1.5 → 1.6 sequence.
The 1.4 installation creates the authored `create.yaml` before mutating admission
is enabled. Its released controller produces the complete resource graph without
an operator-origin annotation. Released 1.5 and 1.6 admission/controllers then
take over the same stored objects on the same API server. No captured annotation
is removed, no lowering code is adapted, and runtime images stay at 1.4.0.

Each stage waits for a completed production DGD reconciliation, checks that origin
is absent, and captures concrete API objects. Capture rejects changes to DGD intent,
worker hash, DGD/PCS identity, or effective clique templates across the sequence.
`provenance.json` records the pinned upstream release revisions, rendered harness
digests, per-stage observation digests, and the final input digest.

This models resources created before CREATE mutation was enabled in an installation.
It does not assert ancestry before the annotation's upstream introduction in 0.7.1.
The active input lives under the shared `testdata/upgrade` publication root, outside
`history/`, so it does not alter the frozen v1.6 capture inventory or replace a
canonical matrix entry. `provenance.json` records the original 1.4 → 1.5 → 1.6
bootstrap; it remains unchanged when later released controllers advance the input.
`../../compatibility.lock.json` records the active input's digest and last released
reconciler. Its release source and version must match the ordinary history lock.

Formal release regeneration captures this graph after successful takeover and
before the intentional worker edit. It publishes the input and compatibility lock
atomically with ordinary history and `RECONCILED_VERSION`. A later candidate consumes
that released-controller result; a stale compatibility lineage blocks the next
minor bump even when ordinary history has already advanced.

Reproduce from the repository root with Go, Helm, PyYAML, GitHub CLI access and
envtest Kubernetes binaries:

```bash
KUBEBUILDER_ASSETS=/absolute/path/to/envtest/binaries \
  python3 deploy/operator/hack/capture_missing_origin.py --output /new/staging/directory
```

Review bootstrap captures and their provenance before initializing this fixture.
Subsequent advancement uses the normal released-controller regeneration command. Ordinary
CI reads the committed snapshot and does not fetch or execute old releases.
`TestClusterOperatorUpgrade/missing-origin` uses the shared permissive historical
seeding phase before current admission. It checks independent takeover safety,
rejects an invented origin through the safety assertion, and verifies an intentional
worker edit reaches the requested immediate Grove workload generation while origin
stays absent. The final bootstrap export and release advancement both use Go
`golden.WriteManifests`, normalizing only API-assigned UID/resourceVersion/creation
metadata and owner UIDs after the raw same-server safety checks. The final export
boundary test covers stable bytes and preservation of controller state. The admission
validator tests cover rejection of origin materialization on update. The Python
lifecycle test advances the actual missing-origin path across two release cycles,
checks state handoff and missing origin, and rejects stale compatibility lineage. Serving
workloads remain blocked; these are CPU API/admission/controller checks, not
inference tests.
