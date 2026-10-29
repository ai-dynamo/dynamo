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
The fixture is separate from the frozen release inventory under `testdata/upgrade`:
it does not advance `RECONCILED_VERSION`, alter the scoped v1.6 capture, or substitute
for any canonical matrix entry. Its final stored state was reconciled by 1.6.
Future development replays are a fixed missing-origin regression, not certification
of skipped-minor upgrades. The rolling history remains the adjacent-upgrade gate.

Reproduce from the repository root with Go, Helm, PyYAML, GitHub CLI access and
envtest Kubernetes binaries:

```bash
KUBEBUILDER_ASSETS=/absolute/path/to/envtest/binaries \
  python3 deploy/operator/hack/capture_missing_origin.py --output /new/staging/directory
```

Review the new capture and its provenance before replacing this fixture. Ordinary
CI reads the committed snapshot and does not fetch or execute old releases.
`TestClusterOperatorUpgrade/missing-origin` uses the shared permissive historical
seeding phase before current admission. It checks independent takeover safety,
rejects an invented origin through the safety assertion, and verifies an intentional
worker edit reaches the requested immediate Grove workload generation while origin
stays absent. Serializer tests cover capture/write/read preservation; the admission
validator tests cover rejection of origin materialization on update. The Python
lifecycle test checks missing-origin preservation through regeneration. Serving
workloads remain blocked; these are CPU API/admission/controller checks, not
inference tests.
