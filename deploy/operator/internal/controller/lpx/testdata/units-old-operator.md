<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Operator-only upgrade fixture

`units-old-operator.json` captures the renderer and request resolver at Dynamo
`20a9321cf16eb11e95fd12194c3033c210f7b2be`. It contains the input DGD, exact
serialized deployment contract (base64), old PCS, LPRs, and ConfigMap. The base
reader ran unchanged in a detached checkout; no new reader was made to emulate
legacy behavior by removing units. The fixture capture passed the base's
`TestCaptureOldUnitsRender` on 2026-11-07.

The artifact is the `units` case of `newUnitsTestRegistry`: LP20 partitions 7 and
8, 16 chips each, devicesPerNode 8, one unit with two nodes=2 reservations and an
XT connector. It also retains the legacy selected chain [7,8]. The old renderer
uses those legacy fields even though the artifact already contains units.

The authored DGD is `loadTestDGD(PipelineSingle, "units")` with the `config`
mounts removed before either reader runs. This isolates the operator lifecycle
from the separate units-template rejection test. It is not a claim that legacy
runtime binaries can run without their configuration. Runtime compatibility is
an external release gate.

To reproduce, put the embedded deployment bytes in a file registry entry named
`units`. In a test in the base controller/lpx package, load the embedded DGD,
construct `newLPXTestDeployment` and `newPreparedLPXTestReconciler`, and call
`renderPodCliqueSet` with the resolved workload and plan. Save its PCS and
resources with `selected.requests`. The current test assigns API identities,
client identity, and Grove observations through the existing test harness.

Do not regenerate this fixture with the new reader. The regression changes only
the operator implementation: DGD spec/generation, input revision, artifact bytes,
and build reference remain fixed. Equal geometry updates the PCS template and
retires requests only after Grove observes new role templates. The test preserves
the old ConfigMap until readiness and checks the settled reconcile for no changes.
