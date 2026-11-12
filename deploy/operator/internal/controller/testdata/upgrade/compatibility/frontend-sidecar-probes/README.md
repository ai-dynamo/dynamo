<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Frontend sidecar probes rendered by 1.6.0

This independent compatibility snapshot holds a Grove worker whose `frontendSidecar`
container authored partial probes: a readiness probe with a handler and
`initialDelaySeconds` only, and a liveness probe with a handler and `periodSeconds`
only. The released 1.6.0 operator created `create.yaml` with mutating admission
enabled, so the DGD carries operator origin `1.6.0`, and field-merged those probes
with its frontend defaults: readiness gained `periodSeconds: 10`,
`timeoutSeconds: 3`, and `failureThreshold: 3`; liveness gained
`initialDelaySeconds: 15`, `timeoutSeconds: 1`, and `failureThreshold: 3`.

Operator 1.7.0 replaces an authored probe as a complete value (#15581). Without the
`FrontendSidecarProbeReplacement` origin gate, taking this workload over drops the
merged fields, changes the worker clique template, and rolls the worker. The
replay's independent safety assertion compares the effective clique templates, so
the scenario fails on an operator that applies replacement to a 1.6-origin workload
and passes on one that keeps the field merge for it.

`provenance.json` records the pinned release revision, the rendered harness digest,
the observation digest, the worker hash, the sidecar probes as 1.6.0 rendered them,
and the final input digest. `../../compatibility.lock.json` records the active
input digest and last released reconciler alongside the other compatibility inputs.

Reproduce from the repository root with Go, PyYAML, GitHub CLI access, and envtest
Kubernetes binaries:

```bash
KUBEBUILDER_ASSETS=/absolute/path/to/envtest/binaries \
  python3 deploy/operator/hack/capture_frontend_sidecar_probes.py --output /new/staging/directory
```

`TestClusterOperatorUpgrade/frontend-sidecar-probes` seeds the snapshot through the
shared permissive historical phase, runs production takeover, checks the independent
safety assertion, and additionally checks the two merged probe fields that
replacement would drop. Formal release regeneration captures the takeover result and
publishes it with the compatibility lock. Serving workloads remain blocked; this is a
CPU admission/controller/API-server check, not an inference test.
