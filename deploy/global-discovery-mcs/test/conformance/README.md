<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Upstream MCS conformance

Run the Kubernetes SIG Multicluster suite against the registry and controller
from this checkout. The wrapper imports the upstream suite without copying or
modifying its assertions. All required and optional tests run, including DNS
and connectivity. Failures remain failures; there is no skip list or expected
failure conversion.

The conformance module is pinned to commit
`32672de8605f9719b9220992c75cb263ba6f3746`, the commit tagged `v0.5.2` in
[kubernetes-sigs/mcs-api](https://github.com/kubernetes-sigs/mcs-api/tree/v0.5.2/conformance).
Its nested Go module resolves as
`sigs.k8s.io/mcs-api/conformance v0.5.1-0.20260713154219-32672de8605f`.
The suite uses this module's existing MCS API and Kubernetes dependency
versions through Go module selection.

## Run on isolated kind clusters

Prerequisites: Docker, Go, kubectl, Helm 3, jq and curl. `make` installs the
pinned kind binary. The registry/controller image is built for the host's
architecture. No GPU or Global Discovery Service image is needed.

```bash
make -C deploy/global-discovery-mcs conformance
```

This creates only `mcs-conformance-hub` and `mcs-conformance-a`, using the
existing e2e image, routing and Helm helpers. Each cluster has a distinct pod
and Service subnet. The hub runs the memory registry; both run the controller.
Controller cluster IDs match the kubeconfig cluster names because the upstream
suite checks `multicluster.kubernetes.io/source-cluster` against those names.
The harness uses private kubeconfig files and never selects the machine's
current context.

The upstream suite resolves Services under `clusterset.local`. The harness does
not install a DNS implementation or rename its queries to the derived local
Service. Missing DNS support is measured as non-conformance, not hidden by
changing the suite or selecting a smaller test set.

Clusters are deleted after a full run, including a failed run. To retain them:

```bash
CONFORMANCE_KEEP=1 make -C deploy/global-discovery-mcs conformance
```

To deploy once and iterate without rebuilding the image:

```bash
make -C deploy/global-discovery-mcs conformance-setup
make -C deploy/global-discovery-mcs conformance-test
make -C deploy/global-discovery-mcs conformance-down
```

`conformance-test` only runs tests against the retained deployment. Re-run
`conformance-setup` to rebuild and redeploy changed implementation code.
Use the same `CONFORMANCE_PREFIX` and `CONFORMANCE_WORK_DIR` for setup, test
and cleanup when overriding either setting. Do not run concurrent suites on
the same clusters.

## Reports and exit status

Reports are written below `deploy/global-discovery-mcs/bin/conformance/reports`:

- `report.yaml` and `report.html`: upstream conformance results and implementation metadata.
- `results.json` and `junit.xml`: complete Ginkgo results, including execution failures.
- `suite.log`: complete test output.
- Controller/registry logs and Kubernetes MCS objects for diagnosis.

The upstream implementation version identifies the deployed content-tagged image,
including on retained-cluster runs. `environment.json` records that image, the
runner checkout revision and the upstream commit separately.
The test binary's nonzero exit status is preserved through logging and cleanup.
Read the full Ginkgo results as well as the conformance report: an execution
failure is not evidence that an implementation requirement passed.

Kubeconfigs stay outside the reports directory and must not be uploaded.
Generated reports and binaries are ignored by Git. The manually dispatched
`MCS Conformance` workflow runs this same target and uploads the reports even
when tests fail.

## Settings

| Variable | Default | Purpose |
| --- | --- | --- |
| `CONFORMANCE_PREFIX` | `mcs-conformance` | Dedicated kind cluster prefix. |
| `CONFORMANCE_WORK_DIR` | module `bin/conformance` | Build/deployment logs, private kubeconfigs and reports. |
| `CONFORMANCE_KEEP` | `0` | Set `1` to retain clusters after a full run. |
| `CONFORMANCE_TIMEOUT` | `60m` | Go test timeout for the complete suite. |
| `E2E_SKIP_BUILD` | `0` | Reuse the existing conformance image during setup. |
| `KIND_NODE_IMAGE`, `E2E_PLATFORM` | existing e2e defaults | Node image and image architecture. |

Distinct subnet overrides supported by the e2e helpers can be supplied when
running alongside another routed e2e deployment. Merely changing the cluster
prefix does not change its subnet assignments.

## Results after API fixes

The [post-fix run](results/17681-api-fixes.md) passes all 15 required checks.
Overall, 19 of 28 specifications pass; nine optional DNS-dependent cases
fail, with none skipped. The original baseline remains below for comparison.

## Recorded baseline

See the [complete #17615 baseline results](results/README.md) and upstream YAML
report. That run completed all 28 specifications with 15 passing, 13 failing
and none skipped; three failures are required API conformance checks.
