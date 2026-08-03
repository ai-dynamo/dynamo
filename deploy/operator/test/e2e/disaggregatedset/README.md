# DisaggregatedSet e2e tests

This suite validates the Dynamo operator's DisaggregatedSet pathway against a
live Kubernetes API server and a real LWS controller. It is intentionally a
separate phase from the controller/envtest coverage in PR #11645.

## Scope

The CPU-only lifecycle test verifies:

1. A DGD with two multinode roles creates one DGD-owned DisaggregatedSet.
2. LWS v0.9.0 creates a ready LeaderWorkerSet for each role.
3. Graph-level labels and annotations propagate to both role pod templates.
4. A graph restart rolls both roles through one DisaggregatedSet revision.
5. Removing the opt-in annotation triggers DS-to-DCD fallback through LWS.
6. Component and shared model Services transfer from DGD to DCD ownership,
   including recreation when the model Service disappears during handoff.

The preparation target installs LWS v0.9.0 and Volcano v1.14.0 when their CRDs
are absent. Existing installations are accepted only when their ready
controller images match the pinned versions; the target refuses to overwrite
different cluster-wide installations. Volcano is required because the Dynamo
operator enables its legacy multinode LWS path only when both LWS and Volcano
APIs are visible at operator startup. The Go suite validates the APIs, both LWS
CRDs, their served/storage version and status subresources, server-side schema
rejection, and both running controller images.

The suite does not special-case OrbStack. OrbStack, Kind, and remote clusters
all use the current kubeconfig and the same Kubernetes API checks.

## Prerequisites

- A reachable Kubernetes cluster.
- The configured CPU-only workload image is pullable by the cluster.

## Run

Install the cluster dependencies first:

```bash
cd deploy/operator
make prepare-e2e-disaggregatedset
```

Then install or restart the Dynamo operator from PR #11645. Feature discovery
runs at operator startup, so installing Volcano after the operator starts does
not enable multinode DCD fallback.

Finally, run the suite:

```bash
make test-e2e-disaggregatedset
```

On a clean cluster the preparation target applies these pinned manifests:

- `https://github.com/kubernetes-sigs/lws/releases/download/v0.9.0/manifests.yaml`
- `https://raw.githubusercontent.com/volcano-sh/volcano/v1.14.0/installer/volcano-development.yaml`

Common overrides:

```bash
make test-e2e-disaggregatedset \
  DISAGGREGATEDSET_NAMESPACE=default \
  DISAGGREGATEDSET_LWS_VERSION=v0.9.0 \
  DISAGGREGATEDSET_VOLCANO_VERSION=v1.14.0 \
  DISAGGREGATEDSET_WORKLOAD_IMAGE=busybox:1.36
```

The suite creates the configured namespace only when it does not already
exist. It always deletes its DGD and workloads, but retains the empty namespace
so repeated runs do not depend on cluster-level namespace finalization.
