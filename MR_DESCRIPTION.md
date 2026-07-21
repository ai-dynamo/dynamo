## Overview:

Update Dynamo's Grove `PodCliqueSet` generation to use the topology constraint fields required by Grove `v0.1.0-alpha.11`.

New resources now emit `topologyName` and `pack.required` instead of the deprecated `packDomain` field. Existing legacy resources are migrated in place without recreating the `PodCliqueSet` or disrupting its workloads.

## Details:

### Problem

Dynamo currently generates the pre-alpha.9 Grove topology constraint shape:

```yaml
topologyConstraint:
  packDomain: rack
```

Grove alpha.11 rejects `packDomain` on newly created workloads and requires every constraint to resolve an explicit or inherited `topologyName`.

This causes newly generated topology-aware `PodCliqueSet` resources to be rejected or remain unresolved.

### Solution

New resources now use the current Grove shape:

```yaml
topologyConstraint:
  topologyName: grove-topology
  pack:
    required: rack
```

The implementation:

- Maps DGD `clusterTopologyName` to Grove `topologyName`.
- Emits `pack.required` instead of deprecated `packDomain` for new resources.
- Allows component constraints to inherit the deployment-level topology name.
- Adds an explicit topology name to service-only constraints when no constrained parent exists.
- Handles both single-node PodClique constraints and multinode PodCliqueScalingGroup constraints.

### Backward compatibility

Grove treats topology constraints as immutable but provides narrow upgrade exceptions for legacy resources. A legacy constraint cannot change both its missing topology name and deprecated packing representation in one update.

The reconciler therefore migrates legacy resources in two steps:

1. Add the missing `topologyName` while retaining `packDomain`.
2. Replace `packDomain` with the equivalent `pack.required` on the next reconciliation.

The existing `PodCliqueSet` update watch drives the second reconciliation. This avoids uncached reads, resource recreation, and workload disruption.

### Validation

Added coverage for:

- Deployment-level topology constraints.
- Service-only single-node constraints.
- Service-only multinode constraints.
- Two-phase migration of legacy constraints.
- Create and update validation against Grove alpha.11's embedded CRD and CEL rules.

The exact ARM64 Docker targets pass:

```bash
docker buildx build \
  --platform linux/arm64 \
  --target linter \
  --progress=plain \
  --build-context snapshot=../snapshot \
  .

docker buildx build \
  --platform linux/arm64 \
  --target tester \
  --progress=plain \
  --build-context snapshot=../snapshot \
  .
```

## Where should the reviewer start?

- `deploy/operator/internal/dynamo/grove.go`: conversion from Dynamo topology constraints to the Grove alpha.11 API shape.
- `deploy/operator/internal/dynamo/graph.go`: propagation of the deployment topology name to single-node and multinode component constraints.
- `deploy/operator/internal/controller/dynamographdeployment_controller.go`: two-phase compatibility migration for existing legacy `PodCliqueSet` resources.
- `deploy/operator/internal/dynamo/grove_schema_test.go`: validation against Grove alpha.11's embedded CRD and CEL rules.
- `deploy/operator/internal/controller/dynamographdeployment_controller_test.go`: compatibility migration coverage.

## Related Issues

**🔗 This PR is linked to an issue:**

- Closes #11399
