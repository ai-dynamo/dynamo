# `snapshotctl`

`snapshotctl` is a lower-level snapshot utility for developers and operators.
It is not the primary Dynamo user workflow. The normal user-facing path is:

```text
DynamoCheckpoint CR -> operator -> snapshot-agent
```

Use `snapshotctl` when you want to exercise checkpoint or restore behavior
directly from a worker pod manifest without going through the operator.

## Requirements

- the snapshot Helm chart must already be installed in the target namespace
- a `snapshot-agent` DaemonSet must be running in that namespace
- the namespace must already have the checkpoint PVC mounted by the agent

## Manifest requirements

`snapshotctl checkpoint --manifest ...` and `snapshotctl restore --manifest ...`
accept a Kubernetes `Pod` manifest, not a Deployment or Job manifest.

That pod manifest must:

- describe the worker pod you want to checkpoint or restore
- use the placeholder image for checkpoint-aware flows
- match the runtime-relevant worker settings you care about preserving

In practice, start from the real worker pod spec you would normally run, then
keep only the pod-level fields needed to recreate that worker accurately.

## Target containers

Which container(s) the operation acts on is controlled by a single annotation
that the snapshot layer treats as mandatory:

```yaml
metadata:
  annotations:
    nvidia.com/snapshot-target-containers: "main"
    # or "engine-0,engine-1" for a failover-style restore
```

- Checkpoints must target **exactly one** container.
- Restores must target **at least one** container; the same checkpoint is
  replayed into every named container.

`snapshotctl` will stamp the annotation for you from the CLI flag:

- `--container <name>` on the `checkpoint` subcommand (single name)
- `--containers <name>[,<name>...]` on the `restore` subcommand (one or more)

You can also pre-stamp the annotation on the manifest and omit the flag. If
both are provided they must agree — `snapshotctl` rejects mismatches instead
of silently picking one.

## Commands

Checkpoint from a manifest:

```bash
snapshotctl checkpoint \
  --manifest ./worker-pod.yaml \
  --container main \
  --namespace ${NAMESPACE}
```

If `--checkpoint-id` is omitted, `snapshotctl` generates one.

Restore by creating a new pod from a manifest:

```bash
snapshotctl restore \
  --manifest ./worker-pod.yaml \
  --containers main \
  --namespace ${NAMESPACE} \
  --checkpoint-id manual-snapshot-123
```

For an intra-pod failover restore, list every engine container:

```bash
snapshotctl restore \
  --manifest ./failover-pod.yaml \
  --containers engine-0,engine-1 \
  --namespace ${NAMESPACE} \
  --checkpoint-id manual-snapshot-123
```

Restore an existing snapshot-compatible pod in place:

```bash
snapshotctl restore \
  --pod existing-restore-target \
  --containers main \
  --namespace ${NAMESPACE} \
  --checkpoint-id manual-snapshot-123
```

## Notes

- `restore --pod` expects a pod that is already compatible with snapshot restore
- `restore --manifest` creates a new restore target pod from the manifest you provide
- `restore` returns after the restore request is submitted; it does not wait for completion
- observe restore progress with pod readiness, events/logs, and per-container
  `nvidia.com/snapshot-restore-status.<container>` annotations
- `snapshotctl` is useful for debugging and lower-level validation, but it does
  not replace the operator-managed checkpoint flow
