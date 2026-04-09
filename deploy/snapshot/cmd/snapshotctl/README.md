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
- contain exactly one worker container
- use the placeholder image for checkpoint-aware flows
- match the runtime-relevant worker settings you care about preserving

In practice, start from the real worker pod spec you would normally run, then
keep only the pod-level fields needed to recreate that worker accurately.

## Commands

Checkpoint from a manifest:

```bash
snapshotctl checkpoint \
  --manifest ./worker-pod.yaml \
  --namespace ${NAMESPACE}
```

If `--checkpoint-id` is omitted, `snapshotctl` generates one.

Restore by creating a new pod from a manifest:

```bash
snapshotctl restore \
  --manifest ./worker-pod.yaml \
  --namespace ${NAMESPACE} \
  --checkpoint-id manual-snapshot-123
```

Restore an existing snapshot-compatible pod in place:

```bash
snapshotctl restore \
  --pod existing-restore-target \
  --namespace ${NAMESPACE} \
  --checkpoint-id manual-snapshot-123
```

## Notes

- `restore --pod` expects a pod that is already compatible with snapshot restore
- `restore --manifest` creates a new restore target pod from the manifest you provide
- `snapshotctl` is useful for debugging and lower-level validation, but it does
  not replace the operator-managed checkpoint flow
