# Dynamo Snapshot — S3 object-store backend

These manifests configure Dynamo Snapshot to store **container checkpoint
artifacts** in an S3-compatible bucket instead of a PVC. Instead of requiring
shared RWX storage, the snapshot-agent stages each checkpoint on its own
filesystem, uploads the artifact tree to the bucket, and on restore downloads
it back before CRIU restore. The backend is endpoint-agnostic (path-style
addressing, configurable endpoint/region) and uses no vendor-specific features.

> **Weights too:** model-weight snapshots (GPU Memory Service) have a parallel
> object-store path. See `lib/gpu_memory_service/snapshot/objectstore.py` and
> set `GMS_SNAPSHOT_S3_BUCKET` + the `FSSPEC_S3_*` env vars on the GMS process.

## Files

| File | Purpose |
|------|---------|
| `s3-credentials.secret.yaml` | Secret with `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`, projected into the agent. |
| `snapshot-agent-values.yaml` | `snapshot` chart override: endpoint, bucket, staging path, credentials ref. |
| `operator-values.yaml` | operator chart override: enable checkpoints, `type: s3`, matching `basePath`. |

## How it fits together

- The **snapshot-agent** (DaemonSet) holds the authoritative S3 endpoint,
  bucket, and credentials. Credentials come from the Secret via `envFrom` —
  they never enter a ConfigMap or pod annotation.
- The **operator** only stamps workload pods with `type=s3` and the agent-local
  staging `basePath`. So `checkpoint.storage.s3.basePath` (operator) **must
  match** `storage.s3.basePath` (agent).
- Artifacts are keyed `s3://<bucket>/<prefix>/<checkpointId>/versions/<n>/...`,
  with `manifest.yaml` uploaded last as the completion marker.

## Apply

```bash
SNAP_NS=dynamo-snapshot       # namespace for the snapshot-agent
OPER_NS=dynamo-system         # namespace for the operator

# 1. Ensure the bucket exists on your S3 endpoint.
aws s3 mb s3://checkpoints    # or your provider's tooling / console

# 2. Credentials Secret (edit values first). Create the namespace first so the
#    Secret has somewhere to land.
kubectl create namespace "$SNAP_NS" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$SNAP_NS" apply -f s3-credentials.secret.yaml

# 3. snapshot-agent pointed at the bucket (edit endpoint/bucket first).
helm upgrade --install snapshot ../../../helm/charts/snapshot \
  -n "$SNAP_NS" --create-namespace \
  -f snapshot-agent-values.yaml

# 4. Operator with s3 checkpoint storage.
helm upgrade --install dynamo-operator \
  ../../../helm/charts/platform/components/operator \
  -n "$OPER_NS" --create-namespace \
  -f operator-values.yaml
```

## Verify the storage path

After a checkpoint is triggered on a workload (`DynamoGraphDeployment` with
checkpoint enabled):

```bash
# Agent log shows the upload:
kubectl -n "$SNAP_NS" logs ds/snapshot-snapshot-agent | grep -i "object store"
#   -> "Uploading checkpoint artifact to object store" ... key_prefix=...

# Artifacts landed in the bucket:
aws s3 ls --recursive s3://checkpoints/
#   -> <checkpointId>/versions/1/manifest.yaml, .../pages-*.img, rootfs-diff.tar, ...

# On restore, the agent log shows the download:
#   -> "Downloading checkpoint artifact from object store"
```

## Automated tests

`.github/workflows/snapshot-s3-tests.yml` exercises this storage path in CI:

- **go-tests** / **python-tests** — run the snapshot/operator unit tests and
  the GMS object-store tests. The live S3 round-trips (`TestS3RoundTrip`,
  `test_s3_round_trip`) run only when an S3 endpoint is supplied via repo
  vars/secrets, and self-skip otherwise (CI pulls no object-store images).
- **helm-render-s3** — renders both charts with the example values above and
  asserts the s3 wiring (staging volume, creds envFrom, no PVC).

## Notes

- `storage.type: s3` requires `storage.accessMode: agentMount` (enforced by the
  chart) — the agent stages artifacts locally before upload.
- Staging volumes hold a full checkpoint artifact and default to an `emptyDir`
  (node ephemeral storage). Size them with `storage.s3.staging.sizeLimit` (agent)
  and `checkpoint.storage.s3.staging.sizeLimit` (operator, restore pods); for
  artifacts larger than node ephemeral storage, override with `staging.hostPath`
  (e.g. local NVMe) or `staging.pvcName`.
- For a plain-HTTP in-cluster endpoint, set `storage.s3.useSSL: false` and use
  the in-cluster Service DNS name as the endpoint.
- Credentials may instead be supplied with the standard `FSSPEC_S3_*` names if
  you reuse the same Secret for the GMS (Python) weight-snapshot path.
