# Test-first PoC: Dynamo Snapshot for a Regolo vLLM container

This directory is a self-contained, non-production PoC. It does not change any
Regolo repository or public Regolo, LiteLLM, KubeAI, or OpenAI-compatible API.
It pins NVIDIA Dynamo `v1.3.0` at commit
`8ce9e22f11576402102ea9d8b8e46233f5430a0d` and targets one L40S on one node.

The local checkout contains no cluster credentials, `kubectl`, `helm`, GPU, or
Regolo image/model coordinates. Consequently, the implementation and frozen
contract can be verified locally, but baseline collection, image construction,
checkpointing, and the 20-run V1 campaign must run in the authorized temporary
GPU environment.

## Layout

- `verification/v0`: frozen charter, protocol, black-box observer, inventory
  collector, acceptance tests and SHA-256 manifest.
- `implementation`: placeholder image, pinned builders, namespace-only RBAC,
  Kubernetes Pod template, protocol primitives and campaign tools.
- `validation`: completed report and sanitized V1 evidence. Full runtime data
  belongs outside Git in `artifacts/`.
- `diagnostics`: reproducible V1 restore-phase analysis and the measured V2
  diagnostic report.
- `verification/v2`: a separate V2 draft that cannot be executed until it is
  explicitly approved.

## V0: verify and collect the unmodified baseline

First verify the immutable suite:

```bash
make verify-v0-checksum
make test
```

Before building the candidate, copy `campaign.example.json`, replace every
placeholder, and use immutable image/model revisions. Render one unmodified
Pod, inventory it read-only, then collect the 10 cold starts. The Pod template
mounts `/var/lib/regolo-vllm-poc/hf-cache` into the standard Hugging Face cache
path so cold and restore runs use identical pre-populated model storage and do
not include network download time:

```bash
python3 verification/v0/harness/inventory.py \
  --namespace "$NAMESPACE" --pod "$BASELINE_POD" --container server \
  --node "$NODE_NAME" --pvc snapshot-pvc \
  > artifacts/v0/environment-inventory.json

implementation/bin/collect-baseline \
  --campaign campaign.json --output-dir artifacts/v0

implementation/bin/seal-artifacts artifacts/v0
```

Review V0 before I1. Mount `verification/v0` read-only in the implementation
and validation sessions. The scripts never modify it.

## I1: build and deploy the candidate

The placeholder build first creates Dynamo's upstream `placeholder` target from
the pinned commit and exact Regolo base digest. A final small layer takes
`cuda-checkpoint`, its helper and `nsrestore` from the same immutable v1.3.0
agent digest used by the chart, then installs the generic `snapshot-entrypoint`.
Because the pinned Regolo vLLM runtime is Ubuntu 22.04 while Dynamo's upstream
placeholder builder defaults to Ubuntu 24.04, the build compiles the same pinned
CRIU source against the exact application base and selects Jammy's
`libgnutls30` runtime package. This avoids copying Noble-linked CRIU binaries
into the Jammy workload without changing the pinned CRIU or Dynamo revisions.
The entrypoint starts the original command,
waits for `SNAPSHOT_READY_URL`, writes `ready-for-snapshot`, and enters inert
standby when Dynamo injects `DYN_SNAPSHOT_RESTORE_STANDBY=1`.
The normal Pod readiness probe remains HTTP `/health`; snapshotctl replaces it
with the `ready-for-snapshot` sentinel only in the checkpoint Job. On restore,
the pinned protocol gates startup with the HTTP probe until CRIU has restored
the server, then the same probe reports serving readiness.

```bash
export REGOLO_IMAGE_DIGEST='registry/regolo-vllm@sha256:<64-hex>'
export PLACEHOLDER_IMAGE='registry/regolo-vllm-snapshot:poc-v0.1'
implementation/bin/build-placeholder
docker push "$PLACEHOLDER_IMAGE"
# Resolve and record registry/regolo-vllm-snapshot@sha256:<digest> as candidate_image_digest.

implementation/bin/build-snapshotctl
```

Install only the pinned `snapshot` chart. The supplied Role and RoleBinding are
namespace-scoped; chart RBAC is disabled to avoid its optional cluster-scoped
DRA ResourceSlice reader. The DaemonSet is privileged but pinned to exactly one
hostname. The PVC is 1 TiB and RWO because this PoC is same-node and sequential.

```bash
export NAMESPACE=dynamo-snapshot-poc
export NODE_NAME=<l40s-node>
export STORAGE_CLASS=<encrypted-storage-class>
export RUNTIME_TYPE=containerd   # or crio
export IMAGE_PULL_SECRET=<temporary-ngc-secret>  # omit for a public agent image
implementation/bin/install-snapshot-chart

implementation/bin/preflight \
  --namespace "$NAMESPACE" --node "$NODE_NAME" --gpu-probe-pod "$BASELINE_POD" \
  --storage-encryption-confirmed
```

Render the snapshot candidate Pod and derive its compatibility identity. The
full hash is the binding identity retained in the campaign. Dynamo v1.3.0 also
places its checkpoint ID in a Kubernetes label, whose value is limited to 63
characters, so the operational checkpoint locator is deterministically mapped
to `h-` plus the first 61 hash characters. The tools verify that mapping before
checkpoint and restore; the full SHA-256 is never truncated for compatibility
comparison:

This single-GPU PoC also uses snapshotctl's pinned
`--disable-cuda-checkpoint-job-file` mode. The v1.3.0 controller otherwise
wraps the workload with `cuda-checkpoint --launch-job`, while the
`cuda-checkpoint` binary shipped in the pinned v1.3.0 agent image does not
implement that option. The upstream source describes the wrapper as required
for multi-GPU checkpoints; multi-GPU is outside this protocol's scope.

```bash
implementation/bin/render-pod --mode snapshot \
  --namespace "$NAMESPACE" --pod-name checkpoint-source --run-id checkpoint \
  --node "$NODE_NAME" --image "$CANDIDATE_IMAGE_DIGEST" \
  --model "$MODEL_ID" --model-revision "$MODEL_REVISION" \
  --original-command-json "$ORIGINAL_COMMAND_JSON" \
  --output artifacts/i1/checkpoint-pod.json

implementation/bin/make-identity \
  --manifest artifacts/i1/checkpoint-pod.json \
  --image-digest "$CANDIDATE_IMAGE_DIGEST" \
  --model-revision "$MODEL_REVISION" --driver-version "$DRIVER_VERSION" \
  --output artifacts/i1/compatibility-identity.json

COMPATIBILITY_HASH=$(implementation/bin/compatibility-hash artifacts/i1/compatibility-identity.json)
CHECKPOINT_ID=$(implementation/bin/checkpoint-id "$COMPATIBILITY_HASH")
implementation/bin/audit-no-secrets \
  --manifest artifacts/i1/checkpoint-pod.json --image "$PLACEHOLDER_IMAGE"
implementation/bin/create-checkpoint --snapshotctl dist/snapshotctl \
  --manifest artifacts/i1/checkpoint-pod.json --namespace "$NAMESPACE" \
  --compatibility-hash "$COMPATIBILITY_HASH" --checkpoint-id "$CHECKPOINT_ID" \
  --output artifacts/i1/checkpoint.json
implementation/bin/seal-artifacts artifacts/i1
```

Do not send any inference request to the checkpoint source. The entrypoint uses
only `/health`, so the KV cache remains empty. The PodSpec explicitly sets
`automountServiceAccountToken: false`; this prevents Kubernetes from injecting
a projected API token into the process being checkpointed. The secret audit
fails closed when that field is absent or true. Copy `COMPATIBILITY_HASH` and
`CHECKPOINT_ID` into `campaign.json`; V1 refuses to run unless both match the
checkpoint metadata and deterministic mapping.

## V1: blinded campaign and report

Generate the registered schedule once. Protect the key from the person/process
that aggregates the opaque records:

```bash
implementation/bin/generate-run-plan --output-dir artifacts/v1
implementation/bin/run-campaign --campaign campaign.json \
  --run-plan artifacts/v1/run-plan.json --key artifacts/v1/unblinding-key.json \
  --checkpoint-metadata artifacts/i1/checkpoint.json --output-dir artifacts/v1/runs

implementation/bin/analyze-results --results artifacts/v1/runs/results.jsonl \
  --key artifacts/v1/unblinding-key.json --output-dir artifacts/v1/report
implementation/bin/seal-artifacts artifacts/v1
```

`analyze-results` writes and hashes the blinded summary before opening the key.
Populate `validation/report-template.md`, retaining all failures and deviations.
An exclusion is manual and allowed only with raw cluster-event evidence; repeat
both runs in that block before analysis.

## Cleanup

Cleanup is deliberately gated and should run only after the report and approved
retention period. Namespace deletion removes the namespace-local PVC and
checkpoint; delete the registry candidate separately under the registry's
approved retention workflow.

```bash
export NAMESPACE=dynamo-snapshot-poc
export CONFIRM_DELETE_NAMESPACE=dynamo-snapshot-poc
implementation/bin/cleanup-namespace
```
