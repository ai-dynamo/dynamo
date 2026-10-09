---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Restore Workers with Dynamo Snapshot
subtitle: Capture an initialized TensorRT-LLM worker and verify that a restored worker serves requests
---

NVIDIA Dynamo Snapshot restores initialized workers so new replicas can skip model
initialization. This walkthrough captures a TensorRT-LLM worker, restores it, and
sends a request through the Dynamo frontend. Choose a single-GPU Qwen3 example
or an eight-GPU GLM 5.3 example.

<Warning>
Experimental. Consult the
[backend compatibility matrix](../../../reference/general/compatibility.mdx) before
using another backend or topology.
</Warning>

## Prerequisites

- A Kubernetes cluster with [NVIDIA GPU support](../../installation/install-dynamo.md)
  and enough full GPUs on one node for the selected example, with MIG disabled. Review Snapshot's
  [driver, runtime, and privilege requirements](https://github.com/ai-dynamo/snapshot#prerequisites).
- A storage class that supports `ReadWriteMany` (RWX) for Snapshot artifacts, and a
  writable `model-cache` PersistentVolumeClaim (PVC) in the workload namespace.
  These are separate volumes. See [Model Storage](../../installation/model-storage/overview.md).
- `helm`, `kubectl`, `curl`, and `jq`, plus access to the runtime image registry and
  Hugging Face. The capture Pod downloads the model if it is not cached.

The examples select `runtimeClassName: nvidia`. Use your cluster's RuntimeClass
name, or omit the field if the default runtime provides GPU support.

## Choose an Example

| Example | GPUs on one node | Model cache | Snapshot artifact storage |
| --- | --- | --- | --- |
| [Qwen3-0.6B](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/trtllm/deploy/snapshot/qwen3-0.6b.yaml) | 1 | 10 GiB | 1 TiB |
| [GLM 5.3 NVFP4](https://github.com/ai-dynamo/dynamo/blob/main/examples/backends/trtllm/deploy/snapshot/glm-5.3.yaml) | 8 B200s, 1 TiB host memory | 2 TiB, RWX | 4 TiB |

The multi-GPU example uses tensor and expert parallelism across eight GPUs.
Both examples are adapted from [Snapshot's framework recipes](https://github.com/ai-dynamo/snapshot/pull/456).
Dynamo retains the KV cache during TensorRT-LLM capture; the GLM configuration
limits its allocation to 10% of free GPU memory. The standalone recipe instead
releases KV before capture, so its artifact sizes and restore timings do not
apply to this adaptation.

Set `RELEASE_VERSION` to your chosen published [Dynamo release](https://github.com/ai-dynamo/dynamo/releases),
without the leading `v`. Use the same value for the
platform chart, runtime image, and example manifest. Replace the storage class
with your cluster's RWX storage class:

```bash
: "${RELEASE_VERSION:?Set RELEASE_VERSION to your published Dynamo release}"
export PLATFORM_NAMESPACE=dynamo-system
export NAMESPACE=dynamo-snapshot
export RWX_STORAGE_CLASS=your-rwx-storage-class
export TRTLLM_IMAGE="nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:${RELEASE_VERSION}"
export RECIPE=qwen3-0.6b
export MODEL=Qwen/Qwen3-0.6B
export DGD=qwen3-0-6b-trtllm-snapshot-restore
export SNAPSHOT_STORAGE_SIZE=1Ti
export CAPTURE_TIMEOUT=20m
export RESTORE_TIMEOUT=5m
```

For the eight-GPU example, replace those recipe settings:

```bash
export RECIPE=glm-5.3
export MODEL=RadixArk/GLM-5.3-NVFP4
export DGD=glm-5-3-trtllm-snapshot-restore
export SNAPSHOT_STORAGE_SIZE=4Ti
export CAPTURE_TIMEOUT=60m
export RESTORE_TIMEOUT=30m
```

Provision `model-cache` in `NAMESPACE` before deploying. The GLM example mounts
it in the frontend as well as the capture and serving Pods, so it needs RWX
access. The storage sizes above are starting allocations; check free space
before capturing, especially if you retain multiple artifacts.

<Steps>
<Step title="Install the platform with Snapshot">

For a test installation, enable the bundled Snapshot chart and the Dynamo
integration together. The platform chart selects the matching Snapshot version;
no separate Snapshot version is needed.

```bash
helm upgrade --install dynamo-platform \
  "https://helm.ngc.nvidia.com/nvidia/ai-dynamo/charts/dynamo-platform-${RELEASE_VERSION}.tgz" \
  --namespace "$PLATFORM_NAMESPACE" --create-namespace --reuse-values \
  --set global.snapshot.install=true \
  --set dynamo-operator.checkpoint.enabled=true \
  --set-string snapshot.storage.pvc.storageClass="$RWX_STORAGE_CLASS" \
  --set-string snapshot.storage.pvc.size="$SNAPSHOT_STORAGE_SIZE" \
  --wait --timeout 10m

kubectl rollout status deployment/dynamo-platform-snapshot-operator \
  -n "$PLATFORM_NAMESPACE" --timeout=300s
kubectl rollout status daemonset/dynamo-platform-snapshot-agent \
  -n "$PLATFORM_NAMESPACE" --timeout=300s
kubectl get crd snapshotjobs.nvidia.com podsnapshots.nvidia.com
```

For an existing platform release, use its name, namespace, and version.
`--reuse-values` preserves its settings. The Snapshot chart creates an RWX
artifact PVC; use `snapshot.storage.pvc` values to customize its size or reuse an
existing claim. See [Snapshot storage configuration](https://github.com/ai-dynamo/snapshot/blob/main/docs/operations/storage.md).

<Note>
If Snapshot is already managed separately, keep `global.snapshot.install=false`
and omit the `snapshot.storage.pvc` settings. Verify that its APIs and agents are
ready before enabling `dynamo-operator.checkpoint.enabled`. Do not install a
second cluster-wide Snapshot operator. Separate installation is recommended for
production; see [Snapshot installation](https://github.com/ai-dynamo/snapshot/blob/main/docs/operations/install.md).
</Note>

</Step>
<Step title="Deploy the capture and serving configuration">

Download the selected recipe from the matching release and set the runtime image:

```bash
export EXAMPLE_URL="https://raw.githubusercontent.com/ai-dynamo/dynamo/v${RELEASE_VERSION}/examples/backends/trtllm/deploy/snapshot"
curl --fail --location "$EXAMPLE_URL/$RECIPE.yaml" -o snapshot-template.yaml
sed "s|my-registry/tensorrtllm-runtime:my-tag|${TRTLLM_IMAGE}|g" \
  snapshot-template.yaml > snapshot-restore.yaml
kubectl get pvc model-cache -n "$NAMESPACE"
```

For GLM, download the pinned model before starting capture. This keeps model
download time outside Dynamo's one-hour capture deadline:

```bash
if [ "$RECIPE" = glm-5.3 ]; then
  curl --fail --location "$EXAMPLE_URL/glm-5.3-model-cache.yaml" \
    -o model-cache-template.yaml
  sed "s|my-registry/tensorrtllm-runtime:my-tag|${TRTLLM_IMAGE}|g" \
    model-cache-template.yaml > model-cache-job.yaml
  kubectl apply -n "$NAMESPACE" -f model-cache-job.yaml
  kubectl wait job/glm-5-3-model-cache -n "$NAMESPACE" \
    --for=condition=Complete --timeout=180m
fi
kubectl apply -n "$NAMESPACE" -f snapshot-restore.yaml
```

The example sets `experimental.checkpoint.enabled: true` and
`startupPolicy: WaitForCheckpoint`. Dynamo creates a `SnapshotJob` to initialize
and capture the worker. Serving replicas stay at zero until capture completes,
then start from the resulting `PodSnapshot`.

<Note>
The default policy, `Immediate`, starts serving workers cold while capture runs.
Only Pods created after capture completes restore; existing workers are not
restarted when the snapshot becomes ready.
</Note>

</Step>
<Step title="Verify capture and restore">

Wait for the DGD to report a usable artifact, then verify the worker itself:

```bash
kubectl wait "dgd/$DGD" -n "$NAMESPACE" \
  --for='jsonpath={.status.checkpoints.TRTLLMWorker.ready}=true' --timeout="$CAPTURE_TIMEOUT"
export SNAPSHOT_NAME=$(kubectl get "dgd/$DGD" -n "$NAMESPACE" \
  -o jsonpath='{.status.checkpoints.TRTLLMWorker.checkpointName}')
kubectl get "podsnapshot/$SNAPSHOT_NAME" -n "$NAMESPACE"
kubectl wait "dgd/$DGD" -n "$NAMESPACE" \
  --for=condition=Ready --timeout="$RESTORE_TIMEOUT"

export WORKER_SELECTOR="nvidia.com/dynamo-graph-deployment-name=$DGD,nvidia.com/dynamo-component=TRTLLMWorker,"'!nvidia.com/snapshot-job'
kubectl wait pod -n "$NAMESPACE" -l "$WORKER_SELECTOR" \
  --for=condition=Ready --timeout="$RESTORE_TIMEOUT"
export WORKER_POD=$(kubectl get pod -n "$NAMESPACE" -l "$WORKER_SELECTOR" \
  -o jsonpath='{.items[0].metadata.name}')
kubectl wait "pod/$WORKER_POD" -n "$NAMESPACE" \
  --for=condition=nvidia.com/Restored --timeout="$RESTORE_TIMEOUT"
test "$(kubectl get "pod/$WORKER_POD" -n "$NAMESPACE" \
  -o jsonpath='{.metadata.annotations.nvidia\.com/restore-from}')" = "$SNAPSHOT_NAME"
```

The selector excludes the capture Pod. Success requires both
`nvidia.com/Restored=True` and the expected `nvidia.com/restore-from` annotation;
artifact readiness alone does not prove a successful restore.

</Step>
<Step title="Send a request to the restored worker">

Forward the frontend port in one terminal:

```bash
kubectl port-forward -n "$NAMESPACE" "svc/$DGD-frontend" 8000:8000
```

In another terminal, set `MODEL` to the selected model name and send a request:

```bash
curl --fail --retry 6 --retry-all-errors --retry-delay 5 \
  http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "$(jq -n --arg model "$MODEL" \
    '{model:$model,messages:[{role:"user",content:"What is the capital of France?"}],max_tokens:256,temperature:0,stream:false,chat_template_kwargs:{enable_thinking:false}}')" \
  | jq -e '.choices[0].message.content | select(test("Paris"; "i"))'
```

</Step>
</Steps>

## If a step fails

Inspect the capture conditions and workload events:

```bash
kubectl describe dgd "$DGD" -n "$NAMESPACE"
kubectl get snapshotjobs,podsnapshots -n "$NAMESPACE" \
  -l "nvidia.com/dynamo-graph-deployment-name=$DGD" -o yaml
kubectl get pods -n "$NAMESPACE"
```

- **Capture is pending:** check GPU scheduling, image pulls, the model-cache PVC,
  and Snapshot agent readiness.
- **Capture fails:** inspect the `SnapshotJob` failure reason and capture Pod logs.
  Keep the recipe's `OMPI_MCA_pml=ob1` and `OMPI_MCA_btl=tcp,self` settings.
- **Restore fails:** inspect the worker's `nvidia.com/Restored` condition and
  [Snapshot troubleshooting](https://github.com/ai-dynamo/snapshot/blob/main/docs/operations/troubleshooting.md).

## Clean Up

Stop port forwarding, then delete the example:

```bash
kubectl delete -n "$NAMESPACE" -f snapshot-restore.yaml
if [ "$RECIPE" = glm-5.3 ]; then
  kubectl delete job/glm-5-3-model-cache -n "$NAMESPACE"
fi
```

The default `deletionPolicy: Delete` removes the DGD-managed capture and artifact.
The model-cache PVC and the Snapshot installation remain available for other
workloads. See the [checkpoint configuration reference](../../../reference/kubernetes-api/dynamo-component-deployment.mdx#experimentalspec)
for reuse and retention rules.
