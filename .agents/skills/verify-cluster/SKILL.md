---
name: verify-cluster
description: Preflight check before deploying a Dynamo recipe -- CRDs, storage class, GPU driver, image-pull access. Returns go/no-go with remediation
user-invocable: true
disable-model-invocation: true
---

# Verify Cluster

Run a preflight against the user's Kubernetes cluster before they deploy a Dynamo recipe. Each check returns **PASS / WARN / FAIL** plus a remediation pointer when not PASS.

Output format: a single summary table at the end, plus the per-check evidence above it.

## Check 1: Dynamo CRDs are installed

```bash
kubectl get crd 2>/dev/null | grep dynamo
```

Expect three CRDs: `dynamocomponentdeployments`, `dynamographdeployments`, `dynamographdeploymentrequests`.

- **PASS** if all three present.
- **FAIL** otherwise. Remediation: install the Dynamo operator. See [Dynamo Operator install guide](../../../docs/kubernetes/dynamo-operator.md).

## Check 2: Default storage class exists

```bash
kubectl get storageclass -o jsonpath='{.items[?(@.metadata.annotations.storageclass\.kubernetes\.io/is-default-class=="true")].metadata.name}'
```

Recipes default to `storageClassName: standard`. Most clusters do not have a `standard` storage class.

- **PASS** if the default class is `standard`.
- **WARN** if the default class exists but is not `standard`. Remediation (auto-applied by `quickstart` Step 3):
  ```bash
  DEFAULT_SC=<name from above>
  yq -i "(.. | select(has(\"storageClassName\")).storageClassName) = \"$DEFAULT_SC\"" recipes/<model>/model-cache/model-cache.yaml
  ```
- **FAIL** if no default storage class. Remediation: `kubectl patch storageclass <name> -p '{"metadata":{"annotations":{"storageclass.kubernetes.io/is-default-class":"true"}}}'` or pick a class manually and pass it as a yq override.

## Check 3: GPU driver matches the runtime image

```bash
kubectl get nodes -l nvidia.com/gpu.present=true -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.nodeInfo.kernelVersion}{"\t"}{.metadata.labels.nvidia\.com/cuda\.driver-version\.full}{"\n"}{end}'
```

If the driver-version label is absent, fall back to:

```bash
kubectl debug node/<one-gpu-node> --image=nvcr.io/nvidia/cuda:12.4.1-base-ubuntu22.04 -- nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1
```

Cross-check the driver version against the image tag the user plans to deploy. Source of truth: [Feature Support Matrix](../../../docs/backends/trtllm/README.md#feature-support-matrix).

- **PASS** if the driver is at or above the matrix minimum for the chosen tag.
- **WARN** if the user has not yet picked a tag (record the driver version, surface it when the tag is picked).
- **FAIL** if the driver is below the minimum. Remediation: upgrade the driver, or pull a lower-CUDA image variant.

## Check 4: Image pull works

```bash
NAMESPACE=${NAMESPACE:-default}
kubectl run dynamo-pull-test \
  --image=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.0.1 \
  --restart=Never \
  --overrides='{"spec":{"nodeSelector":{"nvidia.com/gpu.present":"true"}}}' \
  -n "$NAMESPACE" --command -- sleep 5 2>&1
sleep 10
kubectl get pod dynamo-pull-test -n "$NAMESPACE" -o jsonpath='{.status.phase}'
kubectl delete pod dynamo-pull-test -n "$NAMESPACE" --force --grace-period=0 2>/dev/null
```

- **PASS** if pod reaches `Succeeded` (or `Running` then exits).
- **FAIL** if pod stays `Pending` with `ImagePullBackOff` / `ErrImagePull`. Remediation:
  - All canonical Dynamo images on `nvcr.io/nvidia/ai-dynamo/*` are public -- **no `imagePullSecrets` needed**. If the pull fails for these, the cluster has an egress / DNS issue.
  - Only if pulling from a private registry mirror, create a registry secret per [`docs/troubleshooting.md`](../../../docs/troubleshooting.md#image-pull-failures).

## Check 5: GPU operator is installed

```bash
kubectl get pods -n gpu-operator 2>/dev/null | grep -E 'nvidia-driver|nvidia-device-plugin|nvidia-container-toolkit' | head -5
```

- **PASS** if the device plugin pod is `Running`.
- **FAIL** if absent. Remediation: install the [NVIDIA GPU Operator](https://docs.nvidia.com/datacenter/cloud-native/gpu-operator/latest/getting-started.html).

## Summary table

After running all checks, emit:

```
| Check                  | Status | Notes                                               |
| ---------------------- | ------ | --------------------------------------------------- |
| CRDs installed         | PASS   | 3/3                                                 |
| Default storage class  | WARN   | Default is `gp3`, not `standard` -- auto-fix avail. |
| GPU driver             | PASS   | 570.x on 4 nodes (>= matrix minimum for 1.0.1)      |
| Image pull             | PASS   | nvcr.io reachable, pulled in 7s                     |
| GPU operator           | PASS   | nvidia-device-plugin Running on 4 nodes             |
```

Final verdict:
- Any **FAIL** -> deploy will fail. Fix before running `quickstart`.
- Only **WARN** -> safe to proceed; `quickstart` will auto-apply remediations where possible.
- All **PASS** -> green light.
