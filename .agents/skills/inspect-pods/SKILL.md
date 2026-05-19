---
name: inspect-pods
description: Bundle the right kubectl commands for inspecting a Dynamo deployment -- pod status, logs, recent events, describe, frontend port-forward
user-invocable: true
disable-model-invocation: true
---

# Inspect Pods

Gather everything needed to know what a Dynamo deployment is doing. The non-obvious bit is the pod selector: Dynamo labels every pod with `nvidia.com/dynamo-graph-deployment-name=<DGD-name>`, not the typical `app=` label.

Optional arguments: namespace and DGD (DynamoGraphDeployment) name. If omitted, ask in Step 1.

## Step 1: Resolve the namespace and deployment name

```bash
export NAMESPACE=${NAMESPACE:-dynamo-demo}

# List all DGDs in the namespace
kubectl get dynamographdeployment -n "$NAMESPACE" -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.state}{"\n"}{end}'
```

If exactly one is returned, use it. Otherwise ask the user to pick.

```bash
export DGD=<chosen-name>
export SELECTOR="nvidia.com/dynamo-graph-deployment-name=$DGD"
```

## Step 2: Pod status + role labels

```bash
kubectl get pods -l "$SELECTOR" -n "$NAMESPACE" \
  -o custom-columns='NAME:.metadata.name,READY:.status.containerStatuses[*].ready,STATUS:.status.phase,ROLE:.metadata.labels.nvidia\.com/dynamo-component,RESTARTS:.status.containerStatuses[*].restartCount,AGE:.metadata.creationTimestamp'
```

Look for:
- All `READY=true` -> healthy.
- `STATUS=Pending` -> Step 3 + Step 5 (events) will explain why.
- `RESTARTS > 0` -> previous-container logs (`--previous`) in Step 4.
- Mixed roles -> confirm prefill/decode/frontend are all present per the recipe.

## Step 3: Describe each non-Ready pod

```bash
for POD in $(kubectl get pods -l "$SELECTOR" -n "$NAMESPACE" \
  -o jsonpath='{range .items[?(@.status.phase!="Running")]}{.metadata.name} {end}'); do
  echo "=== $POD ==="
  kubectl describe pod "$POD" -n "$NAMESPACE" | sed -n '/^Events:/,/^$/p'
done
```

Surface:
- `FailedScheduling` -> GPU/CPU/memory shortfall. Compare requests vs `kubectl describe node`.
- `ImagePullBackOff` / `ErrImagePull` -> hand off to [`troubleshoot`](../troubleshoot/SKILL.md) "Image pull failures".
- `CreateContainerConfigError` -> usually a missing secret (HF_TOKEN) or configmap.
- `Unschedulable: 0/N nodes are available` -> taints/tolerations or nodeSelector mismatch.

## Step 4: Tail logs by role

```bash
# Frontend (HTTP server)
kubectl logs -l "$SELECTOR,nvidia.com/dynamo-component=Frontend" \
  -n "$NAMESPACE" --tail=100 --prefix=true

# Workers (decode + prefill)
kubectl logs -l "$SELECTOR,nvidia.com/dynamo-component=VllmDecodeWorker" \
  -n "$NAMESPACE" --tail=100 --prefix=true

kubectl logs -l "$SELECTOR,nvidia.com/dynamo-component=VllmPrefillWorker" \
  -n "$NAMESPACE" --tail=100 --prefix=true 2>/dev/null
```

For SGLang / TRT-LLM recipes, swap `Vllm*Worker` for `Sglang*Worker` / `TrtllmWorker` -- check the role labels printed in Step 2.

For pods that have crashed and restarted, capture the previous logs:

```bash
kubectl logs <pod> -n "$NAMESPACE" --previous --tail=200
```

## Step 5: Recent events in the namespace

```bash
kubectl get events -n "$NAMESPACE" \
  --sort-by=.lastTimestamp \
  -o custom-columns='TIME:.lastTimestamp,TYPE:.type,REASON:.reason,OBJECT:.involvedObject.name,MESSAGE:.message' \
  | tail -30
```

Filter to warnings only if the list is noisy:

```bash
kubectl get events -n "$NAMESPACE" --field-selector type=Warning --sort-by=.lastTimestamp | tail -20
```

## Step 6: Port-forward the frontend

```bash
kubectl port-forward svc/${DGD}-frontend 8000:8000 -n "$NAMESPACE"
```

In another shell:

```bash
curl -s http://localhost:8000/v1/models
curl -s http://localhost:8000/health  # NOTE: /v1/health/ready is not yet wired -- use /health
```

## Output

Return a structured summary:

1. **Pod table** from Step 2.
2. **Per-pod issues** from Steps 3-4 (only for non-Running pods or pods with restarts).
3. **Top 5 warning events** from Step 5.
4. **Verdict**: `HEALTHY` / `DEGRADED` / `BROKEN`, plus the highest-severity finding.
5. **Recommended next action**: `quickstart` smoke test (if HEALTHY), or [`troubleshoot`](../troubleshoot/SKILL.md) (if symptom matches a known entry).
