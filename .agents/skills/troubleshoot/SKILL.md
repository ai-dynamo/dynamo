---
name: troubleshoot
description: Match a Dynamo failure symptom to a ranked cause and fix from the troubleshooting decision tree, with evidence from the live cluster
user-invocable: true
disable-model-invocation: true
---

# Troubleshoot

Map a user-described or observed symptom to the right entry in [`docs/troubleshooting.md`](../../../docs/troubleshooting.md), with cluster evidence to confirm. Cite the entry; never invent a fix.

Optional argument: a free-form symptom description, an error string, or a pod name. If omitted, ask in Step 1.

## Step 1: Get the symptom

If the user already pasted an error or named a failure mode, parse it. Otherwise ask:

- "What command or step failed?"
- "What was the error message? Paste it verbatim."
- "Optional: namespace and pod name, so I can pull live evidence."

If the user has a namespace + DGD, run [`inspect-pods`](../inspect-pods/SKILL.md) first and feed its findings into Step 2.

## Step 2: Match against the troubleshooting tree

The tree lives in [`docs/troubleshooting.md`](../../../docs/troubleshooting.md). The matching rules (one entry per symptom):

| If symptom contains... | Match to entry | Confidence |
| -- | -- | -- |
| `ImagePullBackOff`, `ErrImagePull`, `manifest unknown` | [Image pull failures](../../../docs/troubleshooting.md#image-pull-failures) | high |
| `no matches for kind "DynamoGraphDeployment"` | [Dynamo CRDs missing](../../../docs/troubleshooting.md#dynamo-crds-missing) | high |
| PVC `Pending`, `no persistent volumes available`, `provisioner not found` | [PVC stuck Pending](../../../docs/troubleshooting.md#pvc-stuck-pending) | high |
| `NVIDIA driver on your system is too old`, `CUDA error: no kernel image` | [Driver mismatch](../../../docs/troubleshooting.md#driver-mismatch--cryptic-pytorch-error) | high |
| `/v1/health/ready` returns 404 | [/v1/health/ready returns 404](../../../docs/troubleshooting.md#v1healthready-returns-404) | high |
| `cuda_runtime.h: No such file or directory` | [CUDA dev headers missing](../../../docs/troubleshooting.md#cuda-dev-headers-missing-for-source-build) | high |
| HF 401/403, gated model access | [HF_TOKEN / gated model access](../../../docs/troubleshooting.md#hf_token--gated-model-access) | high |
| Multimodal streaming truncated, vision/audio mid-response failure on vLLM | [Multimodal streaming fails on vLLM](../../../docs/troubleshooting.md#multimodal-streaming-fails-on-vllm) | medium |
| DGDR `Failed`, `pareto_analysis.py` NaN | [DGDR stuck in Failed](../../../docs/troubleshooting.md#dgdr-stuck-in-failed) | high |
| `disagg.sh` first requests 503 | [disagg.sh 503 on first requests](../../../docs/troubleshooting.md#disaggsh-503-on-first-requests) | high |

If no entry matches, say so explicitly. Do not invent. Suggest the user file a bug via the contributor `gh-issue-bug` skill.

## Step 3: Pull live evidence (when applicable)

For each matched entry, gather the evidence that confirms or refutes the match:

**Image pull**:
```bash
kubectl describe pod <pod> -n "$NAMESPACE" | grep -A2 'Failed.*pull'
```

**CRDs missing**:
```bash
kubectl get crd | grep dynamo  # expect 3, blank means missing
```

**PVC Pending**:
```bash
kubectl get pvc -n "$NAMESPACE"
kubectl describe pvc <name> -n "$NAMESPACE" | grep -A1 'Events:\|Status:'
```

**Driver mismatch**: pull the runtime container's CUDA version label, compare to host driver from [`verify-cluster`](../verify-cluster/SKILL.md) Check 3.

**`/v1/health/ready` 404**:
```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v1/health/ready  # expect 404
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health           # expect 200
```

**HF_TOKEN**:
```bash
kubectl get secret hf-token-secret -n "$NAMESPACE" -o jsonpath='{.data.HF_TOKEN}' | base64 -d | head -c 8 ; echo
```

**DGDR**:
```bash
kubectl get dynamographdeploymentrequest -n "$NAMESPACE"
kubectl logs -l app=pareto -n "$NAMESPACE" --tail=50 | grep -i nan
```

## Step 4: Output

Return:

1. **Matched entry** (markdown link to the docs anchor) + confidence.
2. **Evidence** (the kubectl/curl output that confirms or refutes the match).
3. **Fix** (verbatim copy of the docs entry's "Fix" or "Workaround" section -- do not paraphrase, do not invent).
4. **Engineering follow-up** (if the docs entry says one is tracked, surface that fact -- do not pretend the issue is resolved).
5. **Next step**: re-run [`quickstart`](../quickstart/SKILL.md) Step 7 (smoke test) after the fix is applied, or hand back to the user if the fix is non-trivial.

## When to escalate

- Symptom matches no entry in the tree.
- Symptom matches an entry but the fix doesn't resolve the failure.
- Pod logs show a panic or stack trace not covered by any entry.

In all three cases, point the user (or the operating contributor agent) at the [`gh-issue-bug`](../gh-issue-bug/SKILL.md) skill to file a structured bug report against `ai-dynamo/dynamo`.
