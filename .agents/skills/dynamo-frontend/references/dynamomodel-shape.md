# DynamoModel CR Shape (v1alpha1)

Annotated field reference for `DynamoModel`. Per, DynamoModel is
**v1alpha1-only** — no v1beta1 schema exists. Skill examples must not
write `nvidia.com/v1beta1` for this kind.

Source: `deploy/operator/api/v1alpha1/dynamo_model_types.go` and the
CRD at `deploy/operator/config/crd/bases/nvidia.com_dynamomodels.yaml`
on the target release branch (verify with the §11 commands in
[`DYNAMO_REPO_SURVEY.md`](../../../docs/DYNAMO_REPO_SURVEY.md)).

---

## 1. Resource Identifiers

| Field | Value |
|---|---|
| `apiVersion` | `nvidia.com/v1alpha1` |
| `kind` | `DynamoModel` |
| Plural | `dynamomodels` |
| Singular | `dynamomodel` |
| Scope | `Namespaced` |

---

## 2. Top-Level Spec Fields

| Field | Required | Type | Default | Purpose |
|---|---|---|---|---|
| `baseModel` | Yes | string | — | HuggingFace model ID or local path |
| `type` | Yes | enum | — | One of `chat`, `completions`, `embeddings`, `realtime` |
| `aliases` | No | list[string] | — | Additional names clients can use to refer to this model |
| `quantization` | No | object | — | Hint for the Frontend about the model's quantization (informational) |
| `metadata` | No | object | — | Arbitrary annotations the Frontend can surface in `/v1/models` |

---

## 3. Minimal Example

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata:
  name: qwen3-06b
  namespace: dynamo-system
spec:
  baseModel: Qwen/Qwen3-0.6B
  type: chat
```

After apply, `/v1/models` returns:

```json
{
  "data": [
    {"id": "qwen3-06b", "object": "model", "owned_by": "dynamo", ...}
  ]
}
```

The client refers to the model as `qwen3-06b` (the CR's
`metadata.name`), not as `Qwen/Qwen3-0.6B` (the underlying HF id).

---

## 4. Aliases

```yaml
spec:
  baseModel: Qwen/Qwen3-0.6B
  type: chat
  aliases:
    - qwen-small
    - qwen3-tiny
```

Clients can call any of `qwen3-06b`, `qwen-small`, or `qwen3-tiny` and
get routed to the same worker.

---

## 5. Quantization Hint

```yaml
spec:
  baseModel: Qwen/Qwen3-32B-FP8
  type: chat
  quantization:
    algo: fp8
    granularity: per-tensor
```

The hint is informational; the Frontend uses it to populate
`/v1/models` response fields and to tag metrics. The actual
quantization is controlled by the worker's `--load-format` /
`quant_config.json` (per
[`dynamo-optimize/references/modelopt-cli.md`](../../dynamo-optimize/references/modelopt-cli.md)).

---

## 6. Metadata

```yaml
spec:
  baseModel: meta-llama/Llama-3-8B-Instruct
  type: chat
  metadata:
    annotations:
      tenant: "team-alpha"
      tier: "production"
```

Annotations surface in `/v1/models` and on the Frontend's `/metrics`
labels, which is useful for multi-tenant billing or per-tier SLO
tracking.

---

## 7. Multi-Model Registration

Each model gets its own DynamoModel CR. Apply them in one manifest:

```yaml
---
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata: {name: qwen-small, namespace: dynamo-system}
spec: {baseModel: Qwen/Qwen3-0.6B, type: chat}
---
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata: {name: llama-8b, namespace: dynamo-system}
spec: {baseModel: meta-llama/Llama-3-8B-Instruct, type: chat}
---
apiVersion: nvidia.com/v1alpha1
kind: DynamoModel
metadata: {name: text-embed, namespace: dynamo-system}
spec: {baseModel: BAAI/bge-large-en-v1.5, type: embeddings}
```

The DGD's worker services must actually load these models (set the
`--model` arg on each worker accordingly), and the Frontend must have
`--enable-embeddings` for the embeddings type to be exposed.

---

## 8. Status Fields

The operator writes to `.status` after the CR is applied:

| Field | Meaning |
|---|---|
| `.status.ready` | Boolean — is the model registered with the Frontend |
| `.status.workers` | List of worker pods serving this model |
| `.status.lastObserved` | Timestamp of the last reconcile |

Read with:

```bash
kubectl get dynamomodel <name> -n <ns> -o jsonpath='{.status.ready}'
kubectl describe dynamomodel <name> -n <ns>
```

When `.status.ready` stays false for >2 minutes after apply, it's
typically the registration window (workers downloading weights) or
 (HF token not on Frontend).

---

## 9. v1alpha1-Only Note

Per: DynamoModel has no v1beta1 schema. The CRD does not include a
conversion webhook. Skill examples that write
`apiVersion: nvidia.com/v1beta1` for this kind are invalid and will be
rejected by the API server with a clear error.

This differs from DGD / DGDR / DCD / DGDSA, which DO serve both v1alpha1
and v1beta1.

---

## 10. Ownership

The DynamoModel does not set an owner reference on the worker pods —
the relationship is tracked via the worker's `--model` arg and the
operator's reconciliation loop. Deleting a DynamoModel removes the
registration (clients can no longer reach the model by name) but does
not terminate the worker pod; the worker continues to serve its loaded
model on the worker's direct interface.

This is intentional: the Frontend ↔ worker relationship is managed
by the DGD, not by the DynamoModel CR.
