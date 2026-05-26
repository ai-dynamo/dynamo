# Frontend Known Issues

Stable issue patterns specific to the Frontend service, DynamoModel CRs,
and gateway integration. Strict 6-element shape per.

---

### HF token must be on the Frontend (not just the Worker)

**Symptom:** Worker downloads the gated model successfully but `/v1/models` returns empty data. Frontend logs show `401 Unauthorized` against `huggingface.co`.

**Root cause:** The Frontend service registers gated models with the HF Hub API on startup (model card lookup, license check). If the HF token Secret is mounted only on the Worker, the Frontend cannot complete registration.

**Affected:** All Dynamo releases. Gated HF models (Llama family, some Qwen variants).

**Fix:** Mount `hf-token-secret` on **both** `Frontend` and the Worker service(s) in the DGD spec:

```yaml
spec:
  services:
    Frontend:
      envFromSecret: hf-token-secret
    VllmDecodeWorker:
      envFromSecret: hf-token-secret
```

**Verify:** `kubectl exec <frontend-pod> -n <ns> -- env | grep HF_TOKEN` returns the token; `/v1/models` populates within the registration window.

Source:.

---

### `/v1/models` empty for 30-120s after DGD Ready

**Symptom:** DGD reports `state: successful`, Frontend pod is Ready, but `curl /v1/models` returns `{"data": []}` for 30-120 seconds.

**Root cause:** Workers must download weights, initialize the engine, and register endpoints via NATS before `/v1/models` populates. DGD readiness reports pod health, not endpoint registration.

**Affected:** All Dynamo releases.

**Fix:** Poll `/v1/models` until non-empty:

```bash
until curl -s http://<frontend>/v1/models | python3 -c 'import json,sys; sys.exit(0 if json.load(sys.stdin).get("data") else 1)'; do
  sleep 10
done
```

**Verify:** A non-empty `data` array appears.

Source:.

---

### Base model 5xx with chat-template error

**Symptom:** Request to `/v1/chat/completions` returns 5xx; Frontend logs show `chat_template field is required in the tokenizer_config.json file`.

**Root cause:** The model is a base (non-instruction-tuned) checkpoint without a chat template. The `/v1/chat/completions` endpoint requires a chat template to render the message list into the model's expected prompt format.

**Affected:** All releases. Any base model the user attempts to call via `/v1/chat`.

**Fix:** Either (a) use an instruction-tuned variant of the model, (b) supply a chat template explicitly via the model card, or (c) route clients to `/v1/completions` (which does not require a chat template).

**Verify:** A `/v1/completions` request returns a non-empty response.

Source:.

---

### kgateway + Istio sidecar causes Gateway 500 on inference requests

**Symptom:** Requests through the kgateway-fronted path return HTTP 500 with no useful body; the Frontend pod itself reports no errors. Pod has an `istio-proxy` sidecar.

**Root cause:** The Istio sidecar's traffic capture rules interact badly with kgateway's request handling for the inference path. Symptom is open as of 1.2.0 RC5 (DYN-3077 / NVBug 6194957, P0).

**Affected:** Deployments using both kgateway and Istio sidecar injection.

**Fix (interim):** Disable sidecar injection on the Frontend pod via the per-pod annotation pattern in [references/gateway-integration.md](gateway-integration.md):

```bash
kubectl patch deploy <frontend-deploy> -n <ns> --type=merge -p '{
  "spec": {"template": {"metadata": {"annotations": {"sidecar.istio.io/inject": "false"}}}}
}'
```

Then roll the Frontend.

**Verify:** Requests through the gateway return non-500 responses; the Frontend pod has no `istio-proxy` container.

Source: NVBug 6194957 / DYN-3077; for the gateway dependency context.

---

### KServe gRPC ModelReady RPC returns true before workers are ready

**Symptom:** A client using the KServe gRPC ModelReady RPC sees `true` for a model name; subsequent inference requests fail because workers haven't actually registered yet.

**Root cause:** The Frontend's gRPC ModelReady implementation returns true based on the DynamoModel CR existing, not based on whether the worker has registered. Tracked as DYN-3027 / NVBug 6174719 (P0 open as of 1.2.0 RC5).

**Affected:** Deployments using the KServe gRPC surface (`--kserve-grpc` Frontend arg).

**Fix (interim):** Until the upstream fix lands, clients should also probe `/v1/models` via REST as a second readiness check. The gRPC ModelReady is not sufficient on its own.

**Verify:** Cross-check gRPC ModelReady against REST `/v1/models` — both should agree.

Source: NVBug 6174719 / DYN-3027.

---

### Frontend metrics use lowercased model labels

**Symptom:** Metrics from the Frontend's `/metrics` endpoint use lowercased model names; the KV-router's metrics use the original casing; aggregation across the two surfaces is broken.

**Root cause:** The Frontend lowercased model labels on its metric emit path. Fixed in cherry-pick #9775 / DYN-3076 (RC5 → 1.2.0 GA — verify which release line the fix lands in for the user's target version).

**Affected:** Releases before the #9775 cherry-pick. For 1.2.0, the fix is in RC5+.

**Fix:** Upgrade to the post-cherry-pick image tag (per `container/context.yaml` plus the per-release-line tracker).

**Verify:** Frontend `/metrics` model labels preserve the original casing from the DynamoModel CR's `metadata.name`.

Source: PR #9775 / DYN-3076 / NVBug 6194760.

---

### model-download Job evicted under HF_XET_HIGH_PERFORMANCE

**Symptom:** The recipe-style `model-download` Job (which pre-pulls weights to a PVC before the DGD starts) is OOMKilled or evicted; deployment fails to come up.

**Root cause:** `HF_XET_HIGH_PERFORMANCE=1` is set without declaring matching pod resource requests; the Job exceeds the namespace's eviction threshold. Tracked as DYN-3075 / NVBug 6194213 (P0 open as of 1.2.0 RC5).

**Affected:** Recipe-based deploys pulling models from HuggingFace XET-mode. Common with large models (>50 GB).

**Fix (interim):** Either (a) unset `HF_XET_HIGH_PERFORMANCE`, or (b) declare resource requests on the model-download Job pod template to claim the memory and ephemeral-storage XET needs.

**Verify:** The Job completes; the PVC contains the model shards; the DGD comes up.

Source: NVBug 6194213 / DYN-3075.
