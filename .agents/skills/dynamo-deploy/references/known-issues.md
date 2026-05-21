# Known Issues

Stable issue patterns that recur across Dynamo releases. Each entry uses
the strict 6-element shape from `SKILL_AUTHORING.md` §2.8.

Per-RC bugs (transient issues current to a single release candidate) do
not live here — they belong in the skill body and refresh with each
`version` bump (per `SKILL_AUTHORING.md` §11).

The six entries below are sourced from `citations.md` D1-D6.

---

### Helm v4 OCI registry syntax fails against NGC

**Symptom:** `helm install dynamo-platform oci://nvcr.io/nvidia/ai-dynamo/dynamo-platform --version <version>` fails with a registry authentication or syntax error on Helm v4.

**Root cause:** Helm v4 changed the OCI URL parsing in a way that NGC's registry rejects under some auth modes. The chart contents are unchanged.

**Affected:** All Dynamo releases when installed with Helm v4 against NGC.

**Fix:** Fetch the chart tarball directly and install from the local file:

```bash
helm fetch oci://nvcr.io/nvidia/ai-dynamo/dynamo-platform --version <version>
helm install dynamo-platform ./dynamo-platform-<version>.tgz -n <ns> --create-namespace
```

**Verify:**

```bash
helm status dynamo-platform -n <ns>
kubectl get deploy -n <ns> -l app.kubernetes.io/name=dynamo-operator
```

Source:.

---

### HF token must be on Frontend (not just Worker)

**Symptom:** `/v1/models` returns empty data; pod logs on the Frontend show `401 Unauthorized` against `huggingface.co`.

**Root cause:** The Frontend service registers gated models with the HF Hub API on startup. If the HF token is mounted only on the Worker, the Worker can download weights but the Frontend cannot complete model registration, so `/v1/models` stays empty.

**Affected:** All Dynamo releases. Manifests that wire `envFromSecret: hf-token-secret` on the Worker(s) only.

**Fix:** Mount the HF token Secret on **both** the Frontend service and the Worker(s) in the generated DGD:

```yaml
spec:
  services:
    Frontend:
      envFromSecret: hf-token-secret      # required
    VllmDecodeWorker:
      envFromSecret: hf-token-secret      # required
```

**Verify:**

```bash
kubectl get pods -n <ns> -l nvidia.com/dgd-name=<name>
kubectl exec -n <ns> <frontend-pod> -- env | grep HF_TOKEN
curl http://localhost:8000/v1/models | python3 -m json.tool
```

Source:.

---

### DGD "successful" but `/v1/models` returns empty

**Symptom:** `kubectl get dgd <name>` reports `state: successful`, all pods are `Running`, but `curl /v1/models` returns `{"data": []}` for 30-120 seconds.

**Root cause:** DGD readiness reports pod health, not endpoint registration. Workers must (1) download weights from HF or the model cache, (2) initialize the inference engine, and (3) register endpoints with the Frontend via NATS before `/v1/models` populates. The window is typically 30-120 s after Ready.

**Affected:** All Dynamo versions, all backends.

**Fix:** Poll `/v1/models` until non-empty before sending inference traffic:

```bash
until curl -s http://localhost:8000/v1/models | python3 -c 'import json,sys; assert json.load(sys.stdin).get("data")'; do
  echo "waiting for model registration..."
  sleep 10
done
```

**Verify:** A successful curl returns the model entry, not an empty array.

Source:.

---

### Base model fails registration with missing chat template

**Symptom:** Frontend pod fails to register the model; logs show `chat_template field is required in the tokenizer_config.json file`.

**Root cause:** Base (non-instruction-tuned) HuggingFace models do not ship a chat template in `tokenizer_config.json`. The Frontend's `/v1/chat/completions` endpoint requires a chat template to render the OpenAI message format into the model's expected prompt format.

**Affected:** Any base model deployed via `/v1/chat/completions`. Affected at least Dynamo 1.0.0 and later.

**Fix:** Either (a) use an instruction-tuned variant of the model that ships a chat template, (b) supply a chat template explicitly via the model card or operator config, or (c) route to `/v1/completions` instead (which does not require a chat template).

**Verify:**

```bash
curl http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<base-model-id>","prompt":"Once upon a time","max_tokens":16}'
```

Source:.

---

### Disaggregated mode requires `--kv-transfer-config`

**Symptom:** Prefill worker enters `CrashLoopBackOff`; pod logs show `ValueError: --connector is deprecated and the default is no longer nixl`.

**Root cause:** In Dynamo 1.0.0 and later, disaggregated mode requires the `--kv-transfer-config` argument on the prefill worker explicitly naming the NixlConnector. The v0.9.x recipe of using `--disaggregation-mode prefill` alone (relying on the default connector) was removed.

**Affected:** Dynamo 1.0.0 and later. Disaggregated deployments.

**Fix:** Add the explicit transfer config to the prefill worker's args:

```yaml
args:
  - --model
  - <model-id>
  - --disaggregation-mode
  - prefill
  - --kv-transfer-config
  - '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

**Verify:**

```bash
kubectl get pods -n <ns> -l nvidia.com/dgd-name=<name>
kubectl logs -n <ns> <prefill-pod> | tail -50
# Expect no "--connector is deprecated" error.
```

Source:.

---

### Use `kubectl apply --dry-run=server` before any real apply

**Symptom:** A DGDR or DGD applied to the cluster is rejected with a schema error, or accepted silently with an invalid field that the operator later reports as a profiling failure.

**Root cause:** Local YAML validation does not catch CRD schema mismatches or v1alpha1→v1beta1 conversion failures. The API server's server-side dry-run validates the resource (including the CRD schema and the conversion webhook) without persisting it.

**Affected:** All Dynamo deployments. Particularly important for v1alpha1 manifests applied to clusters where the conversion webhook may be misconfigured.

**Fix:** Always run a server-side dry-run before the real apply:

```bash
kubectl apply --dry-run=server -f my-dgdr.yaml
```

If the dry-run reports an error, the real apply will fail the same way. Edit the manifest and re-run the dry-run.

**Verify:** Dry-run output ends with `(server dry run)`. No errors emitted.

Source:.
