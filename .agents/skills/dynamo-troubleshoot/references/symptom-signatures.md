# Symptom Signature Library

Each signature uses the strict 6-element shape from. Signatures
match the broader stable patterns in
[`dynamo-deploy/references/known-issues.md`](../../dynamo-deploy/references/known-issues.md) plus day-2-specific patterns observed in operator and worker logs.

---

### `/v1/models` empty after DGD Ready

**Symptom:** DGD reports `state: successful`, all pods are `Running`, but `curl /v1/models` returns `{"data": []}` for >2 minutes.

**Root cause:** Frontend has not registered the model yet. Common causes (in order of likelihood): (a) HF token missing from Frontend service, (b) workers still downloading weights from HF, (c) NATS connection problem between worker and Frontend, (d) chat_template missing on a base model.

**Affected:** All Dynamo releases, all backends.

**Fix:** Cross-check the evidence bundle in this order:
1. `grep HF_TOKEN describe-frontend-*.txt` — present?
2. `grep -i "downloading" logs-worker-*.txt` — workers still pulling weights?
3. `grep -i "nats.*disconnect\|connection.*refused" logs-worker-*.txt` — NATS issues?
4. `grep "chat_template" logs-frontend-*.txt` — chat template missing?

Apply the matched fix from (HF token), (registration window), or (chat template).

**Verify:** `until curl -s http://localhost:8000/v1/models | python3 -c 'import json,sys; assert json.load(sys.stdin).get("data")'; do sleep 10; done`

Source:,,.

---

### Prefill worker `CrashLoopBackOff` with NIXL deprecation

**Symptom:** Prefill worker `CrashLoopBackOff`; logs show `ValueError: --connector is deprecated and the default is no longer nixl`.

**Root cause:** Disaggregated mode requires `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'` on the prefill worker explicitly. The v0.9.x recipe of relying on the default connector was removed in 1.0.0+.

**Affected:** Dynamo 1.0.0+. Disaggregated deployments missing the explicit transfer config.

**Fix:** Per. Add the explicit config to the prefill worker's args in the DGD spec.

**Verify:** `kubectl logs <prefill-pod> -n <ns> | tail -50` shows no deprecation error after the pod restart.

Source:.

---

### Worker pod stays Pending forever

**Symptom:** `kubectl get pods` shows worker in `Pending` state for >5 minutes; `describe` shows "no nodes available" or "PodAffinity/anti-affinity not satisfied".

**Root cause:** Likely candidates: (a) GPU resource exhaustion (cluster has no free GPUs of the requested SKU), (b) Grove gang scheduling did not converge, (c) KAI scheduler is the binder and rejected the pod, (d) node-selector mismatch.

**Affected:** All Dynamo releases. More common in heterogeneous clusters.

**Fix:**
1. `kubectl get nodes -L nvidia.com/gpu.product -L nvidia.com/gpu.count` — sufficient GPUs of the right SKU?
2. `kubectl get pods -n <ns> -l app.kubernetes.io/name=grove` — Grove pods healthy? (Per, Grove is the GHCR-hosted gang scheduler.)
3. `kubectl describe pod <worker> -n <ns>` — look for events from `kai-scheduler` or `default-scheduler`.

If GPUs are present but Grove rejects, the Planner-latency-mode pod-gang roll issue (DYN-2879) may apply — see [dynamo-plan/references/known-issues.md](../../dynamo-plan/references/known-issues.md).

**Verify:** `kubectl get pods -n <ns> -w` shows worker transition to `Running`.

Source:.

---

### Conversion webhook timeouts

**Symptom:** `kubectl apply` on a DGD or DGDR takes 30+ seconds and fails with "conversion webhook for X failed: dial tcp ... timeout".

**Root cause:** The operator's conversion webhook service is not reachable from the API server. Common causes: webhook TLS certs expired, operator pod not Ready, network policy blocking API-server → operator-pod traffic.

**Affected:** Any release that serves both v1alpha1 and v1beta1 — DGD, DGDR, DCD, DGDSA per.

**Fix:**
1. `kubectl get pods -n <ns> -l app.kubernetes.io/name=dynamo-operator` — operator pod Ready?
2. `kubectl logs <operator-pod> -n <ns> | grep -i "webhook"` — startup errors?
3. `kubectl get apiservice` — any v1beta1.nvidia.com APIService reporting unhealthy?
4. `kubectl get validatingwebhookconfigurations,mutatingwebhookconfigurations | grep dynamo` — both present? TLS cert bundle attached?

If TLS issue: roll the operator. If pod-not-ready: investigate operator logs separately.

**Verify:** Retry the apply; should complete in <1 second.

Source:.

---

### Planner stuck at `num_workers=1`

**Symptom:** Planner-driven DGD never scales up past 1 worker despite TTFT measurement well above the SLA target.

**Root cause:** Latency-mode prefill scale-up triggers a Grove pod-gang roll that erases the scale-up state. Without Grove or with Grove misconfigured, the roll fails silently and the Planner stays at 1 worker.

**Affected:** Dynamo 1.2.x with Grove sub-chart. Tracked as DYN-2879 / NVBug 6109874.

**Fix:** Confirm Grove sub-chart is installed and working (`helm list -A | grep grove`). If Grove is misconfigured, the Planner's `optimizationType: throughput` mode uses a different scale-up path that doesn't require Grove — switch the DGD to throughput mode if latency is not the SLA driver.

**Verify:** `kubectl logs <planner-pod> | grep -i "scaling up"` — confirms scale-up action; `kubectl get pods -l nvidia.com/dgd-name=<name>` shows the new replica.

Source:,.

---

### EFA RDMA falls back to TCP

**Symptom:** Disaggregated KV transfer is functional but throughput is 10× lower than expected; worker logs show "falling back to TCP transport" or "EFA initialization failed".

**Root cause:** NIXL chose the wrong transport. Most often: (a) libfabric / EFA driver mismatch in the container, (b) UCX configuration excluding RDMA, (c) `nixl_ucx_efa_ref` build pin doesn't match the host EFA installer version.

**Affected:** EFA-enabled clusters; vLLM and SGLang backends most observed.

**Fix:**
1. `kubectl exec <worker> -- env | grep -i "ucx\|nixl"` — UCX_TLS, NIXL_TRANSPORT, EFA_FABRIC_VARIANT.
2. `kubectl exec <worker> -- fi_info -p efa` — provider list includes efa?
3. Compare the container's `nixl_ucx_efa_ref` (per `container/context.yaml`) to the host's `aws-efa-installer` version.

If mismatch: rebuild the runtime container with `nixl_ucx_efa_ref` aligned to the host. If config: set `UCX_TLS=rc_x,cuda_copy` explicitly.

**Verify:** Worker logs show NIXL using `efa` provider; benchmarked throughput returns to expected range.

Source:.

---

### Helm install fails with OCI 404

**Symptom:** `helm install dynamo-platform oci://nvcr.io/nvidia/ai-dynamo/dynamo-platform --version <v>` fails with a 404 or "registry authentication required".

**Root cause:** Per. Helm v4 changed OCI URL parsing in a way NGC rejects under some auth modes.

**Affected:** All Dynamo releases on Helm v4.

**Fix:** Per. `helm fetch oci://nvcr.io/nvidia/ai-dynamo/dynamo-platform --version <v>` and `helm install dynamo-platform ./dynamo-platform-<v>.tgz`.

**Verify:** `helm status dynamo-platform -n <ns>` reports `deployed`.

Source:.
