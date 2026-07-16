# Kimi-K2.6 Benchmark Recipe

A single [AIPerf](https://github.com/ai-dynamo/aiperf) trace-replay Job — [perf.yaml](perf.yaml) — covers every Kimi-K2.6 DGD variant (B200/H200 × chat/agent traces). The benchmark is identical across variants; only `ENDPOINT` and `TRACE_FILE` need to change.

The Job waits for `GET /v1/models` on the DGD frontend to return `moonshotai/Kimi-K2.6` (up to ~1h by default), runs a short warmup, then replays the configured trace at a single `CONCURRENCY` value and writes raw artifacts to the shared `model-cache` PVC.

The bench pod is **co-located with the DGD frontend** (`podAffinity` on the frontend's host) so client → server traffic stays on a single node.

## Targeting a variant

Edit the `env` block in [perf.yaml](perf.yaml):


| Variant target           | `ENDPOINT`                                      | `TRACE_FILE` (chat / agent)                                             |
| ------------------------ | ----------------------------------------------- | ----------------------------------------------------------------------- |
| B200 agg, chat workload  | `kimi-k26-agg-b200-chat-frontend:8000`    | `/model-cache/traces/8k_1k_70kv_chat_new_noschedule.jsonl`    |
| B200 agg, agent workload | `kimi-k26-agg-b200-agentic-frontend:8000` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule.jsonl` |
| H200 agg, chat workload  | `kimi-k26-agg-h200-chat-frontend:8000`    | `/model-cache/traces/8k_1k_70kv_chat_new_noschedule.jsonl`    |
| H200 agg, agent workload | `kimi-k26-agg-h200-agentic-frontend:8000` | `/model-cache/traces/64k_400_90kv_agent_new_noschedule.jsonl` |


The chat- and agent-tuned DGDs both serve `moonshotai/Kimi-K2.6`, so either trace can be replayed against either DGD by swapping `TRACE_FILE`.

If you run more than one benchmark in the same namespace, also update `metadata.name` / `labels.app` so jobs and artifact directories stay distinct.

## Dataset

The benchmark replays a [Mooncake-format](https://github.com/kvcache-ai/Mooncake) trace via `aiperf --custom-dataset-type mooncake_trace`. Each JSONL line describes one request (`input_length`, `output_length`, `hash_ids`).

Trace flavours expected on the PVC:

- **Chat** — `/model-cache/traces/8k_1k_70kv_chat_new_noschedule.jsonl`
- **Agent** — `/model-cache/traces/64k_400_90kv_agent_new_noschedule.jsonl`

For shorter runs (smoke tests, faster iteration), point `TRACE_FILE` at a smaller variant of the same trace rather than capping run time. We typically stage 15% and 30% subsets alongside the full dataset, e.g.:

```
/model-cache/traces/<flavour>.jsonl                 # full
/model-cache/traces/<flavour>_short_30perc.jsonl    # ~30% subset
/model-cache/traces/<flavour>_short_15perc.jsonl    # ~15% subset
```

The Dynamo Kimi-K2.5 recipe is the closest reference for trace handling — see [its README](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.5/README.md#dataset-agentic-coding-workflow) and [perf.yaml](https://github.com/ai-dynamo/dynamo/blob/main/recipes/kimi-k2.5/trtllm/agg-gb200-eagle-kv-router/perf.yaml).

## Workflow

```bash
export NAMESPACE=your-namespace
```

### 1. Deploy the DGD

See instructions in the [README](../README.md).

Before deploying, do these adjustments to the DGD:
- Change the `SPECULATIVE_CONFIG` variable to point to `key: speculative-config-synthetic`.
- Modify the worker `replicas` to match your desired target


### 2. Stage the trace on the PVC

Spin up a short-lived helper pod that mounts `model-cache`, then `kubectl cp` the traces in:

```bash
kubectl run pvc-helper -n ${NAMESPACE} \
  --image=busybox:1.36 --restart=Never \
  --overrides='{"spec":{"containers":[{"name":"helper","image":"busybox:1.36","command":["sleep","3600"],"volumeMounts":[{"name":"model-cache","mountPath":"/model-cache"}]}],"volumes":[{"name":"model-cache","persistentVolumeClaim":{"claimName":"model-cache"}}]}}' \
  --command -- sleep 3600

kubectl cp ./traces ${NAMESPACE}/pvc-helper:/model-cache/
```

Keep `pvc-helper` around for fetching artifacts later, or `kubectl delete pod pvc-helper -n ${NAMESPACE}` once you're done staging.

### 3. Run the benchmark

```bash
kubectl apply -f perf.yaml -n ${NAMESPACE}

# Stream logs
kubectl logs -n ${NAMESPACE} -l job-name=kimi-k26-bench -f

# Wait for completion (2h hard cap on the Job)
kubectl wait --for=condition=Complete \
  job/kimi-k26-bench \
  -n ${NAMESPACE} --timeout=7200s
```

### 4. Fetch artifacts

```bash
kubectl cp ${NAMESPACE}/pvc-helper:/model-cache/perf/<epoch>_kimi-k26-bench ./results
```

### 5. Cleanup

```bash
kubectl delete job kimi-k26-bench -n ${NAMESPACE}
kubectl delete pod pvc-helper -n ${NAMESPACE}   # if you kept it around
```

## Running a concurrency sweep

`perf.yaml` runs a **single** `CONCURRENCY` value. To measure multiple concurrencies you must clear server state between runs — otherwise residual KV cache / prefix-cache hits from the previous run skew results.

For each concurrency value you want to measure:

```bash
# 1. Delete the previous bench job
kubectl delete job kimi-k26-bench -n ${NAMESPACE} --ignore-not-found

# 2. Delete the DGD worker pods to drop KV / prefix-cache state.
#    DGDs are Grove-managed (DGD → PodCliqueSet → PodClique → Pod), not Deployments,
#    so `kubectl rollout restart deployment` won't match anything — delete the pods
#    directly and let Grove recreate them.
DGD=kimi-k26-agg-b200-chat
kubectl delete pods -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD},nvidia.com/dynamo-component-type=worker
kubectl wait --for=condition=Ready pod -n ${NAMESPACE} \
  -l nvidia.com/dynamo-graph-deployment-name=${DGD},nvidia.com/dynamo-component-type=worker \
  --timeout=900s

# 3. Bump CONCURRENCY in perf.yaml (or use kubectl create -f perf.yaml \
#    with `--dry-run=client -o yaml | yq '.spec.template.spec.containers[0].env[] |= ...'`)
kubectl apply -f perf.yaml -n ${NAMESPACE}
kubectl wait --for=condition=Complete job/kimi-k26-bench -n ${NAMESPACE} --timeout=7200s
```

(The bench Job's `wait_for_model_ready` loop handles the worker restart window — it re-polls `/v1/models` until the frontend reports ready again.)

## Tunable environment variables

Edit the `env` block on the `Job` to adjust:


| Variable       | Default                                             | Notes                                                                           |
| -------------- | --------------------------------------------------- | ------------------------------------------------------------------------------- |
| `ENDPOINT`     | `kimi-k26-agg-b200-chat-frontend:8000`        | DGD frontend service:port — change per variant                                  |
| `TRACE_FILE`   | `/model-cache/traces/8k_1k_70kv_chat_new_noschedule.jsonl` | Swap to agent or to a smaller subset (`...short_15perc.jsonl`) for shorter runs |
| `CONCURRENCY`  | `24`                                                | Single value — see [Running a concurrency sweep](#running-a-concurrency-sweep)  |
| `TARGET_MODEL` | `moonshotai/Kimi-K2.6`                              | Must match `--served-model-name` on the DGD frontend                            |


## Artifacts

Results are written to:

```
/model-cache/perf/<epoch>_<job-name>/
  warmup/
  Kimi-K2.6_trace_c<concurrency>_<timestamp>/
    profile_export.json
    inputs.json
    ...
```
