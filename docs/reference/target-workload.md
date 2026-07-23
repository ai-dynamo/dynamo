# Target Workload

Canonical path: `runs/<EXP_ID>/target_workload.yaml`

The target workload is the single YAML file the user provides for an optimization job. It is the durable contract from
the user to `recipe-explorer` and says what must be served, measured, and where it may be deployed. It is not a place
for selected manifests, deployment results, tuning history, or secret values.

## Minimal Schema

Provide a standalone YAML file:

```yaml
profile:
  name: kimi-k26-chat              # custom workload identifier
  type: chat                       # chat | agentic | custom
  description: ""                  # optional workload description

model:
  source: moonshotai/Kimi-K2.6     # HF id or local model path requested by the user

hardware:                          # one entry per SKU (e.g. H100, H200, B200, GB300)
  - gpu_type: B200
    gpu_count: 8

kubernetes:
  kube_context: ""                 # required kubectl context for the target cluster
  namespace: ""                    # required existing namespace for all run resources
  storage_class: ""                # optional; needed only when the run must create a PVC

traffic:
  input_tokens: null               # optional rough/median input sequence length when known
  output_tokens: null              # optional rough/median output sequence length when known
  request_rate: null               # optional requests/second target when known
  concurrency: []                  # optional exact benchmark concurrency values when known

preferences:
  precision: ""                    # e.g. fp4, fp8, bf16
  framework: ""                    # vLLM, SGLang, or TRT-LLM
  mode: ""                         # agg, disagg

objectives:
  ttft_ms_p95_max: null            # optional time to first token
  itl_ms_p95_max: null             # optional inter-token-latency
  request_latency_ms_p95_max: null # optional end-to-end request latency
  output_tput_per_gpu_min: null    # optional output tokens/second/GPU
  error_rate_max: null             # optional request error rate
  custom_slos: ""                  # optional custom user-provided SLOs

artifacts:
  supporting_traces: []           # optional paths for JSONL trace files (e.g. production traces)

notes: ""
created_at: ""
```

## Rules

- Do not store token values, kubeconfig contents, registry credentials, or deployment secret names here.
- Treat manifest-referenced Hugging Face and image-pull secrets as pre-existing cluster prerequisites. Verify their
  existence by name; do not ask the user for secret values or create secrets.
- `kube_context` and `namespace` are required. The namespace must already exist.
- `storage_class` is optional when the required PVC already exists or a suitable cluster default is known. If the run
  must create a PVC and no suitable class can be determined safely, stop and ask the user to fill this field.
- Token lengths are optional customer-provided traffic hints, not required benchmark keys. Prefer real traces or
  profile descriptions when exact input/output token counts are unknown.
- Preserve exact user-provided traffic and SLO values. Do not "round" or reinterpret them.
- Preserve heterogeneous hardware as separate `hardware` entries, such as 4 H100 plus 4 H200. Do not collapse mixed
  GPUs into a single string.
- Treat `profile.type` as the traffic shape, not as a complete API contract. The deployer should use the selected
  recipe's documented endpoint, defaulting to `/v1/chat/completions` for current text-generation recipes.
- Keep workload intent separate from candidate configuration. Engine and Dynamo knob changes belong in candidate
  configs, deployment ledgers, or benchmark artifacts.
- Leave unknown values as `null` or `""`. Do not invent a GPU count, recipe variant, model path, or SLO.
- `recipe-explorer` outputs the selected DGD/deploy manifest separately. Do not write selected manifests back into this
  file.
