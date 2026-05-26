# Deployment Patterns

Verbatim DGD YAML patterns for the common Dynamo deployment shapes. Each
pattern names when to use it, what it depends on, and the gotchas.

All examples target `apiVersion: nvidia.com/v1alpha1` because the current
upstream examples are pinned there. Both `v1alpha1` and `v1beta1` are
served (per `SKILL_AUTHORING.md` §4); new authoring should target
`v1beta1` once the upstream examples are bumped.

Source: `ai-dynamo/dynamo` `examples/backends/<backend>/deploy/*.yaml`
on the release branch the skill targets.

---

## Pattern 1: Aggregated

**When to use.** Small models, prototyping, single-replica deploys where
prefill and decode share a worker.

**GPU.** 1 GPU per worker replica (more for tensor-parallel configs).

**Source.** `examples/backends/vllm/deploy/agg.yaml`.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-agg
spec:
  services:
    Frontend:
      envFromSecret: hf-token-secret
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
    VllmDecodeWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      replicas: 1
      resources:
        limits:
          gpu: "1"
        requests:
          custom:
            ephemeral-storage: "2Gi"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          workingDir: /workspace/examples/backends/vllm
          command:
            - python3
            - -m
            - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-0.6B
```

**Key fields.**

- `Frontend.componentType: frontend` — the OpenAI-compatible HTTP layer.
- `VllmDecodeWorker.componentType: worker` — a generic worker (no
  `subComponentType` means aggregated).
- `envFromSecret: hf-token-secret` on **both** Frontend and Worker. The
  Frontend needs the HF token to register gated models with
  `/v1/models`; the Worker needs it to download weights. See
  [known-issues.md](known-issues.md) "HF token must be on Frontend".
- `command: python3 -m dynamo.vllm` — the local-run module form (per
  `SKILL_AUTHORING.md` §4). No `dynamo-run` binary exists.

---

## Pattern 2: Disaggregated (Single-Node)

**When to use.** Throughput-oriented serving where prefill and decode
compute profiles diverge enough to benefit from independent scaling.

**GPU.** Minimum 2 GPUs per replica (1 prefill, 1 decode). Production
deploys typically use 4+ per replica.

**Source.** `examples/backends/vllm/deploy/disagg.yaml`.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-disagg
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
    VllmDecodeWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: decode
      replicas: 1
      resources:
        limits:
          gpu: "1"
        requests:
          custom:
            ephemeral-storage: "2Gi"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          workingDir: /workspace/examples/backends/vllm
          command:
            - python3
            - -m
            - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --disaggregation-mode
            - decode
    VllmPrefillWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: prefill
      replicas: 1
      resources:
        limits:
          gpu: "1"
        requests:
          custom:
            ephemeral-storage: "2Gi"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          workingDir: /workspace/examples/backends/vllm
          command:
            - python3
            - -m
            - dynamo.vllm
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --disaggregation-mode
            - prefill
            - --kv-transfer-config
            - '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'
```

**Key fields.**

- `subComponentType: decode` and `subComponentType: prefill` — split the
  worker role explicitly.
- `--disaggregation-mode` argument on both workers.
- `--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'`
  on the prefill worker. Required at Dynamo >= 1.0.0; the older
  `--disaggregation-mode` alone (the v0.9.x recipe) fails with
  `--connector is deprecated and the default is no longer nixl`. See
  [known-issues.md](known-issues.md) "Disagg KV transfer config".

---

## Pattern 3: KV-Aware Routing

**When to use.** Multi-replica serving where requests benefit from KV
cache locality (a user's follow-up request lands on the worker that
served the prior turn, reusing the warm KV cache).

**GPU.** Adds a Router pod (CPU-only) plus the underlying agg or disagg
worker pool.

**Source.** Adapt `examples/backends/vllm/deploy/agg_router.yaml`. The
key delta from Pattern 1 is the addition of a `Router` service and
`--enable-prefix-caching` plus the KV-aware routing flag on the Frontend.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-agg-router
spec:
  services:
    Frontend:
      envFromSecret: hf-token-secret
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          args:
            - --router
            - kv-aware
    Router:
      componentType: router
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          command: [python3, -m, dynamo.router]
    VllmDecodeWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      replicas: 2   # KV-aware routing benefits from multiple workers
      resources:
        limits:
          gpu: "1"
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          command: [python3, -m, dynamo.vllm]
          args:
            - --model
            - Qwen/Qwen3-0.6B
            - --enable-prefix-caching
```

**Key fields.**

- `Router.componentType: router` — separate service that the Frontend
  delegates routing decisions to.
- `Frontend.args: [--router, kv-aware]` — tells the Frontend to consult
  the router on every request.
- `Worker.args: [--enable-prefix-caching]` — workers must expose their
  prefix-cache state so the router can pick the best match.
- `replicas >= 2` on workers — routing has no effect with a single
  worker.

---

## Pattern 4: Planner-Driven Autoscaling

**When to use.** Production deploys where the load varies and you want
the Planner to scale workers up/down to hit a TTFT/ITL target.

**GPU.** Same as the underlying pattern (Pattern 2 or 3); the Planner
itself is CPU-only.

**Source.** Adapt `examples/backends/vllm/deploy/disagg_planner.yaml`.
The pattern wraps Pattern 2 with a `Planner` service.

```yaml
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: vllm-disagg-planner
spec:
  services:
    Frontend:
      componentType: frontend
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
    Planner:
      componentType: planner
      replicas: 1
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/dynamo-planner:1.2.0
          command: [python3, -m, dynamo.planner]
          args:
            - --target-ttft
            - "50"
            - --target-itl
            - "10"
            - --min-workers
            - "2"
            - --max-workers
            - "8"
    VllmPrefillWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: prefill
      replicas: 2   # initial; Planner scales between min and max
      resources: { limits: { gpu: "1" } }
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          command: [python3, -m, dynamo.vllm]
          args: [--model, Qwen/Qwen3-0.6B, --disaggregation-mode, prefill,
                 --kv-transfer-config, '{"kv_connector":"NixlConnector","kv_role":"kv_both"}']
    VllmDecodeWorker:
      envFromSecret: hf-token-secret
      componentType: worker
      subComponentType: decode
      replicas: 2
      resources: { limits: { gpu: "1" } }
      extraPodSpec:
        mainContainer:
          image: nvcr.io/nvidia/ai-dynamo/vllm-runtime:my-tag
          command: [python3, -m, dynamo.vllm]
          args: [--model, Qwen/Qwen3-0.6B, --disaggregation-mode, decode]
```

**Key fields.**

- `Planner.componentType: planner`. Operator routes scaling decisions
  through it.
- Planner args: `--target-ttft`, `--target-itl`, `--min-workers`,
  `--max-workers`. The Planner reads worker metrics and adjusts
  replicas to stay within the SLA envelope.
- For latency mode, the Planner uses a Grove pod-gang roll for scale-up.
  See [known-issues.md](known-issues.md) "Planner latency-mode Grove
  pod-gang roll" for the current limitation.

---

## Pattern 5: Recipe-Based

**When to use.** A pre-tuned recipe exists for the model and hardware
combination under `ai-dynamo/dynamo` `recipes/<model>/<framework>/<config>/`.

**GPU.** Per-recipe; documented in each recipe's README.

**Source.** Per-recipe under `recipes/<model>/<framework>/<config>/deploy/`.

The recipe ships a complete DGD plus a `benchmark/` subdirectory with the
published numbers and the AIPerf invocation that produced them. Apply as-is
unless the user has a reason to deviate:

```bash
# Pick the recipe matching the model and hardware.
RECIPE=recipes/qwen3-32b-fp8/trtllm/disagg
kubectl apply -f $RECIPE/deploy/ -n <ns>

# Run the recipe's benchmark to confirm reproducible numbers.
bash $RECIPE/benchmark/run.sh
```

Recipes available on the 1.2.0 release line: `deepseek-r1`,
`deepseek-v32-fp4`, `deepseek-v4`, `glm-5-nvfp4`, `gpt-oss-120b`,
`kimi-k2.5`, `llama-3-70b`, `nemotron-3-nano-omni`,
`nemotron-3-super-fp8`, `qwen3-32b`, `qwen3-32b-fp8`,
`qwen3-235b-a22b-fp8`, `qwen3-vl-30b`, `qwen3.6-35b`. Recipe set varies
per release; verify against the target release branch.

---

## Cross-Cutting: HF Token Wiring

Gated HuggingFace models (e.g. Llama family) require an HF access token.
Wire it via a Kubernetes Secret:

```bash
kubectl create secret generic hf-token-secret \
  --from-literal=HF_TOKEN=<your_token> \
  -n <ns>
```

Reference the Secret from **both** the Frontend and the Worker(s) via
`envFromSecret: hf-token-secret` (as shown in Patterns 1 and 2). Missing
the Frontend mount causes `/v1/models` to return empty with a 401
Unauthorized error — see [known-issues.md](known-issues.md) "HF token
must be on Frontend".

---

## Cross-Cutting: Image Tags

The `image` field on each `mainContainer` references the runtime
container for that backend. The registry path is fixed; the tag varies
per release. Look up the current per-release tag in
`container/context.yaml` on the target release branch (per
`SKILL_AUTHORING.md` §4).

Registry paths (per `SKILL_AUTHORING.md` §4):

| Backend | Registry |
|---|---|
| vLLM runtime | `nvcr.io/nvidia/ai-dynamo/vllm-runtime` |
| TensorRT-LLM runtime | `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime` |
| SGLang runtime | `nvcr.io/nvidia/ai-dynamo/sglang-runtime` |
| Frontend | `nvcr.io/nvidia/ai-dynamo/dynamo-frontend` |
| Planner | `nvcr.io/nvidia/ai-dynamo/dynamo-planner` |

Tag pattern during QA: `<release>rc<N>` (cut nightly). GA tag: `<release>`.
