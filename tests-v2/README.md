# tests-v2 — component-oriented harness (skeleton)

A minimal, working slice of the framework proposed in
[DEP 0017](https://github.com/ai-dynamo/enhancements/pull/100). Enough to
rewrite one real test and run it against released containers; not yet the whole
design.

## What is here

| File | Role |
|---|---|
| `dynamo_harness/transport.py` | `Http` — how a component is *reached*. stdlib only. |
| `dynamo_harness/deployment.py` | `Attached`, `Docker` — how a deployment is *controlled*. |
| `dynamo_harness/components.py` | `Frontend` — wire interface + waiting policy. |
| `dynamo_harness/dynamo.py` | `Dynamo` — the facade a test receives. |
| `conftest.py` | Builds `Dynamo` from run-time options. |
| `test_sample.py` | `tests/serve/test_sample.py`, rewritten. |

Transport and deployment are separate on purpose. `Dynamo.attach()` leaves
`deployment=None`, so a query-only test *cannot* restart a container or read
logs — the capability is absent from the object rather than merely discouraged.

## Run it

Deploy a released container and test it (needs Docker + a GPU):

```bash
pytest tests-v2 -v \
  --dynamo-image nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.4.0 \
  --dynamo-model Qwen/Qwen3-0.6B \
  --dynamo-hf-cache "$HOME/.cache/huggingface"
```

Attach to something already running (query-only; lifecycle tests skip):

```bash
pytest tests-v2 -v --dynamo-url http://localhost:8000
```

Other backends — the GA images on hand are vLLM, SGLang and TensorRT-LLM:

```bash
pytest tests-v2 --dynamo-backend sglang \
  --dynamo-image nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4.0
```

Options: `--dynamo-url`, `--dynamo-image`, `--dynamo-backend`, `--dynamo-model`,
`--dynamo-port`, `--dynamo-gpus`, `--dynamo-hf-cache`, `--dynamo-ready-timeout`.

## How the container is run

One container hosting frontend + worker, both using the **`file` discovery
backend**, so no etcd and no NATS are required — the two processes share the
container filesystem, and the default request plane is TCP. That is why the
harness can drive a released image with no sidecar infrastructure.

## Deliberately not here yet

Capability/requirement declaration (`@requires`, `dynamo.require(...)`), the
fleet probe, components beyond the frontend, and the Local / Compose /
Kubernetes lifecycle providers. See DEP 0017 for the full design and phasing.
