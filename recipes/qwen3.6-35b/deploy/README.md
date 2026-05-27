# Qwen3.6-35B-A3B-FP8 — deploy

Standalone deployment of `Qwen/Qwen3.6-35B-A3B-FP8` on a single GPU.
Pick one of the three configs and stand it up; the recipe leaves you
with an OpenAI-compatible HTTP endpoint inside the namespace.

| Config         | Stack         | Stands up                                       |
|----------------|---------------|-------------------------------------------------|
| `vllm-serve`   | vanilla vLLM  | `Deployment` + `Service` `qwen36-vllm-serve`    |
| `dynamo-fd`    | Dynamo + vLLM | `DynamoGraphDeployment` `qwen36-dynamo-fd`      |
| `dynamo-fd-ec` | Dynamo + vLLM | `DynamoGraphDeployment` `qwen36-dynamo-fd-ec`   |

For shared pre-requisites (PVC, namespace, hostname fill-in, HF token),
see the [recipe root README](../README.md).

## Quick start

```bash
export NAMESPACE=<your-namespace>
export HW=gb200   # or h100

# `all` runs pvc check → model download → deploy. The download Job is
# config-agnostic, so the next config you pick reuses the cached weights.
./deploy/deploy.sh -n "$NAMESPACE" --hw "$HW" --config vllm-serve
./deploy/deploy.sh -n "$NAMESPACE" --hw "$HW" --config dynamo-fd
./deploy/deploy.sh -n "$NAMESPACE" --hw "$HW" --config dynamo-fd-ec
```

`deploy.sh` accepts `--step {pvc|download|deploy|clean|all}` for
granular control. `pvc` and `download` are config-agnostic (any
`--config` value works to run them once).

## What you get per config

- `vllm-serve` → a `Deployment` + `Service` named `qwen36-vllm-serve`.
  Hit it at `http://qwen36-vllm-serve:8000/v1/...` from inside the
  namespace.
- `dynamo-fd` / `dynamo-fd-ec` → a `DynamoGraphDeployment`. The Dynamo
  operator stands up a `<dgd>-frontend` Service exposing the same
  OpenAI-compatible API on port 8000.

The vllm command in `deploy/*.yaml` uses `--mm-processor-cache-gb 30`
and `--max-model-len 32768` to handle multimodal contexts up to 5
images per request.

## Cleanup

```bash
./deploy/deploy.sh -n "$NAMESPACE" --hw "$HW" --config <name> --step clean
```

Deletes the Deployment+Service (for `vllm-serve`) or the
DynamoGraphDeployment (for `dynamo-fd` / `dynamo-fd-ec`). PVCs are
intentionally left intact so the cached weights survive across configs.
To wipe everything:

```bash
kubectl -n "$NAMESPACE" delete pvc shared-model-cache
```
