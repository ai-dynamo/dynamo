# CustomEncoder — pluggable in-process vision encoder (aggregated)

K8s recipe that deploys and smoke-tests the Dynamo + vLLM **`CustomEncoder`**
feature from [ai-dynamo/dynamo#10832](https://github.com/ai-dynamo/dynamo/pull/10832):
a custom image encoder that runs **in-process inside a single aggregated
`dynamo.vllm` worker** — no separate encode worker, no NIXL transfer.

This is a **correctness / feature demo**, not a throughput benchmark. It runs
the example `HitchhikersCustomEncoder` on a text-only LM (`Qwen2.5-1.5B-Instruct`)
that fakes an "image" as the embeddings of a fixed phrase, so the assembled
prompt answers **"42"**. Swap `--custom-encoder-class` (and `--model`) for a real
encoder + LM to use it for real.

> **Depends on PR #10832.** The feature code and the `examples/custom_encoder`
> package are not in any published image yet, so this recipe runs against a
> **custom image built from the PR branch** (see [Build the image](#build-the-image)).

## How it works

```
client ── /v1/chat/completions ──▶ Frontend (dynamo.frontend)
                                      │  rewrites image_url → image,
                                      │  minimal jinja emits <|image_pad|>
                                      ▼
                              VllmWorker (dynamo.vllm, 1 GPU)
                                ├─ CustomEncoder.encode(image_urls) → image embeds
                                ├─ assemble mixed token-ids/embeds EmbedsPrompt
                                │    (vLLM embeds text; image embeds spliced in)
                                └─ vLLM engine runs the transformer
```

The encoder never needs the LM's text-embedding table; vLLM embeds the text
positions with the real table and substitutes only the image positions. The
example encoder ignores the image bytes and returns the embeddings of a fixed
phrase, which makes the spliced prompt read as one coherent sentence — a
**semantic** check, not just a shape check.

## Prerequisites

1. Kubectl context pointing at a cluster with **one free GPU** on a node.
2. A namespace you have write access to (`$NAMESPACE` below).
3. A `shared-model-cache` PVC (RWX) in that namespace. Platform-managed
   AWS / FSx clusters pre-provision it; otherwise see [Storage](#storage).
4. `envsubst` on the laptop driving the recipe (Ubuntu: `apt install
   gettext-base`; macOS: `brew install gettext`).
5. **No HuggingFace token** — `Qwen/Qwen2.5-1.5B-Instruct` is public.
6. A **custom image built from PR #10832** — see below.

## Build the image

The published `vllm-runtime` images don't have the `--custom-encoder-class`
handler, the `examples/custom_encoder` package, or vLLM 0.21+. Build one from
the PR branch (`examples/` is `COPY`'d into the image, so the example encoder +
jinja template are baked in):

```bash
cd ~/workspace/dynamo
git fetch origin pull/10832/head:custom-encoder-pr
git checkout custom-encoder-pr

# Render + build the vLLM runtime image (build context = repo root, so
# examples/custom_encoder lands at /workspace/examples/custom_encoder).
python3 container/render.py --framework vllm --target runtime \
        --platform linux/amd64 --output-short-filename     # -> container/rendered.Dockerfile
docker build -t nvcr.io/nvstaging/ai-dynamo/vllm-runtime:qiwa-custom-encoder \
             -f container/rendered.Dockerfile .
docker push nvcr.io/nvstaging/ai-dynamo/vllm-runtime:qiwa-custom-encoder
```

The base image must ship **vLLM >= 0.21** (`EmbedsPrompt.prompt_is_token_ids`);
the PR's base already does. (`/build-image` is a convenience wrapper for the same
render + build.)

## Quick start

```bash
export NAMESPACE=<your-namespace>
export IMAGE=nvcr.io/nvstaging/ai-dynamo/vllm-runtime:qiwa-custom-encoder

# pvc check → model download → deploy (Frontend + 1 GPU worker) → smoke ("42")
./run.sh -n "$NAMESPACE" --image "$IMAGE" --step all

# Optionally pin both pods to a specific node:
./run.sh -n "$NAMESPACE" --image "$IMAGE" --node ip-100-66-1-2.ec2.internal --step all
```

Granular steps: `--step {pvc|download|deploy|smoke|encoder-log|clean}`.
`encoder-log` greps the worker log for the `Loaded CustomEncoder` line to confirm
the plugin loaded in-process.

## Smoke-test details

`smoke.yaml` runs a small CPU pod that hits the auto-created Frontend Service
(`custom-encoder-agg-frontend:8000`) over ClusterIP and POSTs:

```jsonc
{
  "model": "Qwen/Qwen2.5-1.5B-Instruct",
  "max_tokens": 32, "temperature": 0,
  "messages": [{"role": "user", "content": [
    {"type": "text",      "text": "Based on The Hitchhiker's Guide to the Galaxy, The Answer to"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,<1x1 png>"}},
    {"type": "text",      "text": " is?"}
  ]}]
}
```

The completion must contain **"42"** → the pod echoes `SMOKE PASS` (else
`SMOKE FAIL`), which `run.sh` polls for. The `data:` image is self-contained
(the example encoder ignores image content). For a **real** encoder, override
the `IMAGE_URL` env in `smoke.yaml` with a reachable image — `DYN_MM_ALLOW_INTERNAL=1`
on the deploy permits `data:` and localhost URLs.

## Plugging in a real encoder

1. Implement a `CustomEncoder` subclass (load your ViT + projector in `load()`;
   return one `(n_visual_tokens, lm_hidden_dim)` tensor per image in `encode()`).
   See `components/src/dynamo/vllm/multimodal_utils/custom_encoder.py`.
2. Bake your module into the image (or a sibling layer on top of it) so its
   dotted path is importable with `PYTHONPATH=/workspace`.
3. Edit `deploy.yaml`: set `ENCODER_CLASS` to your dotted `module.ClassName`,
   `MODEL_NAME` to your LM, and (if your model defines `<|image_pad|>` itself)
   drop `DYN_IMAGE_PLACEHOLDER_TOKEN_ID` to let the encoder auto-resolve it.
4. Replace the demo's minimal `--custom-jinja-template` if your LM's own chat
   template already renders image content parts.

## Storage

The recipe expects a `shared-model-cache` PVC (RWX), mounted at the HF cache
root (`/home/dynamo/.cache/huggingface`) on the download Job and the worker —
no subPath (the DGD `volumeMounts` grammar has none). If your cluster doesn't
pre-provision it, create it first, picking an RWX storage class:

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: shared-model-cache
spec:
  accessModes: [ReadWriteMany]
  resources:
    requests:
      storage: 100Gi
  storageClassName: <your-rwx-storage-class>   # e.g. dgxc-enterprise-file / FSx Lustre — not ebs
```

## Cleanup

```bash
./run.sh -n "$NAMESPACE" --step clean    # deletes the DGD + smoke pod; keeps the PVC
```

The PVC (model cache) is intentionally retained. To wipe it:
`kubectl -n "$NAMESPACE" delete pvc shared-model-cache`.

## Directory layout

```text
custom-encoder/
├── README.md
├── run.sh                      # driver: pvc | download | deploy | smoke | encoder-log | clean | all
├── deploy.yaml                 # DGD: Frontend + 1 aggregated VllmWorker (custom encoder)
├── smoke.yaml                  # Pod: curl the frontend, assert "42"
└── model-cache/
    └── model-download.yaml     # Job: HF download for Qwen2.5-1.5B-Instruct
```

## Naming & ownership

All resources carry the label `app.kubernetes.io/name: custom-encoder`:

```bash
kubectl -n "$NAMESPACE" get pvc,job,pod,dynamographdeployment \
  -l app.kubernetes.io/name=custom-encoder
```

## Notes

- **Aggregated, single GPU.** The encoder runs in the same process as the worker
  (no encode worker, no NIXL). The PR validates `--custom-encoder-class` as
  agg-only + multimodal; it is rejected under disagg or `--use-vllm-tokenizer`.
- **`--enforce-eager`** skips cudagraph capture for a fast smoke startup and
  sidesteps the un-persistable compilation cache (DGD has no subPath mount).
  Remove it for the realistic engine path.
- **"42" is the deterministic contract** of the example encoder at
  `temperature=0` — it splices the model's own `embed_tokens` embeddings of a
  fixed phrase. If you swap in a real encoder, change the smoke assertion to
  match your model's expected output (or assert pipeline health only).
- Recipes are **local / on-demand reproducers**, not CI artifacts.
