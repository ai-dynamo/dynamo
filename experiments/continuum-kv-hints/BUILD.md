# Building the Continuum KV Hints container

Two Docker images are needed: a Dynamo+vLLM base built from this branch, and a thin
patch layer that overlays the custom vLLM onto it.

## Image names

| Image | Tag |
|---|---|
| Dynamo base (this branch) | `nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-55667792-vllm` |
| + vLLM patch layer | `nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-55667792-vllm-43aeef1b` |

- `55667792` = tip of `karenc/continuum-kv-hints-poc` (this Dynamo branch)
- `43aeef1b` = tip of `karenc/kv-hints-on-v0.29.0` in `~/vllm` (vLLM feature commits
  cherry-picked onto vLLM v0.29.0)

## Step 1 — Dynamo base image

```bash
cd /home/scratch.karenc_coreai/dynamo
git checkout karenc/continuum-kv-hints-poc

# Render the Dockerfile
python3 container/render.py --framework vllm --output-short-filename

# Apply the experiments/ patch — needed because this branch adds Cargo workspace
# members under experiments/ but the Dockerfile template only copies lib/ and
# components/. The patch adds COPY experiments/ at both build stages.
patch container/rendered.Dockerfile \
  experiments/continuum-kv-hints/rendered-dockerfile-experiments.patch

docker build \
  --build-arg ENABLE_MEDIA_FFMPEG=false \
  -t nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-55667792-vllm \
  -f container/rendered.Dockerfile \
  .

docker push nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-55667792-vllm
```

> **Note:** `render.py` regenerates `container/rendered.Dockerfile` and wipes any
> manual edits. Always re-apply the patch after re-running `render.py`.

## Step 2 — vLLM patch layer

The vLLM overlay Dockerfile ([`container/Dockerfile.vllm-kv-hints-patch`](../../container/Dockerfile.vllm-kv-hints-patch))
copies only the Python files changed by the feature branch — no C extensions are
touched. The build context is the vLLM checkout.

```bash
cd ~/vllm
git checkout karenc/kv-hints-on-v0.29.0  # cherry-picks of vLLM feature onto v0.29.0

docker build \
  -t nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-55667792-vllm-43aeef1b \
  -f /home/scratch.karenc_coreai/dynamo/container/Dockerfile.vllm-kv-hints-patch \
  .

docker push nvcr.io/nvidian/dynamo-dev/karenc:dynamo-kv-hints-55667792-vllm-43aeef1b
```

## Why two stages?

vLLM's C extensions (`_C.abi3.so`, flash-attn, etc.) are compiled into the base
`vllm/vllm-openai:v0.29.0-ubuntu2404` image. The feature changes are pure Python, so
they can be layered on top without recompilation. The vLLM branch
(`karenc/kv-hints-g1-actions`) was cherry-picked onto v0.29.0 specifically to isolate
the feature diff from upstream-main changes that could break API compatibility with the
compiled extensions.
