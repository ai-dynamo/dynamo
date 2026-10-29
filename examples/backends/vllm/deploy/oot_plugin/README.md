<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Deploying out-of-tree vLLM plugin models (e.g. Spark-X2.6)

Some model architectures are not registered in vLLM's in-tree model registry.
Their authors instead ship an **out-of-tree (OOT) plugin** — a pip package that
declares a [`vllm.general_plugins`](https://docs.vllm.ai/en/stable/design/plugin_system/)
entry point and registers the architecture (for example
`Spark2_5ForCausalLM`) at engine startup. Dynamo's published
`nvcr.io/nvidia/ai-dynamo/vllm-runtime` images do not contain third-party
model plugins, so serving these models requires a small custom image that adds
the plugin on top of the stock runtime.

This recipe shows the full path — build, deploy, verify — using
[XHToken Spark-X2.6](https://github.com/XHToken/Spark-X2.6) as the worked
example. It applies unchanged to any model whose vLLM support is delivered as
an OOT plugin.

**Why Spark-X2.6 as the example:** it is an Apache-2.0 4B/1.7B agentic model
with a 1M-token native context and a **hybrid attention** architecture (one
full-attention layer for every three 512-token sliding-window layers). Both
properties matter operationally: the long context drives the
`--max-model-len` sizing below, and the hybrid KV cache drives the routing
limitations in [Known limitations](#known-limitations-hybrid-attention-models).

## How it works

```text
┌──────────────────────────── custom image ────────────────────────────┐
│ nvcr.io/nvidia/ai-dynamo/vllm-runtime:<version>                      │
│   + pip install --no-deps <OOT plugin> (registers Spark2_5ForCausalLM)│
└──────────────────────────────────────────────────────────────────────┘
        │
        ▼
DynamoGraphDeployment (agg_oot_plugin.yaml)
  Frontend  ── dynamo.frontend (OpenAI-compatible API)
  worker    ── python3 -m dynamo.vllm --model XHToken/Spark-X2.6-4B ...
               vLLM loads general plugins at engine start and resolves
               the architecture through the plugin's registration.
```

Dynamo itself needs no code changes: the vLLM worker resolves architectures
through vLLM's registry after plugins load, exactly as `vllm serve` does.

## Prerequisites

1. **Dynamo Kubernetes Platform installed** — see the
   [Quickstart Guide](../../../../../docs/fern/pages/kubernetes/getting-started/quickstart.mdx).
2. A **container registry** you can push to.
3. The **plugin distribution** for your model. For Spark-X2.6, follow the
   vLLM deployment section of the
   [model repository](https://github.com/XHToken/Spark-X2.6) / HF model card;
   the plugin is published by the model authors, not by vLLM or Dynamo, and
   its exact package name and any companion pins are documented there.
4. An `hf-token-secret` in the target namespace (the manifests reference it
   through `envFrom.secretRef`; Spark-X2.6 weights are public, but the
   templates keep the secret reference for consistency with the other
   examples — remove it if you mirror weights locally).

## Step 1 — Build the custom runtime image

Pin the base image to the **same Dynamo release you deploy with** (see
[Engine version compatibility](#engine-version-compatibility)):

```bash
cd examples/backends/vllm/deploy/oot_plugin

# Stage a locally downloaded plugin wheel into the build context.
# Skip this when OOT_PLUGIN_SPEC resolves remotely (PyPI name, git+https URL):
# Docker builds cannot read arbitrary host paths, and ./wheels is copied to
# /tmp/oot-wheels inside the image.
cp /path/to/plugin.whl wheels/

docker build \
  --build-arg DYNAMO_VLLM_RUNTIME_IMAGE=nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0 \
  --build-arg OOT_PLUGIN_SPEC=/tmp/oot-wheels/plugin.whl \
  --build-arg OOT_EXPECTED_ARCH=Spark2_5ForCausalLM \
  -t my-registry/vllm-runtime-spark-x2.6:1.6.0 \
  -f Dockerfile .

docker push my-registry/vllm-runtime-spark-x2.6:1.6.0
```

`OOT_PLUGIN_SPEC` accepts a PyPI name, a `git+https://...` URL, or an
in-image path under `/tmp/oot-wheels/` (the `./wheels` build-context
directory is copied there). The plugin is installed with `pip
install --no-deps` so dependency resolution cannot silently replace the
vLLM/torch stack the Dynamo image pins; the build then verifies every
requirement the plugin *declares* against the installed distributions and
**fails when one is missing or version-incompatible** — e.g. a plugin
declaring `vllm==0.32.0` on a v1.6.0 base (which pins vLLM 0.31.0) is
rejected at build time rather than upgrading the engine. When the gate
rejects a plugin over the engine version, find a plugin build targeting the
pinned vLLM version (see below) instead of relaxing the gate.

`OOT_EXPECTED_ARCH` turns the registration check into a hard gate: the build
fails unless the plugin's entry point registers that architecture with
`ModelRegistry` after `load_general_plugins()` runs. Without it the build
still requires that the install **added a new** `vllm.general_plugins` entry
point (entry points already in the base image — vLLM's own and Dynamo's
telemetry plugin — do not count), but does not check which architecture it
registers.

Because `--no-deps` skips resolution entirely, dependencies the plugin needs
that the base image does not already provide must be listed explicitly in
`OOT_PLUGIN_EXTRA_REQUIREMENTS`; the build gate rejects the image when a
declared requirement is absent, so this cannot silently slip to pod startup.
If the plugin needs companion pins (for Spark models the authors document a
`transformers` pin), pass them the same way — these install *with*
dependency resolution, so read the compatibility section below first.
**Downgrading a package the base image pins can break the bundled vLLM
build**; prefer a plugin release built against the pinned vLLM version over
pinning your way backwards.

## Engine version compatibility

OOT plugins are compiled/tested against a specific vLLM API surface, and
vLLM's internal APIs drift between releases. Before building, check which
vLLM version your Dynamo release pins. The authoritative sources, in order of
convenience:

| Source | What it gives you |
| :-- | :-- |
| [Compatibility page — Release Support Matrix](../../../../../docs/fern/pages/reference/general/compatibility.mdx) | Rendered per-release matrix: CUDA toolkit, minimum driver, and the SGLang / TensorRT-LLM / vLLM / NIXL pins for every supported Dynamo release |
| [`docs/fern/components/releases.data.ts`](../../../../../docs/fern/components/releases.data.ts) | The same matrix as machine-readable data (`pins: { sglang, trtllm, vllm, nixl* }` per release) — the page above is generated from it |
| [`container/context.yaml`](../../../../../container/context.yaml) | What `main` currently builds against (base images and tags per framework/device), i.e. the next release's pins |

Recent pins at the time of writing (from the sources above; always re-check
for your release):

| Dynamo release | vLLM | SGLang | TensorRT-LLM |
| :-- | :-- | :-- | :-- |
| v1.6.0 | v0.31.0 | v0.5.21 | v1.3.0rc29 |
| v1.5.0 | v0.28.0 | v0.5.18 | v1.3.0rc25 |
| v1.4.0 | v0.26.0 | v0.5.16 | v1.3.0rc22 |
| main (1.7.0 dev) | v0.32.0 | v0.5.22 | v1.4.0rc1 |

Practical rules for OOT plugins:

- **Plugin ↔ vLLM:** use the plugin release the model authors test against
  the pinned vLLM line. Community plugin builds for popular models (e.g.
  NVFP4/quantized Spark variants on HF) usually state the vLLM version and
  may carry API-drift patches — read their model cards.
- **Plugin ↔ transformers:** the runtime image's transformers version is what
  the pinned vLLM was built against. A plugin demanding an older
  transformers may work (config/tokenizer only) or may break vLLM imports;
  verify inside the built image, not on your laptop:

  ```bash
  docker run --rm my-registry/vllm-runtime-spark-x2.6:1.6.0 \
    python3 -c "import vllm, transformers; print(vllm.__version__, transformers.__version__)"
  ```

- **Dynamo ↔ vLLM:** always take the vLLM version from the Dynamo image, not
  from PyPI. Mixed-version operation between Dynamo components is covered by
  the N-2 window described on the
  [Compatibility page](../../../../../docs/fern/pages/reference/general/compatibility.mdx#mixed-version-compatibility).

## Step 2 — Deploy

```bash
export DYNAMO_VLLM_IMAGE=my-registry/vllm-runtime-spark-x2.6:1.6.0
export OOT_MODEL=XHToken/Spark-X2.6-4B
export OOT_SERVED_MODEL_NAME=spark-x2.6
export OOT_MAX_MODEL_LEN=131072        # see sizing note below
export NAMESPACE=<your-dynamo-namespace>

./deploy.sh              # or: ./deploy.sh --render-only | less
```

Watch the rollout and port-forward the frontend:

```bash
kubectl get pods -n "$NAMESPACE" \
  -l nvidia.com/dynamo-graph-deployment-name=vllm-agg-oot-plugin -w
kubectl port-forward -n "$NAMESPACE" \
  deployment/vllm-agg-oot-plugin-frontend 8000:8000
```

### Sizing `--max-model-len` for long-context models

Spark-X2.6 declares a native 1,048,576-token context. vLLM defaults
`max_model_len` to the model's declared maximum and **profiles KV-cache
memory for it at startup**, which fails on anything short of a very large
GPU. Pick the served context deliberately:

- Start at `131072` on a single 80 GB-class GPU for the 4B model and raise it
  only after measuring.
- The hybrid attention layout works in your favor here: three of every four
  layers keep only a 512-token sliding window, so per-token KV growth is far
  below a dense full-attention model of the same depth.
- If you need the full 1M context, plan for multi-GPU (`tensor_parallel_size`
  via extra worker args) and re-check the limitations below — long-context
  hybrid models are exactly where KV routing features are still stabilizing.

## Step 3 — Test

Note the `model` field must match `--served-model-name`, not the HF id:

```bash
curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "spark-x2.6",
    "messages": [{"role": "user", "content": "Explain KV cache routing in one paragraph."}],
    "max_tokens": 256
  }'
```

Confirm the served name and context cap from the engine itself:

```bash
curl -s localhost:8000/v1/models | python3 -m json.tool
```

## Known limitations: hybrid attention models

The default manifest is an **aggregated deployment without KV-aware routing**
on purpose. Dynamo's KV router and disaggregated paths consume engine KV
cache events, and support for hybrid-attention models (full-attention +
sliding-window groups, or attention + Mamba/linear groups) landed recently
and is still completing. Status at the time of writing (verify against your
Dynamo release — also see the release-level
[Known Issues](../../../../../docs/fern/pages/reference/general/releases/known-issues.mdx)
page):

**Landed:**

- The router filters KV events by cache-spec kind, the DEP for hybrid KV
  cache group support
  ([dynamo#8667](https://github.com/ai-dynamo/dynamo/issues/8667), implemented
  by [dynamo#8751](https://github.com/ai-dynamo/dynamo/pull/8751), shipped in
  v1.2.0).
- The KV-event block-size mismatch that silently dropped events from hybrid
  vLLM models is fixed
  ([dynamo#9376](https://github.com/ai-dynamo/dynamo/issues/9376), shipped in
  v1.6.0).
- Disaggregated serving no longer fails at init when a model exposes two KV
  spec types ([dynamo#10741](https://github.com/ai-dynamo/dynamo/issues/10741),
  shipped in v1.6.0).

**Fixed on `main`, not included in v1.6.0/v1.6.1** at the time of writing:

- Hybrid-model KV events quarantining the KVCR cache owner
  ([dynamo#17124](https://github.com/ai-dynamo/dynamo/issues/17124), fixed by
  [dynamo#15535](https://github.com/ai-dynamo/dynamo/pull/15535)). If you use
  KVCR with a hybrid model, check the release notes before enabling it.

**Still open:**

- The KV router counts attention blocks the engine cannot reuse when
  computing prefix overlap for hybrid models, so overlap estimates are
  optimistic ([dynamo#16794](https://github.com/ai-dynamo/dynamo/issues/16794)).
- Mamba/linear-attention cache groups are not indexed in the KV router
  ([dynamo#12169](https://github.com/ai-dynamo/dynamo/issues/12169)).

Guidance:

1. Start with this aggregated template (`agg_oot_plugin.yaml`) — the known-good
   path for a newly integrated hybrid model.
2. If you need load balancing across replicas, use the router's **approximate
   KV mode** (no engine KV events consumed), as shown in
   [`../agg_router_kv_approx.yaml`](../agg_router_kv_approx.yaml), and
   validate hit behavior with your traffic mix before relying on it.
3. Event-based KV routing (`--router-mode kv`) and the disaggregated
   templates may now work for attention + sliding-window models on v1.6.0 or
   later, but validate against your model and release before production, and
   track [#16794](https://github.com/ai-dynamo/dynamo/issues/16794) /
   [#12169](https://github.com/ai-dynamo/dynamo/issues/12169) for the
   remaining gaps.

## Troubleshooting

| Symptom | Likely cause / fix |
| :-- | :-- |
| Worker crashes at startup with `ValueError: Model architecture 'Spark2_5ForCausalLM' is not supported` | Plugin not installed in the image, or its entry point is in the wrong group. Rebuild with `OOT_EXPECTED_ARCH` set so the Dockerfile gate catches this at build time; check `pip show` and the `vllm.general_plugins` entry points inside the image. |
| Build fails with `the install added no new vllm.general_plugins entry point` | The wheel was installed without its entry-point metadata (e.g. site-packages copied instead of `pip install`ed), or the package simply does not declare a vLLM plugin. Verify the distribution the model authors publish. |
| Build fails with `plugin ... requires vllm==X but this image has vllm==Y` | The `--no-deps` compatibility gate at work: the plugin targets a different vLLM than this Dynamo release pins. Use a plugin build matching the pinned version ([Engine version compatibility](#engine-version-compatibility)); do not install the plugin with dependencies, which would replace the bundled engine. |
| Build fails with `requires X, which is not installed in this image` | The plugin declares a dependency the base image lacks and `--no-deps` did not install it. Add it to `OOT_PLUGIN_EXTRA_REQUIREMENTS` (installed with resolution) and rebuild. |
| Import errors mentioning `transformers` after adding `OOT_PLUGIN_EXTRA_REQUIREMENTS` | The pin downgraded a package the bundled vLLM depends on. Remove the pin and find a plugin build matching the image's vLLM/transformers versions ([Engine version compatibility](#engine-version-compatibility)). |
| Startup OOM / KV-cache profiling failure on a long-context model | `--max-model-len` too large for the GPU. Lower it (see [sizing](#sizing---max-model-len-for-long-context-models)). |
| Image works locally but the pod fails `ErrImagePull` | Image not pushed, or the node lacks registry credentials. The DGD pulls at pod start on the node, not where you ran `docker build`. |
| Requests 404 with `model 'XHToken/Spark-X2.6-4B' does not exist` | You passed the HF id but the worker serves `--served-model-name`. Use the served name (`spark-x2.6` in the examples) in the request body. |
