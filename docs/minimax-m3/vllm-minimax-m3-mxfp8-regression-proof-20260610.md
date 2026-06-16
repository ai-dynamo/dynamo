# MiniMax M3 vLLM MXFP8 Regression Evidence

Date: 2026-06-10

## Verdict

Yes: the pushed Dynamo vLLM runtime image can be used as a standalone `vllm
serve` image. Running `vllm serve` directly inside that image removes Dynamo
frontend, Dynamo worker routing, and Rust parser code from the request path.

With the current Day0 CF vLLM runtime image, clean standalone `vllm serve`
fails during vLLM engine initialization. With a runtime hotpatch it boots but
generates corrupted output. An older 2026-06-04 vLLM image boots and produces
sane base output on the same MXFP8 weights and same GB200 TP=4 shape.

To remove Dynamo completely from the test, we also built and ran a pure
`vllm-openai` image directly from the private AME/vLLM repo at the same
`afbc9ad` commit. That image boots, but returns corrupted text for the same
basic prompt that the 2026-06-04 image answers correctly. This points to a
vLLM/AME model-runtime regression or dependency interaction, not a Dynamo
frontend issue.

## Images

Current failing image:

```text
dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-vllm-runtime-cu130-arm64-day0cf-dyn758cff0-ameafbc9ad-20260610
```

Pull command:

```bash
docker pull dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-vllm-runtime-cu130-arm64-day0cf-dyn758cff0-ameafbc9ad-20260610
```

Code encoded in tag:

- Dynamo: `758cff0`
- AME/vLLM: `afbc9ad`

Last known sane base-inference comparison image:

```text
dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-vllm-runtime-cu130-arm64-ame-tracking-20260604
```

Pull command:

```bash
docker pull dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-vllm-runtime-cu130-arm64-ame-tracking-20260604
```

The comparison image is older and does not register the newer `minimax_m3`
tool/reasoning parser names, so it is only a base inference comparison.

The working image itself reports `VLLM_BUILD_COMMIT=unknown`, but the original
source tarball used to build it still exists on the arm build node. Its git
archive metadata resolves to:

```text
source archive: /tmp/ibhosale-dynamo-vllm-build-ame-tracking-20260604-node0102/ibhosale-dynamo-vllm-ame-tracking-20260604-node0102-ame-tracking.tgz
archive commit: 6647a1a88bf7d52de3d74db37ad0ad7e0e46a9cb
subject: Reduce dtype conversions for AR fusion (#40)
committer date: 2026-06-04 00:43:45 +0800
```

Command used:

```bash
gzip -dc "${AME_TGZ}" | git get-tar-commit-id
```

The copied build log shows:

```text
local vLLM OpenAI image: local/vllm-openai:ame-tracking-minimax-m3-cu130-arm64
local vLLM image id: sha256:7ff19ce0b75ebace72e89543bdfe6b33eb21f51660f08a48d9129e645f93c251
final Dynamo runtime image id: sha256:b5b9f518238f627e39b404e2b283a2d606bafacdb9458875e43d8ab258996c8d
final ACR tag: dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-vllm-runtime-cu130-arm64-ame-tracking-20260604
```

Pure vLLM OpenAI image, no Dynamo runtime:

```text
dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-pure-vllm-openai-cu130-arm64-ameafbc9ad-20260610
```

Digest observed in Kubernetes:

```text
dynamoci.azurecr.io/ai-dynamo/dynamo@sha256:afe15452d17a8efc56d72b0981aa54e24f4335c95b2bfb3e54568b760ba21953
```

Pull command:

```bash
docker pull dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-pure-vllm-openai-cu130-arm64-ameafbc9ad-20260610
```

Pure image source:

- AME/vLLM repo: `/home/ibhosale/ame`
- Branch: `ame_tracking`
- Commit: `afbc9ad1921fe259001edec446de3dff839f11b1`
- Commit subject: `Enable PDL attention decode kernels (#66)`

The failing pure-vLLM build log records:

```text
source_tar=/tmp/ame-afbc9ad1921f-20260610.tar
ame_sha=afbc9ad1921fe259001edec446de3dff839f11b1
local_image=local/pure-vllm-openai:minimax-m3-cu130-arm64-ameafbc9ad-20260610
acr_image=dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-pure-vllm-openai-cu130-arm64-ameafbc9ad-20260610
source extracted from git archive; expected AME SHA afbc9ad1921fe259001edec446de3dff839f11b1
local pure vLLM OpenAI image id: sha256:63e6c930f9992b3184b2400d24d633af78a49463dd21f8771d93c02ae973b355
```

Local build evidence files:

```text
/home/ibhosale/dynamo/test-results/minimax-m3-dynamo-vllm/build-artifacts/20260604-working-image-build.log
/home/ibhosale/dynamo/test-results/minimax-m3-dynamo-vllm/build-artifacts/20260610-failing-pure-vllm-build.log
```

## Shared Test Shape

- Cluster namespace: `ibhosale-dynamo`
- Hardware: one GB200 NVL4 tray, 4 GPUs
- Weights: MXFP8 on `shared-model-cache`
- Model path in pod: `/model-cache/ibhosale-custom-model/payload`
- Parallelism: `--tensor-parallel-size 4 --enable-expert-parallel`
- Sparse attention block size: `--block-size 128`
- Entrypoint: direct `vllm serve`, not `python -m dynamo.vllm`

Representative launch:

```bash
vllm serve "${MODEL_PATH}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --block-size 128 \
  --mm-encoder-attn-backend FLASHINFER \
  --mm-processor-cache-type shm \
  --trust-remote-code \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --no-enable-log-requests
```

The current image also supports:

```bash
--tool-call-parser minimax_m3
--enable-auto-tool-choice
--reasoning-parser minimax_m3
```

Those parser flags are not needed to reproduce the engine-side crash/corruption.

## Pure vLLM OpenAI Image: Boots, Then Corrupts Output

Manifest:

```text
/home/ibhosale/dynamo/test-results/minimax-m3-dynamo-vllm/minimax-m3-vllm-serve-mxfp8-tp4-pure-vllm-ameafbc9ad-20260610.yaml
```

Running pod:

```text
namespace: ibhosale-dynamo
pod: minimax-m3-vllm-serve-mxfp8-tp4-pure
node: gke-brugdizq-dgxc-k8-customer-gpu-w0e-32ae6790-d686
pod_ip: 192.168.83.52
image_id: dynamoci.azurecr.io/ai-dynamo/dynamo@sha256:afe15452d17a8efc56d72b0981aa54e24f4335c95b2bfb3e54568b760ba21953
```

Runtime versions printed by the pod:

```text
vllm=0.11.2.dev278+ame.minimaxm3
torch=2.11.0+cu130
flashinfer-python=0.6.12
flashinfer-cubin=0.6.12
```

Exact direct `vllm serve` args:

```bash
vllm serve /model-cache/ibhosale-custom-model/payload \
  --served-model-name minimax-m3-vllm-serve-mxfp8-pure \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --block-size 128 \
  --mm-encoder-attn-backend FLASHINFER \
  --mm-processor-cache-type shm \
  --tool-call-parser minimax_m3 \
  --enable-auto-tool-choice \
  --reasoning-parser minimax_m3 \
  --trust-remote-code \
  --max-model-len 131072 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.90 \
  --no-enable-log-requests
```

The pure vLLM image booted successfully:

```text
Starting vLLM server on http://0.0.0.0:8000
Application startup complete.
GET /health HTTP/1.1 200 OK
```

Request:

```json
{
  "model": "minimax-m3-vllm-serve-mxfp8-pure",
  "messages": [
    {
      "role": "user",
      "content": "What is 2+2? Answer with just the number."
    }
  ],
  "max_tokens": 32,
  "temperature": 0,
  "chat_template_kwargs": {
    "thinking_mode": "disabled"
  }
}
```

Response summary:

```text
HTTP: 200
finish_reason: length
completion_tokens: 32
content: corrupted text, whitespace, punctuation, repeated ampersands, and
         non-English/replacement characters; not "4"
system_fingerprint: vllm-0.11.2.dev278+ame.minimaxm3-tp4-ep-87027209
```

A second request without `chat_template_kwargs` also returned corrupted text:

```text
HTTP: 200
finish_reason: length
completion_tokens: 64
content: corrupted text with repeated whitespace, ampersands, punctuation,
         partial tag fragments such as "</"; not "4"
system_fingerprint: vllm-0.11.2.dev278+ame.minimaxm3-tp4-ep-87027209
```

Request-time logs show HTTP 200 responses and no Dynamo involvement:

```text
POST /v1/chat/completions HTTP/1.1" 200 OK
Engine 000: Avg prompt throughput: 12.7 tokens/s, Avg generation throughput: 3.2 tokens/s
POST /v1/chat/completions HTTP/1.1" 200 OK
```

## Pure vLLM OpenAI Image: PDL-Revert Hotpatch Still Corrupts Output

To test whether `afbc9ad19` alone caused the bad generations, the same pure
vLLM image was launched with a runtime Python hotpatch that forces MiniMax M3
PDL decode off before `vllm serve` starts.

Hotpatch manifest:

```text
/home/ibhosale/dynamo/test-results/minimax-m3-dynamo-vllm/minimax-m3-vllm-serve-mxfp8-tp4-pure-vllm-pdlrevert-20260610.yaml
```

Running pod:

```text
namespace: ibhosale-dynamo
pod: minimax-m3-vllm-serve-mxfp8-tp4-pdlrevert
node: gke-brugdizq-dgxc-k8-customer-gpu-w0e-32ae6790-06vd
pod_ip: 192.168.90.81
image: dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-pure-vllm-openai-cu130-arm64-ameafbc9ad-20260610
```

Patch applied inside the container:

```text
/usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/common/ops/index_topk.py:
  use_pdl = False
/usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/common/ops/sparse_attn.py:
  use_pdl = False
```

The pod booted cleanly, but the same deterministic request still returned
corrupted output:

```text
HTTP: 200
finish_reason: length
completion_tokens: 32
content: whitespace, punctuation, repeated ampersands, replacement character,
         and unrelated text; not "4"
system_fingerprint: vllm-0.11.2.dev278+ame.minimaxm3-tp4-ep-87027209
```

This means disabling the PDL decode launch path is not sufficient to restore
correctness. PDL may still be one risky change, but the standalone corruption
also implicates another MiniMax M3 runtime path in the current AME/vLLM stack.

## Pure vLLM OpenAI Image: Dense Allreduce Revert Still Corrupts Output

The 2026-06-04 working image already had the post-attention
`fused_allreduce_gemma_rms_norm` path, but it did not have the later
`b1ac5dfdd` dense-MLP deferred allreduce path. To isolate that change, a second
pure-vLLM runtime hotpatch restored dense MLP allreduce and disabled only
`self.fuse_input_allreduce`.

Hotpatch manifest:

```text
/home/ibhosale/dynamo/test-results/minimax-m3-dynamo-vllm/minimax-m3-vllm-serve-mxfp8-tp4-pure-vllm-densear-revert-20260610.yaml
```

Running pod:

```text
namespace: ibhosale-dynamo
pod: minimax-m3-vllm-serve-mxfp8-tp4-densear-revert
node: gke-brugdizq-dgxc-k8-customer-gpu-w0e-32ae6790-5rzq
pod_ip: 192.168.76.15
image: dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-pure-vllm-openai-cu130-arm64-ameafbc9ad-20260610
```

Patch applied inside the container:

```text
/usr/local/lib/python3.12/dist-packages/vllm/models/minimax_m3/nvidia/model.py:
  self.fuse_input_allreduce = False
  dense MiniMaxM3MLP reduce_results=True
```

The pod booted cleanly, but the same deterministic request still returned
corrupted output:

```text
HTTP: 200
finish_reason: length
completion_tokens: 32
content: whitespace, repeated ampersands, punctuation, and unrelated text;
         not "4"
system_fingerprint: vllm-0.11.2.dev278+ame.minimaxm3-tp4-ep-87027209
```

This means `b1ac5dfdd` is not sufficient to explain the corruption either.

## Pure vLLM OpenAI Image: Router GEMV Revert Still Corrupts Output

The old 2026-06-04 working image did not enable the custom fp32 router GEMV for
MiniMax-M3's `(hidden_size=6144, num_experts=128)` gate shape. The current
`afbc9ad` image does enable that path via `513066869` ("M3 router GEMV").
Because router logits control expert selection before the MXFP8 expert GEMMs,
this was a plausible explanation for corrupted generations.

Hotpatch manifest:

```text
/home/ibhosale/dynamo/test-results/minimax-m3-dynamo-vllm/minimax-m3-vllm-serve-mxfp8-tp4-pure-vllm-routergemv-revert-20260610.yaml
```

Running pod:

```text
namespace: ibhosale-dynamo
pod: minimax-m3-vllm-serve-mxfp8-tp4-routergemv-revert
node: gke-brugdizq-dgxc-k8-customer-gpu-w0e-32ae6790-5rzq
pod_ip: 192.168.76.16
image: dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-pure-vllm-openai-cu130-arm64-ameafbc9ad-20260610
```

Patch applied inside the container:

```text
/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/router/gate_linear.py:
  FP32_SUPPORTED_SHAPES = {(3072, 256)}
```

The pod booted cleanly, but the same deterministic request still returned
corrupted output:

```text
HTTP: 200
finish_reason: length
completion_tokens: 32
content: whitespace, repeated ampersands, punctuation, and unrelated text;
         not "4"
system_fingerprint: vllm-0.11.2.dev278+ame.minimaxm3-tp4-ep-87027209
```

This means the MiniMax-M3 custom router GEMV path is not sufficient to explain
the corruption either.

## Pure vLLM OpenAI Image: 20260604 index_topk.py Override Still Corrupts Output

The 2026-06-04 working image used the older MiniMax-M3 indexer decode score
kernel shape, with one Triton program dimension for `num_idx_heads`. The
current `afbc9ad` image includes later `index_topk.py` changes including
vectorized all-head BF16 MMA scoring and PDL launch plumbing. To isolate those
changes, the current pure-vLLM image was run with only `index_topk.py` replaced
by the file copied from the old working 20260604 pod.

Hotpatch file copied to:

```text
/home/ibhosale/dynamo/test-results/minimax-m3-dynamo-vllm/hotpatch-files/index_topk_20260604.py
```

ConfigMap:

```text
namespace: ibhosale-dynamo
configmap: minimax-m3-index-topk-20260604
```

Hotpatch manifest:

```text
/home/ibhosale/dynamo/test-results/minimax-m3-dynamo-vllm/minimax-m3-vllm-serve-mxfp8-tp4-pure-vllm-index-topk-20260604-override-20260610.yaml
```

Running pod:

```text
namespace: ibhosale-dynamo
pod: minimax-m3-vllm-serve-mxfp8-tp4-index-topk-20260604
node: gke-brugdizq-dgxc-k8-customer-gpu-w0e-32ae6790-5rzq
pod_ip: 192.168.76.17
image: dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-pure-vllm-openai-cu130-arm64-ameafbc9ad-20260610
```

Patch confirmation in startup logs:

```text
319:    pid_bc, pid_h = tl.program_id(0), tl.program_id(1)
721:    grid_score = (batch * num_kv_chunks, num_idx_heads)
```

The pod booted cleanly, but the same deterministic request still returned
corrupted output:

```text
HTTP: 200
finish_reason: length
completion_tokens: 32
content: whitespace, punctuation, repeated ampersands, replacement character,
         and unrelated text; not "4"
system_fingerprint: vllm-0.11.2.dev278+ame.minimaxm3-tp4-ep-87027209
```

This means the `index_topk.py` decode-kernel changes after 2026-06-04 are not
sufficient to explain the corruption either.

## Current Image: Clean Failure

Manifest:

```text
/home/ibhosale/dynamo/test-results/minimax-m3-dynamo-vllm/minimax-m3-vllm-serve-mxfp8-tp4-day0cf-clean-20260610.yaml
```

Failure note:

```text
/home/ibhosale/dynamo/test-results/minimax-m3-dynamo-vllm/vllm-serve-mxfp8-tp4-day0cf-clean-failure-20260610.md
```

The current image loads all 31 MXFP8 checkpoint shards, then fails during the
vLLM profile/dummy forward used to size KV cache:

```text
vllm/v1/engine/core.py:_initialize_kv_caches
  -> gpu_model_runner.py:profile_run
  -> gpu_model_runner.py:_dummy_run
  -> vllm/models/minimax_m3/nvidia/model.py
  -> MiniMAXGemmaRMSNorm.forward
  -> flashinfer.norm.gemma_rmsnorm
  -> flashinfer/norm/kernels/rmsnorm.py:rmsnorm_cute
  -> nvidia_cutlass_dsl/.../cutlass.py:_build_gpu_module
```

Root error:

```text
TypeError: __init__(): incompatible function arguments
Invoked with types:
  cutlass._mlir.dialects._gpu_ops_gen.GPUModuleOp,
  str, tuple, NoneType, NoneType,
  kwargs = { attributes: dict, results: list, operands: list, ... }
```

This is before the OpenAI API becomes ready, and does not involve Dynamo request
handling.

## Current Image: Hotpatch Result

A local runtime hotpatch that bypassed FlashInfer Gemma RMSNorm made the server
boot, but the first request then hit the same `GPUModuleOp`/Cutlass DSL failure
through the MiniMax sparse prefill CUTE DSL path.

After also forcing the sparse prefill path away from CUTE DSL, the server stayed
up but generated corrupted text. That is useful diagnostically: the hotpatches
prove which kernel paths block boot, but they are not a correctness fix.

Hotpatch manifest:

```text
/home/ibhosale/dynamo/test-results/minimax-m3-dynamo-vllm/minimax-m3-vllm-serve-mxfp8-tp4-day0cf-gemma-norm-hotpatch-20260610.yaml
```

## Older Image: Same Shape Produces Sane Base Output

Manifest:

```text
/home/ibhosale/dynamo/test-results/minimax-m3-dynamo-vllm/minimax-m3-vllm-serve-mxfp8-tp4-ame-tracking-20260604-noparsers.yaml
```

Running pod metadata:

```text
image: dynamoci.azurecr.io/ai-dynamo/dynamo:ibhosale-minimax-m3-vllm-runtime-cu130-arm64-ame-tracking-20260604
image_id: dynamoci.azurecr.io/ai-dynamo/dynamo@sha256:c3bd6620f8bc2e519b3158a949f08284f97dc9583a15fcad13f13ee376c91bb6
pod_ip: 192.168.84.131
node: gke-brugdizq-dgxc-k8-customer-gpu-w0e-32ae6790-qjmr
```

Runtime versions observed in the running pod:

```text
vllm=0.11.2.dev278+ame.tracking.minimaxm3
torch=2.11.0+cu130
flashinfer-python=0.6.12
nvidia-cutlass-dsl=4.5.2
```

Request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "minimax-m3-vllm-serve-mxfp8-20260604-noparsers",
    "messages": [
      {"role": "user", "content": "What is 2 plus 2? Answer with one number only."}
    ],
    "chat_template_kwargs": {"thinking_mode": "disabled"},
    "temperature": 0,
    "max_tokens": 64
  }'
```

Response content:

```text
4
```

Finish reason:

```text
stop
```

Server fingerprint:

```text
vllm-0.11.2.dev278+ame.tracking.minimaxm3-tp4-ep-09bfc070
```

## Why This Is Strong Evidence

- Same Kubernetes namespace and model PVC.
- Same MXFP8 weights.
- Same one-node GB200x4 topology.
- Same direct `vllm serve` entrypoint.
- Current Dynamo+AME image fails or corrupts without Dynamo request handling.
- Pure AME/vLLM OpenAI image at `afbc9ad` boots but corrupts a trivial request
  without any Dynamo code in the image.
- Older vLLM image gives sane base output under the same serving shape.

The next useful vLLM-side bisect should focus on MiniMax M3 runtime changes
between the recovered working source archive commit
`6647a1a88bf7d52de3d74db37ad0ad7e0e46a9cb` and the failing source archive
commit `afbc9ad1921fe259001edec446de3dff839f11b1`, especially:

- `MiniMAXGemmaRMSNorm` calling `flashinfer.norm.gemma_rmsnorm`
- MiniMax M3 SM100 CUTE DSL sparse prefill path
- Cutlass DSL / FlashInfer compatibility around `GPUModuleOp`
