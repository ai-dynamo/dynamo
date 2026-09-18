# Kimi-K2.6 GMS Shadow-Failover — vLLM 0.25.1 Port (Epoch Log)

Running milestone + findings log for porting the Kimi-K2.6-NVFP4 GMS shadow-failover
setup from vLLM 0.23.0 to 0.25.1 (latest dynamo `main`). Append as we iterate.

- **Branch:** `kimi-failover-0251` (worktree `~/repos/dynamo-kimi-0251`, off latest main)
- **Pod:** `kimi-failover` / ns `mohammed-dev` / node `tx5tk` (tainted `mohammed`) / image `daf44c6bf1…-vllm-runtime`
- **Model:** `/tmp/kimi-k2.6-nvfp4`, TP8, MoE backend **FLASHINFER_TRTLLM**, attention FLASHINFER_MLA

## The reframe
Most of the failover feature is **already merged into `main`** (single-block scratch #11911,
GMS failover #6818, materialization/#9854, kimi_k25 in vLLM). This port is **"validate main +
close version-drift deltas,"** not "re-land our stack."

## Status board
- [x] **L0 — RW→RO weight commit/import** (no shadow, no scratch-KV) — **GREEN** 2026-07-25
- [x] **L0 hardening** — backend fail-fast guard + logprobs A/B (RW vs RO bit-identical) — **DONE** 2026-07-25
- [x] **L1 — scratch-KV routing** — shadow reaches standby, single-block scratch, fail-closed guard never fires — **GREEN** 2026-07-25
- [x] **L2 — two-engine failover** — **GREEN eager+graphs** (promote +8s/+9s, post-failover 200 +5s)
- [x] **Concurrent cold-init** — both engines at once converge, peak 152 GiB (flock+sleep prevents double-peak) — **GREEN**
- [x] **Post-failover replenishment (fail-twice loop)** — failover-1 → new shadow parks beside live active → failover-2 → 200 — **GREEN** 2026-07-25

## Deltas layered on main (the 5 port patches)
| # | Patch | File | Why 0.25.1 needs it |
|---|-------|------|---------------------|
| 1 | meta-safety guard | vLLM `models/kimi_k25.py` (`_on_meta`) | 0.25.1 still does unconditional `.to(device)` on vision_tower/mm_projector → faults building the RO model on meta. Copy: `failover-kimi/kimi_k25_0251-metasafe.py`. |
| 2 | scratch-KV V1 routing | `failover-kimi/dyn_kvroute.py` (gate `DYN_GMS_KVROUTE_V1=1`) | 0.25.1 split the model runner into V1/V2. Main's `patch_kv_cache_pool_scope` targets a V2 symbol that doesn't exist → silent no-op on the default V1 runner. Wraps V1 `_allocate_kv_cache_tensors` in `gms_use_mem_pool("kv_cache")`. |
| 3 | materialize shape-adopt | `lib/gpu_memory_service/client/torch/module.py` | RW reduces NVFP4 MoE scales `(E,2)→(E,)` and commits that shape; RO meta param keeps `(E,2)`. Adopt the committed (authoritative) tensor instead of asserting. |
| 4 | RO MoE-kernel rebuild | `lib/gpu_memory_service/integrations/vllm/model_loader.py` (`_rebuild_ro_moe_kernels`) | RO materializes committed weights but never runs `process_weights_after_loading`, so `quant_method.moe_kernel is None` → MoE forward asserts. Rebuild **only** the kernel object (`get_fused_moe_quant_config` + `make_nvfp4_moe_kernel`); skip the weight re-transform + `fused_experts.process_weights` the writer already committed. Analog of 0.23.0 `66850a4`. |
| 5 | harness GMS-ready string | `failover-kimi/*.sh` | GMS server ready-log changed to "Server started"; old wait string burned a 12-min timeout → now ~9s. |

## L0 evidence (2026-07-25, eager, TP8)
```
RW  registered +223s | mem 141746 MiB/GPU | infer 200 "Paris. Paris is a major European city…"
kill A → weights persist in GMS servers: 73124 MiB/GPU
RO  registered +139s | rebuilt 60 MoE kernels/worker (all 8 TP) | 0 tracebacks/asserts
    imported 71.41 GiB/worker | infer 200 "Paris. Paris is a major European city…" (byte-identical to RW)
```
Harness: `failover-kimi/kimi_rwro.sh`. Raw logs on pod: `/tmp/kimi_rwro/logs/`.

### L0 hardening (2026-07-25)
- **Backend fail-fast guard** (`_rebuild_ro_moe_kernels`): logs the selected NVFP4 backend and
  **refuses to serve** on any backend outside `_RO_VALIDATED_NVFP4_BACKENDS = {FLASHINFER_TRTLLM,
  FLASHINFER_CUTLASS}` (override `DYN_GMS_ALLOW_UNVALIDATED_MOE_BACKEND=1`). Confirmed it logs
  `rebuilding MoE kernels for NVFP4 backend FLASHINFER_TRTLLM` and serves. → **R1 mitigated** (silent-wrong on
  a backend swap is now a loud, actionable error).
- **Logprobs A/B** (`kimi_rwro_ab.sh`, 4 diverse-margin prompts, `logprobs:5`): RW vs RO tokens all match and
  **worst max |Δlogprob| = 0.000e+00 — bit-identical**. Prompts: capital-of-France (high), gold→"Au…79"
  (factual), two-plus-two (numeric), favorite-season (open/low-margin). → **R3 closed** (numerically exact,
  not just argmax-equivalent).

## L1 + L2 results (2026-07-25, TP8, `kimi_failover.sh`)
Two-engine intra-pod shadow failover: engine0 ACTIVE + engine1 SHADOW share GMS weights, kill engine0 → shadow
promotes and serves. Env: `DYN_GMS_SCRATCH_SINGLE_BLOCK=1 DYN_NO_AUTOTUNE=1 DYN_GMS_KVROUTE_V1=1`.

- **L1 (scratch-KV routing):** `[KVROUTE] loaded` on all 8 workers; shadow reached **standby +139s**; the
  scratch fail-closed guard (`no KV allocation was routed through scratch`) fired **0 times**. The `dyn_kvroute.py`
  V1 fix works. Definitive corroboration: shadow overhead is only **+4.3 GiB** (single-block scratch 0.5 GiB) —
  real KV would be ~65 GiB → OOM, per-layer scratch ~30 GiB.
- **L2 eager:** ACTIVE serves 200 → kill whole engine0 → **SHADOW PROMOTED +8s** → **post-failover 200 at +5s**
  ("Paris…"). The +5s is the frontend evicting the dead active's discovery entry (retry-with-backoff, expected).

### Memory / concurrency study (answers: can we bring up a shadow next to a live active?)
Ceiling = **183,359 MiB** (B200). 1 Hz all-GPU sampler (`dev_mem.csv`) + phase markers (`phases.csv`),
analyzed by `analyze_mem.py`. Per-GPU (uniform across all 8):

| metric | eager | graphs |
|---|---|---|
| active resting (weights 73 shared + real KV + rt) | 147,359 MiB (143.9 GiB) | 147,477 MiB |
| colocated resting (active + resting shadow) | 151,721 MiB | 154,071 MiB |
| **Q1 — shadow steady overhead** (coloc − active) | **+4,362 MiB (~4.3 GiB)** | **+6,594 MiB (~6.4 GiB)** |
| **Q2 — replenishment bring-up peak** (shadow init w/ live active) | **155,859 MiB (152.2 GiB)** | **157,397 MiB (153.7 GiB)** |
| headroom at replenishment peak | **27,500 MiB (~26.9 GiB)** | **25,962 MiB (~25.4 GiB)** |
| **failover promotion transient peak** (scratch→real KV swap) | ≤155,859 (no spike) | **182,069 MiB (177.8 GiB) on 2/8 GPUs — 1s** |
| headroom at promotion | ~27 GiB | **~1,290 MiB (~1.3 GiB)** ⚠️ |
| replenishment verdict | **FITS** | **FITS** |

**Takeaways:** (1) The resting shadow is cheap — **~4.3 GiB eager / ~6.4 GiB graphs** on top of a live active (it
shares the 73 GiB weights via GMS and holds *scratch*-KV, not real KV; graphs costs ~2 GiB more for capture
residency). (2) **Replenishment** (bring a shadow up beside a serving active) peaks only ~4–6 GiB above colocated
resting — autotune-off killed the warmup burst, single-block scratch keeps scratch at 0.5 GiB — so it fits with
**~25–27 GiB to spare** in both modes. Green-light for shadow replenishment on promotion (the TRTLLM "fail-once"
limitation). (3) ⚠️ **New risk (graphs only): the failover PROMOTION transient** — the scratch→real KV swap
(`reallocate_all_handles`) — briefly spiked to **177.8 GiB on 2/8 GPUs (GPU1, GPU7), ~1.3 GiB from the ceiling**,
for a single 1s sample at +4s post-kill. Eager showed no such spike (stayed ≤152). So the tightest point in the
graphs lifecycle is promotion, not init/colocation. Worth a headroom margin (lower `--gpu-memory-utilization`, or
free the dying active's KV before the swap) before trusting graphs-mode promotion on larger models. Compare 0.23.0
baseline: eager colocated peak 149 / graphs 160 GiB.

## Post-failover shadow replenishment — GREEN (2026-07-25, `kimi_replenish.sh`, eager)
Lifts the TRTLLM prior-art **"fail-once, no replenishment"** limitation. Full loop, deterministic serialized launches:
```
P2  e0 ACTIVE (+224s) + e1 SHADOW parked (+141s)            colocated 151,721; serve 200
P3  FAILOVER-1: kill e0 -> e1 PROMOTED +7s                  serve 200
P4  REPLENISH: launch engineC (ENGINE_ID=0 reused) -> it races the live active's flock, loses,
    imports RO weights, sets up scratch-KV, PARKS +137s     colocated 151,783; e1 still serving 200
P5  FAILOVER-2: kill e1 -> engineC PROMOTED +8s             serve 200
=> KIMI_REPLENISH_DONE failover1=200 replenish_parked=yes failover2=200
```
**The previously-untested crux works:** a *fresh* shadow acquires a scratch-KV grant + RO weights from the
still-alive GMS servers **after** a promotion (active holds `kv_cache` RW), parks, and is itself promotable.
Replenished shadow's routing confirmed: fail-closed scratch guard fired **0×**; KVROUTE + MoE-kernel-rebuild +
sleep markers all present. **Memory stays bounded across the whole cycle** (per-GPU): colocated_1 155,859 /
failover-1 promo 157,633 / **replenish bring-up 155,921** / colocated_2 (replenished) 155,921 / failover-2 promo
151,039 → **worst-GPU overall peak 157,633 MiB (~25.7 GiB headroom)**. No growth across the cycle (replenish
colocated ≈ initial colocated → no leak). Harness: `failover-kimi/kimi_replenish.sh`, analysis
`analyze_replenish.py`. (Eager only; graphs replenishment not yet run — expect the same graphs promotion-transient
caveat as L2.)

## Deployment (real DGD via operator) — scoping (2026-07-25)
**The operator already supports this topology as a first-class feature — deploying it is config, not a build
(classification "a").** A worker component with:
```yaml
experimental:
  gpuMemoryService: { mode: IntraPod }
  failover:         { mode: IntraPod, numShadows: 1 }   # numShadows Maximum=1 today
```
(+ DGD annotation `nvidia.com/dynamo-kube-discovery-mode: container`) makes the operator auto-render the pod I
hand-roll: a **GMS sidecar** (`python3 -m gpu_memory_service.cli.server` — one supervisor that fans out one server
per-GPU-per-tag, collapsing my 16 processes), **cloned `engine-0`/`engine-1`** racing a flock at
`FAILOVER_LOCK_PATH=/gms-intrapod-control/failover.lock` in a shared emptyDir, auto-injected `ENGINE_ID`,
`DYN_SYSTEM_PORT`, `VLLM_NIXL_SIDE_CHANNEL_PORT`, `DYN_VLLM_KV_EVENT_PORT`, `DYN_VLLM_GMS_SHADOW_MODE=true`, plus a
`frontend` component. Key code: `deploy/operator/internal/gms/gms.go`, `internal/dynamo/failover.go` +
`failover_vllm.go` + `backend_vllm.go`, `internal/controller/failover_cascade_controller.go` (inter-pod only). CRD:
`api/v1beta1/common.go` (`GPUMemoryServiceSpec`, `FailoverSpec`). Sample: `deploy/operator/samples/v1beta1/dgd-gms-failover.yaml`.

**Config-only gaps (add to the worker container spec, NOT a build):** `--load-format gms` in `args` (auto-injected
only for inter-pod), and `DYN_GMS_SCRATCH_SINGLE_BLOCK` / `DYN_NO_AUTOTUNE` / `DYN_GMS_KVROUTE_V1` in `env` (operator
has no refs to these).

**⚠️ The real blocker for deploying MY validated build:** the 5 port fixes are currently **manually overlaid onto the
pod's site-packages** — they are NOT in any image. A real DGD uses the container image (`daf44c6bf1…-vllm-runtime` =
latest main, WITHOUT the fixes). So a DGD deploy needs the fixes in the image OR an overlay: (i) `model_loader.py`
(RO MoE-kernel rebuild + backend guard) and (ii) `module.py` (shape-adopt) are dynamo files → land them in main and
rebuild, or overlay; (iii) `kimi_k25.py` meta-safety is a **vLLM** change → upstream or overlay; (iv) `dyn_kvroute.py`
+ `dyn_noautotune.py` are sitecustomize overlays → ship via ConfigMap+initContainer or bake in. The operator's
schemaless `podTemplate.spec` accepts `initContainers`/`volumes`, so an overlay-init-container is a viable no-rebuild
path. **Net: operator topology = ready; the work is getting the patches into a deployable image/overlay.**

---

## Test coverage & known risks  ⚠️ read before trusting the RO path broadly
L0 is a **happy-path smoke test**: one prompt, greedy (temperature 0), eager, batch-1, ~6 in / 12 out.
It proves the RW→RO weight-sharing plumbing and the kernel rebuild *for this backend/config*. It does
**not** prove general correctness. Grounded findings from reading the 0.25.1 MoE code:

### R1 — RO kernel rebuild is backend-specific (audited: TRTLLM only)
L0 ran **FLASHINFER_TRTLLM**. Its `process_weights_after_loading` (step-5) does: (a) an **in-place fuse**
`w13_weight_scale_2.mul_(w13_input_scale)`, and (b) recompute `g1_scale_c` + register EPLB layer params.
- **Why skip-step-5 is correct here (by construction, not luck):** RW commits the *post-fuse* scales, so
  RO materializes already-fused `w13_weight_scale_2`. The experts `__init__` (called by
  `make_nvfp4_moe_kernel`) recomputes `self.g1_scale_c = quant_config.g1_alphas * a2_gscale` from those
  already-fused scales → correct value with no step-5 needed. Re-running step-5 would **double-fuse** →
  wrong. So skipping is both necessary and sufficient for inference here.
- **Residual risk:** only TRTLLM (+ flashinfer_cutlass by inspection) audited. The other 6 selectable
  backends (CUTEDSL, CUTEDSL_BATCHED, VLLM_CUTLASS, MARLIN, HUMMING, EMULATION, B12x) are **not audited**.
  A backend whose step-5 does work not reproduced by the experts `__init__` (workspace alloc, non-idempotent
  transform) would **break silently** if backend selection ever changes (different GPU / vLLM version /
  `VLLM_*` flag).
- **EPLB:** the skipped `register_parameter` for `g1_scale_c`/`gemm1_clamp_limit`/`gemm1_beta` means
  expert-parallel load-balancing rearrangement would miss them. Safe **only** while EPLB is off.

### R2 — 244 meta attention scales unmaterialized on RO
RO logs `244 meta tensors not in metadata`: `mla_attn.{q,k,v,prob}_scale` across all layers. These fp8 MLA
attention scales are left on the meta device (not committed by RW). L0 still served correctly (likely
defaulted/recomputed at runtime for this config), but this is an unexplained, pre-existing RO-path gap that
could bite a config where those scales are load-bearing.

### R3 — numerical check is argmax-only
Byte-identical greedy output on a *high-confidence* prompt is consistent with correctness but doesn't prove
bit-accuracy — argmax masks small scale errors. Need a **logprobs A/B (RW vs RO)** on a few diverse /
low-margin prompts.

### R4 — untested execution paths
- **Scratch-KV routing + sleep/wake** (patch #2): L0 has no shadow mode → completely unexercised. → L1/L2.
- **CUDA graphs:** L0 eager-only. Rebuilt kernel under graph capture untested. → L2 graphs pass.
- **Concurrency / long sequences / continuous batching:** L0 was batch-1. MoE grouped-GEMM has
  size-dependent paths.
- **Multimodal (vision) path:** text-only; patch #1 touches vision_tower/mm_projector but they're unexercised.
- **Other quant methods:** `_rebuild_ro_moe_kernels` only handles NVFP4; an FP8-MoE model would silently
  skip the guard and hit the same `moe_kernel is None` assert.
- **shape-adopt is permissive:** adopts *any* shape/dtype mismatch; a genuinely corrupt commit would be
  silently accepted rather than flagged.

### Recommended hardening (cheap → do first)
1. **Log + fail-fast** the selected NVFP4 backend in `_rebuild_ro_moe_kernels`; refuse (don't serve)
   unaudited backends. Turns silent-wrong into a loud error.
2. **Logprobs A/B** RW vs RO on several prompts (upgrade "Paris matched" → "numerically equivalent").
3. Investigate R2 (are the 244 meta scales actually load-bearing? default vs recompute?).
4. (later / for upstream) re-cache experts-object derived state + handle EPLB, or option-c
   (commit raw weights, run symmetric `process_weights` on RO) for backend-agnostic generality.

## Repro
```bash
# on pod kimi-failover / ns mohammed-dev / container dev
bash /tmp/kimi_rwro.sh      # L0 (RW→RO)
bash /tmp/kimi_failover.sh  # L2 (EAGER=1 then EAGER=0)
```
Overlays live in `/usr/local/lib/python3.12/dist-packages/` (`dyn_noautotune.py`, `dyn_kvroute.py`),
auto-loaded by `sitecustomize.py` when `DYN_NO_AUTOTUNE=1` / `DYN_GMS_KVROUTE_V1=1`.
