# Dynamo Hero-Page Demo Recording — Design & Implementation Plan

**Status:** Draft for review → hand-off
**Author:** (planning agent)
**Goal:** Produce a short (~90s) asciinema recording of the Dynamo "one YAML → production" deploy flow that loops on the Dynamo hero page. It should *look* like a live deployment but run entirely offline (no cluster), fast-forwarding through the long waits.

---

## 1. Context & Goal

We have a real 17-minute asciinema recording, `qwen-235-dgdr-demo-og.cast`, captured from the `demo` branch script `demo-ecommerce-deployment-narrated.sh` (a `--dgdr-file` variant). It walks the full DGDR lifecycle against a live AKS H100 cluster:

> **DGDR → Profiling → Generated DGD → Deploying (pods) → Serving → live chat request**

For a hero-page background loop, 17 minutes is unusable. **~15 of those minutes are the deploy poll loop** (pods pulling images + loading weights, AGE column ticking up to `15m`). We want to keep the *narrative* and the *real terminal output*, but compress to **~90 seconds** by fast-forwarding the waits.

### What this plan delivers
1. **`hero-demo.sh`** — a new, standalone, **cluster-free** replay script that prints pre-baked (real) output with hero-friendly timing. No `kubectl`, no GPUs, deterministic, re-recordable.
2. **`hero-demo.cast`** — the asciinema v3 recording produced by running the script once under `asciinema rec`.

> **Deliverable scope (confirmed):** stop at the `.cast` file. Embedding/player/styling is handled by a separate agent. This plan does **not** cover the web embed.

### Confirmed design decisions
| Decision | Choice |
|---|---|
| Output format | asciinema `.cast` (v3), leave as-is for downstream embed |
| Framing | **Neutral Dynamo** — "One YAML to Production" (drop e-commerce / 🛒 / Black Friday) |
| Target length | **~90 seconds** (≤110s acceptable if YAML/profiling beats feel rushed) |
| Build approach | **New synthetic replay script** (not post-processing the real cast) |
| Terminal size | **120×32**, pod-name hashes abbreviated for legibility |
| Landing path | Repo root on the **current branch** (`hero-demo.sh` + `hero-demo.cast` next to `hero-demo-plan.md`) |
| ASCII diagram (beat 4) | **Keep, tightened** to a compact 3-box row + planner note |

---

## 2. Source Material Analysis

### 2.1 `qwen-235-dgdr-demo-og.cast` (the 17-min source)
- **Format:** asciinema **v3** (relative time deltas per event), `200×60` terminal.
- **Total:** 1035.5s (17.3 min), 1586 events.
- **Command:** `./demo-ecommerce-deployment-narrated.sh --dgdr-file docs/qwen-dgdr-demo-flow/qwen-235b-fp8-no-planner.yaml --no-cleanup`

**Timeline breakdown (measured):**

| Section | Wall time | Duration | Keep? |
|---|---|---|---|
| Banner + cluster info card | 0–10s | 10s | Condense → 6s title card |
| Step 1: DGDR YAML (typed out) | 15–36s | 21s | Condense → ~16s |
| Step 2: `kubectl apply` + get dgdr | 36–50s | 14s | Condense → ~8s |
| Step 3: Profiling poll (Initializing→Deploying) | 50–94s | ~44s | Compress → ~14s |
| Step 4: Generated DGD (services table + ASCII diagram) | 94–109s | 15s | Condense → ~10s |
| **Step 5: Deploy poll (AGE 36s→15m, Pods 1/7→7/7)** | **109–994s** | **~885s** | **CUT → ~16s fast-forward** |
| Step 6: Verify (dgdr/dgd/pods -o wide/services) | 994–1014s | 20s | Fold into Step 6 → ~6s |
| Step 7: Port-forward + streaming chat response | 1014–1026s | 12s | Keep → ~10s |
| Summary card | 1026–1035s | 9s | Condense → ~4s closing card |

**The single biggest win:** Step 5 is 85% of the runtime and is almost entirely a `printf '\e[H\e[2J'` watch loop re-rendering the same three tables with an incrementing AGE column and `Pods: X/7 ready`. This is exactly what the "fast-forward" mechanic replaces.

### 2.2 `sertac-planner-demo.cast` (the 4.7-min reference)
Confirmed it uses two techniques we should borrow:
- **Countdown compression:** instead of really waiting 60s for the planner, it prints `⏱️  20 sec | Planner: Collecting metrics...` → `⏱️  10 sec | Planner: Analyzing latency...` with short real sleeps.
- **Rapid readiness ticks:** `⏳ Prefill workers: 1/2 ready` → `✅ Prefill workers: 2/2 ready` at ~2s intervals, with the pod AGE column jumping (`169m` old pod alongside a fresh `4m10s` pod).

Takeaway: the audience reads *state transitions*, not real elapsed time. We simulate transitions on a tight clock.

### 2.3 The live script (`demo-ecommerce-deployment-narrated.sh`, 1020 lines)
Reference only — **do not reuse directly**; it depends on a live cluster (`kubectl get dgdr`, `kubectl logs`, `kubectl port-forward`, real `curl`). We lift its **structure, section headers, color palette, and helper functions** (`narrate`, `show_command`, `step_header`, `type_yaml`, `pause`) and its exact on-screen copy. There are also `demo-ecommerce.sh` / `demo-airline-scenario.sh` with a `--dry-run` flag, but their dry-run just skips execution — it does **not** render fake output, so it is not what we want here. We are building a *replay*, not a dry-run.

---

## 3. Storyboard (~90s)

Terminal-only recording. Every beat is real captured text (see Appendix A for verbatim frames). Timings are targets; tune during recording.

| # | Beat | Window | Dur | On-screen | Mechanic |
|---|------|--------|-----|-----------|----------|
| 0 | **Title card** | 0:00–0:06 | 6s | Boxed banner "NVIDIA Dynamo — One YAML to Production" + compact cluster card (Model, 4× H100 nodes, 32× H100, SLA TTFT≤500ms · ITL≤30ms) | `clear`, hold. Loop-safe start frame. |
| 1 | **The only input** | 0:06–0:22 | 16s | `# ecommerce…` → **neutral** header `# qwen3-235b-dgdr.yaml`, then the DGDR YAML typed line-by-line (model, backend, searchStrategy: rapid, autoApply: true, modelCache, workload isl/osl, sla ttft/itl, planner disagg). Short "What you're NOT writing" callout. | `type_yaml` at ~0.03s/line (faster than the 0.06 original). |
| 2 | **Apply** | 0:22–0:30 | 8s | `❯ kubectl apply -f qwen3-235b-dgdr.yaml` → `dynamographdeploymentrequest.../qwen3-235b-fp8-disagg created`; then `❯ kubectl get dgdr …` showing PHASE `Profiling` / PROFILING `Initializing` / AGE `4s`. | Canned output; 1 short pause. |
| 3 | **Profiling** | 0:30–0:44 | 14s | Step 3 header. 3–4 rapid watch ticks cycling profiler phases: `▸ Initializing` → `▸ Sweeping Prefill` → `▸ Sweeping Decode` → `▸ Selecting Config` → `✅ Profiling complete — phase is now: Deploying`. | Clear-screen redraw per tick, ~0.8s each. AGE ticks `4s→27s→…`. |
| 4 | **What Dynamo built** | 0:44–0:54 | 10s | Step 4 header + the `selectedConfig` services table (Frontend×1, TRTLLMDecodeWorker×2 @8 GPU, TRTLLMPrefillWorker×4 @4 GPU, **Total 32 GPUs**) + a **tightened** ASCII Frontend→Prefill→Decode row with a one-line SLA Planner note (see A.4). | Static render; 1 pause. "You didn't write any of this." |
| 5 | **Deploy — fast-forward** ⭐ | 0:54–1:12 | 18s | Step 5 header. The watch frame (DGDR + DGD `READY False` + 7-pod table + `Pods: X/7 ready` + loading bar). **AGE column fast-forwards** and pods flip to Ready. | **Hero mechanic**, see §4. |
| 6 | **Live + verify** | 1:12–1:24 | 12s | `✅ Deployment complete — all workers healthy!` → condensed verify (services: Frontend 1/1, Decode 2/2, Prefill 4/4) → `🟢 MODEL IS LIVE` → port-forward line → `❯ curl …/v1/chat/completions` → streamed **Model response:** "Hooray! I'm now live on Kubernetes with NVIDIA Dynamo, all in one sleek YAML file—let's crush some AI workloads! 🚀" | Stream the response token-ish (chunked prints) for the "it's real" payoff. |
| 7 | **Closing card** | 1:24–1:30 | 6s | "One YAML. One command. Go make coffee. ☕" + one-liner: SLA Planner now watching TTFT/ITL, auto-scales prefill/decode on load. `Learn more: github.com/ai-dynamo/dynamo` | Hold ~2s → loop. Loop-safe end frame. |

**Total: ~90s.** Cushion: beats 1 and 5 are the easiest to trim if we run long.

### Neutral-framing edits (vs. original e-commerce cast)
- Banner: keep "NVIDIA Dynamo — One YAML to Production" (already neutral). Drop 🛒 → use ▸ or 🚀.
- `🛒 E-COMMERCE AI ASSISTANT IS LIVE!` → `🟢 MODEL IS LIVE`.
- DGDR `metadata.name`/labels: `ecommerce-shopping-assistant` → `qwen3-235b-fp8-disagg` (matches the rest of the captured output).
- Summary: drop "Black Friday"; keep generic "when load spikes, the planner scales prefill and decode to hold your SLA."

---

## 4. The Fast-Forward Mechanic (Beat 5) — the centerpiece

Replaces ~885s of real polling with a tight, legible time-lapse. This is the "AGE increments ~1 min every 0.5s" idea from the brief.

**Frame layout (re-rendered each tick via `printf '\e[H\e[2J'`):**
```
  🚀 Step 5: Deployment — Watching Workers Come Online
  Fast-forward ▸ 8m elapsed

❯ kubectl get dgdr qwen3-235b-fp8-disagg -n dynamo-system
NAME                    MODEL                      BACKEND   PHASE       ...   DGD             AGE
qwen3-235b-fp8-disagg   Qwen/Qwen3-235B-A22B-FP8   trtllm    Deploying         trtllm-disagg   8m

❯ kubectl get dgd trtllm-disagg -n dynamo-system
NAME            READY   BACKEND   AGE
trtllm-disagg   False             8m

❯ kubectl get pods -n dynamo-system -l 'nvidia.com/dynamo-graph-deployment-name=trtllm-disagg'
NAME                                          READY   STATUS    RESTARTS   AGE
trtllm-disagg-frontend-…                      1/1     Running   0          8m
trtllm-disagg-trtllmdecodeworker-…-22g9r      0/1     Running   0          8m
trtllm-disagg-trtllmdecodeworker-…-7x9z6      0/1     Running   0          8m
trtllm-disagg-trtllmprefillworker-…-29rsc     0/1     Running   0          8m
… (4 prefill rows)

   Pods: 1/7 ready
   ▸ Workers running — Loading model weights...
   Loading model: [████████████░░░░░░░░░░░░░░░░░░] 40% | 980/2452 shards across 7 workers
```

**Tick schedule (drives AGE + readiness + loading bar together).** ~16–18 frames, `TICK≈0.5s`:

| Frame | AGE shown | Pods ready | Loading bar |
|---|---|---|---|
| 1 | 45s | 1/7 | 8% |
| 2 | 2m | 1/7 | 20% |
| 3 | 4m | 1/7 | 34% |
| 4 | 6m | 1/7 | 48% |
| 5 | 8m | 1/7 | 60% |
| 6 | 10m | 1/7 | 72% |
| 7 | 11m | 2/7 | 80% |
| 8 | 12m | 3/7 | 88% |
| 9 | 13m | 5/7 | 95% |
| 10 | 14m | 7/7 | 100% → `▸ All workers ready` |

- **AGE + readiness curve is faithful to the real cast** (2/7 at ~12m, 3/7 at ~12.7m, 7/7 at ~15m). We just cross the same milestones on a 0.5s clock instead of 5s real polls.
- **Header label:** show `Fast-forward ▸ Nm elapsed` (or a subtle `⏩`) so it reads as intentional time-lapse, not a stall. (Original said `Every 5s · Ns elapsed` — we relabel.)
- On the final frame flip `READY False`→`True`, all pods `1/1`, drop the loading bar, print `✅ Deployment complete — all workers are healthy!`, hold ~1.5s.
- **Implementation:** a bash array of `AGE:READY:PCT` tuples looped with a render function. Pod-row `1/1` vs `0/1` is derived from READY count (frontend first, then decode, then prefill flip to ready last — matches real order where frontend was ready early).

---

## 5. Implementation Notes (`hero-demo.sh`)

Target: a self-contained bash script that produces the storyboard with zero external services.

### 5.1 Principles
- **No cluster calls.** Every `❯ kubectl …` / `❯ curl …` line is `echo`'d as a *prompt string*, immediately followed by a **canned heredoc** of the real captured output (Appendix A). Nothing actually executes.
- **Reuse helpers** from `demo-ecommerce-deployment-narrated.sh`: `narrate`, `show_command`, `step_header`, `type_yaml`, `pause`, and the color palette (`CYAN/MAGENTA/GREEN/DIM/BOLD/NC`, `NAR` cyan). Copy them in; don't source the original.
- **Single timing knob per effect:** `TYPE_SPEED=0.03`, `TICK=0.5`, `PAUSE_SHORT=0.8`, `PAUSE_MED=1.5`. Tuning length = tuning these.
- **Determinism:** no randomness, no `$RANDOM`, fixed strings → identical every recording.

### 5.2 Suggested structure
```bash
#!/usr/bin/env bash
set -euo pipefail
# --- palette + helpers (narrate/show_command/step_header/type_yaml/pause) ---
# --- canned data blocks (heredocs): DGDR_YAML, GET_DGDR_INIT, SERVICES_TABLE,
#     POD_ROWS[], VERIFY_SERVICES, CHAT_RESPONSE ---
# --- render_deploy_frame(age, ready, pct) : clear + print 3 tables + bar ---
beat0_title
beat1_dgdr_yaml
beat2_apply
beat3_profiling      # loop over phase strings
beat4_generated_dgd
beat5_deploy_ffwd     # loop over AGE:READY:PCT tuples, TICK apart
beat6_live_and_chat   # stream CHAT_RESPONSE in chunks
beat7_closing
```

### 5.3 Streaming the chat response (beat 6)
Print `CHAT_RESPONSE` in small word/char chunks with ~0.02s sleeps to mimic token streaming — the single most "alive" moment. (Original did this from a real `curl -N` stream; we fake it from the captured string.)

### 5.4 Terminal size & legibility ⚠️ (needs a call — see §7)
- Original was `200×60`; pod-table rows are ~108 chars (long generated pod names). On a hero page scaled small, 200 cols is unreadable.
- **Recommendation:** record at **`120×32`** and **abbreviate pod-name hashes** (e.g. `trtllm-disagg-trtllmdecodeworker-…-22g9r`) so rows fit ~120 cols. This keeps the 7-pod table shape while staying legible when scaled down. All other frames fit comfortably at 120 cols.
- Drop the `-o wide` verify table (IP/NODE columns push to ~170 cols); use the compact per-service summary (`Frontend 1/1`, `Decode 2/2`, `Prefill 4/4`) instead.

### 5.5 Recording command
```bash
asciinema rec \
  --cols 120 --rows 32 \
  --idle-time-limit 1.5 \
  --command ./hero-demo.sh \
  hero-demo.cast
```
- `--idle-time-limit 1.5` caps any accidental dead air (belt-and-suspenders; our sleeps are all < 1.5s except intentional holds — set holds to ≤1.5s or raise this limit).
- Produces v3 `.cast`. Verify duration with the analysis snippet in Appendix B; iterate on `TICK`/pauses until ~90s.
- **Loop-safety:** first event is a `clear`; last beat holds a static card. Downstream player sets `loop: true` (not our concern, but keep the end frame clean and non-jarring against the start frame).

### 5.6 File locations
- Script: repo root **`hero-demo.sh`** on the **current branch** (next to `hero-demo-plan.md`).
- Output: repo root **`hero-demo.cast`**.
- No new branch — land alongside the plan doc.

---

## 6. Ground-Truth Content

All copy/tables/numbers must stay internally consistent with the captured run:
- **Model:** `Qwen/Qwen3-235B-A22B-FP8`  **Backend:** `trtllm` (disaggregated)
- **Cluster:** AKS, K8s v1.34.4, 4× `Standard_ND_H100_v5`, 32× H100 80GB SXM, 2,560 GB VRAM, Azure Managed Lustre
- **SLA:** TTFT ≤ 500ms · ITL ≤ 30ms · Mode: disagg
- **DGDR name:** `qwen3-235b-fp8-disagg`  **DGD name:** `trtllm-disagg`
- **Selected config:** Frontend×1, TRTLLMDecodeWorker×2 (8 GPU each), TRTLLMPrefillWorker×4 (4 GPU each) → **32 GPUs, 7 pods**
- **Chat response:** `Hooray! I'm now live on Kubernetes with NVIDIA Dynamo, all in one sleek YAML file—let's crush some AI workloads! 🚀`

See **Appendix A** for verbatim frames to bake into the heredocs.

---

## 7. Decisions

**Resolved (baked into this plan):**
- **Terminal width:** record at **120×32**; abbreviate pod-name hashes so the 7-pod table fits and stays legible when scaled down. Drop the `-o wide` verify table (use the compact per-service summary in A.6).
- **Landing path:** repo root, **current branch** — `hero-demo.sh` + `hero-demo.cast` next to `hero-demo-plan.md`. No new branch.
- **ASCII diagram (beat 4):** keep, but **tightened** to a compact 3-box row + one-line planner note (see A.4). It's a simulation, so exact look is flexible — implementer may adjust spacing/box style to fit 120×32 cleanly.
- **Fast-forward label:** use **`Fast-forward ▸ Nm elapsed`** (relabel the original `Every 5s · Ns`) so it reads as an intentional time-lapse.
- **Chat prompt/response:** reuse the captured celebratory line (A.7) — it's the most "alive" payoff and already on-brand.
- **Length latitude:** aim ~90s; up to ~110s acceptable if the YAML/profiling beats feel rushed.

**Still worth a quick gut-check during the first recording (cosmetic, non-blocking):**
1. Does the tightened diagram (A.4) look right at 120×32, or drop it entirely for a tighter services table? (Implementer's call — it's a simulation.)
2. Is the celebratory chat line the right tone for a hero page, or swap to a neutral one-liner? (Trivial string change.)

---

## Appendix A — Verbatim frames captured from `qwen-235-dgdr-demo-og.cast`

> Bake these into `hero-demo.sh` heredocs. Real output from the recorded run.

**A.1 Cluster card (title, beat 0)**
```
Model: Qwen/Qwen3-235B-A22B-FP8
Platform: Azure Kubernetes Service (AKS)
K8s: v1.34.4 · NVIDIA GPU Operator
Nodes: 4× Standard_ND_H100_v5
GPUs: 32× NVIDIA H100 80GB SXM
VRAM: 2,560 GB total
Storage: Azure Managed Lustre (AMLFS)
Backend: trtllm (disaggregated prefill/decode)
SLA: TTFT ≤ 500ms · ITL ≤ 30ms
```

**A.2 `kubectl get dgdr` right after apply (beat 2)**
```
❯ kubectl get dgdr qwen3-235b-fp8-disagg -n dynamo-system
NAME                    MODEL                      BACKEND   PHASE       PROFILING      DGD   AGE
qwen3-235b-fp8-disagg   Qwen/Qwen3-235B-A22B-FP8   trtllm    Profiling   Initializing         4s
```

**A.3 Profiling phase lines (beat 3)** — cycle these:
```
▸ Initializing — Detecting hardware, resolving model architecture...
▸ Sweeping Prefill — Testing TP/PP combinations for prefill latency...
▸ Sweeping Decode — Testing parallelization for decode throughput...
▸ Selecting Config — Filtering candidates against SLA, picking cheapest...
✅ Profiling complete - phase is now: Deploying
```

**A.4 Generated DGD services table (beat 4)**
```
❯ kubectl get dgdr qwen3-235b-fp8-disagg -n dynamo-system -o jsonpath='{.status.profilingResults.selectedConfig}'

   apiVersion:  nvidia.com/v1alpha1
   kind:        DynamoGraphDeployment
   name:        trtllm-disagg

   services:
   Service                                    Replicas   GPUs       Type
   ---------------------------------------- ---------- ------ ----------
   Frontend                                          1      -   frontend
   TRTLLMDecodeWorker (decode)                       2      8     worker
   TRTLLMPrefillWorker (prefill)                     4      4     worker

   Total GPU allocation: 32 GPUs
```
ASCII diagram (tightened, beat 4):
```
   Frontend ──▶ Prefill ×4 ──▶ Decode ×2      [ 32 GPUs · disagg ]
   (router)     (4 GPU ea)     (8 GPU ea)
   SLA Planner ▸ watches TTFT/ITL, auto-scales prefill & decode
```

**A.5 Deploy watch frame — early state, AGE=76s/38s (beat 5 seed)**
```
❯ kubectl get dgd trtllm-disagg -n dynamo-system
NAME            READY   BACKEND   AGE
trtllm-disagg   False             36s

❯ kubectl get pods -n dynamo-system -l 'nvidia.com/dynamo-graph-deployment-name=trtllm-disagg'
NAME                                                          READY   STATUS    RESTARTS   AGE
trtllm-disagg-frontend-bbff75d7-dmtvf                         1/1     Running   0          38s
trtllm-disagg-trtllmdecodeworker-7f586f22-5fc45649cc-22g9r    0/1     Running   0          38s
trtllm-disagg-trtllmdecodeworker-7f586f22-5fc45649cc-7x9z6    0/1     Running   0          38s
trtllm-disagg-trtllmprefillworker-7f586f22-6f5865b6b9-29rsc   0/1     Running   0          38s
trtllm-disagg-trtllmprefillworker-7f586f22-6f5865b6b9-vglb7   0/1     Running   0          38s
trtllm-disagg-trtllmprefillworker-7f586f22-6f5865b6b9-w56kh   0/1     Running   0          38s
trtllm-disagg-trtllmprefillworker-7f586f22-6f5865b6b9-z5m9v   0/1     Running   0          38s

   Pods: 1/7 ready
   ▸ Workers running — Loading model weights...
```
(7 pods: 1 frontend + 2 decode + 4 prefill. Frontend ready first; workers flip last.)

**A.6 Verify — per-service (beat 6, compact form to use instead of `-o wide`)**
```
   Frontend: 1/1 ready  (Deployment)
   TRTLLMDecodeWorker: 2/2 ready  (Deployment)
   TRTLLMPrefillWorker: 4/4 ready  (Deployment)
```

**A.7 Live chat (beat 6)**
```
❯ kubectl port-forward svc/trtllm-disagg-frontend 8000:8000 -n dynamo-system &
   Port-forward active on localhost:8000

❯ curl http://localhost:8000/v1/chat/completions -d '{"model":"Qwen/Qwen3-235B-A22B-FP8", "stream":true, ...}'

   Model response: Hooray! I'm now live on Kubernetes with NVIDIA Dynamo, all in one sleek YAML file—let's crush some AI workloads! 🚀
```

**A.8 DGDR YAML to type out (beat 1)** — derived from the script's `DGDR_YAML`, neutral names:
```yaml
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeploymentRequest
metadata:
  name: qwen3-235b-fp8-disagg
  namespace: dynamo-system
spec:
  model: Qwen/Qwen3-235B-A22B-FP8
  backend: trtllm
  searchStrategy: rapid
  autoApply: true
  modelCache:
    pvcName: model-cache
    pvcMountPath: /home/dynamo/.cache/huggingface
  workload:
    isl: 3000
    osl: 300
  sla:
    ttft: 500
    itl: 30
  features:
    planner:
      max_gpu_budget: 32
      mode: disagg
```

---

## Appendix B — Verify recording length after each take

```python
import json
with open("hero-demo.cast") as f:
    json.loads(f.readline())              # header
    t = sum(json.loads(l)[0] for l in f if l.strip())
print(f"{t:.1f}s  ({t/60:.2f} min)")      # target ~90s
```

---

## Appendix C — Reference files
- Source (17 min, real): `qwen-235-dgdr-demo-og.cast`
- Short-demo reference (4.7 min, skips waits): `sertac-planner-demo.cast`
- Live script (structure/copy source, needs cluster): `demo-ecommerce-deployment-narrated.sh` (on `demo` branch)
- YAML source: `docs/qwen-dgdr-demo-flow/qwen-235b-fp8-no-planner.yaml` (on `demo` branch)
