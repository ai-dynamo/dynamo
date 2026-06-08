# Qwen3.5 — `--async-scheduling` GSM8K agg-vs-disagg (vLLM #42182)

Reproduces the setup of vLLM issue
[#42182](https://github.com/vllm-project/vllm/issues/42182) and checks whether the
reported accuracy collapse still occurs on the Dynamo disagg path. The issue:
`Qwen/Qwen3.5-27B` (a Gated-Delta-Net + full-attention **hybrid**) showed a GSM8K
accuracy **collapse** (~0.12 strict, ~35% invalid) when `--async-scheduling` was
enabled — but **only in disaggregated (P/D) serving**, where the GDN conv/recurrent
state is transferred over the NIXL connector (tracking issue #37285). The original
repro was a vLLM `0.20.2rc1` build (2026-05-12) using vLLM-native
`toy_proxy_server.py` on GB200.

The recipe runs GSM8K on the **same** model/image/config across agg vs disagg and
async on/off, isolating the trigger. It is **parametrized** by model (`MODEL=` env,
default `Qwen/Qwen3.5-27B`) and tensor-parallel size (`[TP]` arg, default 1).

GSM8K: full **1319** questions @ concurrency **64** (`lm-eval[api]`, pulls
`openai/gsm8k`), greedy, `max_gen_toks=2048`. Per-server flags mirror the issue:
bf16 KV cache, prefix caching OFF, `--generation-config vllm`, `NixlConnector`
(`kv_both`, `kv_load_failure_policy:fail`), `VLLM_SSM_CONV_STATE_LAYOUT=DS`,
`--max-num-seqs 256` (GDN: each decode seq needs a Mamba cache block, so the
default 1024 exceeds available blocks and aborts CUDA-graph capture).

## Findings (aws-dev-02, H100, vLLM 0.22.0 from Dynamo main)

**The collapse did NOT reproduce on the Dynamo disagg path.** Across both models,
agg vs disagg, and TP=1 vs TP=2, async-on tracked its async-off control within
~1 pp (≈1 stderr). GSM8K exact-match (strict-match):

| model | agg async-on | disagg async-on | disagg async-off (ctrl) |
|-------|:---:|:---:|:---:|
| Qwen3.5-27B (dense, bf16)      | 0.779 | 0.780 (TP1) · 0.772 (TP2) | 0.785 (TP1) · 0.782 (TP2) |
| Qwen3.5-35B-A3B-FP8 (MoE, fp8) | 0.597–0.614 (TP1/TP2) | 0.613 (TP2) | 0.597 (TP2) |

Disagg genuinely exercised the NIXL path (disagg runs ~1.8× slower than agg from the
prefill→decode hop; `kv_load_failure_policy:fail` requests completed cleanly, so the
GDN conv-state + KV transfer was correct). The 35B-A3B disagg numbers match its own
agg numbers — no silent corruption. (35B absolute is lower than 27B due to a uniform
thinking-output / answer-extraction artifact; the async-vs-control comparison is the
gate and holds.)

**Most likely why it didn't reproduce** (unverified, ranked): (1) fixed in vLLM
0.22.0 — the issue's repro was a ~3-week-older `0.20.2rc1` build; (2) Dynamo's disagg
path (frontend + workers) differs from the issue's vLLM-native `toy_proxy_server`;
(3) GB200/aarch64 (issue) vs H100/x86 (here). The decisive next test is replicating
the issue's exact stack (native vLLM proxy on the `0.20.2rc1` build / GB200).

## Run

```bash
# 1. Download the model (once; public repo). 27B ~54 GB bf16.
kubectl -n qiwa apply -f model-cache/model-download.yaml

# 2. Deploy + eval. Args: <agg|disagg> <DGD_NAME> <ASYNC_FLAG> <RUN_LABEL> [TP]
#    Model via MODEL= env (default Qwen/Qwen3.5-27B).
./run-config.sh agg    q27-agg-async      --async-scheduling    agg-async
./run-config.sh disagg q27-disagg-async   --async-scheduling    disagg-async
./run-config.sh disagg q27-disagg-noasync --no-async-scheduling disagg-noasync
# TP=2 disagg:
./run-config.sh disagg q27-disagg-async-tp2 --async-scheduling disagg-async-tp2 2
# A different model (must be on the PVC):
MODEL=Qwen/Qwen3.5-35B-A3B-FP8 ./run-config.sh disagg q35-disagg-async-tp2 --async-scheduling q35-disagg-async-tp2 2

# 3. Watch / collect
kubectl -n qiwa logs <label>-acc -f
kubectl -n qiwa cp <label>-acc:/perf-cache/accuracy/<label> ./results-<label>
```

`run-config.sh` renders the YAML via envsubst (`hw/h100.env` supplies `VLLM_IMAGE` +
selectors + pull secret), patches the DGD ServiceAccount with `ngc-pull-secret`,
waits for all components Ready, then launches the GSM8K accuracy pod against the
frontend.

## Teardown

```bash
kubectl -n qiwa delete dynamographdeployment <DGD_NAME>
kubectl -n qiwa delete pod <RUN_LABEL>-acc --ignore-not-found
```

## Files
- `model-cache/model-download.yaml` — HF download (`Qwen/Qwen3.5-27B` by default).
- `deploy/agg.yaml` — aggregated worker (envsubst: `$MODEL $TP $GPU_COUNT $ASYNC_FLAG`).
- `deploy/disagg.yaml` — 1P1D NixlConnector + GDN conv-state (same envsubst params).
- `accuracy-job.yaml` — GSM8K 1319 @ conc 64 via `lm-eval[api]`.
- `hw/h100.env` — image / node selector / pull secret.
- `run-config.sh` — deploy + SA patch + readiness wait + accuracy launch.
