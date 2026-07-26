# Dynamo resiliency with shadow failover — 3-node cascade

**Kimi-K2.6 (NVFP4) · Code Agent Traffic (100k/1k ISL/OSL) · 3× B200 nodes**

A 3-worker deployment is driven under a staggered cascade of 3 engine kills (one per worker, 60 s
apart), comparing a **baseline** (no failover) against **GMS shadow-failover**. Both arms run the
identical engine build and configuration, so the only difference is the failover feature.

## Setup

| | |
|---|---|
| **Model** | Kimi-K2.6, NVFP4 quantization |
| **Serving** | vLLM via Dynamo, TP8 per worker, 256K max context, prefix caching, MLA prefill on FlashInfer |
| **Hardware** | 3× B200 nodes (8 GPUs each, 24 GPUs total); one worker per node |
| **Failover** | GMS intra-pod shadow — each worker runs an active engine plus a shadow engine sharing the node's 8 GPUs via the GPU Memory Service; on kill, the shadow promotes and keeps serving |
| **Load** | Code Agent Traffic — KV-reuse trace, 100k/1k ISL/OSL, concurrency 24 |
| **Fault injection** | staggered cascade of 3 engine kills (T+240 / +300 / +360 s), one per worker |

## Headline

| | **baseline** (no failover) | **failover** (GMS shadow) |
|---|---|---|
| serving through the cascade | **blackout** (all 3 cold-restart) | **continuous** |
| truly-failed requests | **524** | **14** |
| decode (steady) | 64 tok/s/user | 65 tok/s/user |
| decode across kills | declines → gap → recovers | **flat** |
| TTFT p50 / max | 738 / **24,160 ms** | 765 / **8,519 ms** |

## TTFT

Baseline spikes to ~24 s during the recovery backlog and blows past the 5 s SLA; failover stays
flat and under SLA throughout. Per-request scatter with a 30 s mean on a shared 0–25 k ms y-axis.

| BASELINE | FAILOVER |
|:---:|:---:|
| ![](baseline/ttft_scatter.png) | ![](failover/ttft_scatter.png) |

## Per-user decode rate

Baseline breaks across the ~150 s blackout, then recovers; failover holds ~65 tok/s/user across all
three kills. Both are continuous clouds (high-variance long-context load), well above the 50
tok/s/user reference.

| BASELINE | FAILOVER |
|:---:|:---:|
| ![](baseline/tok_per_user_over_time.png) | ![](failover/tok_per_user_over_time.png) |

## Request outcome + cumulative

Baseline loses 524 requests during the blackout; failover loses 14. Cumulative successful requests
flat-line through the baseline outage, while failover climbs steadily throughout.

| BASELINE | FAILOVER |
|:---:|:---:|
| ![](baseline/http_status_over_time.png) | ![](failover/http_status_over_time.png) |
| ![](baseline/cumulative_successes.png) | ![](failover/cumulative_successes.png) |
