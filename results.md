
## Iso-SLA ladder at c=8 (matched to agg's knee, SAME screening trace)

Comparator: **agg c=8 on `mooncake_screen_1500` = 55.402 tok/s/user P50, 427.3 ms TTFT P50.**
(Agg's headline 64.88 tok/s/GPU is from the FULL 3541 and is NOT the right comparator for
screening runs -- using it would overstate the gap.)

| Config | GPUs | Conc | P50 TTFT (ms) | P50 tok/s/user | System tok/s | tok/s/GPU | valid/err | SLA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **agg (reference)** | **8** | 8 | **427.3** | **55.402** | - | - | 1496/4 | ✅ pass |
| 1P1D | 16 | 8 | **4572.7** | **48.09** | 444.65 | 27.79 | 1492/4 | ❌ FAIL (user) |
| 1P1D | 16 | 16 | 10802.8 | 38.22 | 573.06 | 35.83 | 1496/4 | ❌ FAIL both |
| 1P2D (RR) | 24 | 16 | 12148.4 | 44.09 | 620.68 | 25.86 | 1496/4 | ❌ FAIL both |

### 1P1D c=8 — TTFT passes, per-user misses

- **TTFT 4572.7 ms clears the 5 s gate**, but is **+970%** vs aggregated's 427.3 ms on the
  identical trace and concurrency. Disaggregation costs roughly an order of magnitude on TTFT
  here -- the prefill/decode handoff and KV transfer dominate at 64K median ISL.
- **Per-user 48.09 misses the >=50 gate** (-13.2% vs agg).
- Per-GPU 27.79 on 16 GPUs, against agg meeting the SLA on **8**.

### External replication (strong)

Cheng's H200 tracker (`chengwa/chengwa-work-tracker`, issues-encountered **C20**) measured
**1P1D = 48.10 tok/s/user at C=8**. Ours is **48.09** on GB200. Different hardware, different
cluster, same topology, agreeing to 0.01. Their 2P1D looked like a pass at 53.80 but repeated to
**N=3 -> mean 49.90, sd 3.38 -> gate fails**; 1P1D's 48.10 sits inside that distribution.

**Implication:** disagg on this model parks at ~48-50 tok/s/user across topologies AND hardware --
persistently just under the gate. Our 48.09 is not an unlucky draw; it is the expected value.
Per Rule 25 any claim that a disagg config crosses 50 needs N>=3, and the prior from two
independent clusters is that it does not.
