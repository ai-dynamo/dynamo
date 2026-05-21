# Quantization Technique Matrix

Decision table for picking a Model Optimizer technique against a Dynamo
deployment target.

---

## Summary Table

| Technique | Precision | Hardware tier | Calibration data | Compute cost | Accuracy impact | Typical recipe |
|---|---|---|---|---|---|---|
| Weight-only FP8 | W8A16 | H100, H200, B200 | None | Minutes | Minimal | Quick wins |
| FP8 PTQ | W8A8 | H100, H200, B200 | C4 / Pile / production slice | 10s of minutes | Very low | Production deploys |
| NVFP4 | W4A8 (4-bit weight, 8-bit activation) | B200, GB200 only | C4 / Pile / production slice | 1-2 hours | Low (with good calib) | DeepSeek-V3.2 NVFP4 recipe; GLM-5-NVFP4 recipe |
| INT8 weight-only | W8A16 | A100, L40s, older | None | Minutes | Modest | Pre-Hopper deployments |
| AWQ | W4A16 | Any GPU | Small calibration set | 2-4 hours | Lowest impact for the precision | Larger MoE / dense models where accuracy matters |
| INT8 PTQ | W8A8 | Any GPU | C4 / Pile | 30-60 min | Modest-to-low | Memory-bound on smaller hardware |

## Per-Hardware Recommendation

**Blackwell (B200, GB200).** Use **NVFP4** for maximum compression and
the smaller memory footprint. Reference recipe:
`recipes/deepseek-v32-fp4/trtllm/disagg/`. NVFP4 is the technique that
unlocks Blackwell's full FP4 throughput.

**Hopper (H100, H200).** Use **FP8 PTQ** as the default. Weight-only
FP8 is a faster path if production accuracy tolerance is loose. Reference
recipe: `recipes/qwen3-32b-fp8/trtllm/agg/`.

**Ampere (A100).** FP8 is not natively supported. Use **INT8 weight-
only** or **AWQ** depending on accuracy tolerance.

**Older / L4 / T4.** INT8 weight-only is the only realistic option.

## When Multiple Techniques Are Viable

For Hopper or newer, the typical decision tree:

```
Want max throughput / smallest memory? → NVFP4 (B200+ only) or FP8 PTQ
Want fast quantization, accuracy-tolerant? → Weight-only FP8
Want best accuracy preservation? → AWQ
Going to production? → FP8 PTQ with calibration on production data
```

## Calibration Data Quick Notes

- **No calibration**: weight-only techniques (W8A16). Quantizes weights from the source distribution alone.
- **Generic calibration**: C4 or Pile. Use 128-512 samples for FP8/INT8 PTQ.
- **Production-aligned calibration**: a slice of your real prompt distribution. Improves accuracy on the workload at the cost of generality.
- **AWQ calibration**: small set (128-256 samples) but the technique is more sensitive to distribution match than PTQ.

## Recipe-First Path

If a Dynamo recipe exists for your model+hardware combination at the
desired precision, prefer the recipe. It ships:

- The exact `quant_config.json`.
- The calibration recipe (data source, sample count, prompt template).
- Published accuracy numbers measured at scale.
- A matching DGD that loads the quantized checkpoint.

The recipe set per release line is documented in
[`DYNAMO_REPO_SURVEY.md` §9.4](../../../docs/DYNAMO_REPO_SURVEY.md).
