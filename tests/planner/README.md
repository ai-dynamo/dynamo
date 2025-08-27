<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# SLA Planner Load Test

This directory contains comprehensive testing tools for validating the SLA planner's scaling behavior.
The SLA planner monitors metrics every 60 seconds (default adjustment interval) and scales
prefill/decode workers based on TTFT, ITL, and request patterns.

## Pre-Requisite: Pre-Deployment Profiling Data

You have two options to obtain the pre-deployment profiling data:

### Option A: Use Test Configuration (Quickstart)

Use the pre-configured test deployment with sample profiling data, we provide the results and the deployment configuration for the following models x hardware configurations:
- `nvidia/Llama-3.1-8B-Instruct-FP8` on H200 with max context length 16384, TP1 Prefill, and TP1 Decode. At ISL/OSL 3000/150, it achieves 40k tokens/s/gpu prefill with 80ms TTFT and 10k tokens/s/gpu decode with 10ms ITL. See `profiling_results/H200_TP1P_TP1D/`.

### Option B: Use Your Own Profiling Results

1. Run pre-deployment profiling for your specific setup. See the [pre-deployment profiling documentation](../../docs/architecture/pre_deployment_profiling.md) for detailed instructions.

## Interpolator Testing

SLA planner uses two interpolators to estimate the performance of prefill and decode. You can test the interpolators with the following command:

```bash
python components/planner/src/dynamo/planner/utils/perf_interpolation.py \
  --profile_results_dir <path_to_profile_results> \
  --isl <ISL> \
  --osl <OSL> \
  --ttft <TTFT(s)> \
  --itl <ITL(s)>
```

The script will perform the interpolation based on ISL, OSL, and TTFT and ITL SLAs and advise the load that can saturate the engine.

For example, to test the interpolator for `nvidia/Llama-3.1-8B-Instruct-FP8` on H200,

```bash
python components/planner/src/dynamo/planner/utils/perf_interpolation.py \
  --profile_results_dir tests/planner/profiling_results/H200_TP1P_TP1D/ \
  --isl 3000 \
  --osl 300 \
  --ttft 0.1 \
  --itl 0.01

# output:
ISL=3000, OSL=300
TTFT=0.1s, ITL=0.01s
Using profile results from tests/planner/profiling_results/H200_TP1P_TP1D/

Interpolating prefill performance ...
        Estimated TTFT=0.060s <= target TTFT=0.100s. Requests can queue 0.040s maximally while meeting TTFT SLA.
        Estimated throughput: 49481.09 tokens/s/gpu. Request rate at 16.49 requests/s will saturate one GPU.

Interpolating decode performance ...
        Average context length: isl + osl/2 = 3150.
        Estimated ITL=0.0097s <= target ITL=0.0100s at 16.16% active kv usage.
        Estimated throughput: 4555.68 token/s/gpu. Request rate at 15.19 requests/s will saturate one GPU.
```

## Generating Load Dataset

We provide a tool to generate load dataset with varying request rate. More details can be found in [sin_load_generator](../../benchmarks/sin_load_generator/README.md).

From previous interpolator testing, ISL 3000 and OSL 300 can handle ~15 request/s/gpu for both prefill and decode.
To test planner's performance for different request rates, we can generate a load dataset with request rate varying between 12 to 36 request/s.
For TP1 H200 engine, planner should scale between 1P1D and 3P3D.

```bash
python benchmarks/sin_load_generator/sin_synth.py \
  --time-duration 1800 \
  --request-rate-min 12 \
  --request-rate-max 36 \
  --request-rate-period 600 \
  --isl1 3000 \
  --osl1 300 \
  --isl2 3000 \
  --osl2 300 \
  --output-file rr-12-36_i3000o300.jsonl
```

The dataset starts at 12 requests/s, increases to 36 requests/s at t=300s, decreases back to 12 requests/s at t=600s, and repeats.
The total duration is 30 minutes or 1800 seconds.
## Planner Dry Run

Before testing SLA planner on real deployments, we provide a dry run feature to test the autoscaling behavior on a given dataset. Specifically, in dry run mode,
- The load predictor will be tested. However, the load metrics will be different from the real deployment because the actual OSL is only known after the requests are processed.
- There will be no SLA predictions. Instead, sla planner will show the safe throughput limit that will ensure the requests can be processed within the SLA.
- The correction factor will be disabled because there is no SLA metrics as reference.

To dry run SLA planner,

```bash
python components/planner/test/planner_sla_dryrun.py \
    --<SLA planner arguments> \
    --dry-run \
    --start-num-p <num_prefill_workers_to_start_with> \
    --start-num-d <num_decode_workers_to_start_with> \
    --output-plot <path_to_output_plot>
```

For example, to dry run SLA planner for the previous FP8 8B on H200 using the generated `rr-12-36_i3000o300.jsonl` dataset,

```bash
python components/planner/test/planner_sla_dryrun.py \
    --ttft 0.1 \
    --itl 0.01 \
    --adjustment-interval 60 \
    --profile-results-dir tests/planner/profiling_results/H200_TP1P_TP1D/ \
    --dataset rr-12-36_i3000o300.jsonl \
    --start-num-p 1 \
    --start-num-d 1 \
    --output-plot dryrun_plot.png
```

Below is the dryrun result:

![Dryrun Plot](./figures/dryrun_plot.png)

The first plot shows the actual request rate and the predicted request rate (in the unit of requests/adjustment_interval).

The second plot shows the actual ISL/OSL and the predicted ISL/OSL. The first two plots are useful when tuning the performance of the load predictor.

The third plot shows the actual prefill throughput, number of prefill workers that planner scales, and the safe throughput limit with the number of prefill workers. If the actual throughput is below the safe throughput limit, the deployment has the capacity to adhere the TTFT SLA. Note that in the real deployment, due to other factors such as queueing, load balancing, KV cache transfer latency, and ISL variance, it is not guaranteed that the actual deployment can adhere the TTFT SLA.

The fourth plot, similar to the third plot, shows the actual decode throughput, number of decode workers that planner scales, and the safe throughput limit with the number of decode workers. If the actual throughput is below the safe throughput limit, the deployment has the capacity to adhere the ITL SLA. Note that in the real deployment, due to other factors such as load balancing and OSL variance, it is not guaranteed that the actual deployment can adhere the ITL SLA.
