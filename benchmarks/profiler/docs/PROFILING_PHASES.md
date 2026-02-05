# SLA Profiler Phases

This document describes the phases executed by the `profile_sla.py` profiler when running an SLA-based profiling job via a `DynamoGraphDeploymentRequest` (DGDR).

## Overview

The SLA profiler automates the process of finding optimal GPU configurations and parallelization strategies for serving LLM models with disaggregated prefill/decode. It profiles performance across different configurations, selects the best options based on SLA requirements, then generates a deployment configuration using the planner.

## Profiling Phases

### Phase 1: Prefill Profiling (Multi-GPU Sweep)

**Purpose:** Find the best GPU count and parallelization mapping for prefill that meets the TTFT (Time To First Token) SLA.

**What happens:**
1. For each GPU count (1, 2, 4, ... up to `max_num_gpus_per_engine`):
   - Generate candidate parallelization mappings (TP, DP combinations)
   - Deploy a prefill-only configuration
   - Run AIPerf benchmark at the target ISL (Input Sequence Length)
   - Measure TTFT latency
   - Calculate throughput per GPU
   - Cleanup deployment
2. Generate performance plots (`prefill_performance.png`)

**Output:** List of (GPU count, parallelization mapping, TTFT, throughput/GPU) tuples

---

### Phase 2: Decode Profiling (Multi-GPU Sweep)

**Purpose:** Find the best GPU count and parallelization mapping for decode that meets the ITL (Inter-Token Latency) SLA.

**What happens:**
1. For each GPU count (1, 2, 4, ... up to `max_num_gpus_per_engine`):
   - Generate candidate parallelization mappings (TP, DP combinations)
   - Deploy a decode-only configuration
   - Determine max KV cache capacity from deployment logs
   - Sweep batch sizes based on KV cache capacity
   - For each batch size, run AIPerf and measure ITL + throughput
   - Cleanup deployment
2. Generate performance plots (`decode_performance.png`)

**Output:** List of (GPU count, parallelization mapping, ITL, throughput/GPU, concurrency) tuples

---

### Phase 3: Configuration Selection

**Purpose:** Select the optimal prefill and decode configurations based on SLA requirements and throughput efficiency.

**What happens:**
1. **Prefill selection:** Among configurations meeting TTFT SLA, pick the one with highest throughput/GPU
2. **Decode selection:** Among configurations meeting ITL SLA, pick the one with highest throughput/GPU
3. If no configuration meets SLA, warn and select the best available (lowest latency)

**Output:** Best prefill mapping + GPU count, best decode mapping + GPU count

---

### Phase 4: Prefill Interpolation

**Purpose:** Build a detailed TTFT vs ISL curve for the selected prefill configuration.

**What happens:**
1. Deploy the selected prefill configuration
2. Sweep across ISL values from minimum to `max_context_length`:
   - Run AIPerf at each ISL
   - Record TTFT for each point
3. Generate interpolation data and plot (`prefill_throughput_interpolation.png`)
4. Cleanup deployment

**Output:** ISL → TTFT mapping for planner consumption

---

### Phase 5: Decode Interpolation

**Purpose:** Build detailed ITL and throughput curves for the selected decode configuration across varying batch sizes and context lengths.

**What happens:**
1. Deploy the selected decode configuration
2. Perform coarse sweep of batch sizes (e.g., 1, 215, 429, 644, 858, 1073)
3. Perform fine-grained sweeps to identify performance inflection points
4. For each batch size:
   - Run AIPerf benchmark
   - Record ITL and throughput
5. Generate interpolation data and plots (`decode_throughput_interpolation.png`)
6. Cleanup deployment

**Output:** Batch size → (ITL, throughput) mappings for planner consumption

---

### Phase 6: DGD Generation with Planner

**Purpose:** Use the SLA planner to compute optimal replica counts and generate the final deployment configuration.

**What happens:**
1. Feed profiling data (interpolation curves) to the planner
2. Planner computes:
   - Optimal number of prefill replicas
   - Optimal number of decode replicas
   - Request rate capacity
3. Generate final `DynamoGraphDeployment` (DGD) configuration
4. Save to `config_with_planner.yaml`

**Output:** Complete DGD YAML ready for deployment

---

### Phase 7: Auto-Apply (Optional)

**Purpose:** Automatically deploy the generated configuration.

**What happens:**
1. If `autoApply: true` in the DGDR spec:
   - The operator creates the DGD from the generated config
   - Workers and frontend are deployed
   - DGDR state transitions to `Deployed`
2. If `autoApply: false`:
   - DGDR state transitions to `Completed`
   - User can review and manually apply the config

**Output:** Running inference deployment (if autoApply enabled)

---

## Phase Timeline Example

```
[00:00] Phase 1: Prefill profiling (1 GPU)
[02:00] Phase 1: Prefill profiling (2 GPUs)
[04:00] Phase 1: Prefill profiling (4 GPUs)
[06:00] Phase 2: Decode profiling (1 GPU) - sweep 6 batch sizes
[12:00] Phase 2: Decode profiling (2 GPUs) - sweep 6 batch sizes
[18:00] Phase 2: Decode profiling (4 GPUs) - sweep 6 batch sizes
[24:00] Phase 3: Configuration selection
[24:01] Phase 4: Prefill interpolation - sweep ~12 ISL values
[28:00] Phase 5: Decode interpolation - coarse + fine sweeps
[40:00] Phase 6: DGD generation with planner
[40:01] Phase 7: Auto-apply deployment
[42:00] Deployment ready
```

*Note: Actual timing varies based on model size, GPU count, and deployment startup time.*

---

## Artifacts Generated

| File | Description |
|------|-------------|
| `prefill_<N>gpus_<mapping>/` | Logs and configs for prefill GPU sweep |
| `decode_<N>gpus_<mapping>/` | Logs and configs for decode GPU sweep |
| `selected_prefill_interpolation/` | Prefill ISL sweep data |
| `selected_decode_interpolation/` | Decode batch size sweep data |
| `prefill_performance.png` | Prefill TTFT vs throughput/GPU plot |
| `decode_performance.png` | Decode ITL vs throughput/GPU plot |
| `prefill_throughput_interpolation.png` | TTFT vs ISL curve |
| `decode_throughput_interpolation.png` | ITL/throughput vs batch size curves |
| `config_with_planner.yaml` | Final DGD configuration |
| `mocker_config_with_planner.yaml` | Config for testing with mocker |
| `profile_sla.log` | Full profiler log |

---

## SLA Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `isl` | Input Sequence Length for profiling | 3000 |
| `osl` | Output Sequence Length for profiling | 150 |
| `ttft` | Target Time To First Token (ms) | 200.0 |
| `itl` | Target Inter-Token Latency (ms) | 20.0 |

---

## See Also

- [profile_sla.py](../profile_sla.py) - Main profiler implementation
- [DynamoGraphDeploymentRequest](../deploy/profile_sla_dgdr.yaml) - Example DGDR manifest
- [Planner documentation](../../../docs/planner.md) - SLA planner details
