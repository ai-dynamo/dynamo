<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Expert Parallelism Load Balancer (EPLB) in SGLang

Mixture-of-Experts (MoE) models utilize a technique called Expert Parallelism (EP), where experts are distributed across multiple GPUs. While this allows for much larger and more powerful models, it can lead to an uneven workload distribution. Because the load on different experts may vary depending on the workload, some GPUs can become bottlenecks, forcing the entire system to wait. This imbalance leads to wasted compute cycles and increased memory usage.

To address this, SGLang implements an Expert Parallelism Load Balancer (EPLB) inspired by the work in the DeepSeek-V3 paper. EPLB analyzes expert usage patterns and dynamically re-arranges the experts across the available GPUs to ensure a more balanced workload.

## The EPLB Algorithm: Core Concepts

The load balancing algorithm revolves around a few key ideas to achieve an optimal distribution of work.

### Redundant Experts for Flexibility

The core strategy is to create **redundant experts**. Instead of being limited to the model's original number of experts, EPLB can create duplicates of heavily-loaded experts. For example, if a model has 256 experts, you can configure EPLB to create an additional 32 "redundant" experts, bringing the total to 288. This pool of replicated experts is then strategically packed onto the available GPUs. A popular expert might be duplicated multiple times, while a moderately used expert might be grouped with several rarely used ones on a single GPU.

### Group-Limited Routing for Efficiency

Modern MoE models like DeepSeek-V3 use **group-limited expert routing**. In this design, experts are organized into groups, and routing decisions are constrained within these groups. EPLB can take advantage of this structure to reduce inter-node data traffic by attempting to place all experts from the same group onto the same node whenever possible.

### Load Balancing Policies

The algorithm comes with two policies for different scenarios:

1.  **Hierarchical Load Balancing**: This policy is used when the number of server nodes evenly divides the number of expert groups. It first harnesses the group-limited routing by packing expert groups onto nodes to balance the load between nodes. Then, within each node, it replicates and packs the experts onto individual GPUs to balance the load locally. This is often used during prefill where the expert-parallel size might be smaller.

2.  **Global Load Balancing**: In all other cases, a global policy is used. It replicates experts globally without regard to their group affiliation and packs them onto individual GPUs. This policy is more general and can be adopted during the decoding stage with a larger expert-parallel size.

## How SGLang Implements EPLB

SGLang provides a robust implementation of EPLB, allowing for dynamic, online rebalancing of expert locations based on real-world traffic.

### Dynamic Rebalancing

You can enable dynamic rebalancing by setting the `--enable-eplb` flag. When enabled, the `EPLBManager` runs in the background. It periodically triggers a rebalance after a certain number of requests, configured with `--eplb-rebalance-num-iterations`. At each rebalance, it computes a new expert placement plan based on the latest usage statistics and updates the model's expert locations on the fly.

### Expert Usage Recording

To make intelligent balancing decisions, SGLang needs to collect data on expert usage. The `ExpertDistributionRecorder` is responsible for this, and its behavior is controlled by the `--expert-distribution-recorder-mode` flag. This flag determines the granularity of the collected data. When `enable_eplb` is on, this mode defaults to `stat` to gather statistics for rebalancing. The available modes are:

- **`per_token`**: This is the most detailed mode. It records the specific expert choices for every single token processed by the model. While it provides the richest data, it also has the highest performance overhead. The raw, unaggregated data for each forward pass is stored.

- **`per_pass`**: In this mode, SGLang records the aggregated expert usage counts for each individual forward pass. The data is not aggregated across different passes, giving you a snapshot of expert popularity for each batch of requests.

- **`stat`**: This mode also records the exact expert usage counts for each forward pass, but it then aggregates these counts across multiple passes (the number of passes is determined by `--expert-distribution-recorder-buffer-size`). This provides a moving average of expert usage statistics and is the default when EPLB is enabled.

- **`stat_approx`**: This mode is similar to `stat` but gathers _approximate_ statistics, usually from the DeepEP dispatcher. This method has lower overhead than `stat` but is less precise, especially for small batch sizes. It is a good choice when performance is critical.

The collected statistics are then fed into the rebalancing algorithm to generate a new expert placement plan.

### Initializing with a Pre-computed Distribution

While SGLang can start with a simple default layout and learn a better one over time, you can also provide it with a pre-computed expert distribution to start with. The `--init-expert-location` flag allows you to specify a file path (`.pt` or `.json`) or a JSON string containing an expert layout. This is useful if you have already analyzed a representative workload offline and want the server to start immediately with a balanced configuration. If this flag is not set, it defaults to a `trivial` sequential layout.

### References and further reading

- [SGLang Large Scale P/D + WideEP Deployment](https://lmsys.org/blog/2025-05-05-large-scale-ep/#expert-parallelism-load-balancer)
- [Deepseek's EPLB repository](https://github.com/deepseek-ai/EPLB)
