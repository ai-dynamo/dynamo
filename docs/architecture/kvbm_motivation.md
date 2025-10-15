<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Motivation behind KVBM

Large language models (LLMs) and other AI workloads increasingly rely on KV caches that extend beyond GPU and local CPU memory into remote storage tiers. However, efficiently managing the lifecycle of KV blocks in remote storage presents challenges:

* Tailored for GenAI use-cases
* Lack of visibility into real-time block usage patterns.
* Need for lightweight, ownership-driven memory management over complex object stores with unneeded overheads.
* Modular and need simplified UX and to be memory safe.
* Inability to differentiate between hot (frequently accessed) and cold (infrequently accessed) blocks across the stack without intrusive application-level changes.
* Difficulty in optimizing storage placement across heterogeneous storage tiers (for example, SSDs, object storage, and cloud storage).

Conventional systems either lack dynamic feedback mechanisms or require deep integration into core storage paths, which both increases complexity and reduces portability.

## Benefits of KV Cache offloading

KV Cache offloading avoids expensive KV Cache recomputation, resulting in faster response times and a better user experience. In the end, providers benefit from higher throughput and lower cost per token, making their inference services more scalable and efficient.

## When to offload KV Cache for reuse

Offloading KV cache to CPU or storage is most effective when KV Cache exceeds GPU memory and cache reuse outweighs the overhead of transferring data. It is especially valuable in long-context, high-concurrency, or resource-constrained inference environments such as:

* **Long sessions and multi-turn conversations:** Offloading preserves large prompt prefixes, avoids recomputation, and improves first-token latency and throughput.
* **High concurrency (future):** Idle or partial conversations can be moved out of GPU memory, allowing active requests to proceed without hitting memory limits.
* **Shared or repeated content (future):** Reuse across users or sessions (for example, system prompts and templates) increases cache hits, especially with remote or cross-instance sharing.
* **Memory- or cost-constrained deployments:** Offloading to RAM or SSD reduces GPU demand, allowing longer prompts or more users without adding hardware.
