---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: "Benchmarking LLM Inference at Scale with AIPerf"
subtitle: "[Francesco Di Natale](https://developer.nvidia.com/blog/author/francescodinatale/), [Elias Bermudez](https://developer.nvidia.com/blog/author/dbermudez/), [Anthony Casagrande](https://developer.nvidia.com/blog/author/acasagrande/), [Matthew Kotila](https://developer.nvidia.com/blog/author/mkotila/), [Harshini Komali](https://developer.nvidia.com/blog/author/lkomali/) and [Ganesh Kudleppanavar](https://developer.nvidia.com/blog/author/ganeshku/) — September 2026"
description: "AIPerf, the successor to GenAI-Perf, is a multiprocess load generator for benchmarking LLM inference without the client becoming the bottleneck."
keywords: AIPerf, GenAI-Perf, LLM benchmarking, inference performance, TTFT, ITL, vLLM, Dynamo
last-updated: September 18, 2026
hide-page-actions: true
---

import { BlogStyles } from "@/components/BlogStyles";
import { BlogArticleMeta } from "@/components/BlogArticleMeta";

<BlogStyles />

<BlogArticleMeta
  authors={[
    { name: "Francesco Di Natale", href: "https://developer.nvidia.com/blog/author/francescodinatale/" },
    { name: "Elias Bermudez", href: "https://developer.nvidia.com/blog/author/dbermudez/" },
    { name: "Anthony Casagrande", href: "https://developer.nvidia.com/blog/author/acasagrande/" },
    { name: "Matthew Kotila", href: "https://developer.nvidia.com/blog/author/mkotila/" },
    { name: "Harshini Komali", href: "https://developer.nvidia.com/blog/author/lkomali/" },
    { name: "Ganesh Kudleppanavar", href: "https://developer.nvidia.com/blog/author/ganeshku/" },
  ]}
  category="Benchmarking"
  date="September 18, 2026"
  readTime="1 min read"
/>

At high concurrency, a benchmark client can saturate before the inference server does, and the numbers then describe the client rather than the deployment. In our blog post, [Benchmarking LLM Inference at Scale with AIPerf](https://developer.nvidia.com/blog/benchmarking-llm-inference-at-scale-with-aiperf/), we introduce [AIPerf](https://github.com/ai-dynamo/aiperf), a ground-up rewrite and the successor to GenAI-Perf. AIPerf spreads load generation across worker processes and hands results to separate record processors, so the client keeps pace with the server. It supports more than 15 endpoint types, synthetic workloads, public datasets such as ShareGPT, and trace replay from Mooncake, Baseten, and WEKA AgentX. The post walks through a first benchmark against vLLM, explains how to read time to first token, inter-token latency, request latency, and output throughput, and then shapes traffic with Poisson arrivals and variable input and output lengths.
