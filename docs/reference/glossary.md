---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: Glossary
---

## B
**Block** - A fixed-size chunk of tokens (typically 16 or 64 tokens) used for efficient KV cache management and memory allocation, serving as the fundamental unit for techniques like PagedAttention.

## C
**Component** - The fundamental deployable unit in Dynamo. A discoverable service entity that can host multiple endpoints and typically maps to a Docker container (such as VllmWorker, Router, Processor).

**Conditional Disaggregation** - Dynamo's intelligent decision-making process within disaggregated serving that determines whether a request is processed locally or sent to a remote prefill engine based on prefill length and queue status.

## D
**Decode Phase** - The second phase of LLM inference that generates output tokens one at a time.

**Disaggregated Serving** - Dynamo's core architecture that separates prefill and decode phases into specialized engines to maximize GPU throughput and improve performance.

**Discovery Plane** - The service discovery layer where components (frontend, router, and workers) register services, discover services, and watch for new service life-cycle events at runtime using Kubernetes or etcd backends.

**Distributed Runtime** - Dynamo's Rust-based core system that manages service discovery, communication, and component lifecycle across distributed clusters.

**Dynamo** - NVIDIA's high-performance distributed inference framework for Large Language Models (LLMs) and generative AI models, designed for multinode environments with disaggregated serving and cache-aware routing.

**Dynamo Kubernetes Platform** - A Kubernetes platform providing managed deployment experience for Dynamo inference graphs.

## E
**Endpoint** - A specific network-accessible API within a Dynamo component, such as `generate` or `load_metrics`.

**Event Plane** - The pub/sub layer for KV cache updates, worker metrics, and sequence tracking; it supports KV-aware routing and disaggregated serving architectures.

## F
**Frontend** - Dynamo's API server component that receives user requests and provides OpenAI-compatible HTTP endpoints.

## G
**Graph** - A collection of interconnected Dynamo components that form a complete inference pipeline with request paths (single-in) and response paths (many-out for streaming). A graph can be packaged into a Dynamo Artifact for deployment.

## I
**Instance** - A running process with a unique `instance_id`. Multiple instances can serve the same namespace, component, and endpoint for load balancing.

**Inter-Token Latency (ITL)** - The latency between consecutive output tokens during the decode phase; typically paired with TTFT to define performance SLAs.

## K
**KV Block Manager (KVBM)** - Dynamo's scalable runtime component that handles memory allocation, management, and remote sharing of Key-Value blocks across heterogeneous and distributed environments.

**KV Cache** - Key-Value cache that stores computed attention states from previous tokens to avoid recomputation during inference.

**KV Router** - Dynamo's intelligent routing system that directs requests to workers with the highest cache overlap to maximize KV cache reuse. Determines routing based on KV cache hit rates and worker metrics.

**KVIndexer** - Dynamo component that maintains a global view of cached blocks across all workers using a prefix tree structure to calculate cache hit rates.

**KVPublisher** - Dynamo component that emits KV cache events (stored/removed) from individual workers to the global KVIndexer.

## L

**LoRA (Low-Rank Adaptation)** - A fine-tuning technique for serving specialized model variants without duplicating full model weights. Dynamo supports dynamic loading and serving of LoRA adapters at runtime using worker APIs (for example, to load/unload,or for discovery in /v1/models).

## M
**Model Deployment Card (MDC)** - A configuration structure containing all information required for distributed model serving. When a worker loads a model, it creates an MDC containing references to components such as the tokenizer, templates, runtime config. Workers publish their MDC to make the model discoverable to frontends. Frontends use the MDC to configure request preprocessing (tokenization, prompt formatting).

## N
**Namespace** - Dynamo's logical grouping mechanism for related components. Similar to directories in a file system, they prevent collisions between different deployments.

**NIXL (NVIDIA Inference tranXfer Library)** - High-performance data transfer library optimized for inference workloads, supporting direct GPU-to-GPU transfers and multiple memory hierarchies.

## P
**PagedAttention** - Memory management technique from vLLM that efficiently manages KV cache by chunking requests into blocks.

**Planner** - Dynamo component that performs dynamic resource scaling based on real-time demand signals and system metrics.

**Prefill Phase** - The first phase of LLM inference that processes the input prompt and generates KV cache.

**Prefix Caching** - Optimization technique that reuses previously computed KV cache for common prompt prefixes.

**Processor** - Dynamo component that handles request preprocessing, tokenization, and routing decisions.

**Profiler** - Dynamo component that analyzes model performance to determine optimal engine configurations, including disagg/agg, parallelization mapping (TP, TEP, DEP), and other engine knobs (batch size, max num tokens), feeding the Planner for SLA-driven autoscaling.

## R
**RadixAttention** - Technique from SGLang that uses a prefix tree structure for efficient KV cache matching, insertion, and eviction.

**RDMA (Remote Direct Memory Access)** - Technology that allows direct memory access between distributed systems, used for efficient KV cache transfers.

**Request Plane** - The transport layer that transmits RPCs between components (frontend-to-worker or router-to-router) utilizing one of these protocols: TCP, HTTP, or NATS.

## S
**SGLang** - Fast LLM inference framework with native embedding support and RadixAttention.

**Speculative Decoding** - An optimization where a draft model proposes tokens for parallel verification by the main model; reduces latency (for example, vLLM with Eagle).

## T
**Tensor Parallelism (TP)** - Model parallelism technique where model weights are distributed across multiple GPUs.

**TensorRT-LLM** - NVIDIA's optimized LLM inference engine with multinode MPI distributed support.

**Time-To-First-Token (TTFT)** - The latency from receiving a request to generating the first output token.

## V
**vLLM** - High-throughput LLM serving engine with distributed tensor/pipeline parallelism and PagedAttention.

## W
**Wide Expert Parallelism (WideEP)** - Mixture-of-Experts deployment strategy that spreads experts across many GPUs (e.g., 64-way EP) so each GPU hosts only a few experts.

## X
**xPyD (x Prefill y Decode)** - Dynamo notation describing disaggregated serving configurations where x prefill workers serve y decode workers. Dynamo supports runtime-reconfigurable xPyD.
