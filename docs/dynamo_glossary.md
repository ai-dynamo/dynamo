# NVIDIA Dynamo Glossary

## B
**Block** - A fixed-size chunk of tokens (typically 16 or 64 tokens) used for efficient KV cache management and memory allocation, serving as the fundamental unit for techniques like PagedAttention.

## C
**Component** - The fundamental deployable unit in Dynamo. A discoverable service entity that can host multiple endpoints and typically maps to a Docker container (such as VllmWorker, Router, Processor).

**Conditional Disaggregation** - Dynamo's intelligent decision-making process within disaggregated serving that determines whether a request is processed locally or sent to a remote prefill engine based on prefill length and queue status.

## D
**Decode Phase** - The second phase of LLM inference that generates output tokens one at a time.

**depends()** - A Dynamo function that creates dependencies between services, enabling automatic client generation and service discovery.

**Disaggregated Serving** - Dynamo's core architecture that separates prefill and decode phases into specialized engines to maximize GPU throughput and improve performance.

**Distributed Runtime** - Dynamo's Rust-based core system that manages service discovery, communication, and component lifecycle across distributed clusters.

**Dynamo** - NVIDIA's high-performance distributed inference framework for Large Language Models (LLMs) and generative AI models, designed for multinode environments with disaggregated serving and cache-aware routing.

**Dynamo Artifact** - A packaged archive containing an inference graph and its dependencies, created using `dynamo build`. It's the containerized, deployable version of a Graph.

**Dynamo Cloud** - A Kubernetes platform providing managed deployment experience for Dynamo inference graphs.

## E
**@endpoint** - A Python decorator used to define service endpoints within a Dynamo component.

**Endpoint** - A specific network-accessible API within a Dynamo component, such as `generate` or `load_metrics`.

## F
**Frontend** - Dynamo's API server component that receives user requests and provides OpenAI-compatible HTTP endpoints.

## G
**Graph** - A collection of interconnected Dynamo components that form a complete inference pipeline with request paths (single-in) and response paths (many-out for streaming). A graph can be packaged into a Dynamo Artifact for deployment.

## I
**Instance** - A running process with a unique `instance_id`. Multiple instances can serve the same namespace, component, and endpoint for load balancing

## K
**KV Block Manager (KVBM)** - Dynamo's scalable runtime component that handles memory allocation, management, and remote sharing of Key-Value blocks across heterogeneous and distributed environments.

**KV Cache** - Key-Value cache that stores computed attention states from previous tokens to avoid recomputation during inference.

**KV Router** - Dynamo's intelligent routing system that directs requests to workers with the highest cache overlap to maximize KV cache reuse. Determines routing based on KV cache hit rates and worker metrics.

**KVIndexer** - Dynamo component that maintains a global view of cached blocks across all workers using a prefix tree structure to calculate cache hit rates.

**KVPublisher** - Dynamo component that emits KV cache events (stored/removed) from individual workers to the global KVIndexer.

## N
**Namespace** - Dynamo's logical grouping mechanism for related components. Similar to directories in a file system, they prevent collisions between different deployments.

**NIXL (NVIDIA Inference tranXfer Library)** - High-performance data transfer library optimized for inference workloads, supporting direct GPU-to-GPU transfers and multiple memory hierarchies.

## P
**PagedAttention** - Memory management technique from vLLM that efficiently manages KV cache by chunking requests into blocks.

**Planner** - Dynamo component that performs dynamic resource scaling based on real-time demand signals and system metrics.

**Prefill Phase** - The first phase of LLM inference that processes the input prompt and generates KV cache.

**Prefix Caching** - Optimization technique that reuses previously computed KV cache for common prompt prefixes.

**Processor** - Dynamo component that handles request preprocessing, tokenization, and routing decisions.

## R
**RadixAttention** - Technique from SGLang that uses a prefix tree structure for efficient KV cache matching, insertion, and eviction.

**RDMA (Remote Direct Memory Access)** - Technology that allows direct memory access between distributed systems, used for efficient KV cache transfers.

## S
**@service** - Python decorator used to define a Dynamo service class.

**SGLang** - Fast LLM inference framework with native embedding support and RadixAttention.

## T
**Tensor Parallelism (TP)** - Model parallelism technique where model weights are distributed across multiple GPUs.

**TensorRT-LLM** - NVIDIA's optimized LLM inference engine with multinode MPI distributed support.

**Time-To-First-Token (TTFT)** - The latency from receiving a request to generating the first output token.

## V
**vLLM** - High-throughput LLM serving engine with Ray distributed support and PagedAttention.

## X
**xPyD (x Prefill y Decode)** - Dynamo notation describing disaggregated serving configurations where x prefill workers serve y decode workers. Dynamo supports runtime-reconfigurable xPyD.
