---
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
title: VLM Performance Optimization
subtitle: Speed up image and video requests with routing, decoding, caching, and encoder disaggregation
---

Once a first request works (see [Image Understanding](overview.md) or
[Video Understanding](../video-understanding/overview.md)), these features cut
the time a vision language model (VLM) spends on the steps around the model
itself: fetching media, finding the right worker, running the vision encoder,
and prefill. They work together or on their own. None of them apply to audio yet.

## Where the Time Goes

A VLM request takes six steps. Each feature below speeds up one of them.

![The six steps of a VLM request: the frontend receives the request and the router picks a worker; the worker loads the image, runs the vision encoder, prefills, and decodes. Multimodal KV routing helps when picking a worker, frontend media decoding and the shared image cache help when loading the image, and the embedding cache and encoder disaggregation help when encoding.](./_assets/vlm-request-path.svg)

## When Each Feature Helps

| Feature | Use it when | Speeds up | Applies to |
| --- | --- | --- | --- |
| [Multimodal KV routing](#multimodal-kv-routing) | The same images or clips repeat across requests and you run several workers | Step 2 and prefill: sends repeats to the worker that already holds their KV cache | Image and video |
| [Frontend media decoding](#frontend-media-decoding) | Fetching and decoding media takes a large share of request latency or worker CPU time | Step 3: moves decoding to the Rust frontend | Image and VP8/VP9 video |
| [Shared image cache](#shared-image-cache) | Several workers download the same image URLs | Step 3: reuses downloaded bytes across workers | Image URLs |
| [Embedding cache](#embedding-cache) | The same images or clips repeat and the vision encoder is busy | Step 4: skips re-encoding repeated content | Image and video |
| [Encoder disaggregation](#encoder-disaggregation) | Vision encoding is the bottleneck, or encoders need different hardware than the LLM | Step 4: scales encoders separately from LLM workers | Image (vLLM video stays on the PD path) |

If your workload is all unique media, the caches and routing give little benefit.
Start with frontend media decoding or encoder disaggregation instead.

For video-specific routing, see
[Exact Video KV Routing](../video-understanding/exact-video-kv-routing.md).
To run your own vision tower, see [Custom Vision Encoders](custom-vision-encoders.md).

## Feature Details

<AccordionGroup>
  <Accordion id="multimodal-kv-routing" title="Multimodal KV routing">
  **Overview**

  Multimodal KV routing extends Dynamo's KV-aware router to account for image or video content when calculating cache overlap. The frontend assigns each supported media object a stable hash and includes its identity in the routing view of the prompt.

  When media appears again, its identity contributes to each worker's KV overlap. The router balances that cache credit against projected prefill and decode load, so the worker with the largest overlap does not necessarily win when it is busy. Cache-aware placement increases prefix reuse without abandoning load balancing.

  > [!IMPORTANT]
  > The KV cache stores attention key/value state so a worker can skip repeated prefill work. The embedding cache stores vision encoder outputs so the encoder can skip repeated image processing. You can use both features together. See [Embedding Cache](#embedding-cache).

  **When to Use**

  Use multimodal KV routing when:

  - Multiple backend workers serve multimodal requests.
  - Images or videos repeat across requests, such as product photos, shared reference images, or several questions about the same video.
  - You want to maximize KV cache reuse for multimodal content.

  Single-worker deployments do not need routing, and workloads with entirely unique media receive little media-specific cache benefit.

  > [!NOTE]
  > This section covers image routing. For video, see [Exact Video KV Routing](../video-understanding/exact-video-kv-routing.md).

  **How It Works**

  The routing flow in general has three steps:

  1. The frontend computes a stable identity for each supported media object.
  2. The frontend represents the media in a routing-only token view that matches the backend's cache identity and token layout.
  3. The KV router includes that overlap in its combined cache-and-load score, selects the lowest-cost eligible worker, and forwards the same media identity.

  **Image Identity**

  By default, an HTTP image's identity hashes the exact URL bytes, including its
  query string. Reusing the identical URL produces the same identity, but two URLs
  for the same image bytes do not. Enable `--frontend-decoding` when you need
  content-stable identity across URLs; the frontend then hashes decoded image
  content. Data URLs follow the same rule: the default path hashes the full data
  URI string, while frontend decoding hashes the decoded bytes.


  <Tabs>
    <Tab title="vLLM" language="vllm">
      vLLM provides two routing paths. Use the default Rust frontend for supported model families when you want minimal frontend processing. Use the Python chat processor when you need vLLM's broader model support or want the frontend to preprocess media and transfer the processed inputs to workers.

      **Default Rust frontend**

      For images, the frontend calculates only the identity and routing token layout unless frontend decoding is enabled. The selected worker still runs the model's multimodal processor. This path depends on the model being registered with Dynamo's Rust processor registry.

      **Alternative: Python chat processor**

      With `--dyn-chat-processor vllm`, the frontend runs vLLM's full multimodal processor. It derives routing positions from vLLM's processed image or video features without requiring a Dynamo Rust processor specification and can transfer processed inputs through shared memory or NIXL. This shifts preprocessing and transfer work to the frontend.

    </Tab>
    <Tab title="SGLang" language="sglang">
      The frontend hashes each image and converts that hash into the same `pad_value` token that SGLang RadixAttention uses for its prefix-cache key. The matching token view lets the router measure image overlap before selecting a worker.

      Dynamo's SGLang image includes the required hash-forwarding support. For video, see [Exact Video KV Routing](../video-understanding/exact-video-kv-routing.md).
    </Tab>
    <Tab title="TensorRT-LLM" language="trtllm">
      The frontend hashes each image, represents that identity in its routing token view, and forwards the hash as `multi_modal_uuids`. TensorRT-LLM workers publish matching KV events so the router can identify cached image blocks.

      Exact video routing is not available on this path; video requests use text-prefix routing.

    </Tab>
  </Tabs>

  **Launch**

  <Tabs>
    <Tab title="vLLM" language="vllm">
      Use the Python chat processor when you need vLLM's model-native multimodal processor or want to transfer processed multimodal inputs from the frontend. Otherwise, use the default Rust frontend.

      **Default Rust frontend**

      ```bash
      cd $DYNAMO_HOME
      bash examples/backends/vllm/launch/agg_multimodal_router.sh
      ```

      **Alternative: Python chat processor**

      ```bash
      cd $DYNAMO_HOME
      bash examples/backends/vllm/launch/agg_multimodal_router_chat_processor.sh
      ```

      See [vLLM Multimodal](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md#multimodal-kv-routing) for model support, hashing behavior, transfer modes, and configuration.
    </Tab>
    <Tab title="SGLang" language="sglang">
      ```bash
      cd $DYNAMO_HOME
      bash examples/backends/sglang/launch/agg_multimodal_router.sh
      ```

      The launcher enables `--frontend-decoding`, configures KV events, and sets matching frontend and worker block sizes.

      See [SGLang Multimodal](../../developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md#multimodal-kv-routing) for prerequisites, configuration, fallback behavior, and verification.
    </Tab>
    <Tab title="TensorRT-LLM" language="trtllm">
      ```bash
      cd $DYNAMO_HOME
      bash examples/backends/trtllm/launch/agg_multimodal_router.sh
      ```

      The launcher enables multimodal serving, KV event publishing, block reuse, and KV-aware routing.

      See [TensorRT-LLM Multimodal](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/multimodal.md#multimodal-kv-routing) for worker requirements, supported models, and limitations.
    </Tab>
  </Tabs>


  **Send an Image Request**

  Send the same image twice. The first request populates the selected worker's KV
  cache, and the second should be routed back to it while workers are idle:

  ```bash
  for i in 1 2; do
    curl -s http://localhost:8000/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen3-VL-2B-Instruct",
        "messages": [{
          "role": "user",
          "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"}}
          ]
        }],
        "max_tokens": 32
      }' > /dev/null
  done
  ```

  Set `"model"` to the model your launcher serves. With
  `DYN_LOG=info,dynamo_kv_router::scheduling=debug,dynamo_llm::kv_router=debug`, the frontend logs a
  `[ROUTING] Best: worker_<id> … with <n>/<total> blocks overlap` line for each
  request; the second request should report more overlap than the first.

  **Support Matrix**

  | Backend | Routing Path | Status | Notes |
  |---------|--------------|--------|-------|
  | [vLLM](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md#multimodal-kv-routing) | Rust frontend (default) | <Badge intent="success" minimal>Yes</Badge> | Exact image routing supports Qwen2-VL, Qwen2.5-VL, Qwen3-VL, LLaVA 1.5, LLaVA-NeXT, Llama 4, Kimi K2.5/K2.6, Qwen3.5, Qwen3.6, and Nemotron 3 Nano Omni. Exact video routing requires frontend decoding and supports Qwen3-VL, Qwen3.5, and Nemotron 3 Nano Omni. Other media layouts use text-prefix routing. |
  | [vLLM](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md#multimodal-kv-routing) | Python chat processor | <Badge intent="success" minimal>Yes</Badge> | Uses vLLM's multimodal processor to derive image and video feature hashes and positions for models supported by vLLM. |
  | [SGLang](../../developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md#multimodal-kv-routing) | Rust frontend (default) | <Badge intent="success" minimal>Yes</Badge> | Exact image routing uses SGLang's hash-forwarding support. Exact video routing supports aggregated Qwen3-VL and Qwen3.5 workers on CUDA with frontend decoding; other video requests use text-prefix routing. |
  | [TensorRT-LLM](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/multimodal.md#multimodal-kv-routing) | Rust frontend (default) | <Badge intent="success" minimal>Images</Badge> | Exact image routing supports the Qwen2-VL family and Kimi K2.5/K2.6. Video requests and other models use text-prefix routing. |

  Nemotron 3 Nano Omni exact image and video routing currently requires the vLLM backend. Other backends fall back to text-prefix routing for this model.
  </Accordion>

  <Accordion id="frontend-media-decoding" title="Frontend media decoding">
  Frontend media decoding moves media fetching and decoding from the inference
  backend to the NVIDIA Dynamo Rust frontend. The frontend decodes supported
  images and videos on a CPU worker pool and transfers the decoded pixel buffers
  to the backend through NIXL.

  The backend still runs its model-specific multimodal processor and vision
  encoder. This feature changes where media input is decoded; it does not skip
  vision encoding.

  **Support Matrix**

  | Input modality | vLLM | SGLang | TensorRT-LLM |
  | --- | --- | --- | --- |
  | Image | Agg | Agg | Agg |
  | Video | Agg (VP8/VP9) | Agg (VP8/VP9) | Not supported |
  | Audio | Not supported | Not supported | Not supported |

  `Agg` refers to an aggregated worker. The entries in this matrix represent the
  supported topologies for frontend decoding.

  SGLang frontend video decoding is available only on CUDA. SGLang XPU images do
  not include the in-tree FFmpeg decoder. Image frontend decoding is not subject
  to that codec restriction.

  This matrix describes frontend media decoding, not the overall multimodal
  support of each backend. A backend can support video or audio by decoding it on
  the worker even when the frontend decoding path does not support that modality.

  **When to Use**

  Use frontend media decoding when media preprocessing consumes a significant
  part of request latency or backend CPU time. It is most useful for workloads
  with:

  - Concurrent requests containing HTTP, HTTPS, or base64-encoded media
  - Multiple images or supported videos in one request
  - Backend workers whose request path is constrained by media fetching or
    decompression

  Frontend media decoding can also be combined with the [embedding
  cache](#embedding-cache). Frontend decoding reduces media input processing
  work, while the embedding cache can skip vision encoding for repeated images.

  **How It Works**

  For each request, the frontend:

  1. Fetches the media URL or decodes the base64 data URL.
  2. Decodes supported images or VP8/VP9 videos on a CPU worker pool.
  3. Registers the decoded pixel buffer with NIXL.
  4. Sends the buffer descriptor to the selected backend worker.

  The backend reads the decoded pixels through NIXL, then continues with its
  normal multimodal processor and vision encoder.

  **Enable Frontend Media Decoding**

  Add `--frontend-decoding` to the backend worker command. Do not add the flag to
  `dynamo.frontend`; the backend advertises the decoder configuration when it
  registers the model.

  | Backend | Worker flags |
  | --- | --- |
  | [vLLM](../../developer-guide/knowledge-base/modular-components/backends/vllm/reference-guide.md) | `--enable-multimodal --frontend-decoding` |
  | [SGLang](../../developer-guide/knowledge-base/modular-components/backends/sglang/reference-guide.md) | `--frontend-decoding` |
  | [TensorRT-LLM](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/reference-guide.md) | `--enable-multimodal --frontend-decoding` |

  **JPEG Decoder**

  The Rust frontend uses libjpeg-turbo for JPEG inputs by default. To use
  `image::ImageReader` instead, set `DYN_MM_ENABLE_LIBJPEG=0` on the frontend
  process. Dynamo backend runtime images include `libturbojpeg`; custom images
  must provide `libturbojpeg.so.0`. If the library is unavailable, Dynamo logs a
  one-time warning and falls back to `image::ImageReader`.

  **NIXL Progress Thread**

  For each worker set with frontend decoding, the frontend creates up to two NIXL
  agents: one for chat requests and one for completions requests. The progress
  thread of each agent sleeps for at most `DYN_MM_NIXL_PROGRESS_DELAY_US`
  microseconds (default `1000`, range `0` to `1000000`), rounded up to whole
  milliseconds, and wakes early on UCX events.

  When the frontend and the backend worker cannot share memory or use RDMA (for
  example, the frontend runs in its own pod with no RDMA device), UCX falls back
  to TCP. Over TCP, some UCX events do not wake the thread, so a read can wait up
  to the full delay. In a same-host TCP test with UCX 1.22, the default delay
  added about 0.7 ms to each 3 MiB read and 3 ms to each 12 MiB read. Larger
  values make each wait longer. Set `DYN_MM_NIXL_PROGRESS_DELAY_US=0` on the
  frontend process to trade one CPU core for each NIXL agent for that latency.

  **Requirements and Limitations**

  - The published `nvcr.io/nvidia/ai-dynamo/dynamo-frontend:1.4.0` image installs
    NIXL wheels but does not expose the native NIXL and UCX libraries and plugins
    required by the Rust frontend. It cannot run frontend media decoding as
    shipped. Run the frontend in a backend runtime image or environment where
    NIXL and UCX are configured and available.
  </Accordion>

  <Accordion id="shared-image-cache" title="Shared image cache">
  The optional shared cache stores encoded bytes from HTTP and HTTPS image URLs
  before image decoding. Workers can reuse those bytes across processes while
  keeping their local decoded-image caches independent. Data URLs,
  frontend-decoded NIXL inputs, video, and audio do not use this cache.

  **Backend Support**

  | Backend | Mode | Shared encoded-image cache |
  | --- | --- | --- |
  | vLLM | Aggregated and disaggregated paths that load URLs through Dynamo `ImageLoader` | Supported |
  | TensorRT-LLM | Aggregated and disaggregated paths that load URLs through Dynamo `ImageLoader` | Supported |
  | SGLang | Standard URL loading | Not supported. Dynamo passes URL strings directly to SGLang. |
  | SGLang | `--frontend-decoding` | Only URL items that still reach Dynamo `ImageLoader` can use the cache. NIXL-decoded items bypass it. |

  Dynamo does not provision or discover a cache service. Deploy Redis Cluster or
  Dragonfly separately and provide a cluster endpoint that every participating
  URL-loading worker can reach. The connection URL carries the endpoint,
  authentication, and TLS choice. See [Cache Service Deployment](#cache-service-deployment)
  for single-node and multi-node layouts.

  Set the following environment variables on every participating worker:

  | Environment variable | Default | Description |
  | --- | --- | --- |
  | `DYN_MM_SHARED_IMAGE_CACHE_ENABLED` | `0` | Set to `1` to enable the shared cache. |
  | `DYN_MM_SHARED_IMAGE_CACHE_URL` | None | Redis Cluster or Dragonfly connection URL. Required when the cache is enabled. |
  | `DYN_MM_SHARED_IMAGE_CACHE_TTL_SECS` | `3600` | Positive cache-entry lifetime in seconds. |
  | `DYN_MM_SHARED_IMAGE_CACHE_CONNECT_TIMEOUT_SECS` | `0.1` | Positive connection timeout in seconds. |
  | `DYN_MM_SHARED_IMAGE_CACHE_IO_TIMEOUT_SECS` | `2.0` | Positive Redis operation timeout in seconds. |
  | `DYN_MM_MAX_FILE_SIZE_MB` | `64` | Maximum encoded image size downloaded from the origin, in MiB. The limit applies whether or not the shared cache is enabled, so oversized images are never stored in it. |
  | `DYN_MM_IMAGE_CACHE_SESSION_SCOPED` | `0` | Set to `1` to partition Dynamo image and image-embedding caches by session affinity. |

  Store the full connection URL in a Kubernetes Secret and inject it into each
  worker rather than placing credentials in a manifest:

  ```yaml
  env:
    - name: DYN_MM_SHARED_IMAGE_CACHE_ENABLED
      value: "1"
    - name: DYN_MM_SHARED_IMAGE_CACHE_URL
      valueFrom:
        secretKeyRef:
          name: multimodal-image-cache
          key: connection-url
    - name: DYN_MM_IMAGE_CACHE_SESSION_SCOPED
      value: "1"
  ```

  > [!WARNING]
  > Use a `rediss://` URL unless Redis traffic is protected by equivalent
  > transport encryption, such as an encrypted service mesh. A `redis://` URL
  > sends cache data and any URL credentials without TLS and should be used only
  > on an appropriately secured in-cluster network.

  Cache reads and fills run on the request path. A miss waits for Redis `SET` so
  the fill completes before the request returns. Redis errors fail open, and each
  cache operation is attempted once. While a cache node is unreachable, an image
  load pays one failed `GET` and one failed `SET`: each is bounded by
  `DYN_MM_SHARED_IMAGE_CACHE_CONNECT_TIMEOUT_SECS` when a new connection is
  refused or times out, and by `DYN_MM_SHARED_IMAGE_CACHE_IO_TIMEOUT_SECS` when
  an open connection stops responding.
  Warnings are emitted initially and at most once per minute; individual
  operation failures remain available at debug level. Dynamo does not currently
  use a bounded asynchronous write queue.

  **Cache Service Deployment**

  Dynamo uses the Redis Cluster protocol for both backends. The client uses
  `DYN_MM_SHARED_IMAGE_CACHE_URL` only to discover the cluster: it reads the slot
  map from that endpoint and then connects to each node at the address the node
  announces. When a node stops responding, the client rediscovers through the same
  URL, so point it at an address that survives cache restarts, such as a
  Kubernetes Service that selects every cache node, not at an individual pod.
  After a restart, the nodes announce their new addresses and workers pick them
  up without restarting. Dynamo requires redis-py 6.2 or later.

  **Single-node Dragonfly.** Run Dragonfly with `--cluster_mode=emulated` so that
  it answers Redis Cluster commands, and keep its default announced address (the
  pod IP). Do not set `--cluster_announce_ip` to the host in
  `DYN_MM_SHARED_IMAGE_CACHE_URL`: with redis-py 6.2 through 7.1, a node that
  announces the same host and port as the URL removes the URL from the client's
  rediscovery list after one failed operation.

  **Redis Cluster.** Run the nodes as a StatefulSet with a headless Service for
  per-pod DNS names, plus a regular Service that selects every node for the URL.
  Configure each node with:

  - `cluster-announce-ip` set to the pod IP, or `cluster-announce-hostname` set to
    the pod's DNS name together with `cluster-preferred-endpoint-type hostname`
    (Redis 7.0 or later).
  - `cluster-require-full-coverage no`, so the remaining nodes keep serving while
    one node is down. Images that hash to the unavailable node fall back to the
    origin.
  - A `nodes.conf` that persists across pod restarts, so a restarted node keeps its
    identity and slots. A rolling restart then keeps the cluster available. If
    every node restarts at once, all peer addresses change and the nodes need
    their peers' new addresses before they can re-form the cluster.

  **Multi-node Dragonfly.** With `--cluster_mode=yes`, Dragonfly nodes do not
  discover each other. A cluster orchestrator must push the slot map with
  `DFLYCLUSTER CONFIG` and push it again whenever a node restarts, because the
  map is kept in memory. Point the URL at a Service that selects every node, as
  for Redis Cluster.

  **Session Scoping**

  Enable `DYN_MM_IMAGE_CACHE_SESSION_SCOPED=1` when the same URL can resolve to
  different content for different sessions. Dynamo derives the
  scope from these headers, in precedence order:

  1. `x-dynamo-session-id`
  2. Recognized agent headers: Claude Code `x-claude-code-session-id` (or
     `x-claude-code-agent-id` for a child agent), Codex `thread-id`, then OpenCode
     `x-session-id`

  Use `x-dynamo-session-id` to provide an explicit Dynamo session identity.
  Dynamo normalizes it and the recognized agent headers into both an
  `AgentContext` and the session affinity used for routing and image-cache
  scoping.

  > [!WARNING]
  > Session scoping partitions cache entries; it is not an authentication
  > boundary. Unless a trusted proxy sets or overwrites `x-dynamo-session-id`,
  > the caller controls the scope. Configure the proxy to derive this header
  > from the authenticated identity on every request, and treat scope identifiers
  > as secrets. A caller that learns another identifier can reuse cached content
  > stored under that scope for the same URL.

  When session scoping is enabled, a missing or blank scope bypasses the local
  decoded-image cache, in-flight request deduplication, the shared encoded-image
  cache, and Dynamo-owned image embedding caches. A valid scope partitions all of
  those caches. The scope does not partition backend KV cache keys; affinity can
  route a session back to the same worker, but KV identity remains content-based.
  </Accordion>

  <Accordion id="embedding-cache" title="Embedding cache">
  **Overview**

  The embedding cache (or encoder cache) is a CPU-side LRU cache that stores vision encoder outputs. When the same multimodal content, such as an image or video, appears in multiple requests, the cached embedding is reused instead of running the vision encoder again. This reduces GPU load on the encoder and lowers latency for repeated content.

  > [!IMPORTANT]
  > Embedding cache is separate from the KV cache, which reuses attention key/value state after prefill to skip prefill and go straight to decode. For KV cache reuse and routing, see [Multimodal KV Routing](#multimodal-kv-routing).

  **When to Use**

  Use the embedding cache when your workload includes repeated multimodal content across requests. Common scenarios:

  - Product catalog queries or multi-turn chat where users ask about the same image(s).
  - Document processing pipelines that reference shared diagrams or figures
  - Video QA or benchmark workloads that repeatedly reference the same clips

  If your workload consists entirely of unique multimodal content, the cache provides no benefit.



  **Configuration**

  <Tabs>
    <Tab title="vLLM" language="vllm">
      Launch a disaggregated multimodal deployment with the embedding cache enabled:

      ```bash
      cd $DYNAMO_HOME/examples/backends/vllm
      bash launch/disagg_multimodal_e_pd.sh --multimodal-embedding-cache-capacity-gb 10
      ```
    </Tab>
    <Tab title="TensorRT-LLM" language="trtllm">
      Launch a disaggregated multimodal deployment with the embedding cache enabled:

      ```bash
      cd $DYNAMO_HOME/examples/backends/trtllm
      ./launch/disagg_e_pd.sh --multimodal-embedding-cache-capacity-gb 10
      ```
    </Tab>
    <Tab title="SGLang" language="sglang">
      Launch a disaggregated multimodal E/PD deployment with the embedding cache enabled:

      ```bash
      cd $DYNAMO_HOME/examples/backends/sglang
      bash launch/multimodal_epd.sh --multimodal-embedding-cache-capacity-gb 10
      ```

      For SGLang encode/prefill/decode deployments that use `--router-mode device-aware-weighted`, set both `--multimodal-embedding-cache-capacity-gb` and `--multimodal-embedding-cache-publisher` on the encode worker. The publisher sends cache-key additions and removals over the configured event plane so the router can track which workers hold each requested embedding.

      When one or more workers hold every distinct embedding-cache key in a request, the router bypasses the CPU-to-non-CPU ratio and selects the least-loaded full-hit worker. Partial cache hits continue through the normal weighted group selection. The publisher is disabled by default and is not needed for round-robin routing or aggregated deployments.
    </Tab>
  </Tabs>

  Set `--multimodal-embedding-cache-capacity-gb` based on your expected working set of unique multimodal content. A larger cache holds more embeddings but consumes more host memory.

  > [!NOTE]
  > If one image URL can return different content for different sessions, enable
  > session-scoped caching as described in [Shared image cache](#shared-image-cache).
  > Dynamo then partitions its image embedding caches by affinity scope and
  > bypasses them when a request has no valid scope.

  See the backend-specific documentation ([vLLM](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md#embedding-cache), [TRT-LLM](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/multimodal.md#embedding-cache)) for more details.

  **How It Works**

  In vLLM/TRT-LLM disaggregated flows, the prefill worker owns the CPU-side LRU cache. On a hit, the encode worker is skipped entirely. On a miss, the encode worker produces the embedding, transfers it via NIXL, and the prefill worker saves it to the cache.

  In SGLang E/PD, the encode worker owns the cache and skips re-encoding on cache hits before forwarding the cached image or video embeddings downstream.

  ```mermaid
  flowchart LR
      req[Request] --> check{CPU cache hit?}
      check -. hit .-> use[Use cached embedding]
      check -- miss --> E[Encode Worker]
      E -- embeddings via NIXL --> save[Save to cache]
      save --> engine[Inference Engine]
      use --> engine
  ```

  **Support Matrix**

  | Backend | Aggregated | Disaggregated (E/PD) | Notes |
  |---------|------------|----------------------|-------|
  | **vLLM** | <Badge intent="success" minimal>Yes</Badge> | <Badge intent="success" minimal>Yes</Badge> | Aggregated uses vLLM-native `ec_both`; disaggregated uses Dynamo `EmbeddingCacheManager` |
  | **TensorRT-LLM** | — | <Badge intent="success" minimal>Yes</Badge> | Dynamo `MultimodalEmbeddingCacheManager` in PD worker |
  | **SGLang** | — | <Badge intent="success" minimal>Yes</Badge> | Dynamo `MultimodalEmbeddingCacheManager` in the encode worker |

  This support requires vLLM `0.17.0` or newer.
  </Accordion>

  <Accordion id="encoder-disaggregation" title="Encoder disaggregation">
  **Overview**

  Encoder disaggregation separates the vision encoder from the prefill/decode pipeline into its own worker. Instead of running image encoding inline, a dedicated encode worker handles media processing and transfers the resulting embeddings to downstream workers via NIXL (RDMA).

  This enables:

  - Independent scaling of encode workers based on vision workload
  - Reduced GPU memory pressure on prefill/decode workers
  - Better GPU utilization by matching worker counts to actual bottlenecks

  **When to Use**

  Use encoder disaggregation when:

  - Vision encoding is a bottleneck and you need to scale encoders independently of LLM workers
  - You want to run the vision encoder on different hardware (e.g., smaller GPUs for encoding, larger GPUs for LLM inference)
  - Your deployment handles high volumes of multimodal requests and encoding throughput is limiting

  For simple deployments or development and testing, the aggregated pattern (EPD, one worker) is easier to set up. See [Deployment Notation](#deployment-notation) for the layout names used in this section.

  **Deployment Patterns**

  <CardGroup cols={2}>
    <Card title="E/PD">
      **Separate encoder, combined prefill and decode**

      <Badge intent="info" minimal>Frontend</Badge> → <Badge intent="info" minimal>Processor</Badge> → <Badge intent="info" minimal>Encode Worker</Badge> → <Badge intent="info" minimal>PD Worker</Badge> → <Badge intent="info" minimal>Response</Badge>

      The encode worker runs the vision model and transfers embeddings over NIXL to the combined prefill and decode worker.
    </Card>
    <Card title="E/P/D">
      **Separate encoder, prefill, and decode**

      <Badge intent="info" minimal>Frontend</Badge> → <Badge intent="info" minimal>Processor</Badge> → <Badge intent="info" minimal>Encode Worker</Badge> → <Badge intent="info" minimal>Prefill Worker</Badge> → <Badge intent="info" minimal>Decode Worker</Badge> → <Badge intent="info" minimal>Response</Badge>

      The encode worker transfers embeddings over NIXL to the prefill worker. The prefill worker then transfers the KV cache to the decode worker.
    </Card>
  </CardGroup>

  **Launching**

  <Tabs>
    <Tab title="vLLM" language="vllm">
      ```bash
      cd $DYNAMO_HOME/examples/backends/vllm

      # E/PD
      bash launch/disagg_multimodal_e_pd.sh --model "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"

      # E/P/D
      bash launch/disagg_multimodal_epd.sh --model "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8"
      ```
    </Tab>
    <Tab title="SGLang" language="sglang">
      ```bash
      cd $DYNAMO_HOME/examples/backends/sglang

      # E/PD
      ./launch/multimodal_epd.sh

      # E/P/D
      ./launch/multimodal_disagg.sh
      ```
    </Tab>
    <Tab title="TensorRT-LLM" language="trtllm">
      ```bash
      cd $DYNAMO_HOME/examples/backends/trtllm

      # E/PD
      bash launch/disagg_e_pd.sh

      # E/P/D
      ./launch/epd_multimodal_image_and_embeddings.sh
      ```
    </Tab>
  </Tabs>

  See the backend-specific documentation ([vLLM](../../developer-guide/knowledge-base/modular-components/backends/vllm/multimodal.md), [TRT-LLM](../../developer-guide/knowledge-base/modular-components/backends/tensorrt-llm/multimodal.md), [SGLang](../../developer-guide/knowledge-base/modular-components/backends/sglang/multimodal.md)) for full configuration details and component flags.

  **Support Matrix**

  | Backend | E/PD | E/P/D | Notes |
  |---------|------|-------|-------|
  | **vLLM** | <Badge intent="success" minimal>Yes</Badge> | <Badge intent="success" minimal>Yes</Badge> | Separate encode worker currently handles `image_url` inputs; `video_url` inputs stay on the prefill/PD path |
  | **TensorRT-LLM** | <Badge intent="success" minimal>Yes</Badge> | <Badge intent="success" minimal>Yes</Badge> | Supports image URLs (via `MultimodalEncoder`) and pre-computed embeddings (via NIXL) |
  | **SGLang** | <Badge intent="success" minimal>Yes</Badge> | <Badge intent="success" minimal>Yes</Badge> | NIXL for embeddings; bootstrap mechanism for P/D KV transfer |
  </Accordion>

</AccordionGroup>

## Deployment Notation

Layouts are named by which stages share a process. A slash marks a process
boundary:

| Notation | Meaning |
| --- | --- |
| EPD | One worker encodes, prefills, and decodes (aggregated) |
| EP/D | The prefill worker also encodes; decode runs separately |
| E/PD | A separate encode worker; one worker prefills and decodes |
| E/P/D | Separate encode, prefill, and decode workers |

## Support Matrix

| Feature | vLLM | SGLang | TensorRT-LLM |
| --- | --- | --- | --- |
| [Multimodal KV routing](#multimodal-kv-routing) | Image and video | Image, and video on aggregated CUDA workers | Image only |
| [Frontend media decoding](#frontend-media-decoding) | Image and VP8/VP9 video, aggregated | Image, and VP8/VP9 video on CUDA, aggregated | Image, aggregated |
| [Shared image cache](#shared-image-cache) | Image URLs loaded through Dynamo | Not supported for standard URL loading | Image URLs loaded through Dynamo |
| [Embedding cache](#embedding-cache) | Aggregated and E/PD | E/PD | E/PD |
| [Encoder disaggregation](#encoder-disaggregation) | E/PD and E/P/D (images only; video stays on the PD path) | E/PD and E/P/D | E/PD and E/P/D (images and pre-computed embeddings) |

Each accordion above lists the supported models, flags, and limitations for its
backends.
