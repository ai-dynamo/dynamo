# Real-Time ASR Service with NVIDIA Dynamo + WebRTC

> **Architecture Proposal Document**  
> A comprehensive guide to building a real-time audio transcription service using NVIDIA Dynamo for inference orchestration and WebRTC for real-time audio transport.

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Understanding NVIDIA Dynamo](#understanding-nvidia-dynamo)
   - [Core Problem Dynamo Solves](#core-problem-dynamo-solves)
   - [Architecture Overview](#architecture-overview)
   - [Key Components](#key-components)
   - [Disaggregated Serving](#disaggregated-serving)
   - [KV Block Manager (KVBM)](#kv-block-manager-kvbm)
3. [Proposed ASR + WebRTC Architecture](#proposed-asr--webrtc-architecture)
   - [High-Level Assessment](#high-level-assessment)
   - [System Architecture](#system-architecture)
   - [Component Deep Dive](#component-deep-dive)
   - [Data Flow](#data-flow)
4. [Why WebRTC Is Not Native to Dynamo](#why-webrtc-is-not-native-to-dynamo)
   - [Current Frontend Architecture](#current-frontend-architecture)
   - [WebRTC Protocol Complexity](#webrtc-protocol-complexity)
   - [Technical Barriers](#technical-barriers)
   - [Could It Be Done?](#could-it-be-done)
5. [Implementation Roadmap](#implementation-roadmap)
6. [ASR Model Options](#asr-model-options)
7. [Key Design Decisions](#key-design-decisions)
8. [Appendix](#appendix)

---

## Executive Summary

This document proposes an architecture for building a **real-time audio transcription service** that:

- **Leverages NVIDIA Dynamo** for distributed ASR model serving, worker orchestration, and autoscaling
- **Uses WebRTC** for low-latency, real-time audio transport from clients
- **Separates concerns** by using a dedicated WebRTC gateway that bridges to Dynamo's inference layer

**Key Insight**: Dynamo excels at inference orchestration but wasn't designed for real-time media protocols. The optimal architecture uses **specialized components for each layer** rather than forcing WebRTC into Dynamo's HTTP frontend.

---

## Understanding NVIDIA Dynamo

### Core Problem Dynamo Solves

Large language models are outgrowing single GPU memory and compute capacity. Tensor-parallelism spreads layers across multiple GPUs and servers, but creates orchestration challenges:

- How do you coordinate shards?
- How do you route requests efficiently?
- How do you share KV cache fast enough?

**NVIDIA Dynamo closes this orchestration gap.**

### Architecture Overview

```
                    ┌─────────────────────────────────────────────────────────────┐
                    │                     HTTP Frontend                           │
                    │  (OpenAI-compatible API, Pre-processor, Router)             │
                    └───────────────────────────┬─────────────────────────────────┘
                                                │
                    ┌───────────────────────────▼─────────────────────────────────┐
                    │                    KV-Aware Router                          │
                    │  (Routes to workers with best cached KV data + load balance)│
                    └───────────────────────────┬─────────────────────────────────┘
                                                │
          ┌─────────────────────────────────────┼─────────────────────────────────┐
          │                                     │                                 │
┌─────────▼─────────┐             ┌─────────────▼─────────────┐      ┌────────────▼────────┐
│  Prefill Worker   │             │     Decode Worker 1       │      │   Decode Worker N   │
│  (TensorRT-LLM,   │ ─── NIXL ──▶│     (vLLM, SGLang,        │      │   (Multi-GPU/Node)  │
│   vLLM, SGLang)   │             │      TensorRT-LLM)        │      │                     │
└───────────────────┘             └───────────────────────────┘      └─────────────────────┘
```

Dynamo is built in **Rust for performance** and **Python for extensibility**, and is fully open-source.

### Key Components

#### 1. Frontend

| Feature | Description |
|---------|-------------|
| **HTTP Server** | OpenAI-compatible REST API written in Rust |
| **Pre-processor** | Request validation, tokenization, prompt templating |
| **Auto-discovery** | Automatically discovers workers via etcd |
| **Entry Point** | `python -m dynamo.frontend --http-port 8000` |

#### 2. Backends/Workers

Dynamo is **inference engine agnostic**:

| Engine | Features |
|--------|----------|
| **vLLM** | Full disaggregation, KV-aware routing, KVBM, multimodal |
| **SGLang** | ZMQ-based communication, disaggregation, KV routing |
| **TensorRT-LLM** | TensorRT acceleration, disaggregated serving |

#### 3. KV-Aware Router

The router intelligently directs requests based on:

- **Global KV Cache State**: Tracks which KV cache blocks exist on each worker
- **Load Balancing**: Considers active prefill tokens and decode blocks
- **Cost Function**: `cost = overlap_score_weight × prefill_blocks + decode_blocks`

#### 4. SLA-Based Planner

Automatically scales workers to meet performance targets:

- Monitors TTFT (Time To First Token) and ITL (Inter-Token Latency)
- Uses predictive modeling (ARIMA, Prophet, Constant)
- Dynamically adjusts prefill/decode worker counts

### Disaggregated Serving

A key Dynamo innovation is **separating prefill and decode phases**:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        Disaggregated Request Flow                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   1. Worker receives request                                             │
│   2. Worker decides: prefill locally or remotely?                       │
│   3. If remote: push to PrefillQueue                                    │
│   4. PrefillWorker pulls from queue                                     │
│   5. PrefillWorker reads cached KVs via NIXL                           │
│   6. PrefillWorker computes prefill                                     │
│   7. PrefillWorker writes KVs back via NIXL                            │
│   8. Worker schedules decoding                                          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

**Why disaggregate?**

| Phase | Characteristic | Optimization |
|-------|---------------|--------------|
| **Prefill** | Compute-bound | Smaller tensor parallelism |
| **Decode** | Memory-bound | Larger tensor parallelism |

### KV Block Manager (KVBM)

The KVBM provides clean separation between:

| Layer | Responsibility |
|-------|---------------|
| **Runtime Connectors** | Translate runtime-specific operations to KVBM interface |
| **KVBM Core** | Block allocation, lifecycle management, eviction policies |
| **NIXL Layer** | Unified data/storage transactions (RDMA, NVLink, P2P GPU) |

---

## Proposed ASR + WebRTC Architecture

### High-Level Assessment

#### What Dynamo Brings ✅

- Distributed serving infrastructure (discovery, routing, scaling)
- Streaming response architecture
- Multi-GPU orchestration
- Python + Rust hybrid (extensible yet performant)
- SLA-based autoscaling

#### Gaps to Address ⚠️

- Dynamo is LLM-optimized (KV cache routing doesn't apply to ASR)
- No native WebRTC support (uses HTTP/TCP/NATS)
- Different computational pattern (ASR is sliding-window, not autoregressive)

### System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT LAYER                                        │
│                                                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐                           │
│  │ Web Browser  │  │ Mobile App   │  │ Desktop App  │                           │
│  │  (WebRTC)    │  │  (WebRTC)    │  │  (WebRTC)    │                           │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘                           │
│         │                 │                 │                                    │
└─────────┼─────────────────┼─────────────────┼────────────────────────────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            │ WebRTC (audio up, text down)
                            ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          WEBRTC GATEWAY LAYER                                    │
│                                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                      WebRTC Media Server                                 │    │
│  │  (LiveKit, Janus, mediasoup, or custom aiortc)                          │    │
│  │                                                                          │    │
│  │  • ICE/STUN/TURN negotiation                                            │    │
│  │  • Audio track extraction (Opus → PCM)                                  │    │
│  │  • Data channel for transcriptions                                      │    │
│  │  • Session management                                                    │    │
│  └──────────────────────────────┬──────────────────────────────────────────┘    │
│                                 │                                                │
└─────────────────────────────────┼────────────────────────────────────────────────┘
                                  │ Audio chunks via gRPC/NATS
                                  ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          DYNAMO LAYER                                            │
│                                                                                  │
│  ┌────────────────────┐                                                         │
│  │   Dynamo Frontend  │  ◄── Modified for audio streaming requests              │
│  │   (Audio Router)   │                                                         │
│  └─────────┬──────────┘                                                         │
│            │                                                                     │
│            │  Route based on: worker load, GPU memory, active streams           │
│            ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │                        ASR Worker Pool                                   │    │
│  │                                                                          │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │    │
│  │  │ ASR Worker 1│  │ ASR Worker 2│  │ ASR Worker 3│  │ ASR Worker N│     │    │
│  │  │ • Whisper   │  │ • Whisper   │  │ • Canary    │  │ • Parakeet  │     │    │
│  │  │ • Streaming │  │ • Streaming │  │ • Streaming │  │ • Streaming │     │    │
│  │  │ • VAD       │  │ • VAD       │  │ • VAD       │  │ • VAD       │     │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘     │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  ┌────────────────────┐                                                         │
│  │   ASR Planner      │  ◄── Scale based on concurrent streams + latency SLAs  │
│  └────────────────────┘                                                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Component Deep Dive

#### 1. WebRTC Gateway Options

| Option | Pros | Cons |
|--------|------|------|
| **LiveKit** | Full-featured, scalable, excellent SDK ecosystem | Another service to manage |
| **Janus Gateway** | Mature, flexible plugin architecture | C-based, steeper learning curve |
| **mediasoup** | Node.js, very performant SFU | Requires custom signaling |
| **aiortc (Python)** | Pure Python, easy Dynamo integration | Less battle-tested at scale |
| **Custom Rust** | Native Dynamo integration | Significant development effort |

**Recommendation**: Start with **LiveKit** for production or **aiortc** for prototyping.

#### 2. Audio Bridge

The bridge between WebRTC and Dynamo must:

```
┌─────────────────────────────────────────────────────────────┐
│                    Audio Bridge Responsibilities             │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. Receive audio packets from WebRTC (Opus codec)          │
│  2. Decode to PCM (16kHz, mono for most ASR models)         │
│  3. Buffer into chunks (e.g., 100ms windows)                │
│  4. Apply VAD (Voice Activity Detection) optionally         │
│  5. Forward to Dynamo worker via streaming RPC              │
│  6. Receive transcription stream back                       │
│  7. Send text via WebRTC data channel                       │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

#### 3. Dynamo ASR Worker

New backend type structure:

```
components/src/dynamo/asr/
├── __init__.py
├── __main__.py
├── main.py
├── streaming_handler.py      # Handles continuous audio stream
├── models/
│   ├── whisper_streaming.py  # Whisper with streaming wrapper
│   ├── canary_streaming.py   # NVIDIA Canary
│   └── parakeet_ctc.py       # NVIDIA Parakeet (CTC-based)
└── vad/
    └── silero_vad.py         # Voice activity detection
```

#### 4. Modified Routing Strategy

| LLM Routing | ASR Routing |
|-------------|-------------|
| KV cache overlap | N/A |
| Prefill tokens | Audio buffer size |
| Decode blocks | Active concurrent streams |
| Prefix matching | **Session affinity** (sticky routing) |

> **Critical**: Once a stream starts on a worker, it should stay there to maintain audio context.

### Data Flow

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         Request Flow: Audio → Transcription                     │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  1. CLIENT                                                                      │
│     └─► WebRTC: Establish connection (ICE, DTLS)                               │
│                                                                                 │
│  2. WEBRTC GATEWAY                                                              │
│     └─► Extract audio track, negotiate codec (Opus → PCM)                      │
│     └─► Open data channel for transcriptions                                   │
│                                                                                 │
│  3. AUDIO BRIDGE                                                                │
│     └─► Decode Opus → PCM 16kHz                                                │
│     └─► Buffer 100ms chunks                                                    │
│     └─► stream = await router.transcribe(audio_stream, session_id)             │
│                                                                                 │
│  4. DYNAMO FRONTEND                                                             │
│     └─► Router: Find worker with capacity + session affinity                   │
│     └─► Route to ASR Worker N                                                  │
│                                                                                 │
│  5. ASR WORKER                                                                  │
│     └─► VAD: Detect speech segments                                            │
│     └─► Model inference: Audio → Text (streaming)                              │
│     └─► Yield partial transcriptions                                           │
│                                                                                 │
│  6. RESPONSE PATH                                                               │
│     └─► Text chunks: Worker → Dynamo → Bridge → WebRTC data channel → Client   │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Latency Budget

For real-time ASR, target **end-to-end < 300-500ms**:

| Component | Target Latency |
|-----------|----------------|
| WebRTC audio capture | ~20ms (codec frame) |
| Network (client → gateway) | ~30-50ms |
| Audio decoding + buffering | ~100ms (buffer size) |
| Dynamo routing | ~5ms |
| ASR inference | ~100-150ms |
| Network (gateway → client) | ~30-50ms |
| **Total** | **~300-400ms** |

---

## Why WebRTC Is Not Native to Dynamo

### Current Frontend Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Dynamo Frontend (Rust)                   │
│                                                             │
│  • HTTP/HTTPS server (Axum/Hyper)                          │
│  • OpenAI-compatible REST API                               │
│  • Request/Response: JSON in, SSE out                      │
│  • Protocol: HTTP/1.1, HTTP/2                              │
│                                                             │
│  Optimized for:                                             │
│  • Text prompts → Token streams                            │
│  • Stateless requests                                       │
│  • High throughput batch processing                         │
└─────────────────────────────────────────────────────────────┘
```

### WebRTC Protocol Complexity

WebRTC is a **completely different protocol stack**:

```
┌─────────────────────────────────────────────────────────────┐
│                     WebRTC Stack                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Application Layer                                    │   │
│  │ • Media Tracks (audio/video)                        │   │
│  │ • Data Channels (arbitrary data)                    │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Session Layer                                        │   │
│  │ • SDP (Session Description Protocol) negotiation    │   │
│  │ • Offer/Answer exchange                             │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Security Layer                                       │   │
│  │ • DTLS (Datagram TLS) for key exchange              │   │
│  │ • SRTP (Secure RTP) for media encryption            │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Connectivity Layer                                   │   │
│  │ • ICE (Interactive Connectivity Establishment)      │   │
│  │ • STUN (Session Traversal Utilities for NAT)       │   │
│  │ • TURN (Traversal Using Relays around NAT)         │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Transport Layer                                      │   │
│  │ • UDP (primary), TCP (fallback)                     │   │
│  │ • SCTP (for data channels)                          │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ Media Layer                                          │   │
│  │ • Opus codec (audio) - encode/decode                │   │
│  │ • Jitter buffer management                          │   │
│  │ • Packet loss concealment                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Technical Barriers

#### 1. Protocol Differences

| HTTP (current) | WebRTC |
|----------------|--------|
| Request → Response | Bidirectional streams |
| TCP-based | Primarily UDP |
| Stateless | Stateful sessions |
| Simple handshake | Complex ICE negotiation |
| Text/JSON | Binary media + data |

#### 2. NAT Traversal Infrastructure

WebRTC requires additional servers:

```
                     Internet
                         │
    ┌────────────────────┼────────────────────┐
    │                    │                    │
┌───▼───┐          ┌─────▼─────┐        ┌────▼────┐
│ STUN  │          │   TURN    │        │Signaling│
│Server │          │  Server   │        │ Server  │
└───────┘          └───────────┘        └─────────┘
    │                    │                    │
    │  NAT discovery     │  Relay fallback   │  SDP exchange
    └────────────────────┼────────────────────┘
                         │
                    ┌────▼────┐
                    │ Dynamo  │
                    └─────────┘
```

#### 3. Media Processing Complexity

WebRTC audio arrives as **Opus-encoded RTP packets**:

```
Client → [48kHz audio] → [Opus encode] → [RTP packets] → UDP → Server

Server must:
  • Reorder packets (jitter buffer)
  • Handle packet loss
  • Decode Opus → PCM
  • Resample to 16kHz (for ASR)
```

#### 4. Different Scaling Patterns

| LLM Serving (HTTP) | Real-time Audio (WebRTC) |
|--------------------|--------------------------|
| Stateless | Stateful sessions |
| Batch-friendly | Stream-per-connection |
| Compute-bound scaling | Connection + bandwidth scaling |
| Latency: 100ms-10s OK | Latency: <50ms required |

### Could It Be Done?

**Yes**, using Rust WebRTC libraries like `webrtc-rs` or `str0m`:

```rust
// Hypothetical native WebRTC in Dynamo
async fn handle_webrtc_connection(peer: WebRtcPeer) {
    let connection = peer.establish_connection().await?;
    let audio_track = connection.get_audio_track().await?;
    let worker = router.select_worker_for_audio_stream().await?;
    let transcriptions = worker.transcribe(audio_track).await?;
    let data_channel = connection.create_data_channel("text").await?;
    
    while let Some(text) = transcriptions.next().await {
        data_channel.send(text).await?;
    }
}
```

**But this would require:**

- ~10,000+ lines of new code
- Deep WebRTC protocol expertise
- Browser compatibility testing
- STUN/TURN infrastructure
- Estimated: **3-6 months** of dedicated development

---

## Implementation Roadmap

### Phase 1: Proof of Concept (2-4 weeks)

| Task | Description |
|------|-------------|
| WebSocket audio transport | Simpler than WebRTC for initial validation |
| Basic `dynamo.asr` backend | Using Faster-Whisper or NVIDIA Canary |
| Audio endpoint in frontend | Accept audio streams, return transcriptions |
| Latency validation | Measure end-to-end performance |

### Phase 2: Production WebRTC (4-8 weeks)

| Task | Description |
|------|-------------|
| Deploy LiveKit | WebRTC gateway infrastructure |
| Audio bridge service | Connect LiveKit rooms to Dynamo |
| Session affinity routing | Sticky routing for audio streams |
| ASR-specific planner | Autoscaling based on stream count |

### Phase 3: Optimization (4-6 weeks)

| Task | Description |
|------|-------------|
| Native Rust WebRTC | If latency requirements demand it |
| Speculative transcription | Similar to speculative decoding in LLMs |
| Multi-language routing | Route to language-specific workers |
| Advanced VAD | Improve word boundary detection |

---

## ASR Model Options

| Model | Type | Streaming | Latency | Quality |
|-------|------|-----------|---------|---------|
| **Whisper** (OpenAI) | Encoder-Decoder | Chunked* | Medium | Excellent |
| **Faster-Whisper** | CTC-optimized | Chunked* | Low | Excellent |
| **NVIDIA Canary** | CTC/Transducer | Native | Low | Excellent |
| **NVIDIA Parakeet** | CTC | Native | Very Low | Good |
| **Moonshine** | Streaming-native | Native | Very Low | Good |

> *Whisper requires chunking workarounds for streaming (not natively streaming)

**Recommendation**: For true real-time, use **NVIDIA Canary** or **Parakeet** which are designed for streaming ASR.

---

## Key Design Decisions

### Decision 1: Separate WebRTC Gateway vs. Native

| Approach | Pros | Cons |
|----------|------|------|
| **Separate Gateway** (recommended) | Battle-tested, faster to implement, specialized tools | Additional service, network hop |
| **Native Dynamo WebRTC** | Single deployment, lower latency potential | Massive development effort, maintenance burden |

**Recommendation**: Use separate gateway (LiveKit).

### Decision 2: WebSocket vs. WebRTC (Phase 1)

| Approach | Pros | Cons |
|----------|------|------|
| **WebSocket** (for PoC) | Simple, works through firewalls, easy to debug | Higher latency, no built-in audio processing |
| **WebRTC** (for production) | Low latency, echo cancellation, noise suppression | Complex setup |

**Recommendation**: Start with WebSocket, migrate to WebRTC.

### Decision 3: Routing Strategy

| LLM Routing | ASR Routing Equivalent |
|-------------|------------------------|
| KV cache overlap | Not applicable |
| Prefill cost | Audio buffer backlog |
| Decode blocks | Active stream count |
| Random/Round-robin | **Session affinity** (required) |

**Recommendation**: Implement session-sticky routing with load-aware fallback.

---

## Appendix

### A. Service Discovery Configuration

```yaml
# etcd configuration for ASR workers
services:
  asr-workers:
    discovery: etcd
    endpoints:
      - host: worker-1.internal
        port: 50051
        capabilities:
          - streaming_asr
          - vad
        gpu_memory: 24GB
        max_concurrent_streams: 10
```

### B. Latency Monitoring Metrics

```
# Prometheus metrics to track
asr_audio_buffer_ms        # Audio buffered before processing
asr_inference_latency_ms   # Model inference time
asr_e2e_latency_ms         # End-to-end (audio in → text out)
asr_active_streams         # Current concurrent streams
asr_worker_utilization     # GPU utilization per worker
```

### C. Questions to Answer Before Implementation

1. **Scale**: How many concurrent streams do you need?
2. **Latency SLA**: What's acceptable word latency? (<200ms is challenging)
3. **Languages**: Single language or multilingual?
4. **Client platforms**: Browser-only or also mobile native?
5. **Audio quality**: Controlled (quiet room) or variable (phone calls)?

---

## References

- [NVIDIA Dynamo GitHub](https://github.com/ai-dynamo/dynamo)
- [LiveKit Documentation](https://docs.livekit.io/)
- [WebRTC Specification](https://www.w3.org/TR/webrtc/)
- [NVIDIA Canary ASR](https://developer.nvidia.com/nemo)
- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper)

---

> **Document Version**: 1.0  
> **Last Updated**: January 2026  
> **Authors**: Architecture Discussion with Claude
