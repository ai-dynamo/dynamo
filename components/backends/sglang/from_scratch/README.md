# Building Custom Inference Flows with Dynamo Runtime

## Introduction

This tutorial series shows you how to build production-ready inference pipelines from scratch using Dynamo's runtime infrastructure. You'll learn to create distributed, scalable systems by connecting components through NATS and service discovery via etcd.

**What makes this different:** No helpers from Dynamo LLM - just pure runtime patterns you can apply to any pipeline architecture.

## Chapter Progression

### [Chapter 1: Simple Chat Pipeline](chapter1/)
**Goal:** Learn the fundamentals of Dynamo components and communication

Build a basic 3-component pipeline:
```
Client → Frontend → Processor → Engine
         (HTTP)     (Tokenizer) (SGLang)
```

**Key Concepts:**
- `@dynamo_worker` decorator and service registration
- Service vs Client patterns  
- Data transport with `chunk.data()`
- Async generator streaming

### Chapter 2: Adding a Router
**Goal:** Load balancing and intelligent request routing

Add a router component for multi-engine deployments:
```
Client → Frontend → Processor → Router → Engine (multiple)
```

**New Concepts:**
- KV cache metrics and engine selection
- ZMQ event publishing/subscribing
- RadixTree for cache-aware routing

### Chapter 3: Custom Router Logic
**Goal:** Implement sophisticated routing strategies

Build custom routing algorithms:
- Prefix caching optimization
- Engine health monitoring
- Request queuing and prioritization

### Chapter 4: Agentic Loops
**Goal:** Multi-step reasoning and tool usage

Transform the pipeline for agent workflows:
- Multi-turn conversations with memory
- Tool calling and execution
- Dynamic request routing based on agent state

### Chapter 5: Kubernetes Deployment
**Goal:** Production deployment and scaling

Deploy the complete system:
- Kubernetes manifests and operators
- Auto-scaling based on load
- Monitoring and observability

## Core Architecture

### Runtime Infrastructure
- **NATS**: Message transport between components
- **etcd**: Service registration and discovery
- **DistributedRuntime**: Unified interface for both

### Communication Patterns
- **Service Registration**: `component.create_service()` → etcd
- **Client Discovery**: `.client()` → etcd lookup → NATS connection  
- **Request Flow**: `client.method(data)` → NATS → service handler

### Key Design Principles
1. **Separation of Concerns**: Each component has one responsibility
2. **Transport Agnostic**: Business logic separate from communication
3. **Discoverable**: Components find each other automatically
4. **Streaming First**: Internal streaming with external aggregation
5. **Fault Tolerant**: Components can restart independently

## Getting Started

### Prerequisites
```bash
# Install Dynamo runtime
pip install ai-dynamo[sglang]
```

### Infrastructure Setup
```bash
# Terminal 1: Start NATS with JetStream
nats-server -js

# Terminal 2: Start etcd
etcd
```

## What You'll Learn

By the end of this series, you'll understand:

- **Distributed System Design**: How to architect scalable inference pipelines
- **Service Communication**: NATS-based messaging and etcd service discovery
- **Load Balancing**: Intelligent routing based on system metrics
- **Production Deployment**: Kubernetes orchestration and monitoring

## Next Steps

Start with [Chapter 1](chapter1/) to build your first pipeline, then progress through the chapters to build increasingly sophisticated systems.

Each chapter builds on the previous ones, so follow them in order for the best learning experience.