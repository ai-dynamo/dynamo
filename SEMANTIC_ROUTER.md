# Semantic Router Guide

Complete guide for setting up and using semantic routing with Dynamo.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Architecture](#architecture)
4. [Model Setup](#model-setup)
5. [Configuration](#configuration)
6. [Testing](#testing)
7. [Routing Modes](#routing-modes)
8. [Metrics](#metrics)
9. [Troubleshooting](#troubleshooting)

## Overview

The semantic router automatically routes requests to the most appropriate model based on content analysis. It uses a trained classifier (CodeIsAbstract/ReasoningTextClassifier) to determine if a query requires reasoning capabilities or can be handled by a general-purpose model.

**Key Features:**
- 🎯 **Unified Architecture**: Single `Classifier` trait supports both binary and multi-class
- 🚀 **ONNX Runtime**: Fast, optimized inference with no Python dependency at runtime
- 🔧 **Flexible**: Binary classification by default, extensible to multi-class
- 📊 **Observable**: Full Prometheus metrics for monitoring
- 🛡️ **Safe**: Configurable fallback model for low-confidence predictions

### Quick Comparison: MockClassifier vs OnnxClassifier

| Feature | MockClassifier | OnnxClassifier |
|---------|---------------|----------------|
| **Setup Time** | < 1 minute | ~10 minutes (model export) |
| **Dependencies** | None (pure Rust) | ONNX Runtime, model files |
| **Classification** | Keyword-based | ML-based (99.9% accuracy) |
| **Use Case** | Development, testing, CI/CD | Production deployments |
| **Build** | `maturin develop --uv` | `maturin develop --uv --features onnx-classifier` |
| **Config** | `semantic-router-binary.yaml` | `semantic-router-binary.yaml` + model paths |
| **Test Script** | `./run_routing_tests.sh` | `./test_semantic_router.sh` |

**👉 Start with MockClassifier for instant testing, then upgrade to OnnxClassifier for production.**

## Quick Start

### Option A: Testing with MockClassifier (Recommended for Development)

The MockClassifier allows you to test the routing architecture without ONNX dependencies. Perfect for development, CI/CD, and validating the routing logic.

#### 1. Configure Binary Routing

Create `semantic-router-binary.yaml`:
```yaml
reasoning_model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
general_model: meta-llama/Meta-Llama-3.1-8B-Instruct
abstain_onprem_model: meta-llama/Meta-Llama-3.1-8B-Instruct
threshold_min_conf: 0.6
```

#### 2. Start Backend Models

**Terminal 1 - General Model:**
```bash
export HF_TOKEN=<your_token>
CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8100
```

**Terminal 2 - Reasoning Model:**
```bash
export HF_TOKEN=<your_token>
CUDA_VISIBLE_DEVICES=1 python -m dynamo.vllm \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --port 8101
```

#### 3. Build and Start Frontend (Without ONNX)

```bash
# Build without onnx-classifier feature (uses MockClassifier automatically)
cd /home/ubuntu/dynamo/lib/bindings/python
maturin develop --uv

# Set environment variables
cd /home/ubuntu/dynamo
export SEMROUTER_ENABLED=true
export SEMROUTER_CONFIG=./semantic-router-binary.yaml

# Start frontend
python -m dynamo.frontend --http-port 8999
```

#### 4. Run Automated Tests

```bash
# In another terminal
./run_routing_tests.sh
```

**Expected output:**
```
🧪 Testing Semantic Router with MockClassifier
==============================================

Test 1: Reasoning query with 'prove' keyword...
✅ PASSED: Routed to reasoning model (deepseek-ai/DeepSeek-R1-Distill-Llama-8B)

Test 2: General query without reasoning keywords...
✅ PASSED: Routed to general model (meta-llama/Meta-Llama-3.1-8B-Instruct)

Test 3: Routing disabled with X-Dynamo-Routing: off...
✅ PASSED: Routing disabled, used requested model

Test 4: Force routing with X-Dynamo-Routing: force...
✅ PASSED: Force routing used specified model

==============================================
🎉 All tests passed!
==============================================
```

**MockClassifier Behavior:**
- Detects keywords: `prove`, `calculate`, `logic`, `deduce` → routes to reasoning model
- All other queries → routes to general model
- Simple, deterministic, perfect for testing routing logic

---

### Option B: Production Setup with ONNX Classifier

For production deployments with ML-based classification:

#### 1. Export Model to ONNX

Install requirements:
```bash
pip install transformers optimum[onnxruntime]
```

Export the model:
```bash
python scripts/export_reasoning_classifier.py ./reasoning-classifier-onnx
```

This downloads `CodeIsAbstract/ReasoningTextClassifier` from HuggingFace and exports it to ONNX format.

#### 2. Configure Routing

Create `semantic-router.yaml` (or use the binary config for simpler setup):

```yaml
# Binary classification configuration
reasoning_model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
general_model: meta-llama/Meta-Llama-3.1-8B-Instruct
abstain_onprem_model: meta-llama/Meta-Llama-3.1-8B-Instruct
threshold_min_conf: 0.6
```

**Configuration Fields:**
- `reasoning_model`: Model for queries requiring reasoning (e.g., math, logic proofs)
- `general_model`: Model for simple queries (e.g., facts, definitions)
- `abstain_onprem_model`: Fallback model when classifier confidence is below threshold
- `threshold_min_conf`: Minimum confidence required (0.0-1.0)

### 3. Start Backend Models

**Terminal 1 - General Model:**
```bash
export HF_TOKEN=<your_token>
CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8100
```

**Terminal 2 - Reasoning Model:**
```bash
export HF_TOKEN=<your_token>
CUDA_VISIBLE_DEVICES=1 python -m dynamo.vllm \
  --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
  --port 8101
```

### 4. Build and Start Frontend with Routing

Set environment variables:
```bash
export SEMROUTER_ENABLED=true
export SEMROUTER_CONFIG=./semantic-router.yaml
export SEMROUTER_MODEL_PATH=./reasoning-classifier-onnx/model.onnx
export SEMROUTER_TOKENIZER_PATH=./reasoning-classifier-onnx/tokenizer.json
export SEMROUTER_MAX_LENGTH=256  # Optional, defaults to 256
```

Build with ONNX classifier support:
```bash
cd /home/ubuntu/dynamo/lib/bindings/python
maturin develop --uv --features onnx-classifier
```

Start the frontend:
```bash
cd /home/ubuntu/dynamo
python -m dynamo.frontend --http-port 8999
```

### 5. Test Routing

**Test reasoning query:**
```bash
curl -s localhost:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Routing: auto' \
  -d '{
    "model": "router",
    "messages": [{
      "role": "user",
      "content": "Prove that the square root of 2 is irrational. Show each step."
    }],
    "max_tokens": 256
  }' | jq '.model'
```

**Expected output:** `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

**Test general query:**
```bash
curl -s localhost:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Routing: auto' \
  -d '{
    "model": "router",
    "messages": [{
      "role": "user",
      "content": "What is the capital of France?"
    }],
    "max_tokens": 64
  }' | jq '.model'
```

**Expected output:** `meta-llama/Meta-Llama-3.1-8B-Instruct`

## Architecture

### Classifier Backends

The semantic router supports two classifier implementations:

#### MockClassifier (Development/Testing)
- **Purpose**: Validate routing architecture without ML dependencies
- **How it works**: Simple keyword detection
  - Keywords: `prove`, `calculate`, `logic`, `deduce` → reasoning model
  - All other queries → general model
- **When to use**:
  - Development and testing
  - CI/CD pipelines
  - Architecture validation
  - Quick prototyping
- **Activation**: Automatically used when building without `--features onnx-classifier`
- **Configuration**: Same YAML format (binary or multi-class)

#### OnnxClassifier (Production)
- **Purpose**: ML-based classification for production
- **Model**: CodeIsAbstract/ReasoningTextClassifier (ModernBERT-based)
- **Accuracy**: 99.9% on test set
- **Latency**: 5-20ms on CPU
- **When to use**: Production deployments requiring accurate classification
- **Activation**: Build with `--features onnx-classifier`
- **Requirements**:
  - ONNX model file (`model.onnx`)
  - Tokenizer configuration (`tokenizer.json`)
  - Environment variables set

### Design Principles

1. **Unified Classifier**: One `Classifier` trait for all classification types
   - Binary: returns `{"non-reasoning": 0.3, "reasoning": 0.7}`
   - Multi-class: returns `{"math": 0.1, "code": 0.2, "reasoning": 0.6, ...}`

2. **Transport-Agnostic**: Single entry point for all transports
   - HTTP/gRPC handlers → `process_chat_request()` → SemRouter
   - Zero-cost abstraction when routing is disabled
   - Only header extraction is transport-specific (unavoidable)

3. **Policy-Based Routing**: Flexible decision logic
   - Binary mode: checks `reasoning_model` in config
   - Multi-class mode: uses rules with label conditions

### Request Flow

```
┌──────────────────────────────────────────────────────┐
│  Transport Handler (HTTP, gRPC, etc)                 │
│  - Extracts X-Dynamo-Routing header                  │
│  - Calls process_chat_request()                      │
└───────────────────┬──────────────────────────────────┘
                    │
                    ▼
         ┌──────────────────────────┐
         │  process_chat_request()  │  ← Single entry point
         │  (request_processor.rs)  │     Zero-cost when disabled
         └───────────┬──────────────┘
                     │
                     ▼
         ┌──────────────────────────┐
         │      SemRouter.apply()   │
         │  ┌────────────────────┐  │
         │  │ Classifier (ONNX)  │  │  ← Analyzes text
         │  │ Policy Engine      │  │  ← Makes decision
         │  └────────────────────┘  │
         └───────────┬──────────────┘
                     │
                     ▼
         ┌──────────────────────────┐
         │ Mutate request.model     │  ← "router" → "llama-3"
         └───────────┬──────────────┘
                     │
                     ▼
         ┌──────────────────────────┐
         │  get_engine(model_name)  │  ← Select correct engine
         └───────────┬──────────────┘
                     │
                     ▼
         ┌──────────────────────────┐
         │  engine.generate(request)│  ← Engine has preprocessor
         └──────────────────────────┘
```

### Why Not in Preprocessor?

Unlike audit/observability (which wraps the entire execution), routing must happen
**before** engine selection because it determines **which** engine/preprocessor to use.

The preprocessor is part of each engine's pipeline. Routing decides which pipeline runs.

**Transport Integration:**

```rust
// HTTP (openai.rs) - 4 lines
let _decision = crate::semrouter::process_chat_request(
    &mut request,
    state.semrouter().map(|r| r.as_ref()),
    headers.get("x-dynamo-routing").and_then(|h| h.to_str().ok()),
    "http",
);

// gRPC would be similar - just extract metadata instead of headers
```

**Performance:**
- Zero allocation when `router: None`
- Inline-friendly for minimal overhead
- No middleware complexity or Boxing

## Model Setup

### CodeIsAbstract/ReasoningTextClassifier

This is a ModernBERT-based binary classifier trained to detect reasoning patterns in text.

**Model Details:**
- Base: `answerdotai/ModernBERT-base`
- Task: Binary sequence classification
- Labels: `["non-reasoning", "reasoning"]`
- Accuracy: 99.9% on test set
- Input: Text (max 256 tokens)
- Output: Probability distribution over 2 labels

**What it detects:**
- ✅ Reasoning: Step-by-step explanations, logical arguments, math proofs
- ❌ Non-reasoning: Simple facts, definitions, direct answers

**Training data bias:**
The model is trained on LLM outputs (DeepSeek, Gemini style reasoning), so it's biased toward detecting LLM reasoning patterns rather than human reasoning styles.

### Exporting to ONNX

The `scripts/export_reasoning_classifier.py` script handles the export:

1. Downloads model from HuggingFace
2. Converts to ONNX format using Optimum
3. Saves tokenizer configuration
4. Creates `model.onnx` (ready for inference)

**Files created:**
```
reasoning-classifier-onnx/
├── model.onnx        # ONNX model (architecture + weights)
├── tokenizer.json    # Tokenizer configuration
└── config.json       # Model metadata
```

## Configuration

### Binary Classification (Recommended)

Use `semantic-router-binary.yaml` for simple reasoning vs. non-reasoning routing:

```yaml
# File: semantic-router-binary.yaml
# Simple binary routing - works with both MockClassifier and OnnxClassifier
reasoning_model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
general_model: meta-llama/Meta-Llama-3.1-8B-Instruct
abstain_onprem_model: meta-llama/Meta-Llama-3.1-8B-Instruct
threshold_min_conf: 0.6
```

**When to use binary configuration:**
- ✅ Testing with MockClassifier
- ✅ Production with ONNX binary classifier (CodeIsAbstract/ReasoningTextClassifier)
- ✅ Simple reasoning vs. general task routing
- ✅ Two-model deployments

### Switching Between Classifiers

| Classifier | Build Command | Env Variables Required | Use Case |
|------------|--------------|------------------------|----------|
| **MockClassifier** | `maturin develop --uv` | `SEMROUTER_ENABLED`, `SEMROUTER_CONFIG` | Development, testing, CI/CD |
| **OnnxClassifier** | `maturin develop --uv --features onnx-classifier` | Above + `SEMROUTER_MODEL_PATH`, `SEMROUTER_TOKENIZER_PATH` | Production with ML |

**To switch classifiers:**
1. Rebuild with/without `--features onnx-classifier`
2. Set appropriate environment variables
3. Restart frontend

Both classifiers use the same YAML configuration format.

### Multi-Class Classification (Advanced)

For multiple model routing based on categories (requires multi-class ONNX model):

```yaml
# Multi-class with rules
abstain_onprem_model: meta-llama/Meta-Llama-3.1-8B-Instruct
threshold_min_conf: 0.5
weights:
  reasoning: 60
  math: 60
  code: 50
  general: 20

rules:
  - when_any:
      - { label: "reasoning", min_conf: 0.55 }
      - { label: "math", min_conf: 0.55 }
    route_onprem_model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    rationale: "reasoning"

  - when_any:
      - { label: "code", min_conf: 0.50 }
    route_onprem_model: deepseek-ai/DeepSeekCoder-V2-Lite-Instruct
    rationale: "code"

  - when_any:
      - { label: "general", min_conf: 0.50 }
    route_onprem_model: meta-llama/Meta-Llama-3.1-8B-Instruct
    rationale: "general"
```

**Note:** Multi-class requires a multi-label classifier. The default CodeIsAbstract/ReasoningTextClassifier is binary only.

### Configuration Files

| File | Purpose | Classifier Support |
|------|---------|-------------------|
| `semantic-router-binary.yaml` | Binary routing (reasoning vs. general) | MockClassifier + OnnxClassifier |
| `semantic-router.yaml` | Multi-class routing with rules | Requires multi-class ONNX model |

**Example locations:**
```bash
/home/ubuntu/dynamo/semantic-router-binary.yaml  # Simple binary config
/home/ubuntu/dynamo/semantic-router.yaml         # Multi-class config
```

## Testing

### Automated Testing with MockClassifier

The fastest way to validate the routing architecture:

```bash
# 1. Ensure backend models are running (see Quick Start)

# 2. Build frontend with MockClassifier (no onnx-classifier feature)
cd /home/ubuntu/dynamo/lib/bindings/python
maturin develop --uv

# 3. Start frontend
cd /home/ubuntu/dynamo
export SEMROUTER_ENABLED=true
export SEMROUTER_CONFIG=./semantic-router-binary.yaml
python -m dynamo.frontend --http-port 8999 &

# 4. Run automated tests
./run_routing_tests.sh
```

The test script validates:
- ✅ Reasoning queries route to reasoning model
- ✅ General queries route to general model
- ✅ Routing disabled with `X-Dynamo-Routing: off`
- ✅ Force mode bypasses classifier
- ✅ Metrics endpoint availability

### Manual Testing with ONNX Classifier

Use the provided test script for ONNX-based testing:
```bash
./test_semantic_router.sh  # Requires ONNX setup
```

Or test manually with curl:

**Shadow Mode** (no override, metrics only):
```bash
curl -s localhost:8999/v1/chat/completions \
  -H 'X-Dynamo-Routing: shadow' \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role":"user","content":"Prove sqrt(2) is irrational"}]
  }' -i
```

Explicit model is used, but routing decision is computed for metrics.

**Auto Mode with Explicit Model** (no override):
```bash
curl -s localhost:8999/v1/chat/completions \
  -H 'X-Dynamo-Routing: auto' \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role":"user","content":"What is 2+2?"}]
  }' | jq '.model'
```

Returns: `meta-llama/Meta-Llama-3.1-8B-Instruct` (explicit model honored)

**Force Mode** (always override):
```bash
curl -s localhost:8999/v1/chat/completions \
  -H 'X-Dynamo-Routing: force' \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [{"role":"user","content":"Prove sqrt(2) is irrational"}]
  }' | jq '.model'
```

Returns: `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` (routing enforced)

## Routing Modes

Control routing behavior via `X-Dynamo-Routing` header:

| Mode | Model Field | Behavior |
|------|-------------|----------|
| **off** (default) | Any | No routing applied |
| **auto** | `"router"` | Routes to appropriate model |
| **auto** | Explicit model | Uses explicit model (no routing) |
| **shadow** | Any | Computes decision but doesn't override (metrics only) |
| **force** | Any | Always routes, ignores explicit model |

### Mode Details

**Auto Mode** (Recommended):
- Safe default for production
- Only routes when user explicitly requests `"model": "router"`
- Honors specific model requests
- Use for opt-in routing

**Shadow Mode**:
- Test routing decisions without affecting traffic
- Compare routing decisions vs actual model usage
- Use for evaluation and A/B testing

**Force Mode**:
- SRE/testing only
- Overrides all model requests
- Use for debugging or controlled experiments

## Metrics

View Prometheus metrics at `http://localhost:8999/metrics`:

### Route Decisions
```promql
semantic_route_decisions_total{
  route="enforce|shadow",
  target="model-name",
  rationale="reasoning|general|abstain_low_confidence",
  winner="reasoning|non-reasoning",
  transport="http"
}
```

### Classifier Latency
```promql
semantic_classifier_latency_ms{transport="http"}
```

**Example queries:**

Routing decision rate:
```promql
rate(semantic_route_decisions_total[5m])
```

Average classifier latency:
```promql
histogram_quantile(0.95, semantic_classifier_latency_ms)
```

Reasoning vs general split:
```promql
sum by (winner) (semantic_route_decisions_total)
```

## Troubleshooting

### Router Not Initializing

**Check logs for initialization message:**

MockClassifier:
```
INFO Initialized MockClassifier (binary_mode=true)
INFO Semantic router initialized with MockClassifier
```

OnnxClassifier:
```
INFO Initialized ONNX classifier with 2 labels, max_len=256
INFO Semantic router initialized
```

**If missing, verify:**

For **MockClassifier**:
1. `SEMROUTER_ENABLED=true`
2. `SEMROUTER_CONFIG` points to valid YAML
3. Built **without** `--features onnx-classifier`

For **OnnxClassifier**:
1. `SEMROUTER_ENABLED=true`
2. `SEMROUTER_CONFIG` points to valid YAML
3. `SEMROUTER_MODEL_PATH` and `SEMROUTER_TOKENIZER_PATH` are set
4. Built **with** `--features onnx-classifier`
5. ONNX model files exist at specified paths

### Which Classifier Am I Using?

Check the frontend logs on startup. You'll see either:
- `MockClassifier (binary_mode=true)` - Using mock classifier
- `Initialized ONNX classifier` - Using ONNX classifier

Or check the build:
```bash
# MockClassifier (default)
maturin develop --uv

# OnnxClassifier
maturin develop --uv --features onnx-classifier
```

### Models Not Found

Ensure backend workers are running and registered:
```bash
curl -s localhost:8999/v1/models | jq '.data[].id'
```

Should list both models from config.

### Routing Not Applied

1. Check `X-Dynamo-Routing` header is set
2. Verify model is `"router"` (in auto mode)
3. Check classifier metrics to see if it's running
4. Review logs for errors

### Low Accuracy

The classifier may not work well for:
- Human-written reasoning (trained on LLM outputs)
- Domain-specific reasoning patterns
- Non-English text

**Solutions:**
1. Adjust `threshold_min_conf` (lower = more aggressive routing)
2. Test with shadow mode first
3. Fine-tune the classifier on your data
4. Use multi-class with domain-specific categories

### Performance Issues

**Classifier too slow:**
- Reduce `SEMROUTER_MAX_LENGTH` (default: 256)
- Monitor `semantic_classifier_latency_ms` metric
- Typical latency: 5-20ms on CPU

**Memory usage:**
- ONNX model is ~350MB in memory
- Consider sharing classifier across workers

## Implementation Details

### Files

**Core Implementation:**
- `lib/llm/src/semrouter/classifier/mod.rs` - Unified Classifier trait
- `lib/llm/src/semrouter/classifier/mock.rs` - MockClassifier implementation
- `lib/llm/src/semrouter/classifier/onnx.rs` - OnnxClassifier implementation
- `lib/llm/src/semrouter/hook.rs` - SemRouter struct
- `lib/llm/src/semrouter/policy/mod.rs` - Routing policy engine
- `lib/llm/src/semrouter/request_processor.rs` - Transport-agnostic entry point
- `lib/llm/src/semrouter/routing.rs` - Routing utilities
- `lib/llm/src/semrouter/types.rs` - RoutingMode and RequestMeta

**Integration:**
- `lib/llm/src/http/service/openai.rs` - HTTP handler integration (4-line change)
- `lib/llm/src/http/service/service_v2.rs` - State initialization

**Configuration:**
- `semantic-router-binary.yaml` - Binary routing config (MockClassifier compatible)
- `semantic-router.yaml` - Multi-class routing config (requires ONNX)

**Testing:**
- `run_routing_tests.sh` - Automated test suite for MockClassifier
- `test_semantic_router.sh` - Manual testing guide (ONNX)

**Utilities:**
- `scripts/export_reasoning_classifier.py` - ONNX model export script

### Design Rationale

**Why ONNX over Candle?**
- ONNX bundles architecture + weights in one file
- Candle requires implementing ModernBERT architecture in Rust (300+ lines)
- ONNX Runtime is battle-tested and optimized
- One-time Python export vs ongoing Rust maintenance

**Why unified Classifier trait?**
- Binary is just multi-class with 2 labels
- Simpler API: one trait instead of two
- Policy layer handles differences
- Easier to extend to multi-class later

**Why in HTTP handler not preprocessor?**
- Routing is transport-specific (needs headers/metadata)
- Preprocessor stays focused on templating & tokenization
- Transport-agnostic utilities provide reusability
- Explicit opt-in at transport level

**Transport-agnostic design:**
Any transport can initialize SemRouter directly:

```rust
// HTTP Service
impl State {
    pub fn init_semrouter_from_config(&mut self, config_path: impl AsRef<Path>) -> Result<()> {
        let router = SemRouter::from_config(config_path)?;  // ← Factory method in semrouter
        self.semrouter = Some(Arc::new(router));
        Ok(())
    }
}

// gRPC Service (hypothetical)
impl GrpcState {
    pub fn init_semrouter(&mut self, config_path: impl AsRef<Path>) -> Result<()> {
        let router = SemRouter::from_config(config_path)?;  // ← Same factory method!
        self.semrouter = Some(Arc::new(router));
        Ok(())
    }
}
```

The initialization logic lives in `SemRouter::from_config()`, not in transport-specific code.

## Getting Started Checklist

### For Development/Testing (5 minutes)
- [ ] Create `semantic-router-binary.yaml` config
- [ ] Start two backend model workers (ports 8100, 8101)
- [ ] Build frontend: `maturin develop --uv` (no features)
- [ ] Export env vars: `SEMROUTER_ENABLED=true`, `SEMROUTER_CONFIG=./semantic-router-binary.yaml`
- [ ] Start frontend: `python -m dynamo.frontend --http-port 8999`
- [ ] Run tests: `./run_routing_tests.sh`
- [ ] ✅ All tests passing → MockClassifier working!

### For Production (30 minutes)
- [ ] Export ONNX model: `python scripts/export_reasoning_classifier.py ./reasoning-classifier-onnx`
- [ ] Set all env vars (including `SEMROUTER_MODEL_PATH`, `SEMROUTER_TOKENIZER_PATH`)
- [ ] Build with ONNX: `maturin develop --uv --features onnx-classifier`
- [ ] Start frontend and verify logs show "Initialized ONNX classifier"
- [ ] Test with shadow mode first (`X-Dynamo-Routing: shadow`)
- [ ] Monitor metrics and tune `threshold_min_conf`
- [ ] Enable auto mode for production (`X-Dynamo-Routing: auto`)

## Next Steps

1. **Test with real traffic** - Use shadow mode first
2. **Tune threshold** - Adjust `threshold_min_conf` based on metrics
3. **Monitor metrics** - Watch routing decisions and latency
4. **A/B test** - Compare routed vs non-routed performance
5. **Evaluate accuracy** - Sample routing decisions for correctness

## Support

For issues or questions:
1. Check troubleshooting section above
2. Review semantic router logs
3. Inspect Prometheus metrics
4. Test with shadow mode to debug decisions
