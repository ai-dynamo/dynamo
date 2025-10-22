# Semantic Router PoC Guide

This guide walks you through testing the semantic routing system locally.

## Architecture

The semantic router sits in the Frontend HTTP service and classifies incoming chat completion requests to route them to the most appropriate on-premises model based on the query content.

**Components:**
- **Frontend (HTTP Service)**: Receives OpenAI-compatible requests, applies semantic routing
- **Classifier**: Analyzes request text and returns category probabilities
- **Policy Engine**: Maps categories to specific models based on YAML configuration
- **Workers**: vLLM or TensorRT-LLM workers serving different models

## Setup

### 1. Start Worker Models

Start two workers on different GPUs:

```bash
export HF_TOKEN=<your_token>

# Terminal 1: General model (Llama)
CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --model meta-llama/Meta-Llama-3.1-8B-Instruct

# Terminal 2: Reasoning model (DeepSeek)
CUDA_VISIBLE_DEVICES=1 python -m dynamo.vllm --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

### 2. Build Frontend with Semantic Router

```bash
# Build the Rust components
cd /home/ubuntu/dynamo/lib/bindings/python
maturin develop --uv

cd /home/ubuntu/dynamo
uv pip install .

uv pip install ai-dynamo[vllm]
```

### 3. Configure and Start Frontend

```bash
# Set environment variables
export SEMROUTER_ENABLED=true
export SEMROUTER_CONFIG=./semantic-router.yaml

# Optional: Use ONNX classifier (requires onnx-classifier feature)
# export ROUTER_CLASSIFIER_ONNX=/models/router-modernbert.onnx
# export ROUTER_CLASSIFIER_TOKENIZER=/models/tokenizer.json

# Start the frontend (Python wrapper calls Rust library)
python -m dynamo.frontend --http-port 8999
```

## Configuration

The `semantic-router.yaml` defines routing rules:

```yaml
abstain_onprem_model: meta-llama/Meta-Llama-3.1-8B-Instruct
threshold_min_conf: 0.40
weights:
  pii:       90
  reasoning: 60
  math:      60
  code:      50
  summarize: 20
  qa:        20
rules:
  - when_any:
      - { label: "pii",       min_conf: 0.40 }
    route_onprem_model: meta-llama/Meta-Llama-3.1-8B-Instruct
    rationale: "pii"
  - when_any:
      - { label: "code",      min_conf: 0.50 }
    route_onprem_model: deepseek-ai/DeepSeekCoderV2-Lite-Instruct
    rationale: "code"
  - when_any:
      - { label: "reasoning", min_conf: 0.55 }
      - { label: "math",      min_conf: 0.55 }
    route_onprem_model: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
    rationale: "reasoning"
  - when_any:
      - { label: "summarize", min_conf: 0.50 }
      - { label: "qa",        min_conf: 0.50 }
    route_onprem_model: meta-llama/Meta-Llama-3.1-8B-Instruct
    rationale: "general"
```

## Testing

### Using the Test Script

```bash
./test_semantic_router.sh
```

### Manual Tests

#### Test 1: Reasoning Query (Router Alias)

```bash
curl -s localhost:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Routing: auto' \
  -d '{
    "model":"router",
    "messages":[{
      "role":"user",
      "content":"Prove that the sum of first n odd numbers is n^2. Think step by step."
    }],
    "max_tokens":128
  }' | jq '.model'
```

**Expected**: Routes to `deepseek-ai/DeepSeek-R1-Distill-Llama-8B`

#### Test 2: Simple Factoid (Router Alias)

```bash
curl -s localhost:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Routing: auto' \
  -d '{
    "model":"router",
    "messages":[{
      "role":"user",
      "content":"What is the capital of Spain?"
    }],
    "max_tokens":64
  }' | jq '.model'
```

**Expected**: Routes to `meta-llama/Meta-Llama-3.1-8B-Instruct`

#### Test 3: Shadow Mode (Explicit Model)

```bash
curl -i -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Routing: shadow' \
  -d '{
    "model":"meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages":[{
      "role":"user",
      "content":"Prove that the sum of first n odd numbers is n^2."
    }],
    "max_tokens":64
  }'
```

**Expected**:
- Uses `meta-llama/Meta-Llama-3.1-8B-Instruct` (no override)
- Metrics show shadow decision would have been "reasoning" → DeepSeek

#### Test 4: Auto Mode with Explicit Model

```bash
curl -s localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'X-Dynamo-Routing: auto' \
  -d '{
    "model":"meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages":[{
      "role":"user",
      "content":"What is 2+2?"
    }],
    "max_tokens":32
  }' | jq '.model'
```

**Expected**: Uses `meta-llama/Meta-Llama-3.1-8B-Instruct` (no override since explicit model given)

## Routing Modes

### `X-Dynamo-Routing` Header Values

- **`auto`** (Recommended): Routes only if `model:"router"` or empty. If explicit model given, operates in shadow mode for telemetry.
- **`force`**: Routes even with explicit model (SRE/testing only).
- **`shadow`**: Never overrides; computes decision for metrics only.
- **Off** (no header): No routing applied.

### Model Alias

Set `"model": "router"` in the request to trigger routing when `X-Dynamo-Routing: auto` is set.

## Metrics

View Prometheus metrics at `http://localhost:8000/metrics`:

```promql
# Route decisions
semantic_route_decisions_total{route="enforce|shadow",target="model-name",rationale="reasoning|general|...",winner="category",transport="http"}

# Classifier latency
semantic_classifier_latency_ms{transport="http"}
```

## Implementation Details

### Files Created

- `lib/llm/src/semrouter/` - Complete semantic router implementation
  - `types.rs` - Core types (RoutingMode, Target, RoutePlan, RequestMeta)
  - `config.rs` - YAML policy configuration
  - `metrics.rs` - Prometheus metrics
  - `classifier/mod.rs` - MultiClassifier trait + HeuristicClassifier
  - `classifier/modernbert.rs` - ModernBERT ONNX adapter (feature-gated)
  - `policy/mod.rs` - Category-to-model mapping
  - `hook.rs` - Main routing logic

### Integration Points

- `lib/llm/src/http/service/service_v2.rs` - Added semantic router to HTTP service State
- `lib/llm/src/http/service/openai.rs` - Integrated routing into chat completions handler
- `lib/llm/src/entrypoint/input/http.rs` - Environment variable configuration
- `semantic-router.yaml` - Example configuration

### Current Classifier

The PoC uses a **HeuristicClassifier** that identifies reasoning queries based on:
- Phrases like "think step by step", "explain your reasoning"
- Query length (> 120 tokens)

For production, replace with ModernBERT or similar:

```rust
#[cfg(feature = "onnx-classifier")]
use crate::semrouter::classifier::modernbert::ModernBertClassifier;

let classifier = Arc::new(ModernBertClassifier::new(
    "router-modernbert.onnx",
    "tokenizer.json",
    vec!["reasoning", "code", "math", "qa", "summarize", "pii"],
    256
)?);
state.init_semrouter_with_classifier(config_path, classifier)?;
```

## Troubleshooting

### Router Not Initializing

Check logs for:
```
INFO Initializing semantic router from config: ./semantic-router.yaml
INFO Semantic router initialized with heuristic classifier
```

If not present, verify:
- `SEMROUTER_ENABLED=true`
- `SEMROUTER_CONFIG` points to valid YAML file
- File is readable and well-formed

### Models Not Found

Ensure workers are registered in etcd and visible to frontend:
```bash
curl -s localhost:8000/v1/models | jq '.data[].id'
```

### Routing Not Applied

- Check `X-Dynamo-Routing` header is set
- Verify model is "router" or request is in appropriate mode
- Check metrics to see if classifier is running

## Next Steps

1. **Test with real queries** from your use cases
2. **Tune confidence thresholds** in `semantic-router.yaml`
3. **Deploy ModernBERT classifier** for production accuracy
4. **Add response headers** to expose shadow routing decisions
5. **Monitor metrics** in Prometheus/Grafana
6. **A/B test** routing decisions vs. baseline

## Design Principles

✅ **No accidental overrides**: Explicit models are honored unless `force` mode
✅ **Opt-in routing**: Via header or `model:"router"` alias
✅ **Classifier-agnostic**: Pluggable via trait
✅ **Production-ready**: Proper error handling, no `.unwrap()`
✅ **Observable**: Full Prometheus metrics

