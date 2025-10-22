# sem-router

Standalone Tower/Axum middleware for semantic routing with fastText classification.

## What is sem-router?

`sem-router` is a reusable Rust crate that provides content-based request routing as a Tower layer. It analyzes incoming HTTP requests, classifies them using fastText, and rewrites the request body's `model` field based on configured routing rules.

**Key Features:**
- **Standalone crate**: Zero dependencies on Dynamo internals - works with any Tower/Axum application
- **FastText classifier**: Sub-millisecond inference on CPU (~300-600 microseconds)
- **Multi-class routing**: Route to different models based on arbitrary classification labels
- **Configurable**: YAML-based configuration with environment variable support
- **Tower middleware**: Composable, idiomatic Rust service architecture

## Quick Start

Get up and running in 5 minutes with an existing fastText model:

```bash
# 1. Copy and customize the example config
cd /home/ubuntu/dynamo
cp semrouter_configs/example_fasttext_config_v2.yaml /tmp/my-semrouter.yaml
sed -i 's|/path/to/your/model.bin|/home/ubuntu/dynamo/fasttext-reasoning-classifier/reasoning.bin|' /tmp/my-semrouter.yaml

# 2. Build Dynamo with semantic router
cd lib/bindings/python
maturin develop --features semrouter-fasttext

# 3. Start Dynamo
export DYN_SEMROUTER_CONFIG=/tmp/my-semrouter.yaml
export RUST_LOG=info,sem_router=debug
python -m dynamo.frontend --http-port 8999

# 4. Test it (in another terminal)
curl -s http://localhost:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "router", "messages": [{"role": "user", "content": "Prove sqrt(2) is irrational"}]}' \
  | jq -r '.model'
# Expected output: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
```

**Example Config:**
- `semrouter_configs/example_fasttext_config_v2.yaml` - Binary reasoning detection with fastText

## Detailed Setup: Dynamo Integration

### 1. Train a fastText Model

You'll need a binary fastText model trained on your data. Here's how to train one for reasoning detection:

```bash
# Install dependencies
pip install fasttext datasets pandas

# Set HuggingFace token (required for dataset access)
export HF_TOKEN=<your_huggingface_token>

# Prepare dataset (uses NVIDIA Nemotron dataset)
python3 scripts/prepare_nemotron_dataset.py \
    --samples 10000 \
    --output data/nemotron_10k

# Train the classifier
python3 scripts/train_fasttext_nemotron.py \
    --input data/nemotron_10k \
    --output fasttext-reasoning-classifier

# Your model is now at: fasttext-reasoning-classifier/reasoning.bin
```

### 2. Create Configuration

Use the provided example config as a starting point:

```bash
# Copy example config
cp semrouter_configs/example_fasttext_config_v2.yaml my-config.yaml

# Edit and update the model_path
sed -i 's|/path/to/your/model.bin|/home/ubuntu/dynamo/fasttext-reasoning-classifier/reasoning.bin|' my-config.yaml
```

Or create your own `semrouter-config.yaml`:

```yaml
semrouter:
  enabled: true
  mode: auto
  model_alias: "router"

  classifier:
    kind: fasttext
    model_path: "/home/ubuntu/dynamo/fasttext-reasoning-classifier/reasoning.bin"

  classes:
    - label: reasoning
      threshold: 0.55
      action: { type: override, model: "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" }

    - label: non-reasoning
      threshold: 0.55
      action: { type: override, model: "meta-llama/Meta-Llama-3.1-8B-Instruct" }

  fallback:
    type: override
    model: "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

### 3. Build Dynamo with Semantic Router

```bash
# Build with the semrouter-fasttext feature
cd /path/to/dynamo
cargo build --release --features semrouter-fasttext

# Or for Python bindings:
cd lib/bindings/python
maturin develop --release --features semrouter-fasttext
```

### 4. Run Dynamo with Semantic Router

```bash
# Set config path
export DYN_SEMROUTER_CONFIG=/path/to/semrouter-config.yaml

# Optional: enable detailed logging
export RUST_LOG=info,sem_router=debug

# Start Dynamo
python -m dynamo.frontend --http-port 8999
```

### 5. Test It Out

```bash
# Send a reasoning query
curl -s http://localhost:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "router",
    "messages": [{"role": "user", "content": "What is the derivative of x^2 + 2x?"}],
    "max_tokens": 100
  }' | jq -r '.model'

# Should return: deepseek-ai/DeepSeek-R1-Distill-Llama-8B

# Send a general query
curl -s http://localhost:8999/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "router",
    "messages": [{"role": "user", "content": "What is the weather like today?"}],
    "max_tokens": 100
  }' | jq -r '.model'

# Should return: meta-llama/Meta-Llama-3.1-8B-Instruct
```

## General Integration (Any Tower/Axum App)

You can use `sem-router` in any Rust application that uses Tower/Axum:

### Add to Your `Cargo.toml`

```toml
[dependencies]
sem-router = { path = "path/to/lib/sem-router", features = ["clf-fasttext"] }
axum = "0.8"
tokio = { version = "1", features = ["full"] }
tower = "0.5"
```

### Use in Your Application

```rust
use sem_router::maybe_semrouter_layer_from_env;
use axum::{Router, routing::post, response::IntoResponse, Json};
use serde_json::{json, Value};

#[tokio::main]
async fn main() {
    // Your regular Axum router
    let mut app = Router::new()
        .route("/v1/chat/completions", post(chat_handler));

    // Add semantic router layer (reads DYN_SEMROUTER_CONFIG env var)
    if let Some(layer) = maybe_semrouter_layer_from_env() {
        println!("Semantic router enabled!");
        app = app.layer(layer);
    }

    // Start server
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

async fn chat_handler(Json(payload): Json<Value>) -> impl IntoResponse {
    // The semantic router middleware has already modified payload["model"]
    // based on the classification result
    let model = payload.get("model").and_then(|v| v.as_str()).unwrap_or("unknown");
    println!("Handling request for model: {}", model);

    Json(json!({
        "model": model,
        "choices": [{
            "message": {
                "role": "assistant",
                "content": "Response from the selected model"
            }
        }]
    }))
}
```

## Configuration Reference

### Top-Level Fields

```yaml
semrouter:
  enabled: bool              # Enable/disable routing (default: false)
  mode: string               # auto | force | shadow | off (default: auto)
  model_alias: string        # Model name that triggers routing (default: "router")
  default_policy: string     # never_when_explicit | allow_when_opt_in | always

  classifier:
    kind: fasttext
    model_path: string       # Path to .bin fastText model file

  classes: array             # Routing rules for each class
    - label: string          # Classification label to match
      threshold: float       # Minimum score (0.0-1.0) to trigger
      action: object         # What to do

  fallback: object           # Action when no class crosses threshold
```

### Actions

```yaml
# Pass through without modification
action: { type: passthrough }

# Override model field
action: { type: override, model: "model-id" }

# Shadow/duplicate request (not fully implemented)
action: { type: shadow, route_to: "http://other-server" }

# Reject with error
action: { type: reject, reason: "This query type is not supported" }
```

### Modes

- **auto**: Route only when `model` field matches `model_alias`
- **force**: Always route, even if an explicit model is specified
- **shadow**: Classify and log but don't modify the request (metrics only)
- **off**: Disable routing

### Headers

- `X-Dynamo-Routing`: Override mode per-request (`auto`, `force`, `shadow`, `off`)
- `X-Dynamo-Routed-Model`: Added by middleware when routing occurs (informational)

## Training fastText Models

### Dataset Requirements

fastText models need labeled text data. For reasoning detection, use:

- **NVIDIA Nemotron Post-Training Dataset**: Contains reasoning and non-reasoning examples
- **LMSYS Chat-1M**: Referenced by Nemotron for additional examples

### Training Pipeline

1. **Prepare Data** (`scripts/prepare_nemotron_dataset.py`):
   - Downloads and merges datasets
   - Extracts text from conversations
   - Creates fastText-format `.train` and `.valid` files
   - Format: `__label__<class> <text>`

2. **Train Model** (`scripts/train_fasttext_nemotron.py`):
   - Trains fastText model with optimized hyperparameters
   - Validates on held-out set
   - Saves `.bin` model file

3. **Evaluate**:
```bash
   fasttext test model.bin data/test.txt
   ```

### Multi-Class Models

To create a multi-class classifier (e.g., reasoning, coding, math):

1. Prepare a dataset with multiple labels:
   ```
   __label__reasoning Prove that sqrt(2) is irrational
   __label__coding Write a Python function to sort a list
   __label__math Solve for x: 2x + 3 = 7
   ```

2. Train with fastText:
```bash
   fasttext supervised -input train.txt -output model -epoch 25 -wordNgrams 2
   ```

3. Configure sem-router with all classes:
   ```yaml
   classes:
     - label: reasoning
       threshold: 0.5
       action: { type: override, model: "reasoning-model" }
     - label: coding
       threshold: 0.5
       action: { type: override, model: "code-model" }
     - label: math
       threshold: 0.5
       action: { type: override, model: "math-model" }
   ```

## Building the Crate

### Standalone (for development/testing)

```bash
# Build the crate
cargo build -p sem-router --features clf-fasttext

# Run tests
cargo test -p sem-router

# Build documentation
cargo doc -p sem-router --open
```

### With Dynamo

```bash
# Without semantic router (default)
cargo build -p dynamo-llm

# With semantic router + fastText
cargo build -p dynamo-llm --features semrouter-fasttext
```

## Architecture

### Request Flow

```
┌─────────────────────────────────────────┐
│  HTTP Request                            │
│  POST /v1/chat/completions               │
│  { "model": "router", "messages": [...] }│
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  SemRouterLayer (Tower Middleware)       │
│  1. Buffer request body                  │
│  2. Parse JSON                           │
│  3. Extract text from messages           │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  FastText Classifier                     │
│  - Predict all class scores              │
│  - Return sorted [(label, score), ...]  │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Decision Engine                         │
│  - Match scores against thresholds       │
│  - Apply routing policy                  │
│  - Choose action                         │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Action: Override                        │
│  - Rewrite model field in JSON           │
│  - Add X-Dynamo-Routed-Model header      │
│  - Rebuild request body                  │
└─────────────────┬────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  Pass to Inner Service                   │
│  (Your handler receives modified request)│
└──────────────────────────────────────────┘
```

### Module Structure

```
lib/sem-router/
├── Cargo.toml                 # Crate manifest
└── src/
    ├── lib.rs                 # Public API & helpers
    ├── layer.rs               # Tower Layer & Service impl
    ├── decision.rs            # Decision types & logic
    ├── config.rs              # YAML configuration
    ├── ctx.rs                 # Request context extraction
    └── classifier/
        ├── mod.rs             # Classifier trait
        └── fasttext.rs        # FastText implementation
```

## Performance

### Latency

- **Classification**: < 1ms on CPU (300-600 microseconds typical)
- **Total overhead**: ~1-2ms per request (includes JSON parsing, body buffering)
- **Memory**: ~50-100MB for fastText model (depends on vocabulary size)

### Throughput

The middleware adds minimal overhead. Benchmark on a 4-core CPU:

- **Without routing**: ~10,000 requests/sec
- **With routing**: ~8,000 requests/sec
- **Overhead**: ~20% (mostly JSON parsing/serialization)

### Optimization Tips

1. **Use `spawn_blocking`** for classification on high-load systems
2. **Reduce model size** by limiting vocabulary (`-maxn`, `-minn` parameters)
3. **Cache classifications** if the same requests are common
4. **Pre-parse JSON** if you control the client (send pre-parsed data)

## Troubleshooting

### Router Not Applied

**Symptom**: Requests pass through unchanged

**Checks:**
1. Is `DYN_SEMROUTER_CONFIG` set and pointing to valid YAML?
2. Is `enabled: true` in config?
3. Does the request's `model` field match `model_alias`?
4. Check logs for initialization messages

### Classification Errors

**Symptom**: "classifier error; falling back to passthrough"

**Causes:**
- fastText model file not found
- Model file corrupted
- Text extraction failed

**Fix:**
- Verify `model_path` is correct
- Check file permissions
- Ensure model has expected labels

### Build Errors

**Symptom**: "FasttextClassifier requires 'clf-fasttext' feature"

**Fix:**
- Add `--features semrouter-fasttext` to build command
- Or add `clf-fasttext` to `default` features in your app's `Cargo.toml`

## Examples

### Example 1: Binary Reasoning Detection

```yaml
semrouter:
  enabled: true
  classifier:
    kind: fasttext
    model_path: "/models/reasoning.bin"
  classes:
    - label: reasoning
      threshold: 0.55
      action: { type: override, model: "deepseek-r1" }
    - label: non-reasoning
      threshold: 0.55
      action: { type: override, model: "llama-3" }
```

### Example 2: Multi-Domain Routing

```yaml
semrouter:
  enabled: true
  classifier:
    kind: fasttext
    model_path: "/models/domain-classifier.bin"
  classes:
    - label: medical
      threshold: 0.6
      action: { type: override, model: "biobert-large" }
    - label: legal
      threshold: 0.6
      action: { type: override, model: "legal-bert" }
    - label: code
      threshold: 0.5
      action: { type: override, model: "codellama-34b" }
    - label: general
      threshold: 0.3
      action: { type: override, model: "gpt-4" }
  fallback:
    type: override
    model: "gpt-3.5-turbo"
```

### Example 3: Shadow Mode (Testing)

```yaml
semrouter:
  enabled: true
  mode: shadow  # Don't modify requests, just log decisions
  classifier:
    kind: fasttext
    model_path: "/models/new-classifier.bin"
  classes:
    - label: reasoning
      threshold: 0.5
      action: { type: override, model: "new-model" }
```

## Contributing

This crate is part of the Dynamo project but designed to be reusable. To contribute:

1. Keep it standalone - no Dynamo-specific dependencies
2. Add tests for new features
3. Update this documentation
4. Follow Rust API guidelines

## License

Apache-2.0 - See LICENSE file

## Support

For issues or questions:
- Check this documentation
- Review the `lib/sem-router/src/` source code
- Open an issue on the Dynamo GitHub repository
