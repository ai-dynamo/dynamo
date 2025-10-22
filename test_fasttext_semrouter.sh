#!/bin/bash
# Quick test script for fastText semantic router

set -e

echo "=== Testing fastText Semantic Router ==="
echo ""

# Step 1: Check if we have a fastText model
if [ -z "$FASTTEXT_MODEL_PATH" ]; then
    echo "❌ Error: FASTTEXT_MODEL_PATH environment variable not set"
    echo "   Please set it to your fastText .bin model file:"
    echo "   export FASTTEXT_MODEL_PATH=/path/to/your/model.bin"
    exit 1
fi

if [ ! -f "$FASTTEXT_MODEL_PATH" ]; then
    echo "❌ Error: Model file not found: $FASTTEXT_MODEL_PATH"
    exit 1
fi

echo "✓ Found fastText model: $FASTTEXT_MODEL_PATH"

# Step 2: Create test config
cat > /tmp/test_semrouter_config.yaml <<EOF
semrouter:
  enabled: true
  mode: auto
  model_alias: "router"

  classifier:
    kind: fasttext
    model_path: "$FASTTEXT_MODEL_PATH"

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
EOF

echo "✓ Created test config: /tmp/test_semrouter_config.yaml"
echo ""

# Step 3: Set environment and build
export DYN_SEMROUTER_CONFIG=/tmp/test_semrouter_config.yaml
export RUST_LOG=info,dynamo_llm::semrouter=debug

echo "=== Building with clf-fasttext feature ==="
cd /home/ubuntu/dynamo
cargo build --lib --manifest-path lib/llm/Cargo.toml --features clf-fasttext 2>&1 | tail -n 30

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
    echo "=== Next Steps ==="
    echo "1. Start your Dynamo server with the fastText feature:"
    echo "   export DYN_SEMROUTER_CONFIG=/tmp/test_semrouter_config.yaml"
    echo "   python -m dynamo.frontend --http-port 8999"
    echo ""
    echo "2. Send a test request:"
    echo '   curl -X POST http://localhost:8787/v1/chat/completions \'
    echo '     -H "Content-Type: application/json" \'
    echo '     -d '"'"'{"model": "router", "messages": [{"role": "user", "content": "Prove that the square root of 2 is irrational"}]}'"'"
    echo ""
    echo "3. Watch the logs for classification results and routing decisions"
else
    echo "❌ Build failed"
    exit 1
fi

