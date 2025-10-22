#!/bin/bash
# Test semantic router with MockClassifier
set -e

echo "========================================="
echo "Semantic Router Test with MockClassifier"
echo "========================================="
echo ""

# Configuration
export SEMROUTER_ENABLED=true
export SEMROUTER_CONFIG=./semantic-router-binary.yaml

echo "✓ Environment configured:"
echo "  SEMROUTER_ENABLED=$SEMROUTER_ENABLED"
echo "  SEMROUTER_CONFIG=$SEMROUTER_CONFIG"
echo ""

# Check if models are running
echo "Checking if backend models are available..."
if ! curl -s http://localhost:8100/health > /dev/null 2>&1; then
    echo "⚠ WARNING: No model found at port 8100 (general model)"
    echo "  Start with: CUDA_VISIBLE_DEVICES=0 python -m dynamo.vllm --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8100"
fi

if ! curl -s http://localhost:8101/health > /dev/null 2>&1; then
    echo "⚠ WARNING: No model found at port 8101 (reasoning model)"
    echo "  Start with: CUDA_VISIBLE_DEVICES=1 python -m dynamo.vllm --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B --port 8101"
fi
echo ""

echo "Starting frontend on port 8999..."
echo "(Press Ctrl+C to stop)"
echo ""
echo "Test commands (run in another terminal):"
echo ""
echo "# Test reasoning query:"
echo "curl -s localhost:8999/v1/chat/completions -H 'Content-Type: application/json' -H 'X-Dynamo-Routing: auto' -d '{\"model\": \"router\", \"messages\": [{\"role\": \"user\", \"content\": \"Prove that sqrt(2) is irrational\"}], \"max_tokens\": 10}' | jq '.model'"
echo ""
echo "# Test general query:"
echo "curl -s localhost:8999/v1/chat/completions -H 'Content-Type: application/json' -H 'X-Dynamo-Routing: auto' -d '{\"model\": \"router\", \"messages\": [{\"role\": \"user\", \"content\": \"What is the capital of France?\"}], \"max_tokens\": 10}' | jq '.model'"
echo ""
echo "# Check metrics:"
echo "curl -s localhost:8999/metrics | grep semrouter"
echo ""
echo "========================================="
echo ""

# Start frontend
python -m dynamo.frontend --http-port 8999

