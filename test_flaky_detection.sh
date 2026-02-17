#!/bin/bash
# Test script to run flaky detection against a specific workflow run

set -e

WORKFLOW_RUN_ID="${1:-21689570943}"
REPO="${2:-ai-dynamo/dynamo}"

echo "Testing flaky detection with workflow run: $WORKFLOW_RUN_ID"
echo "Repository: $REPO"
echo ""

# Set up environment variables
export OPENSEARCH_ENDPOINT="https://gpuwa.nvidia.com/opensearch/df-swdl-triton-ops-gh-tests*"
export SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-}"
export SLACK_OPS_GROUP_ID="${SLACK_OPS_GROUP_ID:-}"
export GITHUB_RUN_ID="$WORKFLOW_RUN_ID"
export GITHUB_REPOSITORY="$REPO"
export WORKFLOW_NAME="Test Flaky Detection"
export FLAKY_THRESHOLD="0.80"
export LOOKBACK_DAYS="7"

# Create test-results directory
mkdir -p test-results
cd test-results

echo "Downloading test artifacts from workflow run..."
echo "Note: You need gh CLI and appropriate permissions"
echo ""

# Download artifacts using gh CLI
if command -v gh &> /dev/null; then
    # List available artifacts
    echo "Available artifacts:"
    gh run view "$WORKFLOW_RUN_ID" --repo "$REPO" --json artifacts --jq '.artifacts[] | select(.name | startswith("test-results-")) | .name'
    echo ""

    # Download test result artifacts
    gh run download "$WORKFLOW_RUN_ID" --repo "$REPO" --pattern "test-results-*" || {
        echo "Warning: Failed to download some artifacts, continuing with available artifacts..."
    }

    echo ""
    echo "Downloaded artifacts:"
    ls -la
    echo ""
else
    echo "ERROR: gh CLI not found. Install with: sudo apt-get install gh"
    exit 1
fi

cd ..

# Set up Python venv
echo "Setting up Python environment..."
python3 -m venv venv
source venv/bin/activate
pip install -q requests

# Run the detection script
echo ""
echo "Running flaky test detection..."
echo "================================"
python3 .github/workflows/detect_flaky_tests.py

echo ""
echo "Test complete!"
