# Using NVIDIA API with Error Classification System

The error classification system now supports NVIDIA's OpenAI-compatible API for accessing Claude models.

## Configuration

Set these environment variables to use NVIDIA API:

```bash
# NVIDIA API key
export ANTHROPIC_API_KEY="$(cat ~/.claude2)"

# API format (openai for NVIDIA, anthropic for direct Anthropic API)
export API_FORMAT="openai"

# NVIDIA API base URL
export API_BASE_URL="https://inference-api.nvidia.com/v1"

# Claude model available through NVIDIA
export ANTHROPIC_MODEL="aws/anthropic/claude-opus-4-5"

# Enable error classification
export ENABLE_ERROR_CLASSIFICATION="true"
```

## Quick Test

Test that NVIDIA API works:

```bash
cd opensearch_upload
python3 test_nvidia_integration.py
```

Expected output:
```
✅ SUCCESS! NVIDIA API Classification Works!

Category: resource_exhaustion
Confidence: 98.00%
Root Cause: This is a classic CUDA out of memory (OOM) error...
```

## Use in GitHub Workflows

Add to your workflow YAML:

```yaml
jobs:
  your-test-job:
    env:
      # NVIDIA API configuration
      ANTHROPIC_API_KEY: ${{ secrets.NVIDIA_API_KEY }}
      API_FORMAT: "openai"
      API_BASE_URL: "https://inference-api.nvidia.com/v1"
      ANTHROPIC_MODEL: "aws/anthropic/claude-opus-4-5"
      ENABLE_ERROR_CLASSIFICATION: "true"

    steps:
      # ... your test steps ...

      - name: Classify Errors on Failure
        if: failure()
        run: |
          python3 .github/workflows/classify_errors_on_failure.py
```

## Supported Models via NVIDIA

Check NVIDIA's API catalog for available Claude models:
- `aws/anthropic/claude-opus-4-5` - Claude Opus 4.5 (most capable)
- `aws/anthropic/claude-sonnet-4-5` - Claude Sonnet 4.5 (balanced)
- Other models may be available through your NVIDIA account

## Cost

NVIDIA API pricing may differ from direct Anthropic API. Check your NVIDIA account for pricing details.

## Advantages of NVIDIA API

- ✅ Access Claude through your NVIDIA infrastructure
- ✅ Unified billing with other NVIDIA services
- ✅ Enterprise support and SLAs
- ✅ Compatible with existing NVIDIA API keys

## Switching Between APIs

### Use NVIDIA API:
```bash
export API_FORMAT="openai"
export API_BASE_URL="https://inference-api.nvidia.com/v1"
export ANTHROPIC_MODEL="aws/anthropic/claude-opus-4-5"
```

### Use Direct Anthropic API:
```bash
export API_FORMAT="anthropic"
unset API_BASE_URL
export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"
```

## Troubleshooting

### Authentication Error
```
Error code: 401 - Authentication failed
```
- Verify your API key: `cat ~/.claude2`
- Check that the key has access to Claude models
- Ensure the key is not expired

### Model Not Found
```
Error: Invalid model
```
- Check available models in your NVIDIA account
- Verify the model name format: `aws/anthropic/claude-opus-4-5`

### Connection Error
```
ConnectionError: Failed to connect
```
- Verify API base URL: `https://inference-api.nvidia.com/v1`
- Check network connectivity to NVIDIA API
- Verify firewall/proxy settings

## Additional Resources

- NVIDIA AI Documentation: https://docs.nvidia.com/ai/
- Anthropic Claude Documentation: https://docs.anthropic.com/
- Error Classification System README: `error_classification/README.md`
