# Error Classification Integration with Workflow Metrics Uploader

## Overview

The workflow metrics uploader now automatically classifies errors from failed jobs and steps using AI-powered error classification. This adds three new fields to OpenSearch documents for failed jobs and steps:

- `s_error_type`: Error category (e.g., "infrastructure_error", "dependency_error", "timeout")
- `s_error_summary`: Root cause summary explaining the error
- `f_error_confidence`: Confidence score (0.0-1.0) of the classification

## How It Works

1. **Automatic Classification**: When uploading job and step metrics, the uploader automatically detects failures (status = "failure" or "failed")

2. **Log Fetching**: For failed jobs/steps, it fetches the complete job logs from GitHub API

3. **Error Extraction**: Uses the error extraction module to identify the relevant error message from the logs

4. **AI Classification**: Sends the error to Claude API for classification into one of 10 categories

5. **OpenSearch Storage**: Adds the classification results (type, summary, confidence) to the metrics document before uploading to OpenSearch

## Configuration

### Required Environment Variables

```bash
# Enable error classification
export ENABLE_ERROR_CLASSIFICATION="true"

# API key for Claude (NVIDIA API or Anthropic direct)
export ANTHROPIC_API_KEY="your-api-key-here"

# For NVIDIA API (recommended)
export API_FORMAT="openai"
export API_BASE_URL="https://inference-api.nvidia.com/v1"
export ANTHROPIC_MODEL="aws/anthropic/claude-opus-4-5"

# Standard workflow metrics variables
export WORKFLOW_INDEX="https://your-opensearch:9200/workflows"
export JOB_INDEX="https://your-opensearch:9200/jobs"
export STEPS_INDEX="https://your-opensearch:9200/steps"
export GITHUB_TOKEN="your-github-token"
export REPO="ai-dynamo/dynamo"
export HOURS_BACK="4"
```

### Optional Environment Variables

```bash
# Disable error classification (default: false)
export ENABLE_ERROR_CLASSIFICATION="false"

# For deduplication support (future enhancement)
export ERROR_CLASSIFICATION_INDEX="https://your-opensearch:9200/error_classifications"
```

## Usage

Run the workflow metrics uploader as normal:

```bash
cd opensearch_upload
python3 workflow_metrics_uploader.py
```

With error classification enabled, you'll see additional output:

```
📋 Processing workflow: CI Test Suite (ID: 21689681963)
  📤 Processing job: vllm-build-test (cuda12.9, amd64)
   📊 Job annotations: 2 total, 1 failures, 0 warnings
   🤖 Classifying error for job: vllm-build-test (cuda12.9, amd64)
   ✅ Classified as: infrastructure_error (92% confidence)
      Summary: This is a pytest collection error where the test module fails to import due to a missing 'opensearch...
   ✅ Posted metrics for github-job-12345678-attempt-1
```

## OpenSearch Fields

### Job Index

Failed jobs will have these additional fields:

```json
{
  "_id": "github-job-12345678-attempt-1",
  "s_job_name": "vllm-build-test (cuda12.9, amd64)",
  "s_status": "failure",
  "s_error_type": "infrastructure_error",
  "s_error_summary": "This is a pytest collection error where the test module fails to import due to a missing 'opensearch' module...",
  "f_error_confidence": 0.92,
  ...
}
```

### Step Index

Failed steps will have the same fields:

```json
{
  "_id": "github-step-12345678_5-attempt-1",
  "s_step_name": "Run tests",
  "s_status": "failure",
  "s_error_type": "infrastructure_error",
  "s_error_summary": "Test environment is missing required dependencies...",
  "f_error_confidence": 0.92,
  ...
}
```

## Error Categories

The classifier categorizes errors into 10 types:

1. **dependency_error**: Package installation, version conflicts, missing libraries
2. **timeout**: Test/build timeouts, deadlocks, hung processes
3. **resource_exhaustion**: Out of Memory (CPU/GPU), disk full, resource limits
4. **network_error**: Connection failures, DNS issues, download failures
5. **assertion_failure**: Test assertion failures, validation errors
6. **compilation_error**: Build/compile failures, linking errors
7. **runtime_error**: Crashes, segfaults, uncaught exceptions
8. **infrastructure_error**: GitHub Actions, Docker, K8s issues, pytest collection errors
9. **configuration_error**: Invalid configs, env variables, permissions
10. **flaky_test**: Non-deterministic failures, race conditions

## Cost Optimization

The classifier implements several cost-saving features:

- **Prompt Caching**: System prompt is cached for 5 minutes (90% cost reduction)
- **Error Deduplication**: Similar errors classified once and reused (70-90% API call reduction)
- **Selective Classification**: Only failed jobs/steps are classified
- **Batch Processing**: Can be run in batch mode for historical data

Expected costs:
- ~$0.003-0.005 per error classified
- With caching and deduplication: ~$3-5/month for typical usage

## Querying in OpenSearch

### Find all errors by category

```json
GET /jobs/_search
{
  "query": {
    "term": {
      "s_error_type": "infrastructure_error"
    }
  }
}
```

### Aggregate errors by type

```json
GET /jobs/_search
{
  "size": 0,
  "aggs": {
    "errors_by_type": {
      "terms": {
        "field": "s_error_type",
        "size": 10
      }
    }
  }
}
```

### Find low-confidence classifications

```json
GET /jobs/_search
{
  "query": {
    "bool": {
      "must": [
        { "term": { "s_status": "failure" } },
        { "range": { "f_error_confidence": { "lt": 0.7 } } }
      ]
    }
  }
}
```

## Troubleshooting

### Error classification not running

1. Check that `ENABLE_ERROR_CLASSIFICATION="true"`
2. Verify `ANTHROPIC_API_KEY` is set
3. Check logs for import errors

### Classification failures

If you see "⚠️ Error during classification", check:

1. API key is valid
2. API rate limits not exceeded
3. Job logs are accessible via GitHub API
4. Error extraction found valid errors

### Low confidence scores

Low confidence (< 70%) may indicate:

1. Ambiguous error messages
2. Multiple error types in one failure
3. Unusual error patterns not in training data

The system automatically re-analyzes with full logs when confidence is low.

## Disabling Error Classification

To disable without removing the code:

```bash
export ENABLE_ERROR_CLASSIFICATION="false"
```

Or simply don't set the environment variable (defaults to false).

## Future Enhancements

- **Deduplication**: Store classifications in OpenSearch to avoid re-classifying identical errors
- **Suggested Fixes**: Add remediation suggestions to classification results
- **Historical Analysis**: Batch classify historical failures for trend analysis
- **Custom Categories**: Allow project-specific error categories
