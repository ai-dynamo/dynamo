# AI-Powered Error Classification System

Automatically categorizes common errors from CI/CD workflows using Claude API and exports classifications to OpenSearch for trend analysis.

## Overview

This system classifies test failures, Docker build errors, Rust tests, and infrastructure errors into 2 broad categories for consistent classification:

1. **infrastructure_error** - Infrastructure/platform issues: network problems, runner/node issues, platform failures, resource limits (OOM, disk full)
2. **code_error** - Code/build/test issues: build failures, test failures, runtime errors, dependency issues, configuration errors

## Architecture

### Components

- **classifier.py** - Core classification orchestration
- **claude_client.py** - Claude API wrapper with caching and rate limiting
- **deduplicator.py** - Error normalization and deduplication
- **error_extractor.py** - Extract errors from JUnit XML, BuildKit logs, annotations
- **config.py** - Configuration management
- **prompts.py** - Claude API prompt templates
- **opensearch_schema.py** - OpenSearch index schema

### Processing Modes

1. **Real-time (CI)** - Classifies critical errors during workflow execution
   - Infrastructure and build errors only
   - Individual test failures deferred to batch
   - Minimal CI overhead (<30s)

2. **Batch (Cronjob)** - Processes all errors periodically
   - Runs every hour via cronjob
   - Full deduplication and classification
   - Cost-optimized with caching

## Installation

```bash
cd opensearch_upload
pip install -r requirements.txt
```

## Configuration

Set environment variables:

```bash
# Required
export ANTHROPIC_API_KEY="your-api-key"
export OPENSEARCH_URL="https://your-opensearch-instance"
export ERROR_CLASSIFICATION_INDEX="error_classifications"

# Optional
export OPENSEARCH_USERNAME="username"
export OPENSEARCH_PASSWORD="password"
export ENABLE_ERROR_CLASSIFICATION="true"
export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"
export MAX_ERROR_LENGTH="10000"
export BATCH_SIZE="10"
```

## Usage

### Phase 0: Validate Categories

Analyze recent errors to validate the 10 categories:

```bash
cd opensearch_upload
export GITHUB_TOKEN=<token>
export REPO=ai-dynamo/dynamo
python3 analyze_recent_errors.py --hours 48 --output error_analysis_report.md
```

### Batch Processing

Classify all unprocessed errors:

```bash
cd opensearch_upload
python3 upload_error_classifications.py --hours 24
```

### CI Integration

Add to workflow YAML:

```yaml
- name: Classify Critical Errors on Failure
  if: failure() && env.ENABLE_ERROR_CLASSIFICATION == 'true'
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
    ERROR_CLASSIFICATION_INDEX: ${{ secrets.ERROR_CLASSIFICATION_INDEX }}
  run: python3 .github/workflows/classify_errors_on_failure.py
```

### Cronjob Setup

Add to crontab on cronjob server:

```bash
# Run every hour
0 * * * * cd /path/to/dynamo/opensearch_upload && \
    /path/to/python3 upload_error_classifications.py --hours 1 >> /var/log/error_classification.log 2>&1
```

## Cost Optimization

### Strategies

1. **Prompt Caching** - System prompt cached across requests (~90% cost reduction)
2. **Error Deduplication** - Classify only unique errors (70-90% fewer API calls)
3. **Selective Real-Time** - Only infrastructure/build errors in CI
4. **Classification Caching** - Reuse classifications for identical errors
5. **Batch Processing** - Efficient API usage for large volumes

### Expected Costs

- Real-time: ~10-50 errors/day × $0.003 = $0.03-0.15/day
- Batch: ~200-500 unique errors/day × $0.0003 (cached) = $0.06-0.15/day
- **Total: ~$3-5/month**

## OpenSearch Schema

Classifications are stored with these fields:

```
s_error_id          - Unique ID
s_error_hash        - Deduplication hash
s_workflow_id       - Link to workflow
s_job_id            - Link to job
s_test_name         - Test name (if applicable)
s_error_source      - pytest|buildkit|annotation
s_framework         - vllm|sglang|trtllm|rust
s_primary_category  - Classification result
f_confidence_score  - 0-1 confidence
s_root_cause_summary - AI explanation
l_occurrence_count  - How many times seen
l_prompt_tokens     - API usage
l_cached_tokens     - Cache hits
@timestamp          - When classified
```

## Development

### Running Tests

```bash
cd tests/error_classification
pytest test_classifier.py
pytest test_deduplicator.py
pytest test_error_extractor.py
```

### Adding New Categories

1. Update `ERROR_CATEGORIES` in `config.py`
2. Add category definition to `SYSTEM_PROMPT_WITH_CACHING` in `prompts.py`
3. Update validation patterns in `analyze_recent_errors.py`
4. Test with real errors

## Troubleshooting

### "ANTHROPIC_API_KEY environment variable is required"

Set the API key:
```bash
export ANTHROPIC_API_KEY="your-key"
```

### "opensearch-py not installed"

Install dependencies:
```bash
pip install -r requirements.txt
```

### Classifications not appearing in OpenSearch

1. Check `ERROR_CLASSIFICATION_INDEX` is set correctly
2. Verify OpenSearch credentials
3. Check index was created: `create_index_if_not_exists()`
4. Review logs for upload errors

### High API costs

1. Verify prompt caching is working (check `l_cached_tokens` > 0)
2. Confirm deduplication is active (check `b_is_duplicate` field)
3. Review `should_classify_realtime()` logic
4. Adjust `CLASSIFY_REALTIME_THRESHOLD` to "infrastructure" or "none"

## License

See repository root LICENSE file.
