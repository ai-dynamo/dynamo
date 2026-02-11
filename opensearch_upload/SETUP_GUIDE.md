# Error Classification System - Setup Guide

Complete setup guide for the AI-powered error classification system.

## Prerequisites

- Python 3.8+
- Anthropic API key
- OpenSearch instance (for storing classifications)
- GitHub token (for analyzing historical errors)

## Installation Steps

### 1. Install Dependencies

```bash
cd opensearch_upload
pip install -r requirements.txt
```

This installs:
- `anthropic>=0.40.0` - Claude API client
- `opensearch-py>=2.3.0` - OpenSearch client
- `requests>=2.28.0` - HTTP library

### 2. Set Environment Variables

Create a `.env` file or export variables:

```bash
# Required for classification
export ANTHROPIC_API_KEY="sk-ant-..."

# Required for OpenSearch upload
export OPENSEARCH_URL="https://your-opensearch-instance:9200"
export ERROR_CLASSIFICATION_INDEX="error_classifications"

# Optional OpenSearch authentication
export OPENSEARCH_USERNAME="admin"
export OPENSEARCH_PASSWORD="password"

# Required for GitHub API (error analysis)
export GITHUB_TOKEN="ghp_..."
export REPO="ai-dynamo/dynamo"

# Optional - tune behavior
export ENABLE_ERROR_CLASSIFICATION="true"
export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"
export MAX_ERROR_LENGTH="10000"
export BATCH_SIZE="10"
export MAX_RPM="50"
export CLASSIFICATION_CACHE_TTL_HOURS="168"
export MIN_CONFIDENCE_FOR_REUSE="0.8"
export CLASSIFY_REALTIME_THRESHOLD="infrastructure"
```

### 3. Create OpenSearch Index

The index is created automatically on first run, but you can create it manually:

```python
from opensearchpy import OpenSearch
from opensearch_upload.error_classification import create_index_if_not_exists

client = OpenSearch(
    hosts=["https://your-opensearch-instance:9200"],
    http_auth=("username", "password"),
    use_ssl=True,
    verify_certs=True,
)

create_index_if_not_exists(client, "error_classifications")
```

### 4. Add GitHub Secrets

For CI integration, add these secrets to your GitHub repository:

1. Go to Settings → Secrets and variables → Actions
2. Add secrets:
   - `ANTHROPIC_API_KEY` - Your Claude API key
   - `ERROR_CLASSIFICATION_INDEX` - OpenSearch index URL
   - `OPENSEARCH_URL` - OpenSearch instance URL
   - `OPENSEARCH_USERNAME` (optional)
   - `OPENSEARCH_PASSWORD` (optional)

## Phase 0: Validate Categories

Before deploying, validate that the 10 categories match your error patterns:

```bash
cd opensearch_upload
export GITHUB_TOKEN="ghp_..."
export REPO="ai-dynamo/dynamo"
python3 analyze_recent_errors.py --hours 48 --output error_analysis_report.md
```

Review `error_analysis_report.md`:
- Check category coverage (should be >90%)
- Review example errors for each category
- Identify any unclassified error patterns
- Adjust categories if needed

## Deployment

### Option A: Batch Processing Only (Recommended First)

Start with batch processing to validate the system:

1. **Test locally**:
```bash
cd opensearch_upload
python3 upload_error_classifications.py --hours 168  # Last week
```

2. **Deploy to cronjob server**:
```bash
# SSH to cronjob server
scp -r opensearch_upload user@cronjob-server:/path/to/dynamo/

# Add to crontab
crontab -e

# Add line (runs every hour):
0 * * * * cd /path/to/dynamo/opensearch_upload && \
    /usr/bin/python3 upload_error_classifications.py --hours 1 \
    >> /var/log/error_classification.log 2>&1
```

3. **Monitor logs**:
```bash
tail -f /var/log/error_classification.log
```

### Option B: Full Deployment (Batch + Real-time)

Once batch processing is working, add CI integration:

1. **Update workflow YAMLs**:

Edit `.github/workflows/ci-test-suite.yml` and other test workflows:

```yaml
env:
  # Add to env section at top
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  ERROR_CLASSIFICATION_INDEX: ${{ secrets.ERROR_CLASSIFICATION_INDEX }}
  ENABLE_ERROR_CLASSIFICATION: 'true'

jobs:
  test:
    steps:
      # ... existing steps ...

      # Add after test steps, only runs on failure
      - name: Classify Critical Errors on Failure
        if: failure() && env.ENABLE_ERROR_CLASSIFICATION == 'true'
        run: |
          python3 .github/workflows/classify_errors_on_failure.py
```

2. **Test with intentional failure**:
   - Create a PR with a failing test
   - Verify classification runs (check workflow logs)
   - Check OpenSearch for classification records

3. **Gradually enable**:
   - Start with one workflow
   - Monitor API costs and performance
   - Roll out to other workflows

## Verification

### Check Classification is Working

1. **Query OpenSearch**:
```bash
curl -X GET "https://your-opensearch-instance:9200/error_classifications/_search?pretty" \
  -u username:password \
  -H 'Content-Type: application/json' \
  -d '{
    "query": {"match_all": {}},
    "size": 10,
    "sort": [{"@timestamp": {"order": "desc"}}]
  }'
```

2. **Verify fields**:
- `s_primary_category` has valid categories
- `f_confidence_score` > 0.7
- `l_cached_tokens` > 0 (prompt caching working)
- `b_is_duplicate` = true for repeated errors
- `l_occurrence_count` > 1 for duplicates

3. **Check category distribution**:
```bash
curl -X GET "https://your-opensearch-instance:9200/error_classifications/_search?pretty" \
  -u username:password \
  -H 'Content-Type: application/json' \
  -d '{
    "size": 0,
    "aggs": {
      "categories": {
        "terms": {
          "field": "s_primary_category",
          "size": 10
        }
      }
    }
  }'
```

### Monitor API Costs

1. **Track token usage**:
```python
# Query for daily token usage
{
  "size": 0,
  "query": {
    "range": {
      "@timestamp": {
        "gte": "now-1d"
      }
    }
  },
  "aggs": {
    "total_prompt_tokens": {"sum": {"field": "l_prompt_tokens"}},
    "total_cached_tokens": {"sum": {"field": "l_cached_tokens"}},
    "cache_hit_rate": {
      "bucket_script": {
        "buckets_path": {
          "cached": "total_cached_tokens",
          "total": "total_prompt_tokens"
        },
        "script": "params.cached / params.total"
      }
    }
  }
}
```

2. **Expected costs** (with caching):
   - First request: ~1000 input tokens × $3/M = $0.003
   - Cached requests: ~50 tokens × $0.30/M = $0.00015
   - Daily (with 70% dedup + caching): ~$0.10-0.30

### Troubleshooting

**Problem: No classifications appearing**

1. Check logs for errors
2. Verify environment variables are set
3. Test OpenSearch connection:
```python
from opensearchpy import OpenSearch
client = OpenSearch(hosts=["https://..."], http_auth=(...))
print(client.info())
```

**Problem: High API costs**

1. Check cache hit rate (should be >80%)
2. Verify deduplication working (`b_is_duplicate` field)
3. Review `should_classify_realtime()` logic
4. Set `CLASSIFY_REALTIME_THRESHOLD=none` to disable CI classification

**Problem: Low classification confidence**

1. Review prompt in `prompts.py`
2. Add more examples to category definitions
3. Check error text quality (truncation, encoding)

**Problem: Wrong categories**

1. Run `analyze_recent_errors.py` to validate categories
2. Adjust category patterns
3. Add more specific examples to prompt
4. Consider sub-categories (Phase 2)

## Next Steps

### Phase 1 Complete - What's Next?

1. **Monitor for 1-2 weeks**:
   - Track category distribution
   - Review confidence scores
   - Identify misclassifications

2. **Build dashboards** (OpenSearch/Kibana):
   - Category trends over time
   - Most common errors by framework
   - Flaky test detection

3. **Phase 2 enhancements**:
   - Hierarchical categories (sub-categories)
   - Suggested fixes in classification
   - Error similarity detection
   - Automated issue creation for recurring errors

## Support

For issues or questions:
1. Check the logs for error messages
2. Review the README in `error_classification/`
3. Open an issue in the repository
4. Check Claude API status: https://status.anthropic.com/

## Useful Commands

```bash
# Analyze recent errors (validation)
python3 analyze_recent_errors.py --hours 48

# Classify last 24 hours (batch)
python3 upload_error_classifications.py --hours 24

# Classify last week (catch-up)
python3 upload_error_classifications.py --hours 168

# Test deduplicator
pytest tests/error_classification/test_deduplicator.py -v

# Check OpenSearch index
curl -X GET "https://your-instance:9200/error_classifications/_count?pretty" -u user:pass

# View recent classifications
curl -X GET "https://your-instance:9200/error_classifications/_search?pretty&size=5&sort=@timestamp:desc" -u user:pass
```
