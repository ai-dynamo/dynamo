# Error Classification System - Quick Reference

## Installation

```bash
cd opensearch_upload
pip install -r requirements.txt
```

## Environment Setup

```bash
# Minimum required
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENSEARCH_URL="https://your-instance:9200"
export ERROR_CLASSIFICATION_INDEX="error_classifications"

# Optional
export OPENSEARCH_USERNAME="admin"
export OPENSEARCH_PASSWORD="password"
export GITHUB_TOKEN="ghp_..."  # For error analysis
export ENABLE_ERROR_CLASSIFICATION="true"  # For CI
```

## Common Commands

### Phase 0: Validate Categories
```bash
python3 analyze_recent_errors.py --hours 48 --output report.md
```

### Batch Classification
```bash
# Last 24 hours
python3 upload_error_classifications.py --hours 24

# Last week (catch-up)
python3 upload_error_classifications.py --hours 168
```

### Test Components
```bash
# Test deduplicator
pytest tests/error_classification/test_deduplicator.py -v

# Test with Python directly
python tests/error_classification/test_deduplicator.py
```

### OpenSearch Queries

**Count classifications:**
```bash
curl -X GET "https://your-instance:9200/error_classifications/_count?pretty" \
  -u username:password
```

**Recent classifications:**
```bash
curl -X GET "https://your-instance:9200/error_classifications/_search?pretty" \
  -u username:password \
  -H 'Content-Type: application/json' \
  -d '{
    "size": 5,
    "sort": [{"@timestamp": {"order": "desc"}}]
  }'
```

**Category distribution:**
```bash
curl -X GET "https://your-instance:9200/error_classifications/_search?pretty" \
  -u username:password \
  -H 'Content-Type: application/json' \
  -d '{
    "size": 0,
    "aggs": {
      "categories": {
        "terms": {"field": "s_primary_category", "size": 10}
      }
    }
  }'
```

**Token usage (daily):**
```bash
curl -X GET "https://your-instance:9200/error_classifications/_search?pretty" \
  -u username:password \
  -H 'Content-Type: application/json' \
  -d '{
    "size": 0,
    "query": {
      "range": {"@timestamp": {"gte": "now-1d"}}
    },
    "aggs": {
      "total_prompt_tokens": {"sum": {"field": "l_prompt_tokens"}},
      "total_cached_tokens": {"sum": {"field": "l_cached_tokens"}}
    }
  }'
```

## Error Categories

1. **dependency_error** - Package/version conflicts
2. **timeout** - Timeouts, deadlocks
3. **resource_exhaustion** - OOM, disk full
4. **network_error** - Connection failures
5. **assertion_failure** - Test failures
6. **compilation_error** - Build failures
7. **runtime_error** - Crashes, exceptions
8. **infrastructure_error** - CI/Docker issues
9. **configuration_error** - Config/permission errors
10. **flaky_test** - Non-deterministic failures

## Cronjob Setup

```bash
# Edit crontab
crontab -e

# Add line (runs every hour)
0 * * * * cd /path/to/dynamo/opensearch_upload && \
    /usr/bin/python3 upload_error_classifications.py --hours 1 \
    >> /var/log/error_classification.log 2>&1

# View logs
tail -f /var/log/error_classification.log
```

## CI Integration

Add to `.github/workflows/*.yml`:

```yaml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  ERROR_CLASSIFICATION_INDEX: ${{ secrets.ERROR_CLASSIFICATION_INDEX }}
  ENABLE_ERROR_CLASSIFICATION: 'true'

jobs:
  test:
    steps:
      - name: Run tests
        continue-on-error: true
        run: pytest tests/ --junitxml=test-results/junit.xml

      - name: Classify Errors on Failure
        if: failure() && env.ENABLE_ERROR_CLASSIFICATION == 'true'
        run: python3 .github/workflows/classify_errors_on_failure.py
```

## Configuration Options

```bash
# Model selection
export ANTHROPIC_MODEL="claude-sonnet-4-5-20250929"

# Rate limiting
export MAX_RPM="50"  # requests per minute

# Caching
export CLASSIFICATION_CACHE_TTL_HOURS="168"  # 1 week
export MIN_CONFIDENCE_FOR_REUSE="0.8"

# Real-time classification threshold
export CLASSIFY_REALTIME_THRESHOLD="infrastructure"  # infrastructure|all|none
```

## Troubleshooting

**No classifications appearing:**
1. Check logs for errors
2. Verify environment variables
3. Test OpenSearch connection
4. Check index exists

**High API costs:**
1. Verify cache hit rate >80%
2. Check deduplication working
3. Review real-time threshold
4. Set `CLASSIFY_REALTIME_THRESHOLD=none`

**Low confidence scores:**
1. Review error text quality
2. Check for truncation
3. Add examples to prompts
4. Run validation script

## Useful Locations

- **Source code**: `opensearch_upload/error_classification/`
- **Scripts**: `opensearch_upload/*.py`
- **CI integration**: `.github/workflows/classify_errors_on_failure.py`
- **Tests**: `tests/error_classification/`
- **Documentation**: `opensearch_upload/*.md`

## Key Files

- `config.py` - Configuration
- `classifier.py` - Main orchestration
- `claude_client.py` - API wrapper
- `deduplicator.py` - Deduplication logic
- `error_extractor.py` - Extract errors
- `prompts.py` - Claude prompts

## Important Fields

- `s_primary_category` - Classification result
- `f_confidence_score` - Confidence (0-1)
- `s_error_hash` - Deduplication key
- `b_is_duplicate` - Is duplicate?
- `l_occurrence_count` - How many times seen
- `l_cached_tokens` - Cache hits
- `s_root_cause_summary` - AI explanation

## Expected Metrics

- **Category coverage**: >95%
- **Confidence scores**: >0.7 average
- **Deduplication rate**: 70-90%
- **Cache hit rate**: >80%
- **CI overhead**: <30s
- **Monthly cost**: $3-5

## Support

- README: `opensearch_upload/error_classification/README.md`
- Setup guide: `opensearch_upload/SETUP_GUIDE.md`
- Implementation summary: `opensearch_upload/IMPLEMENTATION_SUMMARY.md`
- Example workflow: `opensearch_upload/example_workflow_integration.yml`
