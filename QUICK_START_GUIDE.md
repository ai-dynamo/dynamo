# Quick Start Guide - Batch Error Classification

## 🚀 Quick Start

### 1. Enable in Your Workflow

Add to your `.github/workflows/` YAML file:

```yaml
jobs:
  # ... your existing jobs ...

  classify-errors:
    name: Classify Errors
    runs-on: ubuntu-latest
    if: always()  # Run even if previous jobs failed
    needs: [job1, job2, job3]  # List jobs to analyze

    steps:
      - name: Checkout code
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.11'

      - name: Install dependencies
        run: |
          pip install anthropic requests opensearch-py

      - name: Classify Workflow Errors
        env:
          ENABLE_ERROR_CLASSIFICATION: "true"
          ENABLE_PR_COMMENTS: "true"
          MAX_PARALLEL_JOBS: "5"
          ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
          GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: |
          python3 .github/workflows/classify_workflow_errors.py
```

### 2. Set Up Secrets

In your GitHub repository settings:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Add secret: `ANTHROPIC_API_KEY` with your Claude API key
3. `GITHUB_TOKEN` is automatically provided by GitHub Actions

### 3. Test It

```bash
# Create a branch with intentional failures
git checkout -b test-error-classification

# Push and wait for workflow to run
git push origin test-error-classification
```

After workflow completes, check:
- ✅ PR comment with error summary
- ✅ GitHub annotations in Files Changed tab
- ✅ Check run with classification results

## 🎛️ Configuration Options

### Environment Variables

```yaml
env:
  # Required
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}

  # Feature toggles
  ENABLE_ERROR_CLASSIFICATION: "true"   # Enable classification
  ENABLE_GITHUB_ANNOTATIONS: "true"     # Create annotations
  ENABLE_PR_COMMENTS: "true"            # Create PR comments (NEW)

  # Performance tuning
  MAX_PARALLEL_JOBS: "5"                # Concurrent job analyses (NEW)

  # Optional: OpenSearch upload
  OPENSEARCH_URL: ${{ secrets.OPENSEARCH_URL }}
  OPENSEARCH_USERNAME: ${{ secrets.OPENSEARCH_USERNAME }}
  OPENSEARCH_PASSWORD: ${{ secrets.OPENSEARCH_PASSWORD }}
```

### Disable Features Individually

```yaml
# Disable PR comments but keep annotations
env:
  ENABLE_PR_COMMENTS: "false"
  ENABLE_GITHUB_ANNOTATIONS: "true"

# Disable annotations but keep PR comments
env:
  ENABLE_PR_COMMENTS: "true"
  ENABLE_GITHUB_ANNOTATIONS: "false"

# Classify but don't create any GitHub outputs
env:
  ENABLE_PR_COMMENTS: "false"
  ENABLE_GITHUB_ANNOTATIONS: "false"
```

## 📊 Understanding the Output

### PR Comment Example

When a workflow fails, you'll see a comment like:

```markdown
## 🤖 AI Error Classification Summary

Found **5 unique error(s)** across **2 job(s)**

### 🔴 Critical (Immediate attention needed)
| Job | Step | Error Type | Confidence | Summary |
|-----|------|------------|------------|---------|
| build-linux | Run tests | Infrastructure Error | 92% | Pytest collection failed - missing test module |

### 🟠 Important (Should be fixed)
| build-arm64 | Compile | Timeout | 88% | Build exceeded 30min time limit |

### 🔵 Informational
| unit-tests | Test Suite | Assertion Failure | 95% | Expected value 100, got 95 |
```

### Severity Levels

**🔴 Critical** - Fix immediately
- Infrastructure errors (runner problems, Docker issues)
- Compilation errors (build failures)
- Dependency errors (package installation failures)

**🟠 Important** - Should fix soon
- Timeouts (builds/tests taking too long)
- Resource exhaustion (OOM, disk full)
- Configuration errors (missing files, bad configs)
- Network errors (download failures)

**🔵 Informational** - Lower priority
- Runtime errors (crashes in specific tests)
- Assertion failures (test expectations not met)
- Flaky tests (non-deterministic failures)

## 🔍 Troubleshooting

### No PR Comment Created

**Check:**
1. Is it a PR? (PR comments only appear on pull requests)
2. Is `ENABLE_PR_COMMENTS=true`?
3. Does `GITHUB_TOKEN` have write permissions?

**Fix:**
```yaml
permissions:
  pull-requests: write  # Required for PR comments
  checks: write         # Required for annotations
```

### No Annotations Created

**Check:**
1. Is `ENABLE_GITHUB_ANNOTATIONS=true`?
2. Does workflow have `checks: write` permission?

### Classification Taking Too Long

**Reduce parallel jobs:**
```yaml
env:
  MAX_PARALLEL_JOBS: "3"  # Lower concurrency
```

### High Token Usage / Cost

**Solutions:**
1. Reduce log sizes (GitHub Actions truncates at 10MB)
2. Lower `MAX_PARALLEL_JOBS` (sequential = slower but cheaper)
3. Only classify critical jobs:

```python
# Modify classify_workflow_errors.py
failed_jobs = [
    job for job in jobs
    if job.get("conclusion") == "failure"
    and "critical" in job.get("name", "").lower()
]
```

## 📈 Best Practices

### 1. Run After All Jobs Complete

```yaml
classify-errors:
  needs: [build, test, deploy]
  if: always()  # IMPORTANT: Run even if previous jobs fail
```

### 2. Set Reasonable Timeouts

```yaml
classify-errors:
  timeout-minutes: 10  # Reasonable for 5 jobs
```

### 3. Monitor Token Usage

Check workflow logs for:
```
💰 Token usage:
   - Prompt tokens: 150,000
   - Completion tokens: 3,500
   - Cached tokens: 135,000
```

### 4. Test Configuration Changes

Always test in a branch first:
```bash
git checkout -b test-classification-config
# Make changes
git push origin test-classification-config
# Verify in PR
```

## 🧪 Testing Locally

You can test classification locally (requires API key):

```bash
# Set environment variables
export ANTHROPIC_API_KEY="your-key"
export GITHUB_TOKEN="your-token"
export GITHUB_REPOSITORY="owner/repo"
export GITHUB_RUN_ID="12345"
export ENABLE_ERROR_CLASSIFICATION="true"

# Run classifier
python3 .github/workflows/classify_workflow_errors.py
```

## 📚 More Resources

- **Full Documentation**: `BATCH_IMPLEMENTATION_SUMMARY.md`
- **Test Suite**: `test_batch_implementation.py`
- **Error Categories**: See `opensearch_upload/error_classification/prompts.py`

## 💡 Tips

### Get Better Classifications

1. **Descriptive job names**: Use clear names like "build-linux-x86" vs "build-1"
2. **Structured logs**: Claude works best with clear error messages
3. **Keep logs concise**: Remove unnecessary debug output

### Reduce Costs

1. **Filter jobs early**: Only analyze jobs that need attention
2. **Use caching**: System prompt is cached (90% savings)
3. **Batch wisely**: Group related errors in same job

### Improve Accuracy

1. **Provide context**: Good step names help Claude identify errors
2. **Clean logs**: Remove noise, keep signal
3. **Review confidence**: Low confidence (<70%) may need manual review

## 🎯 Common Use Cases

### Use Case 1: CI/CD Monitoring

Monitor all production deployments for failures:

```yaml
on:
  push:
    branches: [main, production]

jobs:
  deploy:
    # ... deployment job ...

  classify-errors:
    needs: [deploy]
    if: failure()  # Only if deployment failed
    # ... classification setup ...
```

### Use Case 2: PR Validation

Provide detailed feedback on PR failures:

```yaml
on:
  pull_request:
    types: [opened, synchronize, reopened]

jobs:
  test:
    # ... test jobs ...

  classify-errors:
    needs: [test]
    if: always()
    # ... classification setup ...
```

### Use Case 3: Nightly Tests

Analyze overnight test runs:

```yaml
on:
  schedule:
    - cron: '0 2 * * *'  # 2 AM daily

jobs:
  nightly-tests:
    # ... extensive tests ...

  classify-errors:
    needs: [nightly-tests]
    if: always()
    # ... classification setup ...
```

---

**Need Help?** Check the full documentation in `BATCH_IMPLEMENTATION_SUMMARY.md`
