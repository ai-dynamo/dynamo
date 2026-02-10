# Test Claude API Connection

Simple test to validate the Claude API connection before deploying the full batch classification system.

## What This Tests

1. ✅ Claude API key is valid and working
2. ✅ Error classification works correctly
3. ✅ Token usage is reasonable
4. ✅ PR comment creation works (if on a PR)

## How to Run

### Option 1: Run Locally

```bash
# Set environment variables
export ANTHROPIC_API_KEY="your-api-key"
export GITHUB_TOKEN="your-token"  # optional, for PR comments

# Run the test
python3 .github/workflows/test_claude_simple.py
```

### Option 2: Run in GitHub Actions (Manual Trigger)

1. Go to **Actions** tab in GitHub
2. Select **Test Claude API Connection** workflow
3. Click **Run workflow**
4. Select branch and run

### Option 3: Push to Test Branch

```bash
# Create a test branch
git checkout -b test/claude-connection

# Commit changes
git add .
git commit -m "Test Claude API connection"

# Push (will auto-trigger workflow)
git push origin test/claude-connection
```

## Expected Output

### Successful Run

```
======================================================================
TESTING CLAUDE API CONNECTION
======================================================================

✅ ANTHROPIC_API_KEY found
   Key length: 108 chars
   Key prefix: sk-ant-api03-...

📝 Initializing configuration...
   Model: claude-sonnet-4-5-20250929
   Max RPM: 50

🔌 Connecting to Claude API...
   ✅ ClaudeClient initialized

----------------------------------------------------------------------
🧪 Testing with fake infrastructure_error...
----------------------------------------------------------------------

Error text (first 200 chars):
ERROR collecting tests/test_server.py
ImportError while importing test module 'tests/test_server.py'.
Hint: make sure your test suite is properly configured...

🤖 Calling Claude API...
✅ Classification successful!

   📊 Results:
      Category: infrastructure_error
      Confidence: 92%
      Summary: Pytest collection failed due to missing module 'vllm'...

   💰 Token Usage:
      Prompt tokens: 1,847
      Completion tokens: 127
      Cached tokens: 1,542

   ✅ Classification CORRECT (expected infrastructure_error)

----------------------------------------------------------------------
🧪 Testing with fake timeout...
----------------------------------------------------------------------

[... similar output for other error types ...]

======================================================================
📊 TEST SUMMARY
======================================================================

Total tests: 3
Correct classifications: 3/3 (100%)
Average confidence: 89%

Details:
  ✅ Expected: infrastructure_error, Got: infrastructure_error, Confidence: 92%
  ✅ Expected: timeout, Got: timeout, Confidence: 88%
  ✅ Expected: assertion_failure, Got: assertion_failure, Confidence: 87%

----------------------------------------------------------------------
💬 Testing PR comment creation...
----------------------------------------------------------------------

   Found PR #123
   ✅ PR comment created successfully

======================================================================
✅ ALL TESTS PASSED - Claude API connection working perfectly!
======================================================================
```

## What Gets Tested

### 1. Infrastructure Error

Tests Claude's ability to identify pytest collection errors:
- Missing modules
- Import failures
- Test environment setup issues

### 2. Timeout Error

Tests Claude's ability to identify timeout issues:
- Test timeouts
- Deadlocks
- Slow operations

### 3. Assertion Failure

Tests Claude's ability to identify test failures:
- Failed assertions
- Expected vs actual mismatches
- Test validation errors

## Validation Criteria

✅ **Pass**: All 3 errors classified correctly with confidence >70%
⚠️ **Partial**: Some errors classified correctly
❌ **Fail**: API key invalid or all classifications wrong

## Token Usage Expectations

Per error classification:
- **Prompt tokens**: ~1,500-2,000 (first call)
- **Cached tokens**: ~1,500 (subsequent calls with caching)
- **Completion tokens**: ~100-200
- **Cost**: ~$0.003 per classification (with caching)

## Troubleshooting

### Error: "ANTHROPIC_API_KEY not set"

**Fix**: Add the API key to GitHub secrets
1. Go to repository **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `ANTHROPIC_API_KEY`
4. Value: Your Claude API key (starts with `sk-ant-api03-`)

### Error: "Failed to initialize Claude client"

**Possible causes:**
- Invalid API key format
- Network connectivity issues
- API endpoint unreachable

**Fix**: Verify API key is correct and has proper permissions

### Error: "Classification failed"

**Possible causes:**
- API rate limit exceeded
- Model not available
- Malformed prompt

**Fix**: Check error details in logs, may need to wait and retry

### PR Comment Not Created

**Possible causes:**
- Not running in PR context
- Missing `GITHUB_TOKEN`
- Insufficient permissions

**Fix**:
1. Ensure workflow has proper permissions:
```yaml
permissions:
  pull-requests: write
```
2. Run from a pull request, not direct push to main

## Next Steps

After this test passes:

1. ✅ Claude API is working
2. ✅ Classification is accurate
3. ✅ Ready to test full batch implementation
4. → Run full workflow with real errors
5. → Verify parallel processing works
6. → Monitor token usage at scale

## Files

- **Workflow**: `.github/workflows/test_claude_connection.yml`
- **Test Script**: `.github/workflows/test_claude_simple.py`
- **This Guide**: `TEST_CLAUDE_CONNECTION.md`

---

**Status**: Ready to test
**Estimated time**: 30-60 seconds
**Cost**: ~$0.01 per run (3 classifications)
