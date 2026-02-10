# Test Claude API Connection

Simple test to validate the Claude API connection works before full deployment.

## What This Tests

1. ✅ Claude API key is valid
2. ✅ Error classification works correctly
3. ✅ Tests 3 error types: infrastructure_error, timeout, assertion_failure
4. ✅ Token usage is tracked

## How to Run

### Automatically (Already Configured)

This workflow will **auto-trigger** when you push to the `nmailhot/test-claude` branch.

Just push your changes and it will run automatically!

### Manually

1. Go to **Actions** tab in GitHub
2. Select **"Test Claude API Connection"** workflow
3. Click **"Run workflow"**
4. Select `nmailhot/test-claude` branch
5. Click green **"Run workflow"** button

## Expected Output

```
✅ ANTHROPIC_API_KEY found
✅ ClaudeClient initialized

🧪 Testing with fake infrastructure_error...
   ✅ Classification successful!
   Category: infrastructure_error
   Confidence: 92%

🧪 Testing with fake timeout...
   ✅ Classification successful!
   Category: timeout
   Confidence: 88%

🧪 Testing with fake assertion_failure...
   ✅ Classification successful!
   Category: assertion_failure
   Confidence: 87%

📊 TEST SUMMARY
Total tests: 3
Correct classifications: 3/3 (100%)
Average confidence: 89%

✅ ALL TESTS PASSED
```

## Cost

- **~$0.01 per run** (3 API calls with prompt caching)
- Takes 30-60 seconds

## Troubleshooting

### "ANTHROPIC_API_KEY not set"

Add the secret in GitHub:
1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Name: `ANTHROPIC_API_KEY`
4. Value: Your Claude API key

### "Classification failed"

- Check API key is valid
- Check network connectivity
- Check error details in logs

## Files

- **Workflow**: `.github/workflows/test_claude_connection.yml`
- **Test Script**: `.github/workflows/test_claude_simple.py`
- **This Guide**: `TEST_CLAUDE_README.md`
