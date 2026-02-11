================================================================================
ERROR CLASSIFICATION REPORT
================================================================================

**Time Range**: Last 6 hours
**Generated**: 2026-02-03T23:19:01.910412+00:00
**Repository**: ai-dynamo/dynamo
**Model**: aws/anthropic/claude-opus-4-5

## Summary

- **Total Errors Classified**: 10
- **Total Error Occurrences**: 18
- **Average Confidence**: 85.6%
- **Total Tokens Used**: 11,919
  - Prompt: 10,784
  - Completion: 1,135

## Category Distribution

| Category | Unique Errors | Total Occurrences | Percentage |
|----------|---------------|-------------------|------------|
| configuration_error | 7 | 13 | 70.0% |
| runtime_error | 1 | 1 | 10.0% |
| assertion_failure | 1 | 2 | 10.0% |
| infrastructure_error | 1 | 2 | 10.0% |

## Sample Classifications

### assertion_failure

**Workflow**: Docs link check
**Job**: Check for broken markdown links
**Confidence**: 85.0%
**Occurrences**: 2

**Root Cause**:
This is a documentation validation check that found broken/invalid markdown links in the project's documentation files. The workflow is asserting that all links should be valid, and this assertion failed because one or more links are broken (404, incorrect paths, etc.). This is a test/validation failure rather than an infrastructure or configuration issue.

**Error Snippet**:
```
❌ WORKFLOW FAILED: Broken links found in documentation files
```

### configuration_error

**Workflow**: Lint PR
**Job**: Validate PR title and add label
**Confidence**: 92.0%
**Occurrences**: 1

**Root Cause**:
The PR title validation failed because the task type 'DRAFT' is not in the allowed list of conventional commit types. This is a configuration/validation error where the PR title format doesn't match the expected schema (feat, fix, docs, etc.). The user needs to update their PR title to use one of the valid task type prefixes.

**Error Snippet**:
```
Invalid or missing task type: 'DRAFT'. Must be one of: feat, fix, docs, test, ci, refactor, perf, chore, revert, style, build
```

**Workflow**: Docs link check
**Job**: Check for broken markdown links
**Confidence**: 82.0%
**Occurrences**: 2

**Root Cause**:
The markdown link checker detected a symlink that traverses multiple parent directories (../../../../), which is flagged as suspicious or potentially broken. This is a configuration/path issue where the symlink structure in the repository doesn't conform to the expected conventions for the link checker tool, likely requiring either fixing the symlink path or configuring the checker to allow such traversals.

**Error Snippet**:
```
Problematic symlink: Suspicious symlink: target requires many directory traversals ('../../../../examples/custom_backend/hello_world/README.md')
```

### infrastructure_error

**Workflow**: NVIDIA Test Lab Validation
**Job**: Trigger CI Pipeline
**Confidence**: 75.0%
**Occurrences**: 2

**Root Cause**:
Git command failed with exit code 128, which typically indicates a git operation failure such as authentication issues, repository access problems, or invalid references. This is an infrastructure-level failure in the CI pipeline's git operations, likely related to checkout, fetch, or push operations failing due to runner or repository configuration issues.

**Error Snippet**:
```
The process '/usr/bin/git' failed with exit code 128
```

### runtime_error

**Workflow**: Lint PR
**Job**: Validate PR title and add label
**Confidence**: 82.0%
**Occurrences**: 1

**Root Cause**:
This is a JavaScript/TypeScript runtime error where code attempted to access the 'type' property on an undefined object. This typically occurs in GitHub Actions workflows when parsing PR metadata or labels, and the expected data structure was not present. The error indicates a null/undefined reference access rather than a configuration or infrastructure issue.

**Error Snippet**:
```
Cannot read properties of undefined (reading 'type')
```

## Cost Analysis

- **Estimated Cost**: $0.0494
  - Input tokens: $0.0324
  - Output tokens: $0.0170

- **Cost per error**: $0.0049

================================================================================