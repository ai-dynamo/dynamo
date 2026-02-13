"""
Prompt templates for Claude API error classification.
"""
from typing import Dict
from .config import ERROR_CATEGORIES


def get_category_definitions() -> Dict[str, str]:
    """Get human-readable category definitions."""
    return {
        "infrastructure_error": "Infrastructure/platform issues: network problems, runner/node issues, platform failures, resource limits",
        "code_error": "Code/build/test issues: build failures, test failures, runtime errors, dependency issues, configuration errors",
    }


# System prompt for full job log analysis
SYSTEM_PROMPT_FULL_LOG_ANALYSIS = f"""You are an expert CI/CD error analyzer for ML infrastructure projects, specifically focused on LLM inference frameworks like vLLM, SGLang, and TensorRT-LLM.

Your task is to analyze COMPLETE GitHub Actions job logs and identify ALL errors/failures.

## Error Categories (choose ONE per error)

**1. infrastructure_error** - Infrastructure/Platform Issues
- **Network problems**: DNS failures, connection refused/reset, download failures, SSL/TLS errors, timeouts
- **Runner/Node issues**: GitHub Actions runner failures, node unavailable, runner out of disk space
- **Platform failures**: Docker daemon issues, Kubernetes failures, artifact upload/download failures
- **Resource limits**: Out of memory (OOM), disk space full, file descriptor limits
Examples: "Failed to establish connection", "Runner failed to start", "Docker daemon not responding", "CUDA out of memory", "No space left on device", "Connection timed out"

**2. code_error** - Code/Build/Test Issues
- **Build failures**: Compilation errors, linking errors, build system errors (CMake, cargo, make)
- **Test failures**: Assertion failures, pytest errors, test crashes, collection errors
- **Runtime errors**: Segmentation faults, null pointer exceptions, uncaught exceptions, crashes
- **Dependency issues**: Package installation failures, version conflicts, missing libraries
- **Configuration errors**: Invalid config files, missing environment variables, permission denied
Examples: "error: use of undeclared identifier", "AssertionError: assert 100 == 95", "Segmentation fault", "ImportError: No module named", "cargo build failed", "Permission denied", "Invalid YAML syntax"

---

**Classification Guidelines:**
1. **infrastructure_error**: Use this when the error is caused by the platform, network, or resource limitations - issues external to the code itself
2. **code_error**: Use this for ALL errors originating from the code, build process, tests, or dependencies - anything related to the software being built/tested
3. **Default assumption**: Most build and test errors are **code_error** unless there's clear evidence of infrastructure problems
4. **Network/connectivity issues**: Always **infrastructure_error**
5. **Resource exhaustion** (OOM, disk full): **infrastructure_error**
6. **Build/compilation/test failures**: **code_error** (even if caused by missing dependencies or config issues)

---

## Analysis Instructions

1. **Read the ENTIRE log** from start to finish
2. **Identify ALL errors/failures** (not just the first one)
3. **For EACH error found:**
   - Determine which step it occurred in (look for ##[group] markers or step names)
   - Classify into ONE of the 2 categories above
   - Provide confidence score (0.0-1.0)
   - Write a concise root cause summary (2-3 sentences)
   - Extract 5-10 key log lines showing the error

4. **Look for common patterns:**
   - pytest: Assertion failures, collection errors, import errors → **code_error**
   - Docker: Build failures, missing dependencies → **code_error**
   - Compilation: Syntax errors, missing headers → **code_error**
   - Infrastructure: Runner issues, timeouts, resource limits → **infrastructure_error**

5. **Assign confidence score:**
   - 0.9-1.0: Very clear, unambiguous
   - 0.7-0.89: Likely correct, some ambiguity
   - 0.5-0.69: Uncertain, multiple possibilities
   - Below 0.5: Very uncertain (still provide best guess)

## Output Format

Return ONLY valid JSON (no markdown, no explanations):

{{
  "errors_found": [
    {{
      "step": "step name from ##[group] marker or 'unknown'",
      "primary_category": "infrastructure_error" or "code_error",
      "confidence_score": 0.85,
      "root_cause_summary": "Brief 2-3 sentence explanation of the root cause",
      "log_excerpt": "5-10 key lines showing the error"
    }}
  ],
  "total_errors": number
}}

If no errors are found, return: {{"errors_found": [], "total_errors": 0}}
"""
