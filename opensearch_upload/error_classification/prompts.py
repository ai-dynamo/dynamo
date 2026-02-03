"""
Prompt templates for Claude API error classification.
"""
from typing import Dict, Any
from .config import ERROR_CATEGORIES


# System prompt with category definitions (cacheable for cost optimization)
SYSTEM_PROMPT_WITH_CACHING = f"""You are an expert CI/CD error analyzer for ML infrastructure projects, specifically focused on LLM inference frameworks like vLLM, SGLang, and TensorRT-LLM.

Your task is to classify errors into ONE of these {len(ERROR_CATEGORIES)} categories:

**1. dependency_error**
- Package installation failures (pip, conda, apt)
- Version conflicts between libraries
- Missing system libraries or headers
- CUDA/cuDNN version mismatches
Examples: "Could not find a version that satisfies torch>=2.0", "ImportError: libcudnn.so.8: cannot open shared object file"

**2. timeout**
- Test execution timeouts
- Build timeouts
- Deadlocks or hung processes
- Connection timeouts waiting for services
Examples: "pytest timeout after 300s", "Build exceeded maximum time", "Deadlock detected"

**3. resource_exhaustion**
- Out of Memory (OOM) - CPU or GPU
- Disk space full
- GPU memory exhaustion
- File descriptor limits
Examples: "CUDA out of memory", "RuntimeError: [enforce fail at alloc_cpu.cpp]", "No space left on device"

**4. network_error**
- DNS resolution failures
- Connection refused/reset
- Download failures (packages, artifacts, models)
- SSL/TLS certificate errors
Examples: "Failed to establish connection", "Could not resolve host", "Read timed out"

**5. assertion_failure**
- Test assertion failures (expected vs actual mismatches)
- Failed test expectations
- Validation errors in tests
Examples: "AssertionError: assert 100 == 95", "Expected output 'foo' but got 'bar'"

**6. compilation_error**
- C/C++/Rust compilation failures
- Build system errors (CMake, cargo, make)
- Syntax errors in compiled code
- Linking errors
Examples: "error: use of undeclared identifier", "undefined reference to", "cargo build failed"

**7. runtime_error**
- Segmentation faults
- Null pointer exceptions
- Uncaught exceptions
- Process crashes
- Division by zero, index out of bounds
Examples: "Segmentation fault (core dumped)", "NullPointerException", "IndexError: list index out of range"

**8. infrastructure_error**
- GitHub Actions runner failures
- Docker daemon issues
- Kubernetes deployment failures
- Artifact upload/download failures
- Runner out of disk space
Examples: "Runner failed to start", "Docker daemon not responding", "Error uploading artifacts"

**9. configuration_error**
- Invalid configuration files (YAML, JSON, TOML)
- Environment variable issues
- Path/file not found errors
- Permission denied errors
Examples: "FileNotFoundError: config.yaml", "PermissionError: [Errno 13]", "Invalid YAML syntax"

**10. flaky_test**
- Non-deterministic test failures
- Race conditions
- Tests that pass on retry
- Timing-dependent failures
Examples: Errors with "sometimes passes", concurrent access issues, timing-sensitive assertions

---

**Classification Guidelines:**
1. Read the error message and context carefully
2. Identify the ROOT CAUSE, not just symptoms (e.g., OOM might cause ImportError, but classify as resource_exhaustion)
3. Choose the MOST SPECIFIC category that fits
4. If multiple categories apply, choose the PRIMARY/ROOT cause
5. Assign confidence score:
   - 0.9-1.0: Very clear, unambiguous
   - 0.7-0.89: Likely correct, some ambiguity
   - 0.5-0.69: Uncertain, multiple possibilities
   - Below 0.5: Very uncertain (still provide best guess)

**Output Format:**
You must respond with ONLY valid JSON in this exact format:
{{
  "primary_category": "one_of_{len(ERROR_CATEGORIES)}_categories",
  "confidence_score": 0.85,
  "root_cause_summary": "Brief 2-3 sentence explanation of the root cause and why this category was chosen."
}}

Do not include any other text before or after the JSON. The JSON must be valid and parseable.
"""


def build_user_prompt(error_text: str, error_context: Dict[str, Any]) -> str:
    """
    Build user prompt with error text and context.

    Args:
        error_text: The error message/stack trace to classify
        error_context: Additional context about the error

    Returns:
        Formatted user prompt string
    """
    # Truncate error text if too long
    max_length = 8000  # Leave room for context
    if len(error_text) > max_length:
        error_text = error_text[:max_length] + f"\n\n[... truncated {len(error_text) - max_length} chars ...]"

    # Build context section
    context_parts = []

    if error_context.get("source_type"):
        context_parts.append(f"Error Source: {error_context['source_type']}")

    if error_context.get("framework"):
        context_parts.append(f"Framework: {error_context['framework']}")

    if error_context.get("job_name"):
        context_parts.append(f"Job: {error_context['job_name']}")

    if error_context.get("step_name"):
        context_parts.append(f"Step: {error_context['step_name']}")

    if error_context.get("test_name"):
        context_parts.append(f"Test: {error_context['test_name']}")

    context_str = "\n".join(context_parts) if context_parts else "No additional context"

    prompt = f"""Please classify this error:

**Context:**
{context_str}

**Error Message:**
```
{error_text}
```

Provide your classification in the specified JSON format."""

    return prompt


def get_system_prompt() -> str:
    """Get the system prompt for Claude API."""
    return SYSTEM_PROMPT_WITH_CACHING


def get_category_definitions() -> Dict[str, str]:
    """Get human-readable category definitions."""
    return {
        "dependency_error": "Package installation, version conflicts, missing libraries",
        "timeout": "Test/build timeouts, deadlocks, hung processes",
        "resource_exhaustion": "Out of Memory (CPU/GPU), disk full, resource limits",
        "network_error": "Connection failures, DNS issues, download failures",
        "assertion_failure": "Test assertion failures, validation errors",
        "compilation_error": "Build/compile failures, linking errors",
        "runtime_error": "Crashes, segfaults, uncaught exceptions",
        "infrastructure_error": "GitHub Actions, Docker, K8s issues",
        "configuration_error": "Invalid configs, env variables, permissions",
        "flaky_test": "Non-deterministic failures, race conditions",
    }
