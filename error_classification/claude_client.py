# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLM inference client with prompt caching and rate limiting.
Supports both Anthropic native API and OpenAI-compatible APIs (e.g., NVIDIA).
"""
import json
import threading
import time
from typing import Any, Dict, List

import anthropic
import requests
from anthropic import Anthropic

from .config import Config
from .prompts import SYSTEM_PROMPT_FULL_LOG_ANALYSIS

# Maximum characters to send to the LLM (approx ~100K tokens, leaves room for system prompt)
MAX_LOG_LENGTH = 400_000


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, max_rpm: int):
        """
        Initialize rate limiter.

        Args:
            max_rpm: Maximum requests per minute
        """
        self.max_rpm = max_rpm
        self.requests = []
        self._lock = threading.Lock()

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        with self._lock:
            now = time.time()

            # Remove requests older than 1 minute
            self.requests = [
                req_time for req_time in self.requests if now - req_time < 60
            ]

            # If at limit, wait
            if len(self.requests) >= self.max_rpm:
                oldest_request = self.requests[0]
                wait_time = 60 - (now - oldest_request)
                if wait_time > 0:
                    print(f"⏳ Rate limit reached, waiting {wait_time:.1f}s")
                    time.sleep(wait_time)

            self.requests.append(time.time())


class ClaudeClient:
    """Client for Claude API with caching and rate limiting."""

    def __init__(self, config: Config):
        """
        Initialize Claude client.

        Args:
            config: Configuration object
        """
        self.config = config
        self.rate_limiter = RateLimiter(max_rpm=config.max_rpm)

        # Initialize appropriate client based on API format
        if config.api_format == "openai":
            # For OpenAI-compatible APIs (like NVIDIA)
            self.client = None  # Will use requests directly
            self.api_base_url = config.api_base_url or "https://api.openai.com/v1"
        else:
            # For native Anthropic API
            self.client = Anthropic(api_key=config.api_key)
            self.api_base_url = None

    def analyze_full_job_log(
        self, job_log: str, job_name: str, job_id: str, use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze complete job log and find/classify all errors.

        Args:
            job_log: Complete raw log from GitHub Actions job
            job_name: Name of the job
            job_id: GitHub job ID
            use_cache: Whether to use prompt caching

        Returns:
            {
                "errors_found": [
                    {
                        "step": "step name",
                        "primary_category": "...",
                        "confidence_score": 0.85,
                        "root_cause_summary": "...",
                        "log_excerpt": "relevant log section"
                    },
                    ...
                ],
                "total_errors": N
            }
        """
        # Route to appropriate API implementation
        if self.config.api_format == "openai":
            return self._analyze_full_log_openai_format(job_log, job_name, job_id)

        # Rate limit
        self.rate_limiter.wait_if_needed()

        # Truncate log if too large (keep last portion - most relevant for failures)
        if len(job_log) > MAX_LOG_LENGTH:
            # Keep last portion of log (failures typically at end)
            truncated_length = len(job_log) - MAX_LOG_LENGTH
            job_log = (
                f"[... truncated first {truncated_length} chars ...]\n\n"
                + job_log[-MAX_LOG_LENGTH:]
            )

        # Build prompt with full log
        user_prompt = self._build_full_log_prompt(job_log, job_name, job_id)

        # Call Claude API
        try:
            if use_cache:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=4096,  # Allow longer response for multiple errors
                    system=[
                        {
                            "type": "text",
                            "text": SYSTEM_PROMPT_FULL_LOG_ANALYSIS,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                    messages=[{"role": "user", "content": user_prompt}],
                )
            else:
                response = self.client.messages.create(
                    model=self.config.model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT_FULL_LOG_ANALYSIS,
                    messages=[{"role": "user", "content": user_prompt}],
                )

            # Extract token usage
            usage = response.usage
            prompt_tokens = getattr(usage, "input_tokens", 0)
            completion_tokens = getattr(usage, "output_tokens", 0)
            cached_tokens = 0
            if use_cache and hasattr(usage, "cache_read_input_tokens"):
                cached_tokens = getattr(usage, "cache_read_input_tokens", 0)

            # Parse JSON response
            response_text = response.content[0].text.strip()
            result = self._parse_full_log_response(response_text)

            # Add token usage to result
            result["prompt_tokens"] = prompt_tokens
            result["completion_tokens"] = completion_tokens
            result["cached_tokens"] = cached_tokens
            result["model_version"] = self.config.model

            return result

        except anthropic.APIError as e:
            print(f"✗ Claude API error: {e}")
            raise

        except Exception as e:
            print(f"✗ Unexpected error during full log analysis: {e}")
            raise

    def _build_full_log_prompt(self, job_log: str, job_name: str, job_id: str) -> str:
        """Build prompt with complete job log."""
        prompt = f"""Here is the complete log from a failed GitHub Actions job.
Please analyze it and identify ALL errors/failures.

Job Name: {job_name}
Job ID: {job_id}
Status: failure

Full Log:
```
{job_log}
```

Please:
1. Identify all errors/failures in this log
2. For each error, determine which step it occurred in
3. Classify each error into one of the 2 categories
4. Provide a root cause summary for each
5. Include a relevant log excerpt showing the error

Return JSON format as specified in the system prompt."""

        return prompt

    def _parse_full_log_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from full log analysis."""
        try:
            # Try to find JSON in response
            start_idx = response_text.find("{")
            end_idx = response_text.rfind("}")

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON found in response")

            json_str = response_text[start_idx : end_idx + 1]
            data = json.loads(json_str)

            # Validate required fields
            if "errors_found" not in data:
                raise ValueError("Missing required field: errors_found")

            if not isinstance(data["errors_found"], list):
                raise ValueError("errors_found must be an array")

            # Validate each error entry
            for i, error in enumerate(data["errors_found"]):
                required_fields = [
                    "step",
                    "primary_category",
                    "confidence_score",
                    "root_cause_summary",
                ]
                for field in required_fields:
                    if field not in error:
                        raise ValueError(f"Error {i}: Missing required field: {field}")

                # Normalize confidence to 0-1 range
                confidence = float(error["confidence_score"])
                if confidence < 0:
                    confidence = 0.0
                elif confidence > 1:
                    confidence = 1.0
                error["confidence_score"] = confidence

            # Set total_errors if not present
            if "total_errors" not in data:
                data["total_errors"] = len(data["errors_found"])

            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

        except Exception as e:
            raise ValueError(f"Error parsing response: {e}")

    def _openai_request_with_retry(self, payload: dict) -> requests.Response:
        """
        Make an OpenAI-compatible API request with retry logic.

        Handles 429 rate-limit errors and timeouts with exponential backoff
        (up to 3 attempts).

        Args:
            payload: JSON payload for the chat completions endpoint

        Returns:
            The successful requests.Response object

        Raises:
            requests.exceptions.HTTPError: If all retries exhausted on 429
            requests.exceptions.Timeout: If all retries exhausted on timeout
            Exception: Any other non-retryable exception
        """
        url = f"{self.api_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
        }

        max_retries = 3
        retry_delay = 2.0
        response = None

        for attempt in range(max_retries):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=120
                )

                # Handle rate limit errors (429)
                if response.status_code == 429:
                    if attempt < max_retries - 1:
                        # Check for Retry-After header
                        retry_after = response.headers.get("Retry-After")
                        if retry_after:
                            try:
                                wait_time = float(retry_after)
                                print(
                                    f"⏳ Rate limit hit (429), Retry-After: {wait_time}s"
                                )
                            except ValueError:
                                wait_time = retry_delay * (2**attempt)
                        else:
                            wait_time = retry_delay * (2**attempt)

                        print(
                            f"⏳ Waiting {wait_time:.1f}s before retry {attempt + 1}/{max_retries}"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"❌ Rate limit exceeded after {max_retries} retries")
                        response.raise_for_status()

                # Retry on 5xx server errors (transient infrastructure issues)
                if response.status_code >= 500:
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2**attempt)
                        print(
                            f"⏳ Server error {response.status_code}, retrying in {wait_time:.1f}s ({attempt + 1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                        continue
                    else:
                        print(
                            f"❌ Server error {response.status_code} after {max_retries} retries"
                        )

                response.raise_for_status()
                return response

            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"⏳ Timeout, retrying {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay)
                    continue
                raise

        # Should not normally reach here, but guard against it
        if response is not None:
            return response
        raise RuntimeError("All retry attempts failed without a response")

    def _analyze_full_log_openai_format(
        self, job_log: str, job_name: str, job_id: str
    ) -> Dict[str, Any]:
        """
        Analyze full log using OpenAI-compatible API (e.g., NVIDIA).

        Args:
            job_log: Complete raw log from GitHub Actions job
            job_name: Name of the job
            job_id: GitHub job ID

        Returns:
            Parsed result dict
        """
        # Apply rate limiting BEFORE making the request
        self.rate_limiter.wait_if_needed()

        # Truncate log if too large
        if len(job_log) > MAX_LOG_LENGTH:
            truncated_length = len(job_log) - MAX_LOG_LENGTH
            job_log = (
                f"[... truncated first {truncated_length} chars ...]\n\n"
                + job_log[-MAX_LOG_LENGTH:]
            )

        # Build prompt
        user_prompt = self._build_full_log_prompt(job_log, job_name, job_id)

        # Make API call with OpenAI format (with retry logic for rate limits)
        payload = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FULL_LOG_ANALYSIS},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.2,
            "max_tokens": 4096,
        }

        response = self._openai_request_with_retry(payload)
        data = response.json()

        # Parse OpenAI format response
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # Parse JSON from response
        result = self._parse_full_log_response(content)

        # Add token usage
        result["prompt_tokens"] = usage.get("prompt_tokens", 0)
        result["completion_tokens"] = usage.get("completion_tokens", 0)
        result["cached_tokens"] = usage.get("cached_tokens", 0)
        result["model_version"] = data.get("model", self.config.model)

        return result

    def _build_summary_prompt(
        self,
        classifications: List[Any],
        workflow_name: str,
        run_id: str,
        run_url: str,
        failed_jobs: int,
    ) -> str:
        """
        Build the summary prompt shared by both Anthropic and OpenAI paths.

        Args:
            classifications: List of ErrorClassification objects
            workflow_name: Name of the workflow that failed
            run_id: GitHub workflow run ID
            run_url: URL to the workflow run
            failed_jobs: Number of failed jobs

        Returns:
            Prompt string for the summary generation request
        """
        # Build list of all errors with context
        error_list = []
        for c in classifications:
            error_list.append(
                {
                    "job_name": c.job_name,
                    "step_name": c.step_name,
                    "category": c.primary_category,
                    "confidence": round(
                        c.confidence_score * 100
                    ),  # Convert to percentage
                    "root_cause": c.root_cause_summary,
                }
            )

        return f"""You are analyzing a failed GitHub Actions workflow. Below are all the error classifications from the failed jobs.

**Workflow**: {workflow_name}
**Run ID**: {run_id}
**Failed Jobs**: {failed_jobs}

**All Errors**:
{json.dumps(error_list, indent=2)}

Your task is to generate a concise, actionable markdown summary for contributors:

1. **Overall Summary** (1-2 sentences): Primary reason(s) the workflow failed.

2. **For each failure or group**: Include exactly:
   - **What failed**: One line describing what broke.
   - **Next step**: One concrete action the contributor can take (e.g., "Fix the import in tests/foo.py", "Re-run the job", "Check Docker login").

3. Keep Infrastructure (🔴) vs Code (🟠) clear. Be short and scannable.

Use this **exact format**:

### 🤖 Overall Failure Summary

[1-2 sentence summary. Mention main pattern if many jobs failed for same reason.]

### ❌ Common Failures (affecting multiple jobs)

**🔴 [Short failure description]** (X jobs affected)

- **What failed**: [One line.]
- **Next step**: [One concrete action.]
- **Jobs**: `job1`, `job2`
- **Root Cause**: [1 sentence.]

[Repeat for each group.]

### 🔍 Unique Job Failures

**🟠 job-name** (step: `step-name`)

- **What failed**: [One line.]
- **Next step**: [One concrete action.]
- **Root Cause**: [1 sentence.]

[Repeat for each unique failure.]

---

**Guidelines**:
- Use 🔴 for infrastructure_error, 🟠 for code_error.
- Every failure or group must have exactly one "What failed" and one "Next step" line; keep them one line each and actionable.
- Group similar failures; only show Common Failures if 2+ jobs share a root cause.
- Keep the whole summary short so external contributors can act quickly.
"""

    def generate_formatted_summary(
        self,
        classifications: List[Any],
        workflow_name: str,
        run_id: str,
        run_url: str,
        failed_jobs: int,
    ) -> str:
        """
        Generate a complete formatted markdown summary with Claude.

        This method sends all error classifications to Claude and asks it to:
        1. Generate an overall workflow failure summary
        2. Group similar failures across jobs
        3. List unique job-specific failures
        4. Format everything as clean, scannable markdown

        Args:
            classifications: List of ErrorClassification objects
            workflow_name: Name of the workflow that failed
            run_id: GitHub workflow run ID
            run_url: URL to the workflow run
            failed_jobs: Number of failed jobs

        Returns:
            Complete markdown-formatted summary string
        """
        # Route to appropriate API implementation
        if self.config.api_format == "openai":
            return self._generate_formatted_summary_openai(
                classifications, workflow_name, run_id, run_url, failed_jobs
            )

        # Rate limit
        self.rate_limiter.wait_if_needed()

        # Build the prompt (shared with OpenAI path)
        prompt = self._build_summary_prompt(
            classifications, workflow_name, run_id, run_url, failed_jobs
        )

        try:
            # Call Claude with the native API
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=2000,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}],
            )

            return response.content[0].text.strip()

        except anthropic.APIError as e:
            print(f"✗ Claude API error during summary generation: {e}")
            # Fallback to a basic summary if Claude fails
            return self._generate_fallback_summary(classifications, failed_jobs)

        except Exception as e:
            print(f"✗ Unexpected error during summary generation: {e}")
            return self._generate_fallback_summary(classifications, failed_jobs)

    def _generate_formatted_summary_openai(
        self,
        classifications: List[Any],
        workflow_name: str,
        run_id: str,
        run_url: str,
        failed_jobs: int,
    ) -> str:
        """Generate formatted summary using OpenAI-compatible API."""
        # Apply rate limiting BEFORE making the request
        self.rate_limiter.wait_if_needed()

        # Build the prompt (shared with Anthropic path)
        prompt = self._build_summary_prompt(
            classifications, workflow_name, run_id, run_url, failed_jobs
        )

        try:
            payload = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 2000,
            }

            response = self._openai_request_with_retry(payload)
            data = response.json()
            content = data["choices"][0]["message"]["content"]

            return content.strip()

        except Exception as e:
            print(f"✗ Error during OpenAI summary generation: {e}")
            return self._generate_fallback_summary(classifications, failed_jobs)

    def _generate_fallback_summary(
        self, classifications: List[Any], failed_jobs: int
    ) -> str:
        """Generate a basic fallback summary if Claude fails."""
        # Count by category
        category_counts = {}
        for c in classifications:
            cat = c.primary_category
            category_counts[cat] = category_counts.get(cat, 0) + 1

        # Build simple summary
        summary = "### 🤖 Overall Failure Summary\n\n"
        summary += f"The workflow failed with {len(classifications)} error(s) across {failed_jobs} job(s).\n\n"

        if category_counts:
            summary += "**Breakdown by category:**\n"
            for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                icon = "🔴" if cat == "infrastructure_error" else "🟠"
                cat_name = cat.replace("_", " ").title()
                summary += f"- {icon} {cat_name}: {count}\n"

        return summary
