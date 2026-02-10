"""
Claude API client with prompt caching and rate limiting.
Supports both Anthropic native API and OpenAI-compatible APIs (e.g., NVIDIA).
"""
import json
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import anthropic
from anthropic import Anthropic
import requests

from .config import Config
from .prompts import get_system_prompt, build_user_prompt, SYSTEM_PROMPT_FULL_LOG_ANALYSIS


@dataclass
class ClassificationResult:
    """Result from Claude API classification."""
    primary_category: str
    confidence_score: float
    root_cause_summary: str
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    model_version: str
    classified_at: str


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

    def wait_if_needed(self):
        """Wait if rate limit would be exceeded."""
        now = time.time()

        # Remove requests older than 1 minute
        self.requests = [req_time for req_time in self.requests if now - req_time < 60]

        # If at limit, wait
        if len(self.requests) >= self.max_rpm:
            oldest_request = self.requests[0]
            wait_time = 60 - (now - oldest_request)
            if wait_time > 0:
                print(f"⏳ Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)

        self.requests.append(now)


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
        self.system_prompt = get_system_prompt()

        # Initialize appropriate client based on API format
        if config.api_format == "openai":
            # For OpenAI-compatible APIs (like NVIDIA)
            self.client = None  # Will use requests directly
            self.api_base_url = config.api_base_url or "https://api.openai.com/v1"
        else:
            # For native Anthropic API
            self.client = Anthropic(api_key=config.anthropic_api_key)
            self.api_base_url = None

    def _classify_with_openai_format(
        self,
        error_text: str,
        error_context: Dict[str, Any]
    ) -> ClassificationResult:
        """
        Classify error using OpenAI-compatible API (e.g., NVIDIA).

        Args:
            error_text: The error message/stack trace
            error_context: Additional context

        Returns:
            ClassificationResult
        """
        # Build user prompt
        user_prompt = build_user_prompt(error_text, error_context)

        # Make API call with OpenAI format
        url = f"{self.api_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.anthropic_api_key}"
        }

        payload = {
            "model": self.config.anthropic_model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 1024
        }

        response = requests.post(url, headers=headers, json=payload, timeout=60)
        response.raise_for_status()

        data = response.json()

        # Parse OpenAI format response
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # Parse JSON from response (handle code blocks)
        try:
            # Try parsing directly
            result_json = json.loads(content)
        except json.JSONDecodeError:
            # Try extracting JSON from markdown code blocks
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
            if json_match:
                result_json = json.loads(json_match.group(1))
            else:
                # Try finding JSON object in the text
                json_match = re.search(r'\{[^{}]*"primary_category"[^{}]*\}', content, re.DOTALL)
                if json_match:
                    result_json = json.loads(json_match.group(0))
                else:
                    raise ValueError(f"Could not parse JSON from response: {content[:200]}")

        return ClassificationResult(
            primary_category=result_json["primary_category"],
            confidence_score=result_json["confidence_score"],
            root_cause_summary=result_json["root_cause_summary"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cached_tokens=usage.get("cached_tokens", 0),
            model_version=data.get("model", self.config.anthropic_model),
            classified_at=datetime.now(timezone.utc).isoformat()
        )

    def classify_error(
        self,
        error_text: str,
        error_context: Dict[str, Any],
        use_cache: bool = True
    ) -> ClassificationResult:
        """
        Classify an error using Claude API.

        Args:
            error_text: The error message/stack trace
            error_context: Additional context (source, framework, job, etc.)
            use_cache: Whether to use prompt caching

        Returns:
            ClassificationResult with category, confidence, and usage stats

        Raises:
            ValueError: If response cannot be parsed
            anthropic.APIError: If API call fails
        """
        # Route to appropriate API implementation
        if self.config.api_format == "openai":
            return self._classify_with_openai_format(error_text, error_context)

        # Rate limit
        self.rate_limiter.wait_if_needed()

        # Truncate error text if needed
        if len(error_text) > self.config.max_error_length:
            error_text = error_text[:self.config.max_error_length]

        # Build user prompt
        user_prompt = build_user_prompt(error_text, error_context)

        # Make API call with caching
        try:
            if use_cache:
                # Use prompt caching for system prompt
                response = self.client.messages.create(
                    model=self.config.anthropic_model,
                    max_tokens=1024,
                    system=[{
                        "type": "text",
                        "text": self.system_prompt,
                        "cache_control": {"type": "ephemeral"}
                    }],
                    messages=[{
                        "role": "user",
                        "content": user_prompt
                    }]
                )
            else:
                # No caching
                response = self.client.messages.create(
                    model=self.config.anthropic_model,
                    max_tokens=1024,
                    system=self.system_prompt,
                    messages=[{
                        "role": "user",
                        "content": user_prompt
                    }]
                )

            # Extract token usage
            usage = response.usage
            prompt_tokens = getattr(usage, 'input_tokens', 0)
            completion_tokens = getattr(usage, 'output_tokens', 0)

            # Extract cached tokens (only available with caching)
            cached_tokens = 0
            if use_cache and hasattr(usage, 'cache_read_input_tokens'):
                cached_tokens = getattr(usage, 'cache_read_input_tokens', 0)

            # Parse response
            response_text = response.content[0].text.strip()
            classification_data = self._parse_response(response_text)

            return ClassificationResult(
                primary_category=classification_data["primary_category"],
                confidence_score=classification_data["confidence_score"],
                root_cause_summary=classification_data["root_cause_summary"],
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                cached_tokens=cached_tokens,
                model_version=self.config.anthropic_model,
                classified_at=datetime.now(timezone.utc).isoformat()
            )

        except anthropic.APIError as e:
            print(f"✗ Claude API error: {e}")
            raise

        except Exception as e:
            print(f"✗ Unexpected error during classification: {e}")
            raise

    def _parse_response(self, response_text: str) -> Dict[str, Any]:
        """
        Parse Claude's JSON response.

        Args:
            response_text: Raw response text from Claude

        Returns:
            Parsed classification data

        Raises:
            ValueError: If response cannot be parsed
        """
        try:
            # Try to find JSON in response
            # Sometimes Claude adds text before/after JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON found in response")

            json_str = response_text[start_idx:end_idx + 1]
            data = json.loads(json_str)

            # Validate required fields
            required_fields = ["primary_category", "confidence_score", "root_cause_summary"]
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            # Validate types
            if not isinstance(data["primary_category"], str):
                raise ValueError("primary_category must be a string")

            if not isinstance(data["confidence_score"], (int, float)):
                raise ValueError("confidence_score must be a number")

            if not isinstance(data["root_cause_summary"], str):
                raise ValueError("root_cause_summary must be a string")

            # Normalize confidence to 0-1 range
            confidence = float(data["confidence_score"])
            if confidence < 0:
                confidence = 0.0
            elif confidence > 1:
                confidence = 1.0

            data["confidence_score"] = confidence

            return data

        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {e}")

        except Exception as e:
            raise ValueError(f"Error parsing response: {e}")

    def classify_batch(
        self,
        errors: list[tuple[str, Dict[str, Any]]],
        use_cache: bool = True
    ) -> list[ClassificationResult]:
        """
        Classify multiple errors in sequence.

        Args:
            errors: List of (error_text, error_context) tuples
            use_cache: Whether to use prompt caching

        Returns:
            List of ClassificationResult objects
        """
        results = []

        for i, (error_text, error_context) in enumerate(errors, 1):
            print(f"  Classifying {i}/{len(errors)}...")

            try:
                result = self.classify_error(error_text, error_context, use_cache)
                results.append(result)

            except Exception as e:
                print(f"  ✗ Failed to classify error {i}: {e}")
                # Continue with remaining errors
                continue

        return results

    def analyze_full_job_log(
        self,
        job_log: str,
        job_name: str,
        job_id: str,
        use_cache: bool = True
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
        max_log_length = 400000  # ~100K tokens, leave room for system prompt
        if len(job_log) > max_log_length:
            # Keep last portion of log (failures typically at end)
            truncated_length = len(job_log) - max_log_length
            job_log = f"[... truncated first {truncated_length} chars ...]\n\n" + job_log[-max_log_length:]

        # Build prompt with full log
        user_prompt = self._build_full_log_prompt(job_log, job_name, job_id)

        # Call Claude API
        try:
            if use_cache:
                response = self.client.messages.create(
                    model=self.config.anthropic_model,
                    max_tokens=4096,  # Allow longer response for multiple errors
                    system=[{
                        "type": "text",
                        "text": SYSTEM_PROMPT_FULL_LOG_ANALYSIS,
                        "cache_control": {"type": "ephemeral"}
                    }],
                    messages=[{
                        "role": "user",
                        "content": user_prompt
                    }]
                )
            else:
                response = self.client.messages.create(
                    model=self.config.anthropic_model,
                    max_tokens=4096,
                    system=SYSTEM_PROMPT_FULL_LOG_ANALYSIS,
                    messages=[{
                        "role": "user",
                        "content": user_prompt
                    }]
                )

            # Extract token usage
            usage = response.usage
            prompt_tokens = getattr(usage, 'input_tokens', 0)
            completion_tokens = getattr(usage, 'output_tokens', 0)
            cached_tokens = 0
            if use_cache and hasattr(usage, 'cache_read_input_tokens'):
                cached_tokens = getattr(usage, 'cache_read_input_tokens', 0)

            # Parse JSON response
            response_text = response.content[0].text.strip()
            result = self._parse_full_log_response(response_text)

            # Add token usage to result
            result['prompt_tokens'] = prompt_tokens
            result['completion_tokens'] = completion_tokens
            result['cached_tokens'] = cached_tokens
            result['model_version'] = self.config.anthropic_model

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
3. Classify each error into one of the 10 categories
4. Provide a root cause summary for each
5. Include a relevant log excerpt showing the error

Return JSON format as specified in the system prompt."""

        return prompt

    def _parse_full_log_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from full log analysis."""
        try:
            # Try to find JSON in response
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')

            if start_idx == -1 or end_idx == -1:
                raise ValueError("No JSON found in response")

            json_str = response_text[start_idx:end_idx + 1]
            data = json.loads(json_str)

            # Validate required fields
            if "errors_found" not in data:
                raise ValueError("Missing required field: errors_found")

            if not isinstance(data["errors_found"], list):
                raise ValueError("errors_found must be an array")

            # Validate each error entry
            for i, error in enumerate(data["errors_found"]):
                required_fields = ["step", "primary_category", "confidence_score", "root_cause_summary"]
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

    def _analyze_full_log_openai_format(
        self,
        job_log: str,
        job_name: str,
        job_id: str
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
        # Truncate log if too large
        max_log_length = 400000
        if len(job_log) > max_log_length:
            truncated_length = len(job_log) - max_log_length
            job_log = f"[... truncated first {truncated_length} chars ...]\n\n" + job_log[-max_log_length:]

        # Build prompt
        user_prompt = self._build_full_log_prompt(job_log, job_name, job_id)

        # Make API call with OpenAI format
        url = f"{self.api_base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.anthropic_api_key}"
        }

        payload = {
            "model": self.config.anthropic_model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_FULL_LOG_ANALYSIS},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.2,
            "max_tokens": 4096
        }

        response = requests.post(url, headers=headers, json=payload, timeout=120)
        response.raise_for_status()

        data = response.json()

        # Parse OpenAI format response
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})

        # Parse JSON from response
        result = self._parse_full_log_response(content)

        # Add token usage
        result['prompt_tokens'] = usage.get("prompt_tokens", 0)
        result['completion_tokens'] = usage.get("completion_tokens", 0)
        result['cached_tokens'] = usage.get("cached_tokens", 0)
        result['model_version'] = data.get("model", self.config.anthropic_model)

        return result
