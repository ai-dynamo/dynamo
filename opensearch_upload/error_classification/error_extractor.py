"""
Extract errors from multiple sources (JUnit XML, BuildKit logs, GitHub annotations).
"""
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


@dataclass
class ErrorContext:
    """Normalized error context for classification."""

    # Error content
    error_text: str
    source_type: str  # pytest|buildkit|rust_test|github_annotation

    # Source references
    workflow_id: Optional[str] = None
    job_id: Optional[str] = None
    step_id: Optional[str] = None
    test_name: Optional[str] = None

    # Framework context
    framework: Optional[str] = None  # vllm|sglang|trtllm|rust

    # Job context
    job_name: Optional[str] = None
    step_name: Optional[str] = None

    # Common metadata
    repo: Optional[str] = None
    workflow_name: Optional[str] = None
    branch: Optional[str] = None
    pr_id: Optional[str] = None
    commit_sha: Optional[str] = None
    user_alias: Optional[str] = None

    # Timestamps
    timestamp: Optional[str] = None

    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        return {
            "error_text": self.error_text,
            "source_type": self.source_type,
            "workflow_id": self.workflow_id,
            "job_id": self.job_id,
            "step_id": self.step_id,
            "test_name": self.test_name,
            "framework": self.framework,
            "job_name": self.job_name,
            "step_name": self.step_name,
            "repo": self.repo,
            "workflow_name": self.workflow_name,
            "branch": self.branch,
            "pr_id": self.pr_id,
            "commit_sha": self.commit_sha,
            "user_alias": self.user_alias,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
        }


class ErrorExtractor:
    """Extract errors from various sources."""

    def extract_from_junit_xml(
        self,
        xml_path: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ErrorContext]:
        """
        Extract errors from pytest JUnit XML results.

        Args:
            xml_path: Path to JUnit XML file
            context: Additional context (workflow_id, job_id, etc.)

        Returns:
            List of ErrorContext objects
        """
        if not os.path.exists(xml_path):
            print(f"  ⚠️  JUnit XML not found: {xml_path}")
            return []

        errors = []
        context = context or {}

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Iterate through testcases
            for testcase in root.findall('.//testcase'):
                # Check for failures or errors
                failure = testcase.find('failure')
                error = testcase.find('error')

                if failure is not None or error is not None:
                    element = failure if failure is not None else error

                    # Extract test information
                    classname = testcase.get('classname', '')
                    test_name = testcase.get('name', '')
                    full_test_name = f"{classname}::{test_name}" if classname else test_name

                    # Extract error message and details
                    error_message = element.get('message', '')
                    error_details = element.text or ''

                    # Combine message and details
                    error_text = f"{error_message}\n{error_details}".strip()

                    if not error_text:
                        continue

                    # Detect framework from test path
                    framework = self._detect_framework(classname, full_test_name)

                    error_context = ErrorContext(
                        error_text=error_text,
                        source_type="pytest",
                        test_name=full_test_name,
                        framework=framework,
                        workflow_id=context.get('workflow_id'),
                        job_id=context.get('job_id'),
                        step_id=context.get('step_id'),
                        job_name=context.get('job_name'),
                        step_name=context.get('step_name'),
                        repo=context.get('repo'),
                        workflow_name=context.get('workflow_name'),
                        branch=context.get('branch'),
                        pr_id=context.get('pr_id'),
                        commit_sha=context.get('commit_sha'),
                        user_alias=context.get('user_alias'),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                        metadata={
                            'test_classname': classname,
                            'test_file': testcase.get('file', ''),
                            'test_time': testcase.get('time', ''),
                        }
                    )

                    errors.append(error_context)

        except ET.ParseError as e:
            print(f"  ✗ Error parsing JUnit XML {xml_path}: {e}")

        except Exception as e:
            print(f"  ✗ Unexpected error extracting from JUnit XML: {e}")

        return errors

    def extract_from_buildkit_log(
        self,
        log_content: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ErrorContext]:
        """
        Extract errors from BuildKit/Docker build logs.

        Args:
            log_content: BuildKit log content
            context: Additional context

        Returns:
            List of ErrorContext objects
        """
        errors = []
        context = context or {}

        try:
            # Split into lines
            lines = log_content.split('\n')

            # Look for ERROR status lines
            # Format: #10 [stage 0/3] RUN pip install ...
            # ERROR [stage 0/3] RUN pip install ...

            error_pattern = re.compile(r'ERROR\s+\[.*?\]\s+(.+)')
            current_error = None
            error_lines = []

            for line in lines:
                match = error_pattern.match(line)

                if match:
                    # Save previous error if exists
                    if current_error and error_lines:
                        error_text = '\n'.join(error_lines)
                        errors.append(self._create_buildkit_error(
                            error_text,
                            current_error,
                            context
                        ))

                    # Start new error
                    current_error = match.group(1)
                    error_lines = [line]

                elif current_error and line.strip():
                    # Continue collecting error lines
                    error_lines.append(line)

                elif current_error and not line.strip():
                    # Empty line, end of error
                    error_text = '\n'.join(error_lines)
                    errors.append(self._create_buildkit_error(
                        error_text,
                        current_error,
                        context
                    ))
                    current_error = None
                    error_lines = []

            # Don't forget last error
            if current_error and error_lines:
                error_text = '\n'.join(error_lines)
                errors.append(self._create_buildkit_error(
                    error_text,
                    current_error,
                    context
                ))

        except Exception as e:
            print(f"  ✗ Error extracting from BuildKit log: {e}")

        return errors

    def extract_from_github_annotations(
        self,
        annotations: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]] = None
    ) -> List[ErrorContext]:
        """
        Extract errors from GitHub annotations.

        Args:
            annotations: List of annotation dictionaries
            context: Additional context

        Returns:
            List of ErrorContext objects
        """
        errors = []
        context = context or {}

        try:
            for annotation in annotations:
                # Only process errors and failures
                level = annotation.get('annotation_level', '')
                if level not in ['failure', 'error']:
                    continue

                message = annotation.get('message', '')
                if not message:
                    continue

                # Extract details
                title = annotation.get('title', '')
                raw_details = annotation.get('raw_details', '')

                # Combine title, message, and details
                error_parts = [part for part in [title, message, raw_details] if part]
                error_text = '\n'.join(error_parts)

                error_context = ErrorContext(
                    error_text=error_text,
                    source_type="github_annotation",
                    workflow_id=context.get('workflow_id'),
                    job_id=context.get('job_id'),
                    step_id=context.get('step_id'),
                    job_name=context.get('job_name'),
                    step_name=context.get('step_name'),
                    repo=context.get('repo'),
                    workflow_name=context.get('workflow_name'),
                    branch=context.get('branch'),
                    pr_id=context.get('pr_id'),
                    commit_sha=context.get('commit_sha'),
                    user_alias=context.get('user_alias'),
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    metadata={
                        'annotation_level': level,
                        'path': annotation.get('path', ''),
                        'start_line': annotation.get('start_line'),
                        'end_line': annotation.get('end_line'),
                    }
                )

                errors.append(error_context)

        except Exception as e:
            print(f"  ✗ Error extracting from GitHub annotations: {e}")

        return errors

    def extract_from_annotation_messages(
        self,
        annotations_text: str,
        context: Optional[Dict[str, Any]] = None
    ) -> List[ErrorContext]:
        """
        Extract errors from s_annotation_messages field (pipe-separated).

        Args:
            annotations_text: Pipe-separated annotation messages
            context: Additional context

        Returns:
            List of ErrorContext objects
        """
        errors = []
        context = context or {}

        try:
            # Split by pipe separator
            messages = annotations_text.split('|')

            for message in messages:
                message = message.strip()

                # Look for error/failure markers
                if '[FAILURE]' in message or '[ERROR]' in message or 'error' in message.lower():
                    error_context = ErrorContext(
                        error_text=message,
                        source_type="github_annotation",
                        workflow_id=context.get('workflow_id'),
                        job_id=context.get('job_id'),
                        step_id=context.get('step_id'),
                        job_name=context.get('job_name'),
                        step_name=context.get('step_name'),
                        repo=context.get('repo'),
                        workflow_name=context.get('workflow_name'),
                        branch=context.get('branch'),
                        pr_id=context.get('pr_id'),
                        commit_sha=context.get('commit_sha'),
                        user_alias=context.get('user_alias'),
                        timestamp=datetime.now(timezone.utc).isoformat(),
                    )

                    errors.append(error_context)

        except Exception as e:
            print(f"  ✗ Error extracting from annotation messages: {e}")

        return errors

    def _detect_framework(self, classname: str, test_name: str) -> Optional[str]:
        """Detect framework from test path."""
        combined = f"{classname} {test_name}".lower()

        if 'vllm' in combined:
            return 'vllm'
        elif 'sglang' in combined or 'sgl' in combined:
            return 'sglang'
        elif 'trtllm' in combined or 'tensorrt' in combined:
            return 'trtllm'
        elif 'rust' in combined or '.rs' in combined:
            return 'rust'

        return None

    def _create_buildkit_error(
        self,
        error_text: str,
        command: str,
        context: Dict[str, Any]
    ) -> ErrorContext:
        """Create ErrorContext for BuildKit error."""
        return ErrorContext(
            error_text=error_text,
            source_type="buildkit",
            workflow_id=context.get('workflow_id'),
            job_id=context.get('job_id'),
            step_id=context.get('step_id'),
            job_name=context.get('job_name'),
            step_name=context.get('step_name'),
            repo=context.get('repo'),
            workflow_name=context.get('workflow_name'),
            branch=context.get('branch'),
            pr_id=context.get('pr_id'),
            commit_sha=context.get('commit_sha'),
            user_alias=context.get('user_alias'),
            timestamp=datetime.now(timezone.utc).isoformat(),
            metadata={
                'command': command,
            }
        )
