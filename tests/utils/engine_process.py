# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from tests.utils.constants import DefaultPort
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import BasePayload, check_health_generate, check_models_api

logger = logging.getLogger(__name__)

FRONTEND_PORT = (
    DefaultPort.FRONTEND.value
)  # Do NOT use this in tests! Use allocate_port() instead.


class EngineResponseError(Exception):
    """Custom exception for engine response errors"""

    pass


class EngineLogError(Exception):
    """Custom exception for engine log validation errors"""

    pass


@dataclass
class EngineConfig:
    """Base configuration for engine test scenarios"""

    name: str
    directory: str
    marks: List[Any]
    request_payloads: List[BasePayload]
    model: str

    script_name: Optional[str] = None
    command: Optional[List[str]] = None
    script_args: Optional[List[str]] = None
    frontend_port: int = DefaultPort.FRONTEND.value
    timeout: int = 600
    delayed_start: int = 0
    env: Dict[str, str] = field(default_factory=dict)
    stragglers: list[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate that either script_name or command is provided, but not both."""
        if not self.script_name and not self.command:
            raise ValueError("Either script_name or command must be provided")
        if self.script_name and self.command:
            raise ValueError("Cannot provide both script_name and command")


class EngineProcess(ManagedProcess):
    """Base class for LLM engine processes (vLLM, TRT-LLM, etc.)"""

    def check_response(
        self,
        payload: BasePayload,
        response: requests.Response,
    ) -> None:
        """
        Check if the response is valid and contains expected content.

        Args:
            payload: The original payload (should have expected_response attribute)
            response: The response object
            response_handler: Function to extract content from response

        Raises:
            EngineResponseError: If the response is invalid or missing expected content
        """

        if response.status_code != 200:
            logger.error(
                "Response returned non-200 status code: %d", response.status_code
            )

            error_msg = f"Response returned non-200 status code: {response.status_code}"
            try:
                error_data = response.json()
                if "error" in error_data:
                    error_msg += f"\nError details: {error_data['error']}"
                logger.error(
                    "Response error details: %s", json.dumps(error_data, indent=2)
                )
            except Exception:
                logger.error("Response text: %s", response.text[:500])

            raise EngineResponseError(error_msg)

        try:
            content = payload.process_response(response)

            logger.info(
                "Extracted content: \n%s",
                content[:200] + "..."
                if isinstance(content, str) and len(content) > 200
                else content,
            )
        except AssertionError as e:
            raise EngineResponseError(str(e))
        except Exception as e:
            raise EngineResponseError(f"Failed to handle response: {e}")

        # Optionally validate expected log patterns after response handling
        if payload.expected_log:
            time.sleep(
                0.5
            )  # The kv event sometimes needs extra time to arrive and be reflected in the log.
            self.validate_expected_logs(payload.expected_log)

    def validate_expected_logs(self, patterns: Any) -> None:
        """Validate that all regex patterns are present in the current logs.

        Reads the full log via ManagedProcess.read_logs and searches for each
        provided regex pattern. Raises EngineLogError if any are missing.
        """
        import re  # local import to keep module load minimal

        content = self.read_logs() or ""
        if not content:
            raise EngineLogError(
                f"Log file not available or empty at path: {self.log_path}"
            )

        compiled = [re.compile(p) for p in patterns]
        missing = []
        for pattern, rx in zip(patterns, compiled):
            if not rx.search(content):
                missing.append(pattern)

        if missing:
            sample = content[-1000:] if len(content) > 1000 else content
            raise EngineLogError(
                f"Missing expected log patterns: {missing}\n\nLog sample:\n{sample}"
            )
        logger.info(f"SUCCESS: All expected log patterns: {patterns} found")

    @classmethod
    def from_config(
        cls,
        config: EngineConfig,
        request: Any,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> "EngineProcess":
        """Factory to create an EngineProcess from configuration (script or command)."""
        assert isinstance(config, EngineConfig), "Must use an instance of EngineConfig"

        if config.script_name:
            command = cls._build_script_command(config)
        elif config.command:
            command = config.command.copy()
        else:
            raise ValueError("Either script_name or command must be provided in config")

        env = os.environ.copy()
        if getattr(config, "env", None):
            env.update(config.env)
        if extra_env:
            env.update(extra_env)

        return cls(
            command=command,
            env=env,
            timeout=config.timeout,
            display_output=True,
            working_dir=config.directory,
            health_check_ports=[],
            health_check_urls=[
                (
                    f"http://localhost:{config.frontend_port}/v1/models",
                    check_models_api,
                ),
                (
                    f"http://localhost:{config.frontend_port}/health",
                    check_health_generate,
                ),
            ],
            delayed_start=config.delayed_start,
            terminate_existing=False,
            stragglers=config.stragglers,
            log_dir=request.node.name,
        )

    @classmethod
    def _build_script_command(cls, config: EngineConfig) -> List[str]:
        """Build command from script configuration."""
        assert (
            config.script_name
        ), "Must provide script_name to run fn _build_script_command"
        directory = config.directory
        script_path = os.path.join(directory, "launch", config.script_name)

        if not os.path.exists(script_path):
            raise FileNotFoundError(f"Script not found: {script_path}")

        command: List[str] = ["bash", script_path]
        if config.script_args:
            command.extend(config.script_args)

        return command

    @classmethod
    def from_script(
        cls,
        config: EngineConfig,
        request: Any,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> "EngineProcess":
        """Factory to create an EngineProcess configured to run a launch script.

        Deprecated: Use from_config() instead.
        """
        return cls.from_config(config, request, extra_env)

    @classmethod
    def from_command(
        cls,
        config: EngineConfig,
        request: Any,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> "EngineProcess":
        """Factory to create an EngineProcess configured to run a direct command.

        Deprecated: Use from_config() instead.
        """
        return cls.from_config(config, request, extra_env)
