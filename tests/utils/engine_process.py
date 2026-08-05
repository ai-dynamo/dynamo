# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import requests

from tests.utils.constants import DefaultPort
from tests.utils.inference_endpoint import InferenceEndpoint
from tests.utils.managed_process import ManagedProcess
from tests.utils.payloads import BasePayload, check_health_generate, check_models_api

# Re-exported for backwards compatibility: these moved to the
# deployment-agnostic verification module so tests/deploy can raise and catch
# the same errors without importing the local-process stack. The class names
# are load-bearing -- tests select them by name in
# @pytest.mark.flaky(only_rerun=[...]).
from tests.utils.verification import (  # noqa: F401
    EngineLogError,
    EngineResponseError,
    ResponseValidationError,
    check_response,
    validate_expected_logs,
)

logger = logging.getLogger(__name__)


FRONTEND_PORT = (
    DefaultPort.FRONTEND.value
)  # Do NOT use this in tests! Use allocate_port() instead.


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
    health_check_workers: bool = False
    # How many worker system ports (DYN_SYSTEM_PORT1..N) the launch script
    # actually binds. The port fixture may allocate more than the script uses
    # (num_system_ports is sized for the largest config in the module), so the
    # health check must not probe beyond this count.
    health_check_worker_count: int = 2
    health_check_funcs: List[Any] = field(default_factory=list)
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

    # Frontend port this engine was launched on, recorded by from_config so the
    # process can hand out a deployment-agnostic InferenceEndpoint. ManagedProcess
    # is a dataclass and this is not a constructor argument, so it is declared
    # here as a class attribute like the other private ManagedProcess state.
    _frontend_port: Optional[int] = None

    @staticmethod
    def worker_health_check_urls(env: Dict[str, str], count: int) -> List[str]:
        """Build /health URLs for the worker system ports a launch script binds.

        The dynamic-port fixture may inject more DYN_SYSTEM_PORT* env vars than
        the script uses (num_system_ports is sized for the largest config in
        the module), so probe exactly DYN_SYSTEM_PORT1..count. A missing or
        malformed value inside the declared range is a configuration error —
        skipping it would silently weaken the readiness gate.
        """
        if count < 1:
            raise ValueError(
                "health_check_worker_count must be >= 1 when "
                f"health_check_workers is enabled, got {count}"
            )
        urls = []
        for idx in range(1, count + 1):
            key = f"DYN_SYSTEM_PORT{idx}"
            val = env.get(key, "")
            if not val.isdigit():
                raise ValueError(
                    f"health_check_workers declares {count} worker(s) but "
                    f"{key} is not a valid port: {val!r}"
                )
            urls.append(f"http://localhost:{val}/health")
        return urls

    def endpoint(self, model: Optional[str] = None) -> InferenceEndpoint:
        """The frontend address, as the deployment-agnostic handle.

        A test that only sends payloads and asserts on responses should take
        this and nothing else from the process.
        """
        if self._frontend_port is None:
            raise RuntimeError(
                "EngineProcess has no frontend port: build it with from_config() "
                "so the launch configuration's frontend_port is recorded"
            )
        return InferenceEndpoint.from_port(self._frontend_port, model=model)

    def check_response(
        self,
        payload: BasePayload,
        response: requests.Response,
    ) -> None:
        """Validate a response, using this process's log for ``expected_log``.

        Thin wrapper over the deployment-agnostic
        :func:`tests.utils.verification.check_response`; the process is passed
        as the log source because it is the only part that needs a handle.
        """
        check_response(payload, response, log_source=self)

    def validate_expected_logs(self, patterns: Any) -> None:
        """Assert every regex in ``patterns`` appears in this process's log."""
        validate_expected_logs(patterns, self)

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

        frontend_checks = [
            (
                f"http://localhost:{config.frontend_port}/v1/models",
                check_models_api,
            ),
            (
                f"http://localhost:{config.frontend_port}/health",
                check_health_generate,
            ),
        ]

        # For disagg-same-gpu deployments, health-check each worker's
        # system port so we wait for ALL workers to be ready, not just the
        # first one to register with the frontend.  Worker liveness checks
        # run FIRST so the frontend has time to discover newly-registered
        # workers before the frontend endpoint checks run.
        #
        # NOTE: DYN_SYSTEM_PORT* env vars are injected by the dynamic port
        # fixtures for ALL tests, so we gate on health_check_workers (only
        # set by same-gpu disagg configs) to avoid health-checking ports
        # that don't serve /health in regular multi-GPU tests.
        #
        # Only probe DYN_SYSTEM_PORT1..health_check_worker_count: the fixture
        # allocates num_system_ports for the largest config in the module
        # (e.g. 4 for disaggregated_router), so a greedy scan of every
        # DYN_SYSTEM_PORT* var would wait forever on ports no process binds.
        delayed = config.delayed_start
        worker_checks: list[tuple] = []
        if config.health_check_workers:
            worker_checks = [
                (url, None)
                for url in cls.worker_health_check_urls(
                    env, config.health_check_worker_count
                )
            ]
            delayed = 0

        health_urls = worker_checks + frontend_checks

        instance = cls(
            command=command,
            env=env,
            timeout=config.timeout,
            display_output=True,
            working_dir=config.directory,
            health_check_ports=[],
            health_check_urls=health_urls,
            health_check_funcs=list(config.health_check_funcs),
            delayed_start=delayed,
            # Must stay False: command[0] is "bash", so True would kill every
            # bash process system-wide.  Stale cleanup relies on stragglers list
            # and process-group termination in __exit__ instead.
            terminate_all_matching_process_names=False,
            stragglers=config.stragglers,
            log_dir=request.node.name,
        )
        instance._frontend_port = config.frontend_port
        return instance

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
