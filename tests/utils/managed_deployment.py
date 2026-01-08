# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import re
import secrets
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, List, Optional

import kr8s
import requests
import yaml
from kr8s.objects import Pod, Service
from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import exceptions

# LogStreamManager removed - using PVC-based log collection only


def _get_workspace_dir() -> str:
    """Get workspace directory without depending on dynamo.common package.

    This allows tests to run without requiring dynamo package to be installed.
    """
    # Start from this file's location and walk up to find workspace root
    current = os.path.dirname(os.path.abspath(__file__))
    while current != os.path.dirname(current):  # Stop at filesystem root
        # Workspace root has pyproject.toml
        if os.path.exists(os.path.join(current, "pyproject.toml")):
            return current
        current = os.path.dirname(current)

    # Fallback: assume workspace is 3 levels up from tests/utils/
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class ServiceSpec:
    """Wrapper around a single service in the deployment spec."""

    def __init__(self, service_name: str, service_spec: dict):
        self._name = service_name
        self._spec = service_spec

    @property
    def name(self) -> str:
        """The service name (read-only)"""
        return self._name

    # ----- Image -----
    @property
    def image(self) -> Optional[str]:
        """Container image for the service"""
        try:
            return self._spec["extraPodSpec"]["mainContainer"]["image"]
        except KeyError:
            return None

    @image.setter
    def image(self, value: str):
        if "extraPodSpec" not in self._spec:
            self._spec["extraPodSpec"] = {"mainContainer": {}}
        if "mainContainer" not in self._spec["extraPodSpec"]:
            self._spec["extraPodSpec"]["mainContainer"] = {}
        self._spec["extraPodSpec"]["mainContainer"]["image"] = value

    @property
    def component_type(self):
        return self._spec["componentType"]

    @property
    def envs(self) -> list[dict[str, str]]:
        """Environment variables for the service"""
        return self._spec.get("envs", [])

    @envs.setter
    def envs(self, value: list[dict[str, str]]):
        self._spec["envs"] = value

    # ----- Replicas -----
    @property
    def replicas(self) -> int:
        return self._spec.get("replicas", 0)

    @replicas.setter
    def replicas(self, value: int):
        self._spec["replicas"] = value

    @property
    def model(self) -> Optional[str]:
        """Model being served by this service (checks both --model and --model-path)"""
        try:
            args_list = self._spec["extraPodSpec"]["mainContainer"]["args"]
        except KeyError:
            return None
        args_str = " ".join(args_list)
        parts = shlex.split(args_str)
        for i, part in enumerate(parts):
            if part in ["--model", "--model-path"]:
                return parts[i + 1] if i + 1 < len(parts) else None
        return None

    @model.setter
    def model(self, value: str):
        if "extraPodSpec" not in self._spec:
            return
        if "mainContainer" not in self._spec["extraPodSpec"]:
            return

        args_list = self._spec["extraPodSpec"]["mainContainer"].get("args", [])
        args_str = " ".join(args_list)
        parts = shlex.split(args_str)

        # Try to update --model first, then --model-path
        model_index = None
        for i, part in enumerate(parts):
            if part in ["--model", "--model-path"]:
                model_index = i
                break

        if model_index is not None:
            if model_index + 1 < len(parts):
                parts[model_index + 1] = value
            else:
                return
        else:
            return

        # Store args as a list of separate strings for proper command-line parsing
        # WRONG: [" ".join(parts)] creates ["--model Qwen/Qwen3-0.6B"] (single string)
        # RIGHT: parts creates ["--model", "Qwen/Qwen3-0.6B"] (separate strings)
        self._spec["extraPodSpec"]["mainContainer"]["args"] = parts

    # ----- GPUs -----
    @property
    def gpus(self) -> int:
        try:
            return int(self._spec["resources"]["limits"]["gpu"])
        except KeyError:
            return 0

    @gpus.setter
    def gpus(self, value: int):
        if "resources" not in self._spec:
            self._spec["resources"] = {}
        if "limits" not in self._spec["resources"]:
            self._spec["resources"]["limits"] = {}
        self._spec["resources"]["limits"]["gpu"] = str(value)

    @property
    def tensor_parallel_size(self) -> int:
        """Get tensor parallel size from vLLM arguments"""
        try:
            args_list = self._spec["extraPodSpec"]["mainContainer"]["args"]
        except KeyError:
            return 1  # Default tensor parallel size

        args_str = " ".join(args_list)
        parts = shlex.split(args_str)
        for i, part in enumerate(parts):
            if part == "--tensor-parallel-size":
                return int(parts[i + 1]) if i + 1 < len(parts) else 1
        return 1

    @tensor_parallel_size.setter
    def tensor_parallel_size(self, value: int):
        if "extraPodSpec" not in self._spec:
            return
        if "mainContainer" not in self._spec["extraPodSpec"]:
            return

        args_list = self._spec["extraPodSpec"]["mainContainer"].get("args", [])
        args_str = " ".join(args_list)
        parts = shlex.split(args_str)

        # Find existing tensor-parallel-size argument
        tp_index = None
        for i, part in enumerate(parts):
            if part == "--tensor-parallel-size":
                tp_index = i
                break

        if tp_index is not None:
            # Update existing value
            if tp_index + 1 < len(parts):
                parts[tp_index + 1] = str(value)
            else:
                parts.append(str(value))
        else:
            # Add new argument
            parts.extend(["--tensor-parallel-size", str(value)])

        # Store args as a list of separate strings for proper command-line parsing
        # When TP > 1, this setter is called and adds --tensor-parallel-size to args.
        # WRONG: [" ".join(parts)] would create ["--model Qwen/Qwen3-0.6B --tensor-parallel-size 2"]
        #        causing argparse to fail with "IndexError: list index out of range"
        # RIGHT: parts creates ["--model", "Qwen/Qwen3-0.6B", "--tensor-parallel-size", "2"]
        self._spec["extraPodSpec"]["mainContainer"]["args"] = parts

        # Auto-adjust GPU count to match tensor parallel size
        self.gpus = value

    # ----- Readiness Probe -----
    def set_readiness_probe(
        self,
        period_seconds: int = 10,
        initial_delay_seconds: int = 0,
        timeout_seconds: int = 4,
        failure_threshold: int = 3,
        path: str = "/health",
        port: int = 9090,
    ):
        """Set readiness probe configuration for this service.

        Args:
            period_seconds: How often to perform the probe (default: 10)
            initial_delay_seconds: Delay before first probe (default: 0)
            timeout_seconds: Probe timeout (default: 4)
            failure_threshold: Failures before marking unready (default: 3)
            path: HTTP path to probe (default: "/health")
            port: Port to probe (default: 9090 for workers, use 8000 for frontend)
        """
        self._spec["readinessProbe"] = {
            "httpGet": {
                "path": path,
                "port": port,
            },
            "periodSeconds": period_seconds,
            "initialDelaySeconds": initial_delay_seconds,
            "timeoutSeconds": timeout_seconds,
            "failureThreshold": failure_threshold,
        }

    # ----- Termination Grace Period -----
    def set_termination_grace_period(self, seconds: int = 60):
        """Set termination grace period for this service's pods.

        Args:
            seconds: Grace period in seconds (default: 60)
        """
        if "extraPodSpec" not in self._spec:
            self._spec["extraPodSpec"] = {}
        self._spec["extraPodSpec"]["terminationGracePeriodSeconds"] = seconds


class DeploymentSpec:
    def __init__(
        self, base: str, endpoint="/v1/chat/completions", port=8000, system_port=9090
    ):
        """Load the deployment YAML file"""
        with open(base, "r") as f:
            self._deployment_spec = yaml.safe_load(f)
        self._endpoint = endpoint
        self._port = port
        self._system_port = system_port

    @property
    def name(self) -> str:
        """Deployment name"""
        return self._deployment_spec["metadata"]["name"]

    @name.setter
    def name(self, value: str):
        self._deployment_spec["metadata"]["name"] = value

    @property
    def port(self) -> int:
        """Deployment port"""
        return self._port

    @property
    def system_port(self) -> int:
        """Deployment port"""
        return self._system_port

    @property
    def endpoint(self) -> str:
        return self._endpoint

    @property
    def frontend_service(self) -> ServiceSpec:
        for service in self.services:
            if service.component_type == "frontend":
                return service

    @property
    def namespace(self) -> str:
        """Deployment namespace"""
        return self._deployment_spec["metadata"]["namespace"]

    @namespace.setter
    def namespace(self, value: str):
        self._deployment_spec["metadata"]["namespace"] = value

    def disable_grove(self):
        if "annotations" not in self._deployment_spec["metadata"]:
            self._deployment_spec["metadata"]["annotations"] = {}
        self._deployment_spec["metadata"]["annotations"][
            "nvidia.com/enable-grove"
        ] = "false"

    def get_model(self, service_name: Optional[str] = None) -> Optional[str]:
        if service_name is None:
            services = self.services
        else:
            services = [self[service_name]]
        for service in services:
            if service.model:
                return service.model
        return None

    def set_model(self, model: str, service_name: Optional[str] = None):
        if service_name is None:
            services = self.services
        else:
            services = [self[service_name]]
        for service in services:
            service.model = model

    def set_image(self, image: str, service_name: Optional[str] = None):
        if service_name is None:
            services = self.services
        else:
            services = [self[service_name]]
        for service in services:
            service.image = image

    def set_tensor_parallel(self, tp_size: int, service_names: Optional[list] = None):
        """Scale deployment for different tensor parallel configurations

        Args:
            tp_size: Target tensor parallel size
            service_names: List of service names to update (defaults to worker services)
        """
        if service_names is None:
            # Auto-detect worker services (services with GPU requirements)
            service_names = [svc.name for svc in self.services if svc.gpus > 0]

        for service_name in service_names:
            service = self[service_name]
            service.tensor_parallel_size = tp_size
            service.gpus = tp_size

    def set_logging(self, enable_jsonl: bool = True, log_level: str = "debug"):
        """Configure logging for the deployment

        Args:
            enable_jsonl: Enable JSON line logging (sets DYN_LOGGING_JSONL=true)
            log_level: Set log level (sets DYN_LOG to specified level)
        """
        spec = self._deployment_spec
        if "envs" not in spec["spec"]:
            spec["spec"]["envs"] = []

        # Remove any existing logging env vars to avoid duplicates
        spec["spec"]["envs"] = [
            env
            for env in spec["spec"]["envs"]
            if env.get("name") not in ["DYN_LOGGING_JSONL", "DYN_LOG"]
        ]

        if enable_jsonl:
            spec["spec"]["envs"].append({"name": "DYN_LOGGING_JSONL", "value": "true"})

        if log_level:
            spec["spec"]["envs"].append({"name": "DYN_LOG", "value": log_level})

    def get_logging_config(self) -> dict:
        """Get current logging configuration

        Returns:
            dict with 'jsonl_enabled' and 'log_level' keys
        """
        envs = self._deployment_spec.get("spec", {}).get("envs", [])

        jsonl_enabled = False
        log_level = None

        for env in envs:
            if env.get("name") == "DYN_LOGGING_JSONL":
                jsonl_enabled = env.get("value") in ["true", "1"]
            elif env.get("name") == "DYN_LOG":
                log_level = env.get("value")

        return {"jsonl_enabled": jsonl_enabled, "log_level": log_level}

    def set_service_env_var(self, service_name: str, name: str, value: str):
        """
        Set an environment variable for a specific service
        """
        service = self.get_service(service_name)
        envs = service.envs if service.envs is not None else []

        # if env var already exists, update it
        for env in envs:
            if env["name"] == name:
                env["value"] = value
                service.envs = envs  # Save back to trigger the setter
                return

        # if env var does not exist, add it
        envs.append({"name": name, "value": value})
        service.envs = envs  # Save back to trigger the setter

    def get_service_env_vars(self, service_name: str) -> list[dict]:
        """
        Get all environment variables for a specific service

        Returns:
            List of environment variable dicts (e.g., [{"name": "VAR", "value": "val"}])
        """
        service = self.get_service(service_name)
        return service.envs

    @property
    def services(self) -> list[ServiceSpec]:
        """List of ServiceSpec objects"""
        return [
            ServiceSpec(svc, spec)
            for svc, spec in self._deployment_spec["spec"]["services"].items()
        ]

    def __getitem__(self, service_name: str) -> ServiceSpec:
        """Allow dict-like access: d['Frontend']"""
        return ServiceSpec(
            service_name, self._deployment_spec["spec"]["services"][service_name]
        )

    def spec(self):
        return self._deployment_spec

    def add_arg_to_service(self, service_name: str, arg_name: str, arg_value: str):
        """
        Add or override a command-line argument for a specific service

        Args:
            service_name: Name of the service (e.g., "VllmDecodeWorker", "TRTLLMWorker")
            arg_name: Argument name (e.g., "--max-model-len", "--max-seq-len")
            arg_value: Argument value (e.g., "1024")
        """
        service = self.get_service(service_name)
        service_spec = service._spec

        # Ensure args list exists
        if "extraPodSpec" not in service_spec:
            service_spec["extraPodSpec"] = {"mainContainer": {}}
        if "mainContainer" not in service_spec["extraPodSpec"]:
            service_spec["extraPodSpec"]["mainContainer"] = {}
        if "args" not in service_spec["extraPodSpec"]["mainContainer"]:
            service_spec["extraPodSpec"]["mainContainer"]["args"] = []

        args_list = service_spec["extraPodSpec"]["mainContainer"]["args"]

        # Convert to list if needed (sometimes it's a single string)
        if isinstance(args_list, str):
            import shlex

            args_list = shlex.split(args_list)
            service_spec["extraPodSpec"]["mainContainer"]["args"] = args_list

        # Find existing argument
        arg_index = None
        for i, arg in enumerate(args_list):
            if arg == arg_name:
                arg_index = i
                break

        if arg_index is not None:
            # Argument found, check if it has a value
            if arg_index + 1 < len(args_list) and not args_list[
                arg_index + 1
            ].startswith("-"):
                # Has a value, replace it
                args_list[arg_index + 1] = arg_value
            else:
                # No value after the argument, insert the value
                args_list.insert(arg_index + 1, arg_value)
        else:
            # Add new argument
            args_list.extend([arg_name, arg_value])

    def get_service(self, service_name: str) -> ServiceSpec:
        """
        Get a specific service from the deployment spec
        """
        if service_name not in self._deployment_spec["spec"]["services"]:
            raise ValueError(f"Service '{service_name}' not found in deployment spec")

        return ServiceSpec(
            service_name, self._deployment_spec["spec"]["services"][service_name]
        )

    def set_service_replicas(self, service_name: str, replicas: int):
        """
        Set the number of replicas for a specific service
        """
        service = self.get_service(service_name)
        service.replicas = replicas

    def set_service_readiness_probe(
        self, service_name: str, period_seconds: int, **kwargs
    ):
        """Set readiness probe for a specific service.

        Args:
            service_name: Name of the service (e.g., "TRTLLMDecodeWorker")
            period_seconds: How often to perform the probe
            **kwargs: Additional probe options (initial_delay_seconds, timeout_seconds, etc.)
        """
        service = self.get_service(service_name)
        service.set_readiness_probe(period_seconds=period_seconds, **kwargs)

    def set_service_termination_grace_period(self, service_name: str, seconds: int):
        """Set termination grace period for a specific service.

        Args:
            service_name: Name of the service (e.g., "TRTLLMDecodeWorker")
            seconds: Grace period in seconds
        """
        service = self.get_service(service_name)
        service.set_termination_grace_period(seconds)

    def save(self, out_file: str):
        """Save updated deployment to file"""
        with open(out_file, "w") as f:
            yaml.safe_dump(self._deployment_spec, f, default_flow_style=False)

    def enable_log_collection(
        self,
        pvc_name=None,
        pvc_size="1Gi",
        storage_class="nebius-shared-fs",
        container_log_dir="/tmp/service_logs",
        enable_all_services=True,
        service_names=None,
    ):
        """
        Enable log collection using a PersistentVolumeClaim with RWX (ReadWriteMany) access.

        This approach creates a PVC that can be shared across all pods in the deployment,
        allowing reliable log collection that persists through pod restarts and deletions.

        IMPORTANT: Requires a storage class that supports ReadWriteMany (RWX) access mode.
        If the cluster does not support RWX, an error will be raised at deployment time.

        Args:
            pvc_name: Name of the PVC to create (auto-generated if not provided)
            pvc_size: Size of the PVC (e.g., "1Gi", "500Mi")
            storage_class: Storage class name (must support RWX). Default: "nebius-shared-fs"
            container_log_dir: Directory inside container where logs are written
            enable_all_services: If True, wrap commands for all services
            service_names: List of specific service names to wrap (used if enable_all_services is False)

        Raises:
            RuntimeError: If the storage class does not support RWX access mode
        """
        if enable_all_services:
            target_services = self.services
        else:
            target_services = [self[name] for name in (service_names or [])]

        # Generate unique PVC name if not provided - do this once and store it
        if pvc_name is None:
            import random
            import time

            # Use timestamp + random to ensure uniqueness even if multiple deployments start simultaneously
            timestamp = int(time.time())
            rand_suffix = random.randint(1000, 9999)
            pvc_name = f"{self.name}-logs-{timestamp}-{rand_suffix}"

        # Store PVC info for later use by ManagedDeployment
        self._log_collection_pvc_name = pvc_name
        self._log_collection_pvc_size = pvc_size
        self._log_collection_storage_class = storage_class
        self._log_collection_container_dir = container_log_dir

        # Debug: Log the generated PVC name for consistency verification
        print(f"[DEBUG] Generated PVC name: {pvc_name}")

        # Add PVC at deployment level (following recipe pattern)
        if "pvcs" not in self._deployment_spec["spec"]:
            self._deployment_spec["spec"]["pvcs"] = []

        # Remove any existing log PVCs to avoid conflicts
        self._deployment_spec["spec"]["pvcs"] = [
            pvc
            for pvc in self._deployment_spec["spec"]["pvcs"]
            if not pvc.get("name", "").endswith("-logs-pvc")
            and pvc.get("name") != pvc_name
        ]

        # Add the log collection PVC (will be created by ManagedDeployment)
        self._deployment_spec["spec"]["pvcs"].append(
            {
                "name": pvc_name,
                "create": False,  # PVC will be created manually by ManagedDeployment
            }
        )

        # Wrap all target services to use PVC-based logging
        for service in target_services:
            self._wrap_service_command_with_pvc(service, container_log_dir, pvc_name)

    def _wrap_service_command_with_pvc(
        self, service: ServiceSpec, container_log_dir: str, pvc_name: str
    ):
        """Wrap service command with PVC volume mounting for log collection."""
        # Check if command is already wrapped to avoid double wrapping
        main_container = service._spec.get("extraPodSpec", {}).get("mainContainer", {})
        existing_command = main_container.get("command", [])

        if (
            len(existing_command) >= 3
            and existing_command[:2] == ["/bin/bash", "-c"]
            and "tee -a" in existing_command[2]
        ):
            return  # Already wrapped, skip

        # Do the standard command wrapping
        self._wrap_service_command(service, container_log_dir)

        # Clean up any existing service-level volume mounts that might conflict
        if "volumeMounts" in service._spec:
            service._spec["volumeMounts"] = [
                mount
                for mount in service._spec["volumeMounts"]
                if not (
                    mount.get("name", "").startswith("service-logs")
                    or mount.get("mountPoint") == container_log_dir
                )
            ]

        # Add volume mount at service level (following Dynamo operator and recipe pattern)
        if "volumeMounts" not in service._spec:
            service._spec["volumeMounts"] = []

        service._spec["volumeMounts"].append(
            {
                "name": pvc_name,  # Use the PVC name directly (not service-logs-pvc)
                "mountPoint": container_log_dir,
            }
        )

    def _get_service_default_command(
        self, service: ServiceSpec, main_container: dict
    ) -> tuple[list[str], list[str]]:
        """
        Get the default command and args for a service.

        Priority:
        1. Explicit command/args in main_container (from YAML)
        2. Component type defaults (for services like frontend that have Go operator defaults)

        Args:
            service: The service specification
            main_container: The mainContainer dict from extraPodSpec

        Returns:
            tuple: (command, args) where both are lists of strings
        """
        # First, check if command/args are explicitly set in the main container
        existing_command = main_container.get("command", [])
        existing_args = main_container.get("args", [])

        if existing_command or existing_args:
            return existing_command, existing_args

        # If no explicit command, use component type defaults
        component_type = service.component_type

        # Frontend default command (from Go operator: component_frontend.go)
        if component_type == "frontend":
            return ["python3"], ["-m", "dynamo.frontend"]

        # Default fallback - this should be rare since most services define their commands
        return ["python3"], []

    def _wrap_service_command(self, service: ServiceSpec, container_log_dir: str):
        """
        Wrap a service's command to tee output to a log file in the volume.

        The wrapping strategy:
        1. Get the original command and args (either explicit or default)
        2. Create a shell script that:
           - Sets up the log directory
           - Creates a unique log file name (with timestamp and random suffix)
           - Runs the original command while teeing output to the log file
           - Handles signals properly for graceful shutdown
        """
        # Ensure extraPodSpec exists
        if "extraPodSpec" not in service._spec:
            service._spec["extraPodSpec"] = {"mainContainer": {}}
        if "mainContainer" not in service._spec["extraPodSpec"]:
            service._spec["extraPodSpec"]["mainContainer"] = {}

        main_container = service._spec["extraPodSpec"]["mainContainer"]

        # Get original command and args
        original_command, original_args = self._get_service_default_command(
            service, main_container
        )

        # Create the log file name with timestamp and random component for uniqueness

        #        timestamp = int(time.time())
        #        log_filename = f"{service.name.lower()}_{timestamp}_$RANDOM.log"
        #        log_path = f"{container_log_dir}/{log_filename}"

        # Build the original command string
        if original_args:
            full_original_command = " ".join(original_command + original_args)
        else:
            full_original_command = " ".join(original_command)

        # Create the wrapper script
        # Uses $POD_NAME (from downward API) for log filename to match kubectl logs naming
        #
        # Key design: Use exec with process substitution so signals go directly to the
        # main process without any trapping. The main process becomes the shell's replacement
        # and receives signals exactly as if it were PID 1.
        wrapper_script = f"""#!/bin/bash
set -e

# Setup log directory
mkdir -p {container_log_dir}

# Use actual pod name (from downward API) + timestamp for unique log file per restart
# This matches the naming used by kubectl logs (e.g., trtllm-disagg-0-frontend-sqfcx.log)
LOG_FILE="{container_log_dir}/${{POD_NAME}}_$(date +%s).log"

# Log startup information
echo "=== SERVICE START at $(date --iso-8601=seconds) ===" | tee -a "$LOG_FILE"
echo "Service: {service.name}" | tee -a "$LOG_FILE"
echo "Component Type: {service.component_type}" | tee -a "$LOG_FILE"
echo "Command: {full_original_command}" | tee -a "$LOG_FILE"
echo "Log File: $LOG_FILE" | tee -a "$LOG_FILE"
echo "Pod: $POD_NAME" | tee -a "$LOG_FILE"
echo "Namespace: $POD_NAMESPACE" | tee -a "$LOG_FILE"
echo "=================================" | tee -a "$LOG_FILE"

# Use exec with process substitution so the main process receives signals directly.
# No signal trapping needed - the main process replaces this shell and handles
# signals exactly as it would if it were PID 1.
exec {full_original_command} > >(tee -a "$LOG_FILE") 2>&1"""

        # Set the wrapped command
        main_container["command"] = ["/bin/bash", "-c", wrapper_script]

        # Remove any existing args since we're now using a shell script
        if "args" in main_container:
            del main_container["args"]

        # Add volume mount at service level (Dynamo operator pattern)
        if "volumeMounts" not in service._spec:
            service._spec["volumeMounts"] = []

        # Check if mount already exists
        log_mount_exists = any(
            mount.get("name") == "service-logs"
            for mount in service._spec["volumeMounts"]
        )
        if not log_mount_exists:
            service._spec["volumeMounts"].append(
                {"name": "service-logs", "mountPoint": container_log_dir}
            )

        # Volume is shared across all services via deployment-level pvcs,
        # so no need to add it to individual service extraPodSpec

        # Add environment variables for pod identification (using Kubernetes downward API)
        if "envs" not in service._spec:
            service._spec["envs"] = []

        # Add POD_NAME env var (actual Kubernetes pod name for log file naming)
        pod_name_env_exists = any(
            env.get("name") == "POD_NAME" for env in service._spec["envs"]
        )
        if not pod_name_env_exists:
            service._spec["envs"].append(
                {
                    "name": "POD_NAME",
                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
                }
            )

        # Add POD_NAMESPACE env var
        namespace_env_exists = any(
            env.get("name") == "POD_NAMESPACE" for env in service._spec["envs"]
        )
        if not namespace_env_exists:
            service._spec["envs"].append(
                {
                    "name": "POD_NAMESPACE",
                    "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}},
                }
            )


class PodProcess:
    def __init__(self, pod: Pod, line: str):
        self.pid = int(re.split(r"\s+", line)[1])
        self.command = " ".join(
            re.split(r"\s+", line)[10:]
        )  # Columns 10+ are the command
        self._pod = pod

    def kill(self, signal=None):
        """Kill this process in the given pod"""

        if not signal:
            if self.pid == 1:
                signal = "SIGINT"
            else:
                signal = "SIGKILL"
        # Python processes need signal handlers for graceful shutdown
        if self.pid == 1 and signal == "SIGKILL" and "python" in self.command.lower():
            logging.info(
                f"PID 1 is a Python process ({self.command[:50]}...), "
                "changing SIGKILL to SIGINT for graceful shutdown"
            )
            signal = "SIGINT"

        logging.info("Killing PID %s with %s", self.pid, signal)

        return self._pod.exec(["kill", f"-{signal}", str(self.pid)])

    def wait(self, timeout: int = 60):
        """Wait for this process to exit in the given pod"""
        # Simple implementation; adjust as needed
        for _ in range(timeout):
            try:
                result = self._pod.exec(
                    ["kill", "-0", str(self.pid)]
                )  # Check if process exists
                if result.returncode != 0:
                    return True  # Process exited
                time.sleep(1)
            except Exception:
                return True
        return False  # Timed out


@dataclass
class ManagedDeployment:
    log_dir: str
    deployment_spec: DeploymentSpec
    namespace: str
    skip_service_restart: bool = False
    enable_volume_log_collection: bool = (
        False  # PVC-based log collection (requires RWX storage class)
    )
    container_log_dir: str = "/tmp/service_logs"  # Directory for volume logs

    _custom_api: Optional[client.CustomObjectsApi] = None
    _core_api: Optional[client.CoreV1Api] = None
    _in_cluster: bool = False
    _logger: logging.Logger = logging.getLogger()
    _port_forward: Optional[Any] = None
    _deployment_name: Optional[str] = None
    _apps_v1: Optional[Any] = None
    _active_port_forwards: List[Any] = field(default_factory=list)

    # PVC-based log collection
    _log_collection_pvc_created: bool = field(
        default=False, init=False
    )  # Track if we created a PVC

    @property
    def frontend_service_name(self):
        return self.deployment_spec.frontend_service.name

    def get_log_pvc_name(self) -> Optional[str]:
        """Return the PVC name if log collection is enabled.

        This allows ManagedLoad to use the same PVC for storing load test results.

        Returns:
            The PVC name if log collection is enabled, None otherwise.
        """
        return getattr(self.deployment_spec, "_log_collection_pvc_name", None)

    def __post_init__(self):
        self._deployment_name = self.deployment_spec.name

    async def _init_kubernetes(self):
        """Initialize kubernetes client"""
        try:
            # Try in-cluster config first (for pods with service accounts)
            config.load_incluster_config()
            self._in_cluster = True
        except Exception:
            # Fallback to kube config file (for local development)
            await config.load_kube_config()
        k8s_client = client.ApiClient()
        self._custom_api = client.CustomObjectsApi(k8s_client)
        self._core_api = client.CoreV1Api(k8s_client)
        self._apps_v1 = client.AppsV1Api()

    async def _wait_for_pods(self, label, expected, timeout=300):
        for _ in range(timeout):
            assert self._core_api is not None, "Kubernetes API not initialized"
            pods = await self._core_api.list_namespaced_pod(
                self.namespace, label_selector=label
            )
            running = sum(
                1
                for pod in pods.items
                if any(
                    cond.type == "Ready" and cond.status == "True"
                    for cond in (pod.status.conditions or [])
                )
            )
            if running == expected:
                return True
            await asyncio.sleep(1)
        raise Exception(f"Didn't Reach Expected Pod Count {label}=={expected}")

    async def _scale_statfulset(self, name, label, replicas):
        body = {"spec": {"replicas": replicas}}
        assert self._apps_v1 is not None, "Kubernetes API not initialized"
        await self._apps_v1.patch_namespaced_stateful_set_scale(
            name, self.namespace, body
        )
        await self._wait_for_pods(label, replicas)

    async def _restart_stateful(self, name, label):
        self._logger.info(f"Restarting {name} {label}")

        await self._scale_statfulset(name, label, 0)
        assert self._core_api is not None, "Kubernetes API not initialized"
        nats_pvc = await self._core_api.list_namespaced_persistent_volume_claim(
            self.namespace, label_selector=label
        )
        for pvc in nats_pvc.items:
            await self._core_api.delete_namespaced_persistent_volume_claim(
                pvc.metadata.name, self.namespace
            )

        await self._scale_statfulset(name, label, 1)

        self._logger.info(f"Restarted {name} {label}")

    async def wait_for_unready(self, timeout: int = 1800, sleep=1, log_interval=60):
        """
        Wait for the custom resource to be unready.

        Args:
            timeout: Maximum time to wait in seconds, default to 30 mins (image pulling can take a while)
        """
        return await self._wait_for_condition(
            timeout, sleep, log_interval, False, "pending"
        )

    async def _wait_for_ready(self, timeout: int = 1800, sleep=1, log_interval=60):
        """
        Wait for the custom resource to be ready.

        Args:
            timeout: Maximum time to wait in seconds, default to 30 mins (image pulling can take a while)
        """
        return await self._wait_for_condition(
            timeout, sleep, log_interval, True, "successful"
        )

    async def _wait_for_condition(
        self,
        timeout: int = 1800,
        sleep=1,
        log_interval=60,
        desired_ready_condition_val: bool = True,
        desired_state_val: str = "successful",
    ):
        start_time = time.time()

        self._logger.info(
            f"Waiting for Deployment {self._deployment_name} to have Ready condition {desired_ready_condition_val} and state {desired_state_val}"
        )

        attempt = 0

        while (time.time() - start_time) < timeout:
            try:
                attempt += 1
                assert self._custom_api is not None, "Kubernetes API not initialized"
                status = await self._custom_api.get_namespaced_custom_object(  # type: ignore[awaitable-is-not-coroutine]
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self._deployment_name,
                )
                # Check both conditions:
                # 1. Ready condition is True
                # 2. State is successful
                status_obj = status.get("status", {})  # type: ignore[attr-defined]
                conditions = status_obj.get("conditions", [])  # type: ignore[attr-defined]
                current_state = status_obj.get("state", "unknown")  # type: ignore[attr-defined]

                observed_ready_condition_val = ""
                for condition in conditions:
                    if condition.get("type") == "Ready":
                        observed_ready_condition_val = condition.get("status")
                        if observed_ready_condition_val == str(
                            desired_ready_condition_val
                        ):
                            break

                observed_state_val = status_obj.get("state")  # type: ignore[attr-defined]

                if (
                    observed_ready_condition_val == str(desired_ready_condition_val)
                    and observed_state_val == desired_state_val
                ):
                    self._logger.info(f"Current deployment state: {current_state}")
                    self._logger.info(f"Current conditions: {conditions}")
                    self._logger.info(
                        f"Elapsed time: {time.time() - start_time:.1f}s / {timeout}s"
                    )

                    self._logger.info(
                        f"Deployment {self._deployment_name} has Ready condition {desired_ready_condition_val} and state {desired_state_val}"
                    )
                    return True
                else:
                    if attempt % log_interval == 0:
                        self._logger.info(f"Current deployment state: {current_state}")
                        self._logger.info(f"Current conditions: {conditions}")
                        self._logger.info(
                            f"Elapsed time: {time.time() - start_time:.1f}s / {timeout}s"
                        )
                        self._logger.info(
                            f"Deployment has Ready condition {observed_ready_condition_val} and state {observed_state_val}, desired condition {desired_ready_condition_val} and state {desired_state_val}"
                        )

            except exceptions.ApiException as e:
                self._logger.info(
                    f"API Exception while checking deployment status: {e}"
                )
                self._logger.info(f"Status code: {e.status}, Reason: {e.reason}")
            except Exception as e:
                self._logger.info(
                    f"Unexpected exception while checking deployment status: {e}"
                )
            await asyncio.sleep(sleep)
        raise TimeoutError("Deployment failed to become ready within timeout")

    async def get_deployment_status(self) -> dict:
        """Get current deployment status (state and conditions).

        Returns:
            dict with 'state' and 'conditions' keys
        """
        assert self._custom_api is not None, "Kubernetes API not initialized"
        status = await self._custom_api.get_namespaced_custom_object(
            group="nvidia.com",
            version="v1alpha1",
            namespace=self.namespace,
            plural="dynamographdeployments",
            name=self._deployment_name,
        )
        status_obj = status.get("status", {})
        return {
            "state": status_obj.get("state"),
            "conditions": status_obj.get("conditions", []),
        }

    async def _restart_nats(self):
        NATS_STS_NAME = "dynamo-platform-nats"
        NATS_LABEL = "app.kubernetes.io/component=nats"

        await self._restart_stateful(NATS_STS_NAME, NATS_LABEL)

    async def _restart_etcd(self):
        ETCD_STS_NAME = "dynamo-platform-etcd"
        ETCD_LABEL = "app.kubernetes.io/component=etcd"

        await self._restart_stateful(ETCD_STS_NAME, ETCD_LABEL)

    async def _create_deployment(self):
        """
        Create a DynamoGraphDeployment from either a dict or yaml file path.

        Args:
            deployment: Either a dict containing the deployment spec or a path to a yaml file
        """

        # Extract service names

        self._services = self.deployment_spec.services

        self._logger.info(
            f"Starting Deployment {self._deployment_name} with spec {self.deployment_spec}"
        )

        try:
            assert self._custom_api is not None, "Kubernetes API not initialized"
            await self._custom_api.create_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="dynamographdeployments",
                body=self.deployment_spec.spec(),
            )
            self._logger.info(self.deployment_spec.spec())
            self._logger.info(f"Deployment Started {self._deployment_name}")
        except exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                # Fallback: Replace the existing deployment with the new spec
                # This ensures the new spec (e.g., different replica counts) is applied
                self._logger.info(
                    f"Deployment {self._deployment_name} already exists, replacing with new spec..."
                )
                try:
                    await self._custom_api.replace_namespaced_custom_object(
                        group="nvidia.com",
                        version="v1alpha1",
                        namespace=self.namespace,
                        plural="dynamographdeployments",
                        name=self._deployment_name,
                        body=self.deployment_spec.spec(),
                    )
                    self._logger.info(
                        f"Deployment {self._deployment_name} replaced successfully"
                    )
                except exceptions.ApiException as replace_e:
                    self._logger.error(
                        f"Failed to replace deployment {self._deployment_name}: {replace_e}"
                    )
                    raise
            else:
                self._logger.info(
                    f"Failed to create deployment {self._deployment_name}: {e}"
                )
                raise

    async def trigger_rolling_upgrade(self, service_names: list[str]):
        """
        Triggers a rolling update for a list of services
        This is a dummy update - sets an env var on the service
        """

        if not service_names:
            raise ValueError(
                "service_names cannot be empty for trigger_rolling_upgrade"
            )

        patch_body: dict[str, Any] = {"spec": {"services": {}}}

        for service_name in service_names:
            self.deployment_spec.set_service_env_var(
                service_name, "TEST_ROLLING_UPDATE_TRIGGER", secrets.token_hex(8)
            )

            updated_envs = self.deployment_spec.get_service_env_vars(service_name)
            patch_body["spec"]["services"][service_name] = {"envs": updated_envs}

        try:
            assert self._custom_api is not None, "Kubernetes API not initialized"
            await self._custom_api.patch_namespaced_custom_object(
                group="nvidia.com",
                version="v1alpha1",
                namespace=self.namespace,
                plural="dynamographdeployments",
                name=self._deployment_name,
                body=patch_body,
                _content_type="application/merge-patch+json",
            )
        except exceptions.ApiException as e:
            self._logger.info(
                f"Failed to patch deployment {self._deployment_name}: {e}"
            )
            raise

    async def get_pod_names(self, service_names: list[str] | None = None) -> list[str]:
        if not service_names:
            service_names = [service.name for service in self.deployment_spec.services]

        pod_names: list[str] = []

        for service_name in service_names:
            label_selector = (
                f"nvidia.com/selector={self._deployment_name}-{service_name.lower()}"
            )
            assert self._core_api is not None, "Kubernetes API not initialized"
            pods: client.V1PodList = await self._core_api.list_namespaced_pod(
                self.namespace, label_selector=label_selector
            )
            for pod in pods.items:
                pod_names.append(pod.metadata.name)

        return pod_names

    def get_processes(self, pod: Pod) -> list[PodProcess]:
        """Get list of processes in the given pod"""
        result = pod.exec(["ps", "-aux"])
        lines = result.stdout.decode().splitlines()
        # Skip header line
        processes = [PodProcess(pod, line) for line in lines[1:]]
        return processes

    def get_service(self, service_name=None):
        if not service_name:
            service_name = ""
        full_service_name = f"{self._deployment_name}-{service_name.lower()}"

        return Service.get(full_service_name, namespace=self.namespace)

    def get_pods(self, service_names: list[str] | None = None) -> dict[str, list[Pod]]:
        result: dict[str, list[Pod]] = {}

        if not service_names:
            service_names = [service.name for service in self.deployment_spec.services]

        for service_name in service_names:
            # List pods for this service using the selector label
            # nvidia.com/selector: deployment-name-service
            label_selector = (
                f"nvidia.com/selector={self._deployment_name}-{service_name.lower()}"
            )

            pods: list[Pod] = []

            for pod in kr8s.get(
                "pods", namespace=self.namespace, label_selector=label_selector
            ):
                pods.append(pod)  # type: ignore[arg-type]

            result[service_name] = pods

        return result

    def get_pod_manifest_logs_metrics(self, service_name: str, pod: Pod, suffix=""):
        directory = os.path.join(self.log_dir, service_name)
        os.makedirs(directory, exist_ok=True)

        try:
            with open(os.path.join(directory, f"{pod.name}{suffix}.yaml"), "w") as f:
                f.write(pod.to_yaml())
        except Exception as e:
            self._logger.error(e)
        try:
            with open(os.path.join(directory, f"{pod.name}{suffix}.log"), "w") as f:
                f.write("\n".join(pod.logs()))
        except Exception as e:
            self._logger.error(e)
        try:
            previous_logs = pod.logs(previous=True)
            with open(
                os.path.join(directory, f"{pod.name}{suffix}.previous.log"), "w"
            ) as f:
                f.write("\n".join(previous_logs))
        except Exception as e:
            self._logger.debug(e)

        self._get_pod_metrics(pod, service_name, suffix)

    def _get_service_logs(self, service_name=None, suffix=""):
        service_names = None
        if service_name:
            service_names = [service_name]

        service_pods = self.get_pods(service_names)

        for service, pods in service_pods.items():
            for pod in pods:
                self.get_pod_manifest_logs_metrics(service, pod, suffix)

    def _get_pod_metrics(self, pod: Pod, service_name: str, suffix=""):
        directory = os.path.join(self.log_dir, service_name)
        os.makedirs(directory, exist_ok=True)
        port = None
        if service_name == self.frontend_service_name:
            port = self.deployment_spec.port
        else:
            port = self.deployment_spec.system_port

        pf = self.port_forward(pod, port)

        if not pf:
            self._logger.error(f"Unable to get metrics for {service_name}")
            return

        content = None

        try:
            url = f"http://localhost:{pf.local_port}/metrics"

            response = requests.get(url, timeout=30)
            content = None
            try:
                content = response.text
            except ValueError:
                pass

        except Exception as e:
            self._logger.error(str(e))

        if content:
            with open(
                os.path.join(directory, f"{pod.name}.metrics{suffix}.log"), "w"
            ) as f:
                f.write(content)

    async def _delete_deployment(self):
        """
        Delete the DynamoGraphDeployment CR.
        """
        try:
            if self._deployment_name and self._custom_api is not None:
                await self._custom_api.delete_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self._deployment_name,
                )
        except exceptions.ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                raise

    async def _wait_for_deletion(self, timeout: int = 120):
        """
        Wait for the deployment CR to be fully deleted from Kubernetes.

        This prevents a race condition where _create_deployment() is called
        before the previous deployment is fully removed, resulting in a 409
        error and the new deployment spec never being applied.

        Args:
            timeout: Maximum time to wait in seconds (default 120s)
        """
        if not self._deployment_name or self._custom_api is None:
            return

        start_time = time.time()
        self._logger.info(
            f"Waiting for deployment {self._deployment_name} to be fully deleted..."
        )

        while (time.time() - start_time) < timeout:
            try:
                await self._custom_api.get_namespaced_custom_object(
                    group="nvidia.com",
                    version="v1alpha1",
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self._deployment_name,
                )
                # CR still exists, wait
                await asyncio.sleep(2)
            except exceptions.ApiException as e:
                if e.status == 404:
                    self._logger.info(
                        f"Deployment {self._deployment_name} fully deleted"
                    )
                    return
                raise

        self._logger.warning(
            f"Deployment {self._deployment_name} deletion timed out after {timeout}s, "
            "proceeding anyway (will attempt to patch if exists)"
        )

    def port_forward(
        self, pod: Pod, remote_port: int, max_connection_attempts: int = 3
    ):
        """Attempt to connect to a pod and return the port-forward object on success.

        Note: Port forwards run in background threads. When pods are terminated,
        the async cleanup may fail, which is expected and can be safely ignored.
        """
        try:
            # Create port forward - this runs in a background thread
            # Use 127.0.0.1 (localhost) instead of 0.0.0.0 to prevent port conflicts
            port_forward = pod.portforward(
                remote_port=remote_port,
                local_port=0,  # Auto-assign an available port
                address="127.0.0.1",  # Use localhost for better isolation and conflict prevention
            )
            port_forward.start()

            # Try to connect with exponential backoff
            backoff_delay = 0.5  # Start with 500ms

            for attempt in range(max_connection_attempts):
                time.sleep(backoff_delay)
                backoff_delay = min(
                    backoff_delay * 1.5, 5.0
                )  # Double delay, max 5 seconds

                # Check if port is assigned
                if port_forward.local_port == 0:
                    self._logger.debug(
                        f"Port not yet assigned for pod {pod.name} (attempt {attempt+1}/{max_connection_attempts})"
                    )
                    continue

                # Try to connect to the port forwarded service
                test_url = f"http://localhost:{port_forward.local_port}/"
                try:
                    # Send HEAD request to test connection
                    response = requests.head(test_url, timeout=5)
                    if response.status_code in (200, 404):  # 404 is acceptable
                        self._active_port_forwards.append(port_forward)
                        return port_forward
                except (requests.ConnectionError, requests.Timeout) as e:
                    self._logger.warning(
                        f"Connection test failed for pod {pod.name} (attempt {attempt+1}/{max_connection_attempts}): {e}"
                    )

                # Restart port-forward for next attempt (except on last attempt)
                if attempt == max_connection_attempts - 1:
                    continue
                try:
                    port_forward.stop()
                    port_forward.start()
                except Exception as e:
                    self._logger.debug(
                        f"Error restarting port forward for pod {pod.name}: {e}"
                    )
                    break

            # All attempts failed
            self._logger.warning(
                f"Port forward failed after {max_connection_attempts} attempts for pod {pod.name}"
            )
            try:
                port_forward.stop()
            except Exception:
                pass  # Ignore errors during cleanup
            return None

        except Exception as e:
            self._logger.warning(
                f"Failed to create port forward for pod {pod.name}: {e}"
            )
            return None

    async def _cleanup(self):
        try:
            # Collect traditional logs/metrics while pods are still running
            self._get_service_logs()

            # Stop port forwards
            self._logger.info(
                f"Cleaning up {len(self._active_port_forwards)} active port forwards"
            )
            for port_forward in self._active_port_forwards:
                try:
                    port_forward.stop()
                except RuntimeError as e:
                    if "anext()" in str(e) or "already running" in str(e):
                        self._logger.debug(f"Port forward cleanup: {e}")
                    else:
                        self._logger.warning(
                            f"Unexpected error stopping port forward: {e}"
                        )
                except Exception as e:
                    self._logger.debug(f"Error stopping port forward: {e}")
            self._active_port_forwards.clear()
        finally:
            # Clean up all resources (deployment, then extract logs from PVC, then delete PVC)
            await self._cleanup_all_resources()

    async def __aenter__(self):
        try:
            self._logger = logging.getLogger(self.__class__.__name__)
            self.deployment_spec.namespace = self.namespace
            self._deployment_name = self.deployment_spec.name
            logging.getLogger("httpx").setLevel(logging.WARNING)
            await self._init_kubernetes()

            # Run delete deployment and service restarts in parallel
            tasks = [self._delete_deployment()]
            if not self.skip_service_restart:
                tasks.extend([self._restart_etcd(), self._restart_nats()])
            await asyncio.gather(*tasks)

            # Wait for CR to be fully deleted to avoid race condition
            # where _create_deployment() gets 409 (Already exists) and
            # the new deployment spec is never applied
            await self._wait_for_deletion()

            # Enable PVC-based log collection before deployment creation
            if self.enable_volume_log_collection:
                # Check if PVC-based logging is already configured (by enable_log_collection)
                pvc_configured = hasattr(
                    self.deployment_spec, "_log_collection_pvc_name"
                )

                if pvc_configured:
                    self._logger.info(
                        f"PVC-based log collection configured: {self.deployment_spec._log_collection_pvc_name}"
                    )
                    # Validate RWX support and create PVC
                    await self._create_log_collection_pvc()
                else:
                    # Auto-configure PVC-based logging with defaults
                    self._logger.info("Auto-configuring PVC-based log collection")
                    self.deployment_spec.enable_log_collection(
                        container_log_dir=self.container_log_dir,
                        enable_all_services=True,
                    )
                    await self._create_log_collection_pvc()

            await self._create_deployment()
            await self._wait_for_ready()
            # Note: Download job will be created during cleanup after pods exit

        except:
            await self._cleanup()
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()

    async def create_log_download_job(
        self,
        local_output_dir: str,
        container_log_dir="/tmp/service_logs",
        job_name=None,
        download_timeout=300,
    ):
        """
        Create a Kubernetes job to download log files from the PVC-based service-logs volume.

        This job will:
        1. Mount the same PVC as the deployment services
        2. Create a tar archive of all log files
        3. Keep the job pod alive for extraction (similar to ManagedAIPerfDeployment pattern)

        Args:
            local_output_dir: Local directory to save the downloaded logs
            container_log_dir: Container directory where logs are stored (should match enable_log_collection)
            job_name: Optional custom job name (defaults to deployment-name-log-download)
            download_timeout: Timeout in seconds for the download job

        Returns:
            dict: Information about the created job
        """
        if not self._custom_api:
            raise RuntimeError(
                "Kubernetes API not initialized. Call _init_kubernetes() first."
            )

        # Generate job name
        if not job_name:
            job_name = (
                f"{self.deployment_spec.name}-log-download-{secrets.token_hex(4)}"
            )

        os.makedirs(local_output_dir, exist_ok=True)

        # Create the log download script - keeps container alive for on-demand extraction
        download_script = f"""#!/bin/bash
set -e

echo "=== LOG DOWNLOAD JOB STARTED at $(date --iso-8601=seconds) ==="
echo "Deployment: {self.deployment_spec.name}"
echo "Namespace: {self.namespace}"
echo "Container log dir: {container_log_dir}"
echo "Job name: {job_name}"

# Create archive directory
mkdir -p /tmp/log_archive

if [ ! -d "{container_log_dir}" ]; then
    echo "Log directory {container_log_dir} does not exist, creating it..."
    mkdir -p {container_log_dir}
fi

# Mark job as ready (tar will be created on-demand at extraction time)
echo "ready" > /tmp/log_archive/job_ready.txt

echo "=== LOG DOWNLOAD JOB READY at $(date --iso-8601=seconds) ==="
echo "Waiting for extraction signal. Logs will be archived on-demand."

# Keep container alive for extraction
while true; do
    LOG_COUNT=$(find {container_log_dir} -name "*.log" -type f 2>/dev/null | wc -l)
    echo "[$(date '+%H:%M:%S')] Log download job alive - $LOG_COUNT log files available"
    sleep 60
done"""

        # Define the job spec
        job_spec = {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": job_name,
                "namespace": self.namespace,
                "labels": {
                    "app": "log-download",
                    "managed-by": "managed-deployment",
                    "deployment": self.deployment_spec.name,
                },
            },
            "spec": {
                "backoffLimit": 1,
                "completions": 1,
                "parallelism": 1,
                "template": {
                    "metadata": {
                        "labels": {
                            "app": "log-download",
                            "job-name": job_name,
                        }
                    },
                    "spec": {
                        "restartPolicy": "Never",
                        "containers": [
                            {
                                "name": "log-downloader",
                                "image": "busybox:1.35",  # Lightweight image with tar support
                                "command": ["/bin/sh", "-c", download_script],
                                "volumeMounts": [
                                    {
                                        "name": "service-logs",
                                        "mountPath": container_log_dir,
                                        "readOnly": True,  # Read-only since we're just downloading
                                    }
                                ],
                                "resources": {
                                    "requests": {"cpu": "100m", "memory": "128Mi"},
                                    "limits": {"cpu": "500m", "memory": "512Mi"},
                                },
                            }
                        ],
                        "volumes": [
                            {
                                "name": "service-logs",
                                **self._get_download_job_volume_config(),
                            }
                        ],
                    },
                },
            },
        }

        # Create the job using BatchV1Api
        try:
            batch_api = client.BatchV1Api()
            await batch_api.create_namespaced_job(
                namespace=self.namespace, body=job_spec
            )
            self._logger.info(f"Log download job created: {job_name}")

            return {
                "success": True,
                "job_name": job_name,
                "namespace": self.namespace,
                "local_output_dir": local_output_dir,
            }

        except exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                self._logger.warning(f"Job {job_name} already exists")
                return {
                    "success": True,
                    "job_name": job_name,
                    "namespace": self.namespace,
                    "local_output_dir": local_output_dir,
                    "note": "job_already_existed",
                }
            else:
                self._logger.error(f"Failed to create job {job_name}: {e}")
                raise

    async def extract_logs_from_download_job(
        self, job_name: str, local_output_dir: str
    ):
        """
        Extract logs from a log download job pod by creating tar on-demand.

        This method creates the tar archive at extraction time (not at job start),
        ensuring all logs up to this point are captured.

        Args:
            job_name: Name of the log download job
            local_output_dir: Local directory to save the extracted logs

        Returns:
            dict: Extraction results including file count and paths
        """
        container_log_dir = self.container_log_dir

        try:
            # Find the job pod
            pods = []
            job_label = f"job-name={job_name}"
            pod_generator = kr8s.get(
                "pods",
                namespace=self.namespace,
                label_selector=job_label,
            )
            for pod in pod_generator:
                pods.append(pod)

            if not pods:
                raise Exception(f"No pods found for job {job_name}")

            pod = pods[0]

            # Wait for job to be ready
            self._logger.info("Waiting for log download job to be ready...")
            for attempt in range(60):  # Wait up to 60 seconds
                try:
                    result = await asyncio.wait_for(
                        asyncio.create_task(
                            asyncio.to_thread(
                                pod.exec,
                                ["test", "-f", "/tmp/log_archive/job_ready.txt"],
                            )
                        ),
                        timeout=5.0,
                    )
                    if result.returncode == 0:
                        break
                except Exception:
                    pass

                self._logger.info(
                    f"Waiting for download job to be ready... (attempt {attempt + 1}/60)"
                )
                await asyncio.sleep(1)
            else:
                self._logger.warning(
                    "Download job did not become ready in expected time, proceeding anyway..."
                )

            # Create tar archive ON-DEMAND (captures all logs up to this point)
            self._logger.info("Creating tar archive of logs on-demand...")
            create_tar_script = f"""
cd {container_log_dir} 2>/dev/null || exit 1
LOG_COUNT=$(find . -name "*.log" -type f | wc -l)
echo "LOG_COUNT:$LOG_COUNT"
if [ "$LOG_COUNT" -gt 0 ]; then
    tar -czf /tmp/log_archive/service_logs.tar.gz *.log 2>/dev/null
    echo "TAR_CREATED:true"
else
    echo "TAR_CREATED:false"
fi
"""
            tar_result = await asyncio.wait_for(
                asyncio.create_task(
                    asyncio.to_thread(pod.exec, ["sh", "-c", create_tar_script])
                ),
                timeout=30.0,
            )

            # Parse the output to get log count
            output = tar_result.stdout.decode() if tar_result.stdout else ""
            log_count = 0
            tar_created = False
            for line in output.split("\n"):
                if line.startswith("LOG_COUNT:"):
                    log_count = int(line.split(":")[1])
                elif line.startswith("TAR_CREATED:"):
                    tar_created = line.split(":")[1] == "true"

            self._logger.info(f"Found {log_count} log files, tar_created={tar_created}")

            extracted_files = []

            if log_count > 0 and tar_created:
                # Extract the tar archive
                self._logger.info("Extracting service logs archive...")
                cat_result = await asyncio.wait_for(
                    asyncio.create_task(
                        asyncio.to_thread(
                            pod.exec, ["cat", "/tmp/log_archive/service_logs.tar.gz"]
                        )
                    ),
                    timeout=60.0,
                )

                if cat_result.returncode != 0:
                    raise Exception(
                        f"Archive extraction failed with return code {cat_result.returncode}"
                    )

                # Save the archive locally
                local_archive = os.path.join(local_output_dir, "service_logs.tar.gz")
                with open(local_archive, "wb") as f:
                    f.write(cat_result.stdout)

                # Extract the archive locally
                import tarfile

                with tarfile.open(local_archive, "r:gz") as tar:
                    tar.extractall(path=local_output_dir)
                    extracted_files = tar.getnames()

                # Remove the temporary archive file
                os.remove(local_archive)

                self._logger.info(
                    f"Extracted {len(extracted_files)} log files to {local_output_dir}"
                )

            else:
                self._logger.info("No log files were available for download")

            # Create metadata file locally
            import json

            metadata = {
                "deployment_name": self.deployment_spec.name,
                "namespace": self.namespace,
                "extraction_timestamp": asyncio.get_event_loop().time(),
                "job_name": job_name,
                "log_count": log_count,
                "container_log_dir": container_log_dir,
            }
            metadata_path = os.path.join(local_output_dir, "download_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)
            extracted_files.append("download_metadata.json")

            return {
                "success": True,
                "extracted_files": extracted_files,
                "log_count": log_count,
                "local_output_dir": local_output_dir,
            }

        except Exception as e:
            self._logger.error(
                f"Failed to extract logs from download job {job_name}: {e}"
            )
            return {
                "success": False,
                "error": str(e),
                "local_output_dir": local_output_dir,
            }

    async def cleanup_log_download_job(self, job_name: str):
        """
        Clean up a log download job and its associated pods.

        Args:
            job_name: Name of the job to clean up
        """
        try:
            # Delete the job with foreground propagation to cascade to pods
            from kubernetes_asyncio.client.models import V1DeleteOptions

            delete_options = V1DeleteOptions(propagation_policy="Foreground")

            batch_api = client.BatchV1Api()
            await batch_api.delete_namespaced_job(
                name=job_name,
                namespace=self.namespace,
                body=delete_options,
            )
            self._logger.info(f"Log download job {job_name} deleted")

        except exceptions.ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                self._logger.warning(
                    f"Failed to delete log download job {job_name}: {e}"
                )

    async def download_volume_logs_now(self, local_output_dir=None):
        """
        Download logs from volume-based collection immediately.

        This creates a temporary download job, extracts logs, and cleans up the job.
        Useful for downloading logs during test execution (not just at cleanup time).

        Args:
            local_output_dir: Optional local directory to save logs (defaults to log_dir/volume_logs_manual)

        Returns:
            dict: Download results
        """
        if not self.enable_volume_log_collection:
            return {"success": False, "error": "Volume log collection is not enabled"}

        if local_output_dir is None:
            local_output_dir = os.path.join(self.log_dir, "volume_logs_manual")

        self._logger.info("Downloading volume logs on demand...")

        try:
            # Create a temporary download job
            download_job_result = await self.create_log_download_job(
                local_output_dir=local_output_dir,
                container_log_dir=self.container_log_dir,
            )

            if not download_job_result.get("success"):
                return {"success": False, "error": "Failed to create download job"}

            job_name = download_job_result["job_name"]

            # Extract logs
            result = await self.extract_logs_from_download_job(
                job_name, local_output_dir
            )

            if result["success"]:
                self._logger.info(
                    f"Successfully downloaded {result['log_count']} log files to {local_output_dir}"
                )

            # Cleanup the temporary job
            await self.cleanup_log_download_job(job_name)

            return result

        except Exception as e:
            self._logger.error(f"Failed to download volume logs: {e}")
            return {"success": False, "error": str(e)}

    async def _create_log_collection_pvc(self) -> str:
        """
        Create a PVC for log collection based on the deployment spec configuration.

        Requires a storage class that supports ReadWriteMany (RWX) access mode.

        Returns:
            The name of the created PVC

        Raises:
            RuntimeError: If the storage class does not support RWX access mode
        """
        pvc_name = getattr(
            self.deployment_spec, "_log_collection_pvc_name", "dynamo-logs-pvc"
        )
        pvc_size = getattr(self.deployment_spec, "_log_collection_pvc_size", "1Gi")
        storage_class = getattr(
            self.deployment_spec, "_log_collection_storage_class", "nebius-shared-fs"
        )

        # Validate RWX support
        await self._validate_rwx_support(storage_class)

        self._logger.info(
            f"Creating PVC {pvc_name} with storage class {storage_class} (RWX)"
        )

        # Check if PVC already exists and delete it to start fresh
        try:
            await self._core_api.read_namespaced_persistent_volume_claim(
                name=pvc_name, namespace=self.namespace
            )
            self._logger.info(
                f"PVC {pvc_name} already exists, deleting it to start fresh"
            )
            await self._core_api.delete_namespaced_persistent_volume_claim(
                name=pvc_name, namespace=self.namespace
            )
            # Wait a moment for deletion to complete
            await asyncio.sleep(2)
            self._logger.info(f"Deleted existing PVC {pvc_name}")
        except client.ApiException as e:
            if e.status != 404:  # Not found is expected, other errors are issues
                self._logger.warning(
                    f"Error checking/deleting existing PVC {pvc_name}: {e}"
                )
            else:
                self._logger.info(f"PVC {pvc_name} does not exist, will create new one")

        # Create PVC with RWX access mode
        pvc_spec = {
            "apiVersion": "v1",
            "kind": "PersistentVolumeClaim",
            "metadata": {
                "name": pvc_name,
                "namespace": self.namespace,
                "labels": {
                    "managed-by": "managed-deployment",
                    "deployment": self.deployment_spec.name,
                    "purpose": "log-collection",
                    "app": f"{self.deployment_spec.name}-logs",
                },
            },
            "spec": {
                "accessModes": [
                    "ReadWriteMany"
                ],  # RWX required for multi-pod log collection
                "storageClassName": storage_class,
                "resources": {"requests": {"storage": pvc_size}},
            },
        }

        try:
            await self._core_api.create_namespaced_persistent_volume_claim(
                namespace=self.namespace, body=pvc_spec
            )
            self._log_collection_pvc_created = True
            self._logger.info(f"Created PVC {pvc_name} for log collection ({pvc_size})")

            # Simple verification - just wait a moment for PVC to be available
            self._logger.info(
                f"Waiting 5 seconds for PVC {pvc_name} to be available in cluster..."
            )
            await asyncio.sleep(5)

            self._logger.info(
                f"Proceeding with deployment assuming PVC {pvc_name} was created successfully"
            )
            return pvc_name

        except client.ApiException as e:
            self._logger.error(f"Failed to create PVC {pvc_name}: {e}")
            raise

    async def _validate_rwx_support(self, storage_class_name: str):
        """
        Validate that the storage class supports ReadWriteMany (RWX) access mode.

        Args:
            storage_class_name: Name of the storage class to validate

        Raises:
            RuntimeError: If the storage class does not exist or doesn't support RWX
        """
        try:
            storage_api = client.StorageV1Api()
            sc = await storage_api.read_storage_class(name=storage_class_name)

            # Note: StorageClass doesn't explicitly list supported access modes
            # The validation happens when the PVC is bound to a PV
            # We just verify the storage class exists
            self._logger.info(
                f"Storage class '{storage_class_name}' exists (provisioner: {sc.provisioner})"
            )

            # Known RWX-capable provisioners
            rwx_provisioners = [
                "nfs",
                "cephfs",
                "glusterfs",
                "azurefile",
                "efs.csi.aws.com",
                "file.csi.azure.com",
                "csi.cloudscale.ch",
                "mounted-fs-path.csi.nebius",  # Nebius mounted filesystem (RWX capable)
            ]

            # Check if provisioner is known to support RWX
            provisioner_lower = sc.provisioner.lower() if sc.provisioner else ""
            supports_rwx = any(p in provisioner_lower for p in rwx_provisioners)

            if not supports_rwx:
                self._logger.warning(
                    f"Storage class '{storage_class_name}' uses provisioner '{sc.provisioner}' "
                    f"which may not support ReadWriteMany (RWX). "
                    f"Log collection requires RWX for multi-pod access. "
                    f"Known RWX provisioners: {rwx_provisioners}"
                )
                raise RuntimeError(
                    f"Storage class '{storage_class_name}' (provisioner: {sc.provisioner}) "
                    f"may not support ReadWriteMany (RWX) access mode. "
                    f"Log collection requires RWX for reliable multi-pod log capture. "
                    f"Please use a storage class with an RWX-capable provisioner."
                )

            self._logger.info(
                f"Storage class '{storage_class_name}' appears to support RWX"
            )

        except client.ApiException as e:
            if e.status == 404:
                raise RuntimeError(
                    f"Storage class '{storage_class_name}' does not exist. "
                    f"Please specify a valid storage class that supports ReadWriteMany (RWX)."
                )
            raise

    async def _cleanup_log_collection_pvc(self):
        """
        Clean up the log collection PVC if we created it.
        """
        if not self._log_collection_pvc_created:
            return

        pvc_name = getattr(
            self.deployment_spec, "_log_collection_pvc_name", "dynamo-logs-pvc"
        )

        try:
            await self._core_api.delete_namespaced_persistent_volume_claim(
                name=pvc_name, namespace=self.namespace
            )
            self._logger.info(f"Cleaned up log collection PVC {pvc_name}")

        except client.ApiException as e:
            if e.status != 404:  # Not found is acceptable during cleanup
                self._logger.warning(f"Failed to cleanup PVC {pvc_name}: {e}")

    def _get_download_job_volume_config(self) -> dict:
        """
        Get the PVC volume configuration for the log download job.

        Returns:
            dict: Volume configuration for the download job using PVC

        Raises:
            RuntimeError: If PVC-based logging is not configured
        """
        if hasattr(self.deployment_spec, "_log_collection_pvc_name"):
            pvc_name = self.deployment_spec._log_collection_pvc_name
            return {"persistentVolumeClaim": {"claimName": pvc_name}}
        else:
            raise RuntimeError(
                "PVC-based log collection is not configured. "
                "Call deployment_spec.enable_log_collection() before using ManagedDeployment."
            )

    async def _cleanup_all_resources(self):
        """
        Comprehensive cleanup of all resources created by this deployment.

        Order for volume log collection:
        1. Delete deployment (pods exit gracefully, write final logs to PVC)
        2. Wait for pods to terminate
        3. Create download job to extract logs from PVC
        4. Extract logs
        5. Delete download job
        6. Delete PVC
        """
        try:
            # Delete the main deployment first (pods write final logs to PVC on exit)
            self._logger.info("Deleting deployment...")
            await self._delete_deployment()

            # Wait for pods to fully terminate before extracting logs
            if self.enable_volume_log_collection:
                self._logger.info("Waiting for pods to terminate...")
                await self._wait_for_pods_terminated()

                # Now create download job to extract logs from PVC (after pods are gone)
                await self._extract_logs_from_pvc()

                # Clean up PVC
                await self._cleanup_log_collection_pvc()

            # Clean up any orphaned jobs related to this deployment
            await self._cleanup_orphaned_jobs()

        except Exception as e:
            self._logger.warning(f"Error during comprehensive cleanup: {e}")

    async def _wait_for_pods_terminated(self, timeout: int = 120):
        """Wait for all deployment pods to terminate."""
        label_selector = f"dynamo-deployment={self.deployment_spec.name}"
        for attempt in range(timeout):
            try:
                pods = await self._core_api.list_namespaced_pod(
                    namespace=self.namespace, label_selector=label_selector
                )
                if not pods.items:
                    self._logger.info("All pods terminated")
                    return
                running = [
                    p.metadata.name
                    for p in pods.items
                    if p.status.phase not in ("Succeeded", "Failed")
                ]
                if not running:
                    self._logger.info("All pods completed")
                    return
                self._logger.info(
                    f"Waiting for {len(running)} pods to terminate... ({attempt + 1}/{timeout})"
                )
            except Exception as e:
                self._logger.debug(f"Error checking pods: {e}")
            await asyncio.sleep(1)
        self._logger.warning(f"Timeout waiting for pods to terminate after {timeout}s")

    async def _extract_logs_from_pvc(self):
        """Create download job, extract logs, and cleanup job."""
        try:
            volume_logs_dir = os.path.join(self.log_dir, "volume_logs")
            os.makedirs(volume_logs_dir, exist_ok=True)

            # Create download job
            self._logger.info("Creating log download job...")
            download_job_result = await self.create_log_download_job(
                local_output_dir=volume_logs_dir,
                container_log_dir=self.container_log_dir,
            )

            if not download_job_result.get("success"):
                self._logger.warning("Failed to create log download job")
                return

            job_name = download_job_result["job_name"]
            self._logger.info(f"Log download job created: {job_name}")

            # Extract logs
            extraction_result = await self.extract_logs_from_download_job(
                job_name, volume_logs_dir
            )
            if extraction_result.get("success"):
                self._logger.info(
                    f"Successfully extracted {extraction_result.get('log_count', 0)} log files"
                )
            else:
                self._logger.warning(
                    f"Log extraction failed: {extraction_result.get('error', 'Unknown error')}"
                )

            # Cleanup download job
            await self.cleanup_log_download_job(job_name)

        except Exception as e:
            self._logger.warning(f"Error extracting logs from PVC: {e}")

    async def _cleanup_orphaned_jobs(self):
        """Clean up any jobs that might be left behind."""
        try:
            # Get all jobs in the namespace that are related to this deployment
            batch_api = client.BatchV1Api()
            jobs = await batch_api.list_namespaced_job(
                namespace=self.namespace,
                label_selector=f"deployment={self.deployment_spec.name}",
            )

            for job in jobs.items:
                job_name = job.metadata.name
                try:
                    # Delete the job with cascade to clean up pods
                    await batch_api.delete_namespaced_job(
                        name=job_name,
                        namespace=self.namespace,
                        body=client.V1DeleteOptions(propagation_policy="Background"),
                    )
                    self._logger.info(f"Cleaned up orphaned job: {job_name}")
                except client.ApiException as e:
                    if e.status != 404:
                        self._logger.warning(f"Failed to cleanup job {job_name}: {e}")

        except Exception as e:
            self._logger.warning(f"Error cleaning up orphaned jobs: {e}")


async def main():
    LOG_FORMAT = "[TEST] %(asctime)s %(levelname)s %(name)s: %(message)s"
    DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,  # ISO 8601 UTC format
    )

    # Get workspace directory
    workspace_dir = _get_workspace_dir()

    deployment_spec = DeploymentSpec(
        os.path.join(workspace_dir, "examples/backends/vllm/deploy/agg.yaml")
    )

    deployment_spec.disable_grove()

    print(deployment_spec._deployment_spec)

    deployment_spec.name = "foo"

    deployment_spec.set_image("nvcr.io/nvidia/ai-dynamo/vllm-runtime:0.4.1")

    # Configure logging
    deployment_spec.set_logging(enable_jsonl=True, log_level="debug")

    print(f"Logging config: {deployment_spec.get_logging_config()}")

    async with ManagedDeployment(
        namespace="test", log_dir=".", deployment_spec=deployment_spec
    ):
        time.sleep(60)


if __name__ == "__main__":
    asyncio.run(main())
