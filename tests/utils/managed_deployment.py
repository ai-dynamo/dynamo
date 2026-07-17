# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
import os
import re
import secrets
import time
from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional

import kr8s
import requests
import yaml
from kr8s.objects import Pod, Service
from kubernetes_asyncio import client, config
from kubernetes_asyncio.client import exceptions

from tests.utils.test_output import resolve_test_output_path


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


# Supported DynamoGraphDeployment CRD schemas. v1alpha1 uses ``spec.services``
# (a dict of service name -> ServiceSpec); v1beta1 uses ``spec.components``
# (a list of components with podTemplate.spec.containers).
SCHEMA_V1ALPHA1 = "v1alpha1"
SCHEMA_V1BETA1 = "v1beta1"


class ServiceSpec:
    """Wrapper around a single service/component in the deployment spec.

    Supports both v1alpha1 (``spec.services[<name>]``) and v1beta1
    (``spec.components[*]``) schemas. Accessors dispatch on ``schema`` so
    callers see a uniform interface regardless of the underlying CRD version.
    """

    # Default sidecar container name used when ``frontendSidecar`` references a
    # container by name in v1beta1. Matches the convention used by the
    # examples and recipes that ship with dynamo.
    _DEFAULT_SIDECAR_NAME = "sidecar-frontend"

    def __init__(
        self,
        service_name: str,
        service_spec: dict,
        schema: str = SCHEMA_V1ALPHA1,
    ):
        self._name = service_name
        self._spec = service_spec
        self._schema = schema

    @property
    def name(self) -> str:
        """The service name (read-only)"""
        return self._name

    @property
    def schema(self) -> str:
        """CRD schema this ServiceSpec is bound to."""
        return self._schema

    # ----- v1beta1 container helpers -----
    def _containers_list(self, create: bool = False) -> Optional[list]:
        """Return the v1beta1 ``podTemplate.spec.containers`` list, or None."""
        if self._schema != SCHEMA_V1BETA1:
            return None
        if create:
            return (
                self._spec.setdefault("podTemplate", {})
                .setdefault("spec", {})
                .setdefault("containers", [])
            )
        return self._spec.get("podTemplate", {}).get("spec", {}).get("containers")

    def _find_container(
        self, container_name: str, create: bool = False
    ) -> Optional[dict]:
        """Find a container by name in v1beta1; optionally create if missing."""
        containers = self._containers_list(create=create)
        if containers is None:
            return None
        for c in containers:
            if c.get("name") == container_name:
                return c
        if create:
            new_c: dict[str, Any] = {"name": container_name}
            containers.append(new_c)
            return new_c
        return None

    def _main_container(self, create: bool = False) -> Optional[dict]:
        return self._find_container("main", create=create)

    def _sidecar_container_name(self) -> str:
        # v1beta1 components reference the sidecar container by name via the
        # ``frontendSidecar`` field; default to the canonical name otherwise.
        return self._spec.get("frontendSidecar") or self._DEFAULT_SIDECAR_NAME

    def _sidecar_container(self, create: bool = False) -> Optional[dict]:
        return self._find_container(self._sidecar_container_name(), create=create)

    # ----- Image -----
    @property
    def image(self) -> Optional[str]:
        """Container image for the service"""
        if self._schema == SCHEMA_V1BETA1:
            container = self._main_container()
            return container.get("image") if container else None
        try:
            return self._spec["extraPodSpec"]["mainContainer"]["image"]
        except KeyError:
            return None

    @image.setter
    def image(self, value: str):
        if self._schema == SCHEMA_V1BETA1:
            container = self._main_container(create=True)
            assert container is not None
            container["image"] = value
            return
        if "extraPodSpec" not in self._spec:
            self._spec["extraPodSpec"] = {"mainContainer": {}}
        if "mainContainer" not in self._spec["extraPodSpec"]:
            self._spec["extraPodSpec"]["mainContainer"] = {}
        self._spec["extraPodSpec"]["mainContainer"]["image"] = value

    @property
    def frontend_sidecar_image(self) -> Optional[str]:
        """Container image for the frontendSidecar (if present)."""
        if self._schema == SCHEMA_V1BETA1:
            container = self._sidecar_container()
            return container.get("image") if container else None
        try:
            return self._spec["frontendSidecar"]["image"]
        except KeyError:
            return None

    @frontend_sidecar_image.setter
    def frontend_sidecar_image(self, value: str):
        if self._schema == SCHEMA_V1BETA1:
            container = self._sidecar_container(create=True)
            assert container is not None
            container["image"] = value
            return
        if "frontendSidecar" not in self._spec:
            self._spec["frontendSidecar"] = {}
        self._spec["frontendSidecar"]["image"] = value

    @property
    def envs(self) -> list[dict[str, str]]:
        """Environment variables for the service.

        v1alpha1 stores envs at the service level (``spec.envs``); v1beta1
        stores them per-container under ``podTemplate.spec.containers[main].env``.
        """
        if self._schema == SCHEMA_V1BETA1:
            container = self._main_container()
            if container is None:
                return []
            return container.get("env", [])
        return self._spec.get("envs", [])

    @envs.setter
    def envs(self, value: list[dict[str, str]]):
        if self._schema == SCHEMA_V1BETA1:
            container = self._main_container(create=True)
            assert container is not None
            container["env"] = value
            return
        self._spec["envs"] = value

    def add_pvc_mount(self, pvc_name: str, mount_point: str) -> None:
        """Add a service-level volumeMount for a PVC declared in ``spec.pvcs``. Idempotent."""
        mounts = self._spec.setdefault("volumeMounts", [])
        if not any(m.get("name") == pvc_name for m in mounts):
            mounts.append({"name": pvc_name, "mountPoint": mount_point})

    def _get_main_container_for_args(self, create: bool = False) -> Optional[dict]:
        """Locate the main container dict for argv access in either schema."""
        if self._schema == SCHEMA_V1BETA1:
            return self._main_container(create=create)
        if "extraPodSpec" not in self._spec:
            if not create:
                return None
            self._spec["extraPodSpec"] = {"mainContainer": {}}
        if "mainContainer" not in self._spec["extraPodSpec"]:
            if not create:
                return None
            self._spec["extraPodSpec"]["mainContainer"] = {}
        return self._spec["extraPodSpec"]["mainContainer"]

    @staticmethod
    def _is_shell_style(container: dict) -> bool:
        """Detect ``command: [..., "-c"]`` + ``args: ["<shell-string>"]``.

        v1beta1 manifests frequently invoke workers via a shell so that
        ``$VAR`` references in the args expand at pod start. Mutating those
        args naively (e.g. shlex-splitting and writing back as argv tokens)
        breaks the shell's contract — ``sh -c`` takes exactly one command
        string; everything after it becomes ``$0``/``$1``/…, so the rest of
        the flags would be silently dropped at runtime.
        """
        cmd = container.get("command")
        if not isinstance(cmd, list) or len(cmd) < 2 or cmd[-1] != "-c":
            return False
        args = container.get("args", [])
        return (
            isinstance(args, list)
            and len(args) == 1
            and isinstance(args[0], str)
            and " " in args[0]
        )

    def _get_args(self) -> list[str]:
        """Return parsed argv tokens for the main container.

        - Argv-style args (already a list of tokens, or a single-string scalar
          that's whitespace-separable) return the **live** list stored in the
          spec, so in-place mutations propagate immediately.
        - Shell-style args (``command: [/bin/sh, -c]`` + a single args string)
          return a **detached** parsed copy. Callers that mutate must persist
          changes via :meth:`_set_args` so the shell command remains a single
          ``args[0]`` string when the manifest is serialised back.
        """
        container = self._get_main_container_for_args(create=True)
        if container is None:
            return []
        if "args" not in container:
            container["args"] = []
        args = container["args"]
        if isinstance(args, str):
            args = args.split()
            container["args"] = args
            return args
        if self._is_shell_style(container):
            import shlex

            return shlex.split(args[0]) if args[0].strip() else []
        return args

    # Bare-safe characters for re-joining shell-style argv tokens. Includes ``$``
    # so that ``$MODEL_PATH``-style references survive the round-trip without
    # being single-quoted (which would suppress runtime expansion under
    # ``/bin/sh -c``). Compare with ``shlex.quote`` which is purely literal.
    _SHELL_BARE_SAFE = re.compile(r"^[A-Za-z0-9_./:=,@%+\-$]+$")

    @classmethod
    def _shell_quote_preserving_vars(cls, tok: str) -> str:
        """Quote ``tok`` for inclusion in a shell command while preserving
        ``$VAR`` expansion when the variable name uses safe characters.

        Tokens that are entirely composed of safe characters (alnum, ``-_./
        :=,@%+`` and ``$``) are left bare; everything else is wrapped in
        single quotes (with embedded single quotes escaped via the standard
        ``'\\''`` trick) to preserve literal contents.
        """
        if not tok:
            return "''"
        if cls._SHELL_BARE_SAFE.match(tok):
            return tok
        return "'" + tok.replace("'", "'\\''") + "'"

    def _set_args(self, tokens: list[str]) -> None:
        """Write argv ``tokens`` back into the main container.

        For shell-style invocations the tokens are re-joined into a single
        shell-quoted string and stored as ``args[0]`` so the original
        ``/bin/sh -c`` contract is preserved. ``$VAR`` references are left
        bare so the shell still expands them at pod start.
        """
        container = self._get_main_container_for_args(create=True)
        if container is None:
            return
        if self._is_shell_style(container):
            container["args"] = [
                " ".join(self._shell_quote_preserving_vars(t) for t in tokens)
            ]
        else:
            container["args"] = list(tokens)

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
        args = self._get_args()
        for i, arg in enumerate(args):
            if arg in ["--model", "--model-path"]:
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    return args[i + 1]
        return None

    @model.setter
    def model(self, value: str):
        args = self._get_args()
        for i, arg in enumerate(args):
            if arg in ["--model", "--model-path"]:
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    args[i + 1] = value
                self._set_args(args)
                return

    # ----- GPUs -----
    # v1alpha1 stores GPU limits at ``spec.resources.limits.gpu``; v1beta1
    # stores them per-container at
    # ``podTemplate.spec.containers[main].resources.limits["nvidia.com/gpu"]``.
    @property
    def gpus(self) -> int:
        if self._schema == SCHEMA_V1BETA1:
            container = self._main_container()
            if container is None:
                return 0
            try:
                return int(container["resources"]["limits"]["nvidia.com/gpu"])
            except KeyError:
                return 0
        try:
            return int(self._spec["resources"]["limits"]["gpu"])
        except KeyError:
            return 0

    @gpus.setter
    def gpus(self, value: int):
        # Kubernetes requires ``requests == limits`` for extended resources
        # like ``nvidia.com/gpu``; the GAIE fixture and other v1beta1
        # manifests declare both fields explicitly. Update them in lockstep
        # so callers like ``set_tensor_parallel`` can't produce a spec the
        # operator (or kube-scheduler) will reject.
        if self._schema == SCHEMA_V1BETA1:
            container = self._main_container(create=True)
            assert container is not None
            resources = container.setdefault("resources", {})
            resources.setdefault("limits", {})["nvidia.com/gpu"] = str(value)
            resources.setdefault("requests", {})["nvidia.com/gpu"] = str(value)
            return
        resources = self._spec.setdefault("resources", {})
        resources.setdefault("limits", {})["gpu"] = str(value)
        resources.setdefault("requests", {})["gpu"] = str(value)

    @property
    def tensor_parallel_size(self) -> int:
        """Get tensor parallel size from vLLM arguments"""
        args = self._get_args()
        for i, arg in enumerate(args):
            if arg == "--tensor-parallel-size":
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    return int(args[i + 1])
                return 1
        return 1

    @tensor_parallel_size.setter
    def tensor_parallel_size(self, value: int):
        args = self._get_args()
        for i, arg in enumerate(args):
            if arg == "--tensor-parallel-size":
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    args[i + 1] = str(value)
                else:
                    args.append(str(value))
                self._set_args(args)
                self.gpus = value
                return
        args.extend(["--tensor-parallel-size", str(value)])
        self._set_args(args)
        self.gpus = value


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
        self._schema = self._detect_schema()

    def _detect_schema(self) -> str:
        """Detect whether the loaded manifest is v1alpha1 (services dict) or
        v1beta1 (components list).

        We trust ``apiVersion`` first and fall back to inspecting ``spec`` so
        manifests without an explicit version still work.
        """
        api_version = self._deployment_spec.get("apiVersion", "")
        if api_version.endswith("/v1beta1"):
            return SCHEMA_V1BETA1
        if api_version.endswith("/v1alpha1"):
            return SCHEMA_V1ALPHA1
        spec = self._deployment_spec.get("spec", {})
        if isinstance(spec.get("components"), list):
            return SCHEMA_V1BETA1
        return SCHEMA_V1ALPHA1

    @property
    def schema(self) -> str:
        """CRD schema of the loaded manifest (``v1alpha1`` or ``v1beta1``)."""
        return self._schema

    @property
    def api_version(self) -> str:
        """CRD version string suitable for CustomObjectsApi calls."""
        return self._schema

    def _component_by_name(self, service_name: str) -> dict:
        """Look up a v1beta1 component dict by its ``name`` field."""
        for comp in self._deployment_spec["spec"].get("components", []):
            if comp.get("name") == service_name:
                return comp
        raise KeyError(service_name)

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

    def mount_model_cache_pvc(
        self, pvc_name: str, mount_point: str = "/models"
    ) -> None:
        """Reference a pre-existing PVC and mount it at ``mount_point`` on every
        service, with ``HF_HOME`` pointed at it so models come from the shared cache
        instead of HuggingFace. Idempotent; used by CI via --model-cache-pvc.
        """
        spec = self._deployment_spec["spec"]
        pvcs = spec.setdefault("pvcs", [])
        if not any(p.get("name") == pvc_name for p in pvcs):
            pvcs.append({"name": pvc_name, "create": False})
        for service in self.services:
            service.add_pvc_mount(pvc_name, mount_point)
            self.set_service_env_var(service.name, "HF_HOME", mount_point)

    def set_frontend_sidecar_image(
        self, image: str, service_name: Optional[str] = None
    ):
        if service_name is None:
            services = self.services
        else:
            services = [self[service_name]]
        for service in services:
            service.frontend_sidecar_image = image

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
        if self._schema == SCHEMA_V1BETA1:
            return [
                ServiceSpec(comp["name"], comp, schema=self._schema)
                for comp in self._deployment_spec["spec"].get("components", [])
                if "name" in comp
            ]
        return [
            ServiceSpec(svc, spec, schema=self._schema)
            for svc, spec in self._deployment_spec["spec"]["services"].items()
        ]

    def __getitem__(self, service_name: str) -> ServiceSpec:
        """Allow dict-like access: d['Frontend']"""
        if self._schema == SCHEMA_V1BETA1:
            return ServiceSpec(
                service_name,
                self._component_by_name(service_name),
                schema=self._schema,
            )
        return ServiceSpec(
            service_name,
            self._deployment_spec["spec"]["services"][service_name],
            schema=self._schema,
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
        # _get_args() returns parsed argv tokens. The list it returns may be
        # the live spec list (argv-style args) OR a detached parsed copy
        # (shell-style ``command: [/bin/sh, -c]``); always persist via
        # ``_set_args`` so the latter case is re-joined into a single string
        # rather than written back as argv tokens (which the shell would drop).
        args_list = service._get_args()

        arg_index = None
        for i, arg in enumerate(args_list):
            if arg == arg_name:
                arg_index = i
                break

        if arg_index is not None:
            if arg_index + 1 < len(args_list) and not args_list[
                arg_index + 1
            ].startswith("-"):
                args_list[arg_index + 1] = arg_value
            else:
                args_list.insert(arg_index + 1, arg_value)
        else:
            args_list.extend([arg_name, arg_value])

        service._set_args(args_list)

    def get_service(self, service_name: str) -> ServiceSpec:
        """
        Get a specific service from the deployment spec
        """
        if self._schema == SCHEMA_V1BETA1:
            try:
                comp = self._component_by_name(service_name)
            except KeyError:
                raise ValueError(
                    f"Service '{service_name}' not found in deployment spec"
                )
            return ServiceSpec(service_name, comp, schema=self._schema)

        if service_name not in self._deployment_spec["spec"]["services"]:
            raise ValueError(f"Service '{service_name}' not found in deployment spec")

        return ServiceSpec(
            service_name,
            self._deployment_spec["spec"]["services"][service_name],
            schema=self._schema,
        )

    def set_service_replicas(self, service_name: str, replicas: int):
        """
        Set the number of replicas for a specific service
        """
        service = self.get_service(service_name)
        service.replicas = replicas

    def save(self, out_file: str):
        """Save updated deployment to file"""
        with open(out_file, "w") as f:
            yaml.safe_dump(self._deployment_spec, f, default_flow_style=False)


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
class PodStatusDetail:
    """Container-level status snapshot for a single container in a pod."""

    pod_name: str
    container_name: str
    state: Literal["Waiting", "Terminated", "Running", "Unknown"]
    reason: str = ""
    message: str = ""
    exit_code: Optional[int] = None
    restart_count: int = 0

    def format(self) -> str:
        result = f"{self.pod_name}/{self.container_name}: {self.state}"
        if self.reason:
            result += f": {self.reason}"
        if self.message:
            result += f" ({self.message})"
        if self.exit_code is not None:
            result += f" (exit_code={self.exit_code})"
        if self.restart_count > 0:
            result += f" [restarts={self.restart_count}]"
        return result


@dataclass
class ManagedDeployment:
    log_dir: str
    deployment_spec: DeploymentSpec
    namespace: str
    # TODO: this should be determined by the deployment_spec
    # the service containing component_type: Frontend determines what is actually the frontend service
    frontend_service_name: str = "Frontend"
    skip_service_restart: bool = False

    _custom_api: Optional[client.CustomObjectsApi] = None
    _core_api: Optional[client.CoreV1Api] = None
    _in_cluster: bool = False
    _logger: logging.Logger = logging.getLogger()
    _port_forward: Optional[Any] = None
    # Initialized from deployment_spec.name in __post_init__; placeholder needed for dataclass ordering
    _deployment_name: str = field(default="")
    _apps_v1: Optional[Any] = None
    _active_port_forwards: List[Any] = field(default_factory=list)
    # Per ``pod_name/container_name`` -> highest restart count for which we
    # have already dumped the previous-instance log inline during the wait
    # loop. Used by ``_dump_in_flight_restart_logs`` to avoid re-spamming
    # the same crash log every poll while CrashLoopBackOff is still active.
    _logged_restart_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        self._deployment_name = self.deployment_spec.name
        self.log_dir = resolve_test_output_path(self.log_dir)

    async def _init_kubernetes(self):
        """Initialize kubernetes client.

        Priority order:
        1. KUBECONFIG environment variable (CI scenario with proper RBAC)
        2. In-cluster config (for pods without explicit kubeconfig)
        3. Default kubeconfig (~/.kube/config)
        """
        kubeconfig_path = os.environ.get("KUBECONFIG")

        if kubeconfig_path and os.path.exists(kubeconfig_path):
            # Explicit kubeconfig provided (CI scenario) - use it first
            self._logger.info(f"Loading kubeconfig from KUBECONFIG: {kubeconfig_path}")
            await config.load_kube_config(config_file=kubeconfig_path)
            self._in_cluster = False
            self._logger.info("Successfully loaded kubeconfig from KUBECONFIG")
        else:
            try:
                # Try in-cluster config (for pods without explicit kubeconfig)
                self._logger.info("Attempting in-cluster kubernetes config")
                config.load_incluster_config()
                self._in_cluster = True
                self._logger.info("Successfully loaded in-cluster kubernetes config")
            except Exception as e:
                # Fallback to default kube config file (for local development)
                self._logger.warning(
                    f"In-cluster config failed ({type(e).__name__}: {e}), "
                    f"falling back to default kubeconfig (~/.kube/config)"
                )
                await config.load_kube_config()
                self._in_cluster = False
                self._logger.info("Successfully loaded default kubeconfig")

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
                    version=self.deployment_spec.api_version,
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
                    # Surface the previous-instance log tail for any DGD
                    # container that restarted while we are still waiting
                    # for Ready. Dedup'd per (pod, container, restart_count)
                    # so we emit each crash exactly once. Runs every poll
                    # (not just on log_interval) so we catch the first
                    # CrashLoopBackOff event as soon as kubelet reports it,
                    # without waiting up to 60s to learn that prefill is
                    # cycling.
                    in_flight = await self._dump_in_flight_restart_logs()
                    if in_flight:
                        self._logger.warning(
                            "Detected in-flight container restarts (deployment not yet Ready):"
                        )
                        for line in in_flight:
                            self._logger.warning(f"  {line}")

                    if attempt % log_interval == 0:
                        self._logger.info(f"Current deployment state: {current_state}")
                        self._logger.info(f"Current conditions: {conditions}")
                        self._logger.info(
                            f"Elapsed time: {time.time() - start_time:.1f}s / {timeout}s"
                        )
                        self._logger.info(
                            f"Deployment has Ready condition {observed_ready_condition_val} and state {observed_state_val}, desired condition {desired_ready_condition_val} and state {desired_state_val}"
                        )
                        pod_details = await self._get_pod_status_details()
                        if pod_details:
                            for d in pod_details:
                                self._logger.info(f"  Pod status: {d.format()}")
                        pod_events = await self._get_pod_events()
                        if pod_events:
                            self._logger.info("  Pod warning events:")
                            for ev in pod_events:
                                self._logger.info(f"    {ev}")

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

        # Collect pod diagnostics before raising
        pod_details = await self._get_pod_status_details()
        elapsed = time.time() - start_time
        msg = (
            f"Deployment {self._deployment_name} failed to reach "
            f"Ready={desired_ready_condition_val}, state={desired_state_val} "
            f"within {elapsed:.0f}s (timeout={timeout}s)"
        )
        if pod_details:
            detail_lines = "\n".join(f"  {d.format()}" for d in pod_details)
            msg += f"\n\nPod status at timeout:\n{detail_lines}"
        raise TimeoutError(msg)

    async def _get_pod_status_details(self) -> List[PodStatusDetail]:
        """Collect container-level status for all pods owned by this deployment.

        Returns a list of PodStatusDetail objects. Returns empty list on any
        API failure so callers never need to guard against exceptions.
        """
        try:
            assert self._core_api is not None, "Kubernetes API not initialized"
            label = f"nvidia.com/dynamo-graph-deployment-name={self._deployment_name}"
            pods = await self._core_api.list_namespaced_pod(
                self.namespace, label_selector=label
            )

            details: List[PodStatusDetail] = []
            for pod in pods.items:
                pod_name = pod.metadata.name
                pod_status = pod.status
                phase = pod_status.phase if pod_status else "Unknown"

                container_statuses = (
                    pod_status.container_statuses if pod_status else None
                )
                if not container_statuses:
                    details.append(
                        PodStatusDetail(
                            pod_name=pod_name,
                            container_name="*",
                            state="Unknown",
                            reason=f"{phase} (no container status)",
                        )
                    )
                    continue

                for cs in container_statuses:
                    state: Literal[
                        "Waiting", "Terminated", "Running", "Unknown"
                    ] = "Unknown"
                    reason = ""
                    message = ""
                    exit_code: Optional[int] = None

                    if cs.state and cs.state.waiting:
                        state = "Waiting"
                        reason = cs.state.waiting.reason or ""
                        message = cs.state.waiting.message or ""
                    elif cs.state and cs.state.terminated:
                        state = "Terminated"
                        reason = cs.state.terminated.reason or ""
                        exit_code = cs.state.terminated.exit_code
                    elif cs.state and cs.state.running:
                        state = "Running"

                    details.append(
                        PodStatusDetail(
                            pod_name=pod_name,
                            container_name=cs.name,
                            state=state,
                            reason=reason,
                            message=message,
                            exit_code=exit_code,
                            restart_count=cs.restart_count or 0,
                        )
                    )

            return details

        except exceptions.ApiException as e:
            self._logger.debug(f"Failed to collect pod status details: {e}")
            return []

    async def _dump_in_flight_restart_logs(
        self, prev_log_tail_lines: int = 80
    ) -> List[str]:
        """Surface the previous-instance log tail for any DGD container that
        has restarted *while we are still waiting for Ready*.

        This is the diagnostic that converts a "deployment timed out, no idea
        why" CI failure into a self-diagnosing one when a worker is in
        CrashLoopBackOff during startup.
        Each container is reported at most once per restart_count value via
        ``_logged_restart_counts``: the next time the same container restarts
        (count increments), we dump the new previous-instance log. Restarts
        that have already been reported are skipped, so the wait loop does
        not re-spam the same crash log on every poll.

        Returns the list of human-readable warnings (one entry per newly
        observed restart) so callers can log them with proper formatting.
        """
        try:
            assert self._core_api is not None, "Kubernetes API not initialized"
            label = f"nvidia.com/dynamo-graph-deployment-name={self._deployment_name}"
            pods = await self._core_api.list_namespaced_pod(
                self.namespace, label_selector=label
            )
            warnings: List[str] = []
            for pod in pods.items:
                pod_name = pod.metadata.name if pod.metadata else "?"
                statuses = (pod.status.container_statuses if pod.status else None) or []
                for cs in statuses:
                    after = cs.restart_count or 0
                    if after <= 0:
                        continue
                    key = f"{pod_name}/{cs.name}"
                    already = self._logged_restart_counts.get(key, 0)
                    if after <= already:
                        # We've already dumped the previous-instance log
                        # for this exact restart count; nothing new.
                        continue

                    last_reason = ""
                    last_exit: Optional[int] = None
                    if cs.last_state and cs.last_state.terminated:
                        last_reason = cs.last_state.terminated.reason or ""
                        last_exit = cs.last_state.terminated.exit_code

                    warning = (
                        f"{key}: in-flight restart count {after} "
                        f"(last seen {already}) while waiting for Ready"
                    )
                    if last_reason:
                        warning += f" lastReason={last_reason}"
                    if last_exit is not None:
                        warning += f" lastExitCode={last_exit}"

                    prev_log = await self._fetch_previous_container_log(
                        pod_name, cs.name, tail_lines=prev_log_tail_lines
                    )
                    if prev_log:
                        warning += (
                            f"\n      --- last {prev_log_tail_lines} lines of "
                            f"previous {cs.name} log ({pod_name}) ---\n"
                        )
                        for line in prev_log.splitlines():
                            warning += f"      {line}\n"
                        warning += f"      --- end of previous {cs.name} log ---"
                    else:
                        warning += (
                            f"\n      (previous log unavailable for "
                            f"{cs.name} on {pod_name})"
                        )
                    warnings.append(warning)
                    # Mark this restart_count as reported so we don't re-emit
                    # the same log on the next poll. We update only after we
                    # have produced the warning so a transient API failure
                    # does not silently consume the event.
                    self._logged_restart_counts[key] = after
            return warnings
        except exceptions.ApiException as e:
            # API access can fail mid-poll (vCluster syncer hiccup, transient
            # connection reset). Best-effort diagnostic: skip this round and
            # let the next poll retry. Bugs in our own field access (e.g.
            # ``cs.last_state.terminated`` on an unexpected status shape)
            # propagate to the wait loop's "Unexpected exception" handler so
            # they don't get silently swallowed.
            self._logger.debug(f"Failed to check in-flight restart counts: {e}")
            return []

    async def _fetch_previous_container_log(
        self,
        pod_name: str,
        container: str,
        tail_lines: int = 100,
    ) -> Optional[str]:
        """Fetch the previous (pre-restart) instance log for a container.

        Returns the tail of the log as a single string, or None if no previous
        instance exists or the API call fails. This is the artifact that
        normally lives in ``<pod>.<container>.previous.log`` on disk; we
        surface it inline so failed CI runs are self-diagnosing without
        needing an artifact download.
        """
        try:
            assert self._core_api is not None, "Kubernetes API not initialized"
            log = await self._core_api.read_namespaced_pod_log(
                name=pod_name,
                namespace=self.namespace,
                container=container,
                previous=True,
                tail_lines=tail_lines,
            )
            return log if isinstance(log, str) else str(log)
        except exceptions.ApiException as e:
            # 400 "previous terminated container … not found" is the common
            # case when restart_count > 0 but the previous instance log was
            # rotated. 404 means the pod is gone (cleanup race). Anything
            # else (e.g. an AttributeError in our own formatting) is a real
            # bug -- let it bubble to the wait loop's outer handler instead
            # of silently returning None.
            self._logger.debug(
                f"No previous log for {pod_name}/{container} "
                f"(status={e.status}, reason={e.reason})"
            )
            return None

    async def _get_pod_events(self) -> List[str]:
        """Fetch warning events for pods in this deployment's namespace."""
        try:
            assert self._core_api is not None, "Kubernetes API not initialized"
            events = await self._core_api.list_namespaced_event(self.namespace)
            warnings = []
            for event in events.items:
                if event.type != "Normal" and event.involved_object.kind == "Pod":
                    name = event.involved_object.name or "unknown"
                    reason = event.reason or ""
                    msg = event.message or ""
                    warnings.append(f"{name}: {reason} - {msg}")
            return warnings[-10:]
        except Exception as e:
            self._logger.debug(f"Failed to collect pod events: {e}")
            return []

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
                version=self.deployment_spec.api_version,
                namespace=self.namespace,
                plural="dynamographdeployments",
                body=self.deployment_spec.spec(),
            )
            self._logger.info(self.deployment_spec.spec())
            self._logger.info(f"Deployment Started {self._deployment_name}")
        except exceptions.ApiException as e:
            if e.status == 409:  # Already exists
                self._logger.info(f"Deployment {self._deployment_name} already exists")
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

        # Apply env-var edits to the in-memory spec first; then build a
        # schema-appropriate merge patch. JSON merge-patch replaces lists
        # wholesale, so for v1beta1 we send the full components list.
        for service_name in service_names:
            self.deployment_spec.set_service_env_var(
                service_name, "TEST_ROLLING_UPDATE_TRIGGER", secrets.token_hex(8)
            )

        patch_body: dict[str, Any]
        if self.deployment_spec.api_version == SCHEMA_V1BETA1:
            patch_body = {
                "spec": {
                    "components": self.deployment_spec.spec()["spec"].get(
                        "components", []
                    )
                }
            }
        else:
            patch_body = {"spec": {"services": {}}}
            for service_name in service_names:
                updated_envs = self.deployment_spec.get_service_env_vars(service_name)
                patch_body["spec"]["services"][service_name] = {"envs": updated_envs}

        try:
            assert self._custom_api is not None, "Kubernetes API not initialized"
            await self._custom_api.patch_namespaced_custom_object(
                group="nvidia.com",
                version=self.deployment_spec.api_version,
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

        for original_name in service_names:
            label_selector = (
                f"nvidia.com/dynamo-graph-deployment-name={self._deployment_name},"
                f"nvidia.com/dynamo-component={original_name}"
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

        for original_name in service_names:
            # List pods using stable labels that are not affected by worker hash suffixes.
            label_selector = (
                f"nvidia.com/dynamo-graph-deployment-name={self._deployment_name},"
                f"nvidia.com/dynamo-component={original_name}"
            )

            pods: list[Pod] = []

            for pod in kr8s.get(
                "pods", namespace=self.namespace, label_selector=label_selector
            ):
                pods.append(pod)  # type: ignore[arg-type]

            result[original_name] = pods

        return result

    def get_pod_manifest_logs_metrics(self, service_name: str, pod: Pod, suffix=""):
        directory = os.path.join(self.log_dir, service_name)
        os.makedirs(directory, exist_ok=True)

        try:
            with open(os.path.join(directory, f"{pod.name}{suffix}.yaml"), "w") as f:
                f.write(pod.to_yaml())
        except Exception as e:
            self._logger.error(e)

        # Resolve the container list from the pod manifest. Multi-container pods
        # (e.g. workers with a frontendSidecar, or pods running under vCluster
        # with the rewrite-hosts init container) reject ``pod.logs()`` calls
        # that omit ``container=`` with a "container name must be specified"
        # so we have to be more detailed here. Iterate every container and write one
        # log file per container.
        container_names: List[str] = []
        try:
            spec = pod.raw.get("spec", {}) if hasattr(pod, "raw") else {}
            for c in spec.get("initContainers", []) or []:
                if c.get("name"):
                    container_names.append(c["name"])
            for c in spec.get("containers", []) or []:
                if c.get("name"):
                    container_names.append(c["name"])
        except Exception as e:
            self._logger.debug(f"Failed to resolve containers for {pod.name}: {e}")

        if not container_names:
            container_names = [""]

        for container in container_names:
            file_suffix = f".{container}" if container else ""
            try:
                logs = pod.logs(container=container) if container else pod.logs()
                with open(
                    os.path.join(directory, f"{pod.name}{file_suffix}{suffix}.log"),
                    "w",
                ) as f:
                    f.write("\n".join(logs))
            except Exception as e:
                self._logger.error(
                    f"Failed to fetch logs for {pod.name} container={container or '<default>'}: {e}"
                )
            try:
                previous_logs = (
                    pod.logs(container=container, previous=True)
                    if container
                    else pod.logs(previous=True)
                )
                with open(
                    os.path.join(
                        directory,
                        f"{pod.name}{file_suffix}{suffix}.previous.log",
                    ),
                    "w",
                ) as f:
                    f.write("\n".join(previous_logs))
            except Exception as e:
                # Previous-instance logs are absent unless the container has
                # restarted. Common case is "no previous terminated container"
                # -- log at debug so we don't spam the test output.
                self._logger.debug(
                    f"No previous logs for {pod.name} container={container or '<default>'}: {e}"
                )

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
                    version=self.deployment_spec.api_version,
                    namespace=self.namespace,
                    plural="dynamographdeployments",
                    name=self._deployment_name,
                )
        except exceptions.ApiException as e:
            if e.status != 404:  # Ignore if already deleted
                raise

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
                except Exception as e:
                    self._logger.debug(
                        f"Error stopping port forward for pod {pod.name}: {e}"
                    )
                # kr8s' sync stop() can return before the background thread has
                # fully torn down (or even finished starting), so we can't assume
                # the old forward is dead once stop() returns. Track it so
                # _cleanup() stops it later regardless of whether stop() raised,
                # rather than losing the reference when we replace the object.
                if port_forward not in self._active_port_forwards:
                    self._active_port_forwards.append(port_forward)
                # Create a fresh portforward object so local_port=0 picks a new
                # ephemeral port rather than re-binding the previously assigned
                # port that may still be in TIME_WAIT.
                try:
                    port_forward = pod.portforward(
                        remote_port=remote_port,
                        local_port=0,
                        address="127.0.0.1",
                    )
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
            # Collect logs/metrics first; any PFs opened here will be tracked and stopped below.
            self._get_service_logs()
            self._logger.info(
                f"Cleaning up {len(self._active_port_forwards)} active port forwards"
            )
            for port_forward in self._active_port_forwards:
                try:
                    port_forward.stop()
                except RuntimeError as e:
                    # Expected error when pod is terminated:
                    # "anext(): asynchronous generator is already running"
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
            await self._delete_deployment()

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

            await self._create_deployment()
            await self._wait_for_ready()

        except:
            await self._cleanup()
            raise
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._cleanup()


class ManagedDGDR:
    """Async helper for managing DynamoGraphDeploymentRequest custom resources.

    Provides CRUD operations and phase-polling against the DGDR CRD using the
    ``kubernetes_asyncio`` client, following the same patterns as
    ``ManagedDeployment`` (shared kubeconfig initialisation, timeout logic,
    structured error messages).

    Typical usage from a pytest fixture::

        dgdr = ManagedDGDR(namespace="default")
        await dgdr.init()
        await dgdr.create(manifest)
        phase = await dgdr.wait_for_phase(name, "Ready", timeout=600)
        await dgdr.delete(name)
        await dgdr.close()
    """

    # CRD coordinates for DGDR
    DGDR_GROUP = "nvidia.com"
    DGDR_VERSION = "v1beta1"
    DGDR_PLURAL = "dynamographdeploymentrequests"

    # CRD coordinates for DGD (for mocker cleanup)
    DGD_PLURAL = "dynamographdeployments"

    DEFAULT_POLL_INTERVAL = 10  # seconds

    def __init__(
        self,
        namespace: str = "default",
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        self.namespace = namespace
        self._custom_api: Optional[client.CustomObjectsApi] = None
        self._api_client: Optional[client.ApiClient] = None
        self._logger = logging.getLogger(self.__class__.__name__)
        self._loop = loop

    def run(self, coro):
        """Run an async coroutine synchronously using the stored event loop.

        Convenience for callers that are not themselves async (e.g. pytest
        fixtures and synchronous test methods).
        """
        if self._loop is None:
            raise RuntimeError(
                "No event loop set on ManagedDGDR; pass loop= at construction or call init() first"
            )
        return self._loop.run_until_complete(coro)

    async def init(self) -> None:
        """Initialise the kubernetes_asyncio client.

        Priority: KUBECONFIG env → in-cluster → ~/.kube/config  (same as
        ManagedDeployment._init_kubernetes).
        """
        kubeconfig_path = os.environ.get("KUBECONFIG")

        if kubeconfig_path and os.path.exists(kubeconfig_path):
            self._logger.info("Loading kubeconfig from KUBECONFIG: %s", kubeconfig_path)
            await config.load_kube_config(config_file=kubeconfig_path)
        else:
            try:
                self._logger.info("Attempting in-cluster kubernetes config")
                config.load_incluster_config()
            except Exception as e:
                self._logger.warning(
                    "In-cluster config failed (%s: %s), falling back to default kubeconfig",
                    type(e).__name__,
                    e,
                )
                await config.load_kube_config()

        self._api_client = client.ApiClient()
        self._custom_api = client.CustomObjectsApi(self._api_client)

    async def close(self) -> None:
        """Close the underlying API client."""
        if self._api_client:
            await self._api_client.close()
            self._api_client = None
            self._custom_api = None

    # ----- CRUD -----

    async def create(self, manifest: dict) -> str:
        """Create a DGDR custom resource.  Returns the resource name."""
        assert self._custom_api is not None, "call init() first"
        name = manifest["metadata"]["name"]
        await self._custom_api.create_namespaced_custom_object(
            group=self.DGDR_GROUP,
            version=self.DGDR_VERSION,
            namespace=self.namespace,
            plural=self.DGDR_PLURAL,
            body=manifest,
        )
        self._logger.info("Created DGDR %s/%s", self.namespace, name)
        return name

    async def get(self, name: str) -> Optional[dict]:
        """Get a DGDR as a dict, or ``None`` if not found."""
        assert self._custom_api is not None, "call init() first"
        try:
            return await self._custom_api.get_namespaced_custom_object(
                group=self.DGDR_GROUP,
                version=self.DGDR_VERSION,
                namespace=self.namespace,
                plural=self.DGDR_PLURAL,
                name=name,
            )
        except exceptions.ApiException as e:
            if e.status == 404:
                return None
            raise

    async def delete(self, name: str, ignore_not_found: bool = True) -> None:
        """Delete a DGDR."""
        assert self._custom_api is not None, "call init() first"
        try:
            await self._custom_api.delete_namespaced_custom_object(
                group=self.DGDR_GROUP,
                version=self.DGDR_VERSION,
                namespace=self.namespace,
                plural=self.DGDR_PLURAL,
                name=name,
            )
            self._logger.info("Deleted DGDR %s/%s", self.namespace, name)
        except exceptions.ApiException as e:
            if e.status == 404 and ignore_not_found:
                return
            raise

    async def list(self, label_selector: str = "") -> List[dict]:
        """List DGDRs, optionally filtered by label selector.  Returns items."""
        assert self._custom_api is not None, "call init() first"
        resp = await self._custom_api.list_namespaced_custom_object(
            group=self.DGDR_GROUP,
            version=self.DGDR_VERSION,
            namespace=self.namespace,
            plural=self.DGDR_PLURAL,
            label_selector=label_selector,
        )
        return resp.get("items", [])

    async def server_dry_run(self, manifest: dict) -> dict:
        """Apply with server-side dry-run to validate admission webhooks.

        Returns the API response dict.  Raises ``ApiException`` on rejection.
        """
        assert self._custom_api is not None, "call init() first"
        return await self._custom_api.create_namespaced_custom_object(
            group=self.DGDR_GROUP,
            version=self.DGDR_VERSION,
            namespace=self.namespace,
            plural=self.DGDR_PLURAL,
            body=manifest,
            dry_run="All",
        )

    # ----- Phase helpers -----

    async def get_phase(self, name: str) -> Optional[str]:
        """Return ``status.phase`` of the named DGDR, or ``None``."""
        obj = await self.get(name)
        if obj is None:
            return None
        return obj.get("status", {}).get("phase")

    async def get_condition(self, name: str, condition_type: str) -> Optional[dict]:
        """Return the named condition dict from ``status.conditions``."""
        obj = await self.get(name)
        if obj is None:
            return None
        for c in obj.get("status", {}).get("conditions", []):
            if c.get("type") == condition_type:
                return c
        return None

    async def wait_for_phase(
        self,
        name: str,
        target_phase: str,
        timeout: int = 3600,
        fail_fast_phases: Optional[List[str]] = None,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> str:
        """Poll until the DGDR reaches *target_phase* or times out.

        Returns the final observed phase.  Raises ``AssertionError`` on
        fail-fast and ``TimeoutError`` on timeout.
        """
        if fail_fast_phases is None:
            fail_fast_phases = ["Failed"]

        deadline = time.monotonic() + timeout
        last_phase: Optional[str] = None

        while time.monotonic() < deadline:
            current = await self.get_phase(name)
            if current != last_phase:
                self._logger.info("DGDR %s/%s phase: %s", self.namespace, name, current)
                last_phase = current

            if current == target_phase:
                return current
            if current in fail_fast_phases:
                obj = await self.get(name)
                conditions = obj.get("status", {}).get("conditions", []) if obj else []
                raise AssertionError(
                    f"DGDR {self.namespace}/{name} reached fail-fast phase {current!r} "
                    f"while waiting for {target_phase!r}. conditions={conditions}"
                )
            await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"Timed out after {timeout}s waiting for DGDR {self.namespace}/{name} "
            f"to reach phase {target_phase!r}. Last phase: {last_phase!r}"
        )

    async def wait_for_any_phase(
        self,
        name: str,
        target_phases: List[str],
        timeout: int = 3600,
        poll_interval: int = DEFAULT_POLL_INTERVAL,
    ) -> str:
        """Poll until the DGDR reaches any of *target_phases*.  Returns matched phase."""
        deadline = time.monotonic() + timeout
        last_phase: Optional[str] = None

        while time.monotonic() < deadline:
            current = await self.get_phase(name)
            if current != last_phase:
                self._logger.info("DGDR %s/%s phase: %s", self.namespace, name, current)
                last_phase = current
            if current in target_phases:
                return current
            await asyncio.sleep(poll_interval)

        raise TimeoutError(
            f"Timed out after {timeout}s waiting for DGDR {self.namespace}/{name} "
            f"to reach any of {target_phases!r}. Last phase: {last_phase!r}"
        )

    # ----- DGD helpers (for mocker cleanup) -----

    async def delete_dgd(self, name: str, ignore_not_found: bool = True) -> None:
        """Delete a DynamoGraphDeployment resource."""
        assert self._custom_api is not None, "call init() first"
        try:
            await self._custom_api.delete_namespaced_custom_object(
                group=self.DGDR_GROUP,
                version="v1alpha1",
                namespace=self.namespace,
                plural=self.DGD_PLURAL,
                name=name,
            )
            self._logger.info("Deleted DGD %s/%s", self.namespace, name)
        except exceptions.ApiException as e:
            if e.status == 404 and ignore_not_found:
                return
            raise

    async def get_dgd(self, name: str) -> Optional[dict]:
        """Get a DynamoGraphDeployment, or ``None`` if not found."""
        assert self._custom_api is not None, "call init() first"
        try:
            return await self._custom_api.get_namespaced_custom_object(
                group=self.DGDR_GROUP,
                version="v1alpha1",
                namespace=self.namespace,
                plural=self.DGD_PLURAL,
                name=name,
            )
        except exceptions.ApiException as e:
            if e.status == 404:
                return None
            raise


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
