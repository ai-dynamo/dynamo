# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import logging
from typing import Any, Protocol, Tuple

from dynamo.planner.defaults import SubComponentType
from dynamo.profiler.utils.config import (
    Config,
    Container,
    PodSpec,
    ServiceResources,
    break_arguments,
    get_service_name_by_type,
    set_argument_value,
    update_image,
)
from dynamo.profiler.utils.defaults import EngineType

logger = logging.getLogger(__name__)


class ConfigModifierProtocol(Protocol):
    @classmethod
    def convert_config(
        cls,
        config: dict,
        target: EngineType,
        is_moe_model: bool = False,
    ) -> dict:
        ...

    @classmethod
    def set_config_tp_size(
        cls,
        config: dict,
        tp_size: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        ...

    @classmethod
    def set_config_tep_size(
        cls,
        config: dict,
        tep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        ...

    @classmethod
    def set_config_dep_size(
        cls,
        config: dict,
        dep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        ...

    @classmethod
    def get_model_name(cls, config: dict) -> Tuple[str, str]:
        ...

    @classmethod
    def set_prefill_config(
        cls,
        config: dict,
        max_batch_size: int,
        max_num_tokens: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        ...

    @classmethod
    def get_port(cls, config: dict) -> int:
        ...

    @classmethod
    def get_kv_cache_size_from_dynamo_log(
        cls, dynamo_log_fn: str, attention_dp_size: int = 1
    ) -> int:
        ...

    @classmethod
    def load_default_config(cls, mode: str = "disagg") -> dict:
        ...

    @classmethod
    def update_model(
        cls, config: dict, model_name: str, model_path: str | None = None
    ) -> dict:
        ...

    @classmethod
    def update_image(cls, config: dict, image: str) -> dict:
        ...

    @classmethod
    def update_model_from_pvc(
        cls,
        config: dict,
        model_name: str,
        pvc_name: str,
        pvc_mount_path: str,
        pvc_path: str,
    ) -> dict:
        ...


class BaseConfigModifier:
    """
    Shared helper base class for profiler config modifiers.

    This class intentionally lives in `protocol.py` so all backends can inherit
    common PVC + volumeMount + frontend CLI patching behavior.
    """

    # Subclasses should override, e.g. "vllm" / "sglang" / "trtllm"
    BACKEND: str = ""
    # Worker CLI arg name for model path / name. vLLM uses "--model"; others use "--model-path".
    WORKER_MODEL_PATH_ARG: str = "--model-path"
    WORKER_SERVED_MODEL_NAME_ARG: str = "--served-model-name"

    @classmethod
    def _get_model_name_and_path_from_args(cls, args: list[str]) -> Tuple[str, str]:
        """
        Extract model name and path from worker args.

        Checks --served-model-name first (API name), then falls back to
        backend-specific path argument (--model-path or --model).

        Args:
            args: Broken argument list

        Returns:
            Tuple of (model_name, model_path)

        Raises:
            ValueError: If neither --served-model-name nor model path arg is found
        """
        model_name = ""
        # Check for --served-model-name first (API model name)
        for i, arg in enumerate(args):
            if arg == cls.WORKER_SERVED_MODEL_NAME_ARG and i + 1 < len(args):
                model_name = args[i + 1]
                break

        # Check for backend-specific path argument
        model_path = ""
        for i, arg in enumerate(args):
            if arg == cls.WORKER_MODEL_PATH_ARG and i + 1 < len(args):
                model_path = args[i + 1]
                break

        # Require at least one to be specified
        if not model_name and not model_path:
            raise ValueError(
                f"Cannot determine model: neither {cls.WORKER_MODEL_PATH_ARG} nor "
                f"{cls.WORKER_SERVED_MODEL_NAME_ARG} found in worker configuration. "
                f"Please specify a model name/path in your config."
            )

        # If only one is specified, use it for both
        if not model_path:
            model_path = model_name
        elif not model_name:
            model_name = model_path

        return model_name, model_path

    @classmethod
    def _normalize_model_path(cls, pvc_mount_path: str, pvc_path: str) -> str:
        mount = (pvc_mount_path or "").rstrip("/")
        sub = (pvc_path or "").lstrip("/")
        if not sub:
            return mount
        return f"{mount}/{sub}"

    @classmethod
    def _ensure_spec_pvc(cls, cfg: Config, pvc_name: str) -> None:
        pvcs = getattr(cfg.spec, "pvcs", None)
        if pvcs is None:
            pvcs = []

        for pvc in pvcs:
            if isinstance(pvc, dict) and pvc.get("name") == pvc_name:
                # Ensure create is false (do not create PVC in profiling flows)
                pvc["create"] = False
                setattr(cfg.spec, "pvcs", pvcs)
                return

        pvcs.append({"name": pvc_name, "create": False})
        setattr(cfg.spec, "pvcs", pvcs)

    @classmethod
    def _ensure_service_volume_mount(
        cls, service: Any, pvc_name: str, mount_path: str
    ) -> None:
        volume_mounts = getattr(service, "volumeMounts", None)
        if volume_mounts is None:
            volume_mounts = []
        if not isinstance(volume_mounts, list):
            volume_mounts = []

        for vm in volume_mounts:
            if isinstance(vm, dict) and vm.get("name") == pvc_name:
                vm["mountPoint"] = mount_path
                setattr(service, "volumeMounts", volume_mounts)
                return

        volume_mounts.append({"name": pvc_name, "mountPoint": mount_path})
        setattr(service, "volumeMounts", volume_mounts)

    @classmethod
    def _update_container_args_preserving_shell_form(
        cls, container: Container, update_fn
    ) -> None:
        """
        Update container args while preserving a common shell form:
        - If `command` is `sh -c` and args is a single-string list, keep it that way.
        """
        original_args = container.args
        cmd = container.command or []

        is_shell_c = (
            isinstance(cmd, list)
            and len(cmd) >= 2
            and cmd[0] in ("/bin/sh", "sh")
            and cmd[1] == "-c"
        )
        is_single_string_args = (
            isinstance(original_args, list)
            and len(original_args) == 1
            and isinstance(original_args[0], str)
        )

        tokens = break_arguments(original_args)
        tokens = update_fn(tokens)

        if is_shell_c and is_single_string_args:
            # Keep as one string for `sh -c`
            import shlex

            container.args = [shlex.join(tokens)]
        else:
            container.args = tokens

    @classmethod
    def _update_frontend_cli(
        cls, cfg: Config, model_name: str, model_path: str
    ) -> None:
        frontend = cfg.spec.services.get("Frontend")
        if not frontend:
            return

        if frontend.extraPodSpec is None:
            frontend.extraPodSpec = PodSpec(mainContainer=Container())
        if frontend.extraPodSpec.mainContainer is None:
            frontend.extraPodSpec.mainContainer = Container()

        c = frontend.extraPodSpec.mainContainer

        # If operator defaults are being used (no command/args), we must provide full CLI.
        if not c.command and not c.args:
            c.command = ["python3", "-m", "dynamo.frontend"]
            c.args = []

        def _patch(tokens: list[str]) -> list[str]:
            tokens = set_argument_value(tokens, "--model-name", model_name)
            tokens = set_argument_value(tokens, "--model-path", model_path)
            return tokens

        cls._update_container_args_preserving_shell_form(c, _patch)

    @classmethod
    def _apply_model_update_to_cfg(
        cls,
        cfg: Config,
        model_name: str,
        model_path: str,
        patch_frontend: bool,
    ) -> None:
        """
        Apply model updates to a validated DGD config object.

        This is the shared implementation for both:
        - update_model()
        - update_model_from_pvc()
        """

        def _patch_service(service: Any) -> None:
            if not service.extraPodSpec or not service.extraPodSpec.mainContainer:
                return
            c = service.extraPodSpec.mainContainer

            def _patch(tokens: list[str]) -> list[str]:
                tokens = set_argument_value(
                    tokens, cls.WORKER_MODEL_PATH_ARG, model_path
                )
                tokens = set_argument_value(
                    tokens, cls.WORKER_SERVED_MODEL_NAME_ARG, model_name
                )
                return tokens

            cls._update_container_args_preserving_shell_form(c, _patch)

        # Update workers (prefill + decode) if present.
        patched_services: set[str] = set()
        for sct in (SubComponentType.PREFILL, SubComponentType.DECODE):
            try:
                svc_name = get_service_name_by_type(cfg, cls.BACKEND, sct)
            except Exception:
                continue
            if svc_name not in cfg.spec.services:
                continue
            _patch_service(cfg.spec.services[svc_name])
            patched_services.add(svc_name)

        # Fallback for agg mode: if no worker was patched via subComponentType
        # lookup, patch any non-Frontend/Planner worker service.
        if not patched_services:
            for name, service in cfg.spec.services.items():
                if name not in cls._NON_WORKER_SERVICES:
                    _patch_service(service)
                    patched_services.add(name)

        if patch_frontend:
            cls._update_frontend_cli(cfg, model_name=model_name, model_path=model_path)

    @classmethod
    def update_model(
        cls, config: dict, model_name: str, model_path: str | None = None
    ) -> dict:
        """
        Unified model update API.

        Args:
            config: DGD config dict
            model_name: served model name (HF id)
            model_path: model path inside container (if using PVC/local path). If omitted,
                defaults to model_name (HF download case for workers).
        """
        cfg = Config.model_validate(config)
        if model_path is None:
            model_path = model_name

        # Frontend requires a real filesystem path (validate_model_path checks isdir),
        # so only inject model args when `model_path` looks like a path.
        patch_frontend = bool(
            isinstance(model_path, str)
            and (model_path.startswith("/") or model_path.startswith("."))
        )
        cls._apply_model_update_to_cfg(
            cfg,
            model_name=model_name,
            model_path=model_path,
            patch_frontend=patch_frontend,
        )

        return cfg.model_dump()

    @classmethod
    def update_model_from_pvc(
        cls,
        config: dict,
        model_name: str,
        pvc_name: str,
        pvc_mount_path: str,
        pvc_path: str,
    ) -> dict:
        """
        Update a DGD config to serve `model_name`, with weights located in a mounted PVC.

        Common steps across backends:
        - Add `spec.pvcs`
        - Add `volumeMounts` for Frontend + prefill + decode (if present)
        - Patch Frontend CLI (`--model-name`, `--model-path`)
        - Delegate worker CLI patching to backend-specific implementation.
        """
        if not pvc_name:
            return config

        cfg = Config.model_validate(config)
        model_path = cls._normalize_model_path(pvc_mount_path, pvc_path)

        cls._ensure_spec_pvc(cfg, pvc_name)

        # Mount to Frontend + prefill + decode services if present.
        if "Frontend" in cfg.spec.services:
            cls._ensure_service_volume_mount(
                cfg.spec.services["Frontend"], pvc_name, pvc_mount_path
            )

        for sct in (SubComponentType.PREFILL, SubComponentType.DECODE):
            svc_name = get_service_name_by_type(cfg, cls.BACKEND, sct)
            if svc_name in cfg.spec.services:
                cls._ensure_service_volume_mount(
                    cfg.spec.services[svc_name], pvc_name, pvc_mount_path
                )

        # Patch workers + frontend with PVC model path.
        cls._apply_model_update_to_cfg(
            cfg,
            model_name=model_name,
            model_path=model_path,
            patch_frontend=True,
        )

        return cfg.model_dump()

    @classmethod
    def build_dgd_config(
        cls,
        mode: str,
        model_name: str,
        image: str,
        # Disagg workers (used when mode=="disagg")
        prefill_cli_args: list[str] | None = None,
        prefill_replicas: int = 1,
        prefill_gpus: int = 1,
        decode_cli_args: list[str] | None = None,
        decode_replicas: int = 1,
        decode_gpus: int = 1,
        # Agg worker (used when mode=="agg")
        agg_cli_args: list[str] | None = None,
        agg_replicas: int = 1,
        agg_gpus: int = 1,
        # Optional
        namespace: str | None = None,
        model_path: str | None = None,
        pvc_name: str | None = None,
        pvc_mount_path: str | None = None,
    ) -> dict:
        """
        Build a complete DynamoGraphDeployment config by loading a base YAML
        and injecting pre-computed CLI args, model, image, replicas, and GPU resources.

        This is intended for use by external tools (e.g. AIConfigurator) that
        have already computed the per-worker CLI arguments and just need them
        placed into a valid DGD config structure.

        Args:
            mode: "agg" or "disagg"
            model_name: Model name / HuggingFace ID (e.g. "Qwen/Qwen3-32B")
            image: Container image for all services
            prefill_cli_args: Pre-computed CLI args list for prefill worker
            prefill_replicas: Number of prefill worker replicas
            prefill_gpus: GPUs per prefill worker
            decode_cli_args: Pre-computed CLI args list for decode worker
            decode_replicas: Number of decode worker replicas
            decode_gpus: GPUs per decode worker
            agg_cli_args: Pre-computed CLI args list for agg worker
            agg_replicas: Number of agg worker replicas
            agg_gpus: GPUs per agg worker
            namespace: K8s namespace (optional)
            model_path: Model path if different from model_name (e.g. PVC path)
            pvc_name: PVC claim name for model cache (optional)
            pvc_mount_path: PVC mount path (optional)

        Returns:
            Complete DGD config dict ready for YAML serialization

        Raises:
            ValueError: If mode is not "agg" or "disagg"
        """
        if mode not in ("agg", "disagg"):
            raise ValueError(f"Invalid mode '{mode}': must be 'agg' or 'disagg'")

        config = cls.load_default_config(mode=mode)
        cfg = Config.model_validate(config)

        # Set metadata
        cfg.metadata.name = f"{cls.BACKEND}-{mode}"
        if namespace and hasattr(cfg.metadata, "namespace"):
            cfg.metadata.namespace = namespace

        # Update image for all services
        config = update_image(cfg.model_dump(), image)
        cfg = Config.model_validate(config)

        if mode == "disagg":
            cls._apply_disagg_workers(
                cfg,
                prefill_cli_args=prefill_cli_args or [],
                prefill_replicas=prefill_replicas,
                prefill_gpus=prefill_gpus,
                decode_cli_args=decode_cli_args or [],
                decode_replicas=decode_replicas,
                decode_gpus=decode_gpus,
            )
        else:
            cls._apply_agg_worker(
                cfg,
                agg_cli_args=agg_cli_args or [],
                agg_replicas=agg_replicas,
                agg_gpus=agg_gpus,
            )

        # Update model (handles worker args + frontend patching)
        effective_model_path = model_path or model_name
        if pvc_name and pvc_mount_path:
            result = cls.update_model_from_pvc(
                cfg.model_dump(),
                model_name=model_name,
                pvc_name=pvc_name,
                pvc_mount_path=pvc_mount_path,
                pvc_path="",
            )
        else:
            result = cls.update_model(
                cfg.model_dump(),
                model_name=model_name,
                model_path=effective_model_path,
            )

        return result

    _NON_WORKER_SERVICES = {"Frontend", "Planner"}

    @classmethod
    def _resolve_service_name(
        cls,
        cfg: Config,
        component_type: SubComponentType,
    ) -> str | None:
        """Resolve the service name for a given component type, with fallback."""
        try:
            return get_service_name_by_type(cfg, cls.BACKEND, component_type)
        except Exception:
            # Fallback: find the first worker service (skip Frontend, Planner)
            for name in cfg.spec.services:
                if name not in cls._NON_WORKER_SERVICES:
                    return name
            return None

    @staticmethod
    def _apply_worker_config(
        service: Any,
        cli_args: list[str],
        replicas: int,
        gpus: int,
    ) -> None:
        """Apply CLI args, replicas, and GPU resources to a single worker service."""
        service.replicas = replicas

        if service.resources is None:
            service.resources = ServiceResources()
        if service.resources.limits is None:
            service.resources.limits = {}
        service.resources.limits["gpu"] = str(gpus)

        if service.extraPodSpec and service.extraPodSpec.mainContainer:
            service.extraPodSpec.mainContainer.args = list(cli_args)

    @classmethod
    def _apply_disagg_workers(
        cls,
        cfg: Config,
        prefill_cli_args: list[str],
        prefill_replicas: int,
        prefill_gpus: int,
        decode_cli_args: list[str],
        decode_replicas: int,
        decode_gpus: int,
    ) -> None:
        """Apply CLI args, replicas, and GPU resources to disagg worker services."""
        for sct, cli_args, replicas, gpus in [
            (
                SubComponentType.PREFILL,
                prefill_cli_args,
                prefill_replicas,
                prefill_gpus,
            ),
            (SubComponentType.DECODE, decode_cli_args, decode_replicas, decode_gpus),
        ]:
            svc_name = cls._resolve_service_name(cfg, sct)
            if svc_name is None or svc_name not in cfg.spec.services:
                logger.warning(
                    "Could not find %s service for backend %s, skipping",
                    sct.value,
                    cls.BACKEND,
                )
                continue
            cls._apply_worker_config(
                cfg.spec.services[svc_name], cli_args, replicas, gpus
            )

    @classmethod
    def _apply_agg_worker(
        cls,
        cfg: Config,
        agg_cli_args: list[str],
        agg_replicas: int,
        agg_gpus: int,
    ) -> None:
        """Apply CLI args, replicas, and GPU resources to the agg worker service.

        In agg mode, the default config template may use a generic worker
        service name (e.g. ``TRTLLMWorker``) that does not match the disagg
        naming convention (``TRTLLMDecodeWorker``).  We first try the standard
        DECODE lookup, then fall back to any non-Frontend/Planner service.
        """
        svc_name = cls._resolve_service_name(cfg, SubComponentType.DECODE)
        if svc_name is None or svc_name not in cfg.spec.services:
            # Fallback: find any worker service in the config
            for name in cfg.spec.services:
                if name not in cls._NON_WORKER_SERVICES:
                    svc_name = name
                    break
        if svc_name is None or svc_name not in cfg.spec.services:
            logger.warning("Could not find worker service for agg mode")
            return
        cls._apply_worker_config(
            cfg.spec.services[svc_name], agg_cli_args, agg_replicas, agg_gpus
        )
