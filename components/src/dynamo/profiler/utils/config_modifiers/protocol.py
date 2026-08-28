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
import shlex
from collections.abc import Iterator
from typing import Protocol, Tuple
from uuid import uuid4

from dynamo.planner.config.defaults import SubComponentType
from dynamo.profiler.utils.config import (
    Component,
    Config,
    Container,
    break_arguments,
    get_component_by_name,
    get_component_name_by_type,
    get_main_container,
    get_main_container_dict,
    sanitize_cli_args,
    set_unique_argument_value,
    setup_worker_component_resources,
    update_image,
)
from dynamo.profiler.utils.defaults import EngineType
from dynamo.profiler.utils.model_cache_paths import normalize_model_cache_path

logger = logging.getLogger(__name__)


class ConfigModifierProtocol(Protocol):
    @classmethod
    def apply_runtime_defaults(cls, config: dict) -> dict:
        ...

    @classmethod
    def apply_model_runtime_constraints(
        cls, config: dict, model_name_or_path: str | None
    ) -> dict:
        ...

    @classmethod
    def finalize_dgd(cls, config: dict, model_name_or_path: str | None = None) -> dict:
        ...

    @classmethod
    def supports_remote_code_trust(cls) -> bool:
        ...

    @classmethod
    def has_remote_code_trust(cls, config: dict) -> bool:
        ...

    @classmethod
    def enable_remote_code_trust(cls, config: dict) -> dict:
        ...

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

    @classmethod
    def build_dgd_config(
        cls,
        mode: str,
        model_name: str,
        image: str,
        prefill_cli_args: list[str] | None = None,
        prefill_replicas: int = 1,
        prefill_gpus: int = 1,
        decode_cli_args: list[str] | None = None,
        decode_replicas: int = 1,
        decode_gpus: int = 1,
        agg_cli_args: list[str] | None = None,
        agg_replicas: int = 1,
        agg_gpus: int = 1,
        namespace: str | None = None,
        model_path: str | None = None,
        pvc_name: str | None = None,
        pvc_mount_path: str | None = None,
        num_gpus_per_node: int | None = None,
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
    REMOTE_CODE_TRUST_FLAG: str | None = None

    @classmethod
    def apply_runtime_defaults(cls, config: dict) -> dict:
        """Apply idempotent backend defaults required by every deployment."""
        return config

    @classmethod
    def apply_model_runtime_constraints(
        cls, config: dict, model_name_or_path: str | None
    ) -> dict:
        """Apply idempotent constraints derived from the selected model."""
        return config

    @classmethod
    def finalize_dgd(cls, config: dict, model_name_or_path: str | None = None) -> dict:
        """Apply backend defaults and model constraints at the DGD boundary."""
        config = cls.apply_runtime_defaults(config)
        return cls.apply_model_runtime_constraints(config, model_name_or_path)

    @classmethod
    def supports_remote_code_trust(cls) -> bool:
        """Return whether this backend expresses remote-code trust as a CLI flag."""
        return cls.REMOTE_CODE_TRUST_FLAG is not None

    @classmethod
    def _worker_containers(cls, config: dict) -> Iterator[dict]:
        """Yield main containers for worker components in a DGD."""
        components = config.get("spec", {}).get("components", [])
        if not isinstance(components, list):
            return
        for component in components:
            if not isinstance(component, dict) or component.get("type") not in {
                "worker",
                "prefill",
                "decode",
            }:
                continue
            main_container = get_main_container_dict(component)
            if main_container is not None:
                yield main_container

    @staticmethod
    def _is_mocker_container(container: dict) -> bool:
        command = container.get("command") or []
        args = container.get("args") or []
        return "dynamo.mocker" in " ".join(str(token) for token in [*command, *args])

    @staticmethod
    def _is_shell_command(container: dict) -> bool:
        command = container.get("command") or []
        args = container.get("args") or []
        return (
            isinstance(command, list)
            and len(command) >= 2
            and command[0] in ("/bin/sh", "sh")
            and command[1] == "-c"
            and len(args) == 1
            and isinstance(args[0], str)
        )

    @classmethod
    def _container_has_flag(cls, container: dict, flag: str) -> bool:
        args = container.get("args") or []
        if cls._is_shell_command(container):
            return flag in args[0]
        return flag in args

    @classmethod
    def has_remote_code_trust(cls, config: dict) -> bool:
        """Return whether every real worker already enables remote-code trust."""
        flag = cls.REMOTE_CODE_TRUST_FLAG
        if flag is None:
            return False

        workers = list(cls._worker_containers(config))
        return bool(workers) and all(
            cls._is_mocker_container(container)
            or cls._container_has_flag(container, flag)
            for container in workers
        )

    @classmethod
    def enable_remote_code_trust(cls, config: dict) -> dict:
        """Enable remote-code trust on real workers without changing shell form."""
        flag = cls.REMOTE_CODE_TRUST_FLAG
        if flag is None:
            return config

        for container in cls._worker_containers(config):
            if cls._is_mocker_container(container) or cls._container_has_flag(
                container, flag
            ):
                continue
            args = container.get("args") or []
            if cls._is_shell_command(container):
                args[0] = f"{args[0]} {flag}"
            else:
                container["args"] = [*args, flag]
        return config

    @classmethod
    def load_default_config(cls, mode: str = "disagg") -> dict:
        """Load default DGD config for the given mode. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement load_default_config")

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
        """Resolve a PVC model-cache path to the mounted container path."""
        return normalize_model_cache_path(pvc_mount_path, pvc_path)

    @classmethod
    def _ensure_component_volume_mount(
        cls, component: Component, pvc_name: str, mount_path: str
    ) -> None:
        """Mount an existing PVC into a v1beta1 component's main container."""
        pod_spec = component.podTemplate.spec
        volumes = pod_spec.volumes or []
        for volume in volumes:
            if isinstance(volume, dict) and volume.get("name") == pvc_name:
                volume["persistentVolumeClaim"] = {"claimName": pvc_name}
                break
        else:
            volumes.append(
                {
                    "name": pvc_name,
                    "persistentVolumeClaim": {"claimName": pvc_name},
                }
            )
        pod_spec.volumes = volumes

        main_container = get_main_container(component)
        volume_mounts = main_container.volumeMounts or []
        for volume_mount in volume_mounts:
            if isinstance(volume_mount, dict) and volume_mount.get("name") == pvc_name:
                volume_mount["mountPath"] = mount_path
                main_container.volumeMounts = volume_mounts
                return

        volume_mounts.append({"name": pvc_name, "mountPath": mount_path})
        main_container.volumeMounts = volume_mounts

    @staticmethod
    def _ensure_component_hf_home_env(component: Component, hf_home: str) -> None:
        main_container = get_main_container(component)
        env_list = main_container.env or []

        env_list[:] = [
            e
            for e in env_list
            if not (isinstance(e, dict) and e.get("name") == "HF_HOME")
        ]
        env_list.append({"name": "HF_HOME", "value": hf_home})
        main_container.env = env_list

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
            container.args = [shlex.join(tokens)]
        else:
            container.args = tokens

    @classmethod
    def _update_frontend_cli(
        cls, cfg: Config, model_name: str, model_path: str
    ) -> None:
        frontend = get_component_by_name(cfg, "Frontend")
        if not frontend:
            return

        main_container = get_main_container(frontend)

        # If operator defaults are being used (no command/args), we must provide full CLI.
        if not main_container.command and not main_container.args:
            main_container.command = ["python3"]
            main_container.args = ["-m", "dynamo.frontend"]

        def _patch(tokens: list[str]) -> list[str]:
            tokens = set_unique_argument_value(tokens, "--model-name", model_name)
            tokens = set_unique_argument_value(tokens, "--model-path", model_path)
            return tokens

        cls._update_container_args_preserving_shell_form(main_container, _patch)

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

        def _patch_component(component: Component) -> None:
            main_container = get_main_container(component)

            def _patch(tokens: list[str]) -> list[str]:
                tokens = set_unique_argument_value(
                    tokens, cls.WORKER_MODEL_PATH_ARG, model_path
                )
                tokens = set_unique_argument_value(
                    tokens, cls.WORKER_SERVED_MODEL_NAME_ARG, model_name
                )
                return tokens

            cls._update_container_args_preserving_shell_form(main_container, _patch)

        # Update workers (prefill + decode) if present.
        patched_components: set[str] = set()
        for sub_component_type in (
            SubComponentType.PREFILL,
            SubComponentType.DECODE,
        ):
            try:
                component_name = get_component_name_by_type(
                    cfg, cls.BACKEND, sub_component_type
                )
            except (KeyError, ValueError):
                continue
            component = get_component_by_name(cfg, component_name)
            if component is None:
                continue
            _patch_component(component)
            patched_components.add(component_name)

        if not patched_components:
            for component in cfg.spec.components:
                if component.component_type == "worker":
                    _patch_component(component)
                    patched_components.add(component.name)

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
        - Add a PVC volume and main-container mount to every component
        - Patch Frontend CLI (`--model-name`, `--model-path`)
        - Delegate worker CLI patching to backend-specific implementation.
        """
        if not pvc_name:
            return config

        cfg = Config.model_validate(config)
        model_path = cls._normalize_model_path(pvc_mount_path, pvc_path)

        for component in cfg.spec.components:
            cls._ensure_component_volume_mount(component, pvc_name, pvc_mount_path)

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
        num_gpus_per_node: int | None = None,
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
            num_gpus_per_node: GPUs per physical node. When provided, worker
                GPU limits are capped per node and multinode.nodeCount is set
                for workers that span multiple nodes.

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
        cfg.metadata.name = f"{cls.BACKEND}-{mode}-{uuid4().hex[:8]}"
        if namespace and hasattr(cfg.metadata, "namespace"):
            cfg.metadata.namespace = namespace

        # Update image for all components
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
                num_gpus_per_node=num_gpus_per_node,
            )
        else:
            cls._apply_agg_worker(
                cfg,
                agg_cli_args=agg_cli_args or [],
                agg_replicas=agg_replicas,
                agg_gpus=agg_gpus,
                num_gpus_per_node=num_gpus_per_node,
            )

        # Update model (handles worker args + frontend patching)
        effective_model_path = model_path or model_name
        if pvc_name and pvc_mount_path and model_path:
            # pvcModelPath was explicitly provided — model weights live at a
            # known path inside the PVC.  Let update_model_from_pvc handle
            # volume mount + CLI patching.
            pvc_path = ""
            if effective_model_path and (
                effective_model_path == pvc_mount_path
                or effective_model_path.startswith(pvc_mount_path + "/")
            ):
                pvc_path = effective_model_path[len(pvc_mount_path) :].strip("/")
            if pvc_path:
                result = cls.update_model_from_pvc(
                    cfg.model_dump(),
                    model_name=model_name,
                    pvc_name=pvc_name,
                    pvc_mount_path=pvc_mount_path,
                    pvc_path=pvc_path,
                )
            else:
                for component in cfg.spec.components:
                    cls._ensure_component_volume_mount(
                        component, pvc_name, pvc_mount_path
                    )
                    cls._ensure_component_hf_home_env(component, pvc_mount_path)
                result = cls.update_model(
                    cfg.model_dump(),
                    model_name=model_name,
                    model_path=effective_model_path,
                )
        elif pvc_name and pvc_mount_path:
            for component in cfg.spec.components:
                cls._ensure_component_volume_mount(component, pvc_name, pvc_mount_path)
                cls._ensure_component_hf_home_env(component, pvc_mount_path)
            result = cls.update_model(
                cfg.model_dump(),
                model_name=model_name,
            )
        else:
            result = cls.update_model(
                cfg.model_dump(),
                model_name=model_name,
                model_path=effective_model_path,
            )

        return cls.finalize_dgd(result, effective_model_path)

    @classmethod
    def _resolve_component_name(
        cls,
        cfg: Config,
        component_type: SubComponentType,
    ) -> str | None:
        """Resolve a worker component name, with an aggregate-mode fallback."""
        try:
            component_name = get_component_name_by_type(
                cfg, cls.BACKEND, component_type
            )
        except (KeyError, ValueError):
            component_name = None
        if component_name and get_component_by_name(cfg, component_name) is not None:
            return component_name
        for component in cfg.spec.components:
            if component.component_type == "worker":
                return component.name
        return None

    @staticmethod
    def _apply_worker_config(
        component: Component,
        cli_args: list[str],
        replicas: int,
        gpus: int,
        num_gpus_per_node: int | None = None,
    ) -> None:
        """Apply CLI args, replicas, and GPU resources to one worker component."""
        component.replicas = replicas
        get_main_container(component).args = sanitize_cli_args(list(cli_args))

        # Apply resources after args so multinode sizing can inspect final TP/PP flags.
        setup_worker_component_resources(component, gpus, num_gpus_per_node)

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
        num_gpus_per_node: int | None = None,
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
            component_name = cls._resolve_component_name(cfg, sct)
            component = (
                get_component_by_name(cfg, component_name) if component_name else None
            )
            if component is None:
                logger.warning(
                    "Could not find %s component for backend %s, skipping",
                    sct.value,
                    cls.BACKEND,
                )
                continue
            cls._apply_worker_config(
                component,
                cli_args,
                replicas,
                gpus,
                num_gpus_per_node=num_gpus_per_node,
            )

    @classmethod
    def _apply_agg_worker(
        cls,
        cfg: Config,
        agg_cli_args: list[str],
        agg_replicas: int,
        agg_gpus: int,
        num_gpus_per_node: int | None = None,
    ) -> None:
        """Apply CLI args, replicas, and GPU resources to the agg worker service.

        In agg mode, the default config template may use a generic worker
        service name (e.g. ``TRTLLMWorker``) that does not match the disagg
        naming convention (``prefill`` / ``decode``).  We first try the standard
        DECODE lookup, then fall back to any non-Frontend/Planner service.
        """
        component_name = cls._resolve_component_name(cfg, SubComponentType.DECODE)
        component = (
            get_component_by_name(cfg, component_name) if component_name else None
        )
        if component is None:
            logger.warning("Could not find worker component for agg mode")
            return
        cls._apply_worker_config(
            component,
            agg_cli_args,
            agg_replicas,
            agg_gpus,
            num_gpus_per_node=num_gpus_per_node,
        )

    @classmethod
    def set_config_replicas(
        cls,
        config: dict,
        replicas: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """Apply the evaluated Candidate's replica count to the worker component.

        Backend-agnostic (component.replicas is a plain Kubernetes field, no
        CLI flag involved), unlike set_config_model/set_config_kv_cache, so
        this lives once here rather than duplicated per backend. Mirrors the
        one relevant line of _apply_worker_config (component.replicas =
        replicas) without touching CLI args or GPU resources, since those
        are already set by set_config_tp_size/tep/dep separately.

        Without this call, a candidate's total materialized GPU footprint
        (replicas x gpus-per-worker) silently diverges from what was
        actually evaluated: setup_worker_component_resources correctly sets
        gpus-per-worker via the tp/tep/dep setters, but replicas is left at
        whatever the base template hardcodes (confirmed: replicas: 1 in
        every agg.yaml template), regardless of the candidate's real
        replicas value. Confirmed in review: a real 4-GPU candidate
        (tp=2, replicas=2) materialized as a 2-GPU DGD (tp=2, replicas=1).
        """
        cfg = Config.model_validate(config)
        component_name = cls._resolve_component_name(cfg, component_type)
        component = (
            get_component_by_name(cfg, component_name) if component_name else None
        )
        if component is None:
            raise ValueError(
                f"could not find worker component for {component_type} to set replicas"
            )
        component.replicas = replicas
        return cfg.model_dump()