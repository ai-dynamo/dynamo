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

from typing import Any, Protocol

from benchmarks.profiler.utils.config import (
    Config,
    Container,
    PodSpec,
    break_arguments,
    get_service_name_by_type,
    set_argument_value,
)
from benchmarks.profiler.utils.defaults import EngineType
from dynamo.planner.defaults import SubComponentType


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
    def get_model_name(cls, config: dict) -> str:
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
    def load_default_config(cls) -> dict:
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
        # Update workers (prefill + decode) if present.
        for sct in (SubComponentType.PREFILL, SubComponentType.DECODE):
            try:
                svc_name = get_service_name_by_type(cfg, cls.BACKEND, sct)
            except Exception:
                continue
            if svc_name not in cfg.spec.services:
                continue

            service = cfg.spec.services[svc_name]
            if not service.extraPodSpec or not service.extraPodSpec.mainContainer:
                continue

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
