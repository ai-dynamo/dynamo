# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from typing import Tuple
from uuid import uuid4

from dynamo.planner.config.defaults import SubComponentType
from dynamo.profiler.utils.config import (
    Config,
    append_argument,
    break_arguments,
    find_main_container,
    get_component_by_name,
    get_component_name_by_type,
    get_main_container,
    get_worker_component_from_config,
    remove_valued_arguments,
    set_argument_value,
    setup_worker_component_resources,
    update_image,
    validate_and_get_worker_args,
)
from dynamo.profiler.utils.config_modifiers.protocol import BaseConfigModifier
from dynamo.profiler.utils.defaults import DYNAMO_RUN_DEFAULT_PORT, EngineType
from dynamo.profiler.utils.dgd_template import load_dgd_template

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def _arg_value_any(args: list[str], keys: tuple[str, ...]) -> str | None:
    for idx in range(len(args) - 1, -1, -1):
        arg = args[idx]
        for key in keys:
            if arg.startswith(f"{key}="):
                return arg.split("=", 1)[1]
            if arg == key:
                return args[idx + 1] if idx + 1 < len(args) else None
    return None


def _arg_value(args: list[str], key: str) -> str | None:
    return _arg_value_any(args, (key,))


def _int_arg_value(args: list[str], keys: tuple[str, ...], default: int) -> int:
    try:
        return int(_arg_value_any(args, keys) or "")
    except ValueError:
        return default


def _has_arg(args: list[str], key: str) -> bool:
    return any(arg == key or arg.startswith(f"{key}=") for arg in args)


def _ensure_safe_prefill_cuda_graph_bs(args: list[str]) -> list[str]:
    args = list(args)
    parsed_args = break_arguments(args)
    if _has_arg(parsed_args, "--cuda-graph-bs") or _has_arg(
        parsed_args, "--disable-cuda-graph"
    ):
        return args

    max_running_requests = _arg_value(parsed_args, "--max-running-requests")
    try:
        max_running_requests_int = int(max_running_requests or "")
    except ValueError:
        return args

    if max_running_requests_int == 1:
        dp_size = _int_arg_value(
            parsed_args,
            ("--data-parallel-size", "--dp-size", "--dp"),
            default=1,
        )
        safe_bs = max(1, dp_size)
        args = append_argument(args, ["--cuda-graph-bs", str(safe_bs)])
    return args


class SGLangConfigModifier(BaseConfigModifier):
    BACKEND = "sglang"
    REMOTE_CODE_TRUST_FLAG = "--trust-remote-code"

    @classmethod
    def load_default_config(cls, mode: str = "disagg") -> dict:
        return load_dgd_template(cls.BACKEND, mode)

    @classmethod
    def update_image(cls, config, image: str) -> dict:
        """Update container image for all DGD components."""
        return update_image(config, image)

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
        prefill_cli_args = _ensure_safe_prefill_cuda_graph_bs(prefill_cli_args)
        super()._apply_disagg_workers(
            cfg,
            prefill_cli_args=prefill_cli_args,
            prefill_replicas=prefill_replicas,
            prefill_gpus=prefill_gpus,
            decode_cli_args=decode_cli_args,
            decode_replicas=decode_replicas,
            decode_gpus=decode_gpus,
            num_gpus_per_node=num_gpus_per_node,
        )

    @classmethod
    def convert_config(
        cls,
        config: dict,
        target: EngineType,
        is_moe_model: bool = False,
    ) -> dict:
        cfg = Config.model_validate(config)

        # set metadata name
        cfg.metadata.name = f"sglang-agg-{uuid4().hex[:8]}"

        cfg.spec.components = [
            component
            for component in cfg.spec.components
            if component.component_type != "planner"
        ]

        if target == EngineType.PREFILL:
            prefill_component_name = get_component_name_by_type(
                cfg, "sglang", SubComponentType.PREFILL
            )
            decode_component_name = get_component_name_by_type(
                cfg, "sglang", SubComponentType.DECODE
            )

            prefill_component = get_component_by_name(cfg, prefill_component_name)
            if prefill_component is None:
                raise ValueError(
                    f"Missing prefill component {prefill_component_name!r}"
                )
            if prefill_component_name != decode_component_name:
                cfg.spec.components = [
                    component
                    for component in cfg.spec.components
                    if component.name != decode_component_name
                ]
            prefill_component.name = decode_component_name
            prefill_component.component_type = "decode"

            worker_service = get_worker_component_from_config(
                cfg,
                backend="sglang",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="sglang")
            args = break_arguments(args)

            # remove disagg flags
            args = remove_valued_arguments(args, "--disaggregation-mode")
            args = remove_valued_arguments(args, "--disaggregation-transfer-backend")
            args = remove_valued_arguments(args, "--disaggregation-bootstrap-port")

            # disable prefix caching
            if "--disable-radix-cache" not in args:
                args = append_argument(args, "--disable-radix-cache")

            get_main_container(worker_service).args = args

        elif target == EngineType.DECODE:
            prefill_component_name = get_component_name_by_type(
                cfg, "sglang", SubComponentType.PREFILL
            )
            decode_component_name = get_component_name_by_type(
                cfg, "sglang", SubComponentType.DECODE
            )

            if prefill_component_name != decode_component_name:
                cfg.spec.components = [
                    component
                    for component in cfg.spec.components
                    if component.name != prefill_component_name
                ]
            decode_component = get_component_by_name(cfg, decode_component_name)
            if decode_component is None:
                raise ValueError(f"Missing decode component {decode_component_name!r}")
            decode_component.component_type = "decode"

            worker_service = get_worker_component_from_config(
                cfg,
                backend="sglang",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="sglang")
            args = break_arguments(args)

            # remove disagg flags
            args = remove_valued_arguments(args, "--disaggregation-mode")
            args = remove_valued_arguments(args, "--disaggregation-transfer-backend")
            args = remove_valued_arguments(args, "--disaggregation-bootstrap-port")

            # enable prefix caching
            if "--disable-radix-cache" in args:
                args.remove("--disable-radix-cache")

            if is_moe_model:
                # need to use round_robin dp attention routing for MoE models to ensure kv reuse can skip prefill
                if "--load-balance-method" in args:
                    idx = args.index("--load-balance-method")
                    args[idx + 1] = "round_robin"
                else:
                    args = append_argument(
                        args, ["--load-balance-method", "round_robin"]
                    )

            get_main_container(worker_service).args = args

        # set num workers to 1
        # Use the inferred decode service name
        final_decode_component_name = get_component_name_by_type(
            cfg, "sglang", SubComponentType.DECODE
        )
        decode_component = get_component_by_name(cfg, final_decode_component_name)
        if decode_component is None:
            raise ValueError(
                f"Missing decode component {final_decode_component_name!r}"
            )
        decode_component.replicas = 1

        return cfg.model_dump()

    @classmethod
    def set_config_tp_size(
        cls,
        config: dict,
        tp_size: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        cfg = Config.model_validate(config)
        worker_service = get_worker_component_from_config(
            cfg, backend="sglang", sub_component_type=component_type
        )

        # Set up resources
        setup_worker_component_resources(worker_service, tp_size)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="sglang")

        # Set --tp argument
        args = set_argument_value(args, "--tp", str(tp_size))
        args = remove_valued_arguments(args, "--tp-size")
        args = remove_valued_arguments(args, "--tensor-parallel-size")

        # Remove --ep if present
        args = remove_valued_arguments(args, "--ep")
        args = remove_valued_arguments(args, "--ep-size")
        args = remove_valued_arguments(args, "--expert-parallel-size")

        # remove --dp if present
        args = remove_valued_arguments(args, "--dp")
        args = remove_valued_arguments(args, "--dp-size")
        args = remove_valued_arguments(args, "--data-parallel-size")

        get_main_container(worker_service).args = args
        return cfg.model_dump()

    @classmethod
    def set_config_tep_size(
        cls,
        config: dict,
        tep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        cfg = Config.model_validate(config)
        worker_service = get_worker_component_from_config(
            cfg, backend="sglang", sub_component_type=component_type
        )

        # Set up resources with multinode configuration
        setup_worker_component_resources(worker_service, tep_size, num_gpus_per_node)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="sglang")

        # 1. Set --tp=tep_size, if not present add it
        args = set_argument_value(args, "--tp", str(tep_size))
        args = remove_valued_arguments(args, "--tp-size")
        args = remove_valued_arguments(args, "--tensor-parallel-size")

        # 2. Set --ep=tep_size, if not present add it
        args = set_argument_value(args, "--ep", str(tep_size))
        args = remove_valued_arguments(args, "--ep-size")
        args = remove_valued_arguments(args, "--expert-parallel-size")

        # 3. Remove --dp if present
        args = remove_valued_arguments(args, "--dp")
        args = remove_valued_arguments(args, "--dp-size")
        args = remove_valued_arguments(args, "--data-parallel-size")

        # 4. Remove --enable-dp-attention if present
        if "--enable-dp-attention" in args:
            args.remove("--enable-dp-attention")

        get_main_container(worker_service).args = args
        return cfg.model_dump()

    @classmethod
    def set_config_dep_size(
        cls,
        config: dict,
        dep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        cfg = Config.model_validate(config)
        worker_service = get_worker_component_from_config(
            cfg, backend="sglang", sub_component_type=component_type
        )

        # Set up resources with multinode configuration
        setup_worker_component_resources(worker_service, dep_size, num_gpus_per_node)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="sglang")

        # 1. Set --tp=dep_size
        args = set_argument_value(args, "--tp", str(dep_size))
        args = remove_valued_arguments(args, "--tp-size")
        args = remove_valued_arguments(args, "--tensor-parallel-size")

        # 2. Set --dp=dep_size (data parallelism across experts)
        args = set_argument_value(args, "--dp", str(dep_size))
        args = remove_valued_arguments(args, "--dp-size")
        args = remove_valued_arguments(args, "--data-parallel-size")

        # 3. Enable --enable-dp-attention
        if "--enable-dp-attention" not in args:
            args = append_argument(args, "--enable-dp-attention")

        # 4. Set --ep=dep_size (expert parallelism size)
        args = set_argument_value(args, "--ep", str(dep_size))
        args = remove_valued_arguments(args, "--ep-size")
        args = remove_valued_arguments(args, "--expert-parallel-size")

        get_main_container(worker_service).args = args
        return cfg.model_dump()

    @classmethod
    def get_model_name(cls, config: dict) -> Tuple[str, str]:
        cfg = Config.model_validate(config)
        worker_service = get_worker_component_from_config(cfg, backend="sglang")
        args = validate_and_get_worker_args(worker_service, backend="sglang")
        args = break_arguments(args)
        return cls._get_model_name_and_path_from_args(args)

    @classmethod
    def get_port(cls, config: dict) -> int:
        cfg = Config.model_validate(config)
        frontend_component = get_component_by_name(cfg, "Frontend")
        if frontend_component is None:
            logger.warning(
                "Frontend component not found, using default port: %s",
                DYNAMO_RUN_DEFAULT_PORT,
            )
            return DYNAMO_RUN_DEFAULT_PORT

        main_container = find_main_container(frontend_component)
        if main_container is None:
            logger.warning(
                "Frontend main container not found, using default port: %s",
                DYNAMO_RUN_DEFAULT_PORT,
            )
            return DYNAMO_RUN_DEFAULT_PORT

        args = main_container.args
        if not args:
            logger.warning(
                "No args found in Frontend configuration, using default port: %s",
                DYNAMO_RUN_DEFAULT_PORT,
            )
            return DYNAMO_RUN_DEFAULT_PORT

        args = break_arguments(args)
        try:
            idx = args.index("--http-port")
            return int(args[idx + 1])
        except (ValueError, IndexError):
            logger.warning(
                "Port not found in configuration args, using default port: %s",
                DYNAMO_RUN_DEFAULT_PORT,
            )
            return DYNAMO_RUN_DEFAULT_PORT

    @classmethod
    def get_kv_cache_size_from_dynamo_log(
        cls, dynamo_log_fn: str, attention_dp_size: int = 1
    ) -> int:
        try:
            with open(dynamo_log_fn, "r") as f:
                for line in f:
                    if "KV Cache is allocated" in line and "#tokens:" in line:
                        # Extract the number after "#tokens:"
                        match = re.search(r"#tokens:\s*(\d+)", line)
                        if match:
                            return int(match.group(1)) * attention_dp_size
        except Exception as e:
            logger.warning(f"Failed to parse KV cache size from log file. Error: {e}")
        return 0

    @classmethod
    def set_prefill_config(
        cls,
        config: dict,
        max_batch_size: int,
        max_num_tokens: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """
        Configure prefill-related limits for aggregated prefill runs.
        - Batch size is applied as server concurrency.
        - Max tokens is applied as a total token cap to avoid chunked prefill.
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_component_from_config(
            cfg, backend="sglang", sub_component_type=component_type
        )
        args = validate_and_get_worker_args(worker_service, backend="sglang")
        args = break_arguments(args)

        # Set max concurrency to control effective batch size
        args = set_argument_value(args, "--max-running-requests", str(max_batch_size))
        args = _ensure_safe_prefill_cuda_graph_bs(args)

        # Cap total tokens processed in a batch to avoid chunked prefill
        args = set_argument_value(args, "--chunked-prefill-size", str(max_num_tokens))

        args = append_argument(args, "--enable-dp-lm-head")

        get_main_container(worker_service).args = args
        return cfg.model_dump()

    @classmethod
    def set_config_kv_cache(
        cls,
        config: dict,
        block_size: int,
        memory_fraction: float,
        prefix_caching: bool,
        component_type: SubComponentType = SubComponentType.DECODE,
    ) -> dict:
        """Apply KV-cache block size, memory budget, and prefix-caching policy.

        SGLang naming/polarity differs from vLLM, confirmed against the real
        base template (--page-size) and real deploy examples/launch scripts
        (--mem-fraction-static, present at examples/backends/sglang/deploy and
        launch/_test_agg.sh). Prefix caching (SGLang: radix cache) defaults ON
        and is a single-flag toggle, not a paired enable/disable flag like
        vLLM -- presence of --disable-radix-cache means OFF, its absence means
        the (default) ON state. Do not copy vLLM's flag names or polarity here.
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_component_from_config(
            cfg, backend="sglang", sub_component_type=component_type
        )
        args = validate_and_get_worker_args(worker_service, backend="sglang")
        args = break_arguments(args)

        args = set_argument_value(args, "--page-size", str(block_size))
        args = set_argument_value(args, "--mem-fraction-static", str(memory_fraction))

        if "--disable-radix-cache" in args:
            args.remove("--disable-radix-cache")
        if not prefix_caching:
            args = append_argument(args, "--disable-radix-cache")

        get_main_container(worker_service).args = args
        return cfg.model_dump()
