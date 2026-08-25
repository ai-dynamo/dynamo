# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import re
from typing import Tuple
from uuid import uuid4

from pydantic import ValidationError

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
    set_unique_argument_value,
    setup_worker_component_resources,
    update_image,
    validate_and_get_worker_args,
)
from dynamo.profiler.utils.config_modifiers.protocol import BaseConfigModifier
from dynamo.profiler.utils.defaults import DYNAMO_RUN_DEFAULT_PORT, EngineType
from dynamo.profiler.utils.dgd_template import load_dgd_template
from dynamo.profiler.utils.model_info import (
    get_mamba_cache_align_block_size,
    get_model_context_length,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", "%Y-%m-%d %H:%M:%S"
)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

DEFAULT_VLLM_KV_TRANSFER_CONFIG = '{"kv_connector":"NixlConnector","kv_role":"kv_both"}'


def _get_valued_arg(args: list[str], key: str) -> str | None:
    value = None
    for i, arg in enumerate(args):
        if arg == key and i + 1 < len(args):
            value = args[i + 1]
        if isinstance(arg, str) and arg.startswith(f"{key}="):
            value = arg.split("=", 1)[1]
    return value


def _set_valued_arg(args: list[str], key: str, value: str) -> list[str]:
    return set_unique_argument_value(args, key, value)


def _parse_human_readable_int(value: str) -> int:
    """Match vLLM's decimal lowercase and binary uppercase size suffixes."""
    value = value.strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?)([kKmMgGtT])", value)
    if match is None:
        return int(value)

    number, suffix = match.groups()
    if suffix.islower():
        multiplier = {
            "k": 10**3,
            "m": 10**6,
            "g": 10**9,
            "t": 10**12,
        }[suffix]
        return int(float(number) * multiplier)

    multiplier = {
        "K": 2**10,
        "M": 2**20,
        "G": 2**30,
        "T": 2**40,
    }[suffix]
    return int(number) * multiplier


def _finalize_disagg_cli_args(args: list[str], role: SubComponentType) -> list[str]:
    """Restore the Dynamo runtime arguments omitted by AIC engine tuning."""
    tokens = break_arguments(args)
    # AIC may still emit the removed vLLM role flags. Normalize them at this
    # ingestion boundary so they never reach the backend CLI.
    cleaned_args = [
        arg
        for arg in tokens
        if arg not in ("--is-prefill-worker", "--is-decode-worker")
    ]
    finalized = set_unique_argument_value(
        cleaned_args, "--disaggregation-mode", role.value
    )
    if (
        role == SubComponentType.PREFILL
        and _get_valued_arg(finalized, "--kv-transfer-config") is None
    ):
        finalized = set_unique_argument_value(
            finalized,
            "--kv-transfer-config",
            DEFAULT_VLLM_KV_TRANSFER_CONFIG,
        )
    return finalized


class VllmV1ConfigModifier(BaseConfigModifier):
    BACKEND = "vllm"
    REMOTE_CODE_TRUST_FLAG = "--trust-remote-code"
    # vllm uses a different arg for model path
    WORKER_MODEL_PATH_ARG = "--model"

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
        super()._apply_disagg_workers(
            cfg,
            prefill_cli_args=_finalize_disagg_cli_args(
                prefill_cli_args, SubComponentType.PREFILL
            ),
            prefill_replicas=prefill_replicas,
            prefill_gpus=prefill_gpus,
            decode_cli_args=_finalize_disagg_cli_args(
                decode_cli_args, SubComponentType.DECODE
            ),
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
        model_name_or_path: str | None = None,
    ) -> dict:
        cfg = Config.model_validate(config)

        # MoE flags (--enable-expert-parallel) are set in set_config_tep_size/set_config_dep_size

        # set metadata name
        cfg.metadata.name = f"vllm-agg-{uuid4().hex[:8]}"

        cfg.spec.components = [
            component
            for component in cfg.spec.components
            if component.component_type != "planner"
        ]

        if target == EngineType.PREFILL:
            prefill_component_name = get_component_name_by_type(
                cfg, "vllm", SubComponentType.PREFILL
            )
            decode_component_name = get_component_name_by_type(
                cfg, "vllm", SubComponentType.DECODE
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
                backend="vllm",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="vllm")
            args = break_arguments(args)

            # Remove role selection when converting the prefill worker to aggregated.
            args = remove_valued_arguments(args, "--disaggregation-mode")
            # AIC may still emit this removed vLLM role flag.
            if "--is-prefill-worker" in args:
                args.remove("--is-prefill-worker")

            # disable prefix caching
            if "--enable-prefix-caching" in args:
                args.remove("--enable-prefix-caching")
            if "--no-enable-prefix-caching" not in args:
                args = append_argument(args, "--no-enable-prefix-caching")

            get_main_container(worker_service).args = args

        elif target == EngineType.DECODE:
            prefill_component_name = get_component_name_by_type(
                cfg, "vllm", SubComponentType.PREFILL
            )
            decode_component_name = get_component_name_by_type(
                cfg, "vllm", SubComponentType.DECODE
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
                backend="vllm",
                sub_component_type=SubComponentType.DECODE,
            )
            args = validate_and_get_worker_args(worker_service, backend="vllm")
            args = break_arguments(args)

            # enable prefix caching
            if "--enable-prefix-caching" not in args:
                args = append_argument(args, "--enable-prefix-caching")
            if "--no-enable-prefix-caching" in args:
                args.remove("--no-enable-prefix-caching")

            args = cls._apply_mamba_cache_align_token_floor(args, model_name_or_path)
            get_main_container(worker_service).args = args

        # set num workers to 1
        # Use the inferred decode service name
        final_decode_component_name = get_component_name_by_type(
            cfg, "vllm", SubComponentType.DECODE
        )
        decode_component = get_component_by_name(cfg, final_decode_component_name)
        if decode_component is None:
            raise ValueError(
                f"Missing decode component {final_decode_component_name!r}"
            )
        decode_component.replicas = 1

        return cfg.model_dump()

    @classmethod
    def _apply_mamba_cache_align_token_floor(
        cls, args: list[str], model_name_or_path: str | None
    ) -> list[str]:
        if not model_name_or_path:
            return args

        mamba_cache_mode = _get_valued_arg(args, "--mamba-cache-mode")
        if mamba_cache_mode:
            mamba_cache_mode = mamba_cache_mode.lower()
        if mamba_cache_mode != "align":
            return args

        try:
            mamba_align_floor = get_mamba_cache_align_block_size(model_name_or_path)
        except (OSError, ValueError, TypeError):
            logger.debug(
                "Could not derive Mamba cache align token floor for %s",
                model_name_or_path,
                exc_info=True,
            )
            return args
        if not mamba_align_floor:
            return args

        current = _get_valued_arg(args, "--max-num-batched-tokens")
        try:
            current_value = int(current) if current is not None else 0
        except ValueError:
            current_value = 0
        if current_value >= mamba_align_floor:
            return args

        logger.info(
            "Raising vLLM --max-num-batched-tokens from %s to Mamba align floor %d",
            current if current is not None else "unset",
            mamba_align_floor,
        )
        return _set_valued_arg(args, "--max-num-batched-tokens", str(mamba_align_floor))

    @classmethod
    def _apply_model_context_window_ceiling(
        cls, args: list[str], model_name_or_path: str | None
    ) -> list[str]:
        """Keep generated ``--max-model-len`` within the model's context window."""
        if not model_name_or_path:
            return args

        current = _get_valued_arg(args, "--max-model-len")
        if current is None:
            return args
        try:
            current_value = _parse_human_readable_int(current)
        except ValueError:
            logger.warning(
                "Cannot validate non-integer vLLM --max-model-len value %r",
                current,
            )
            return args

        try:
            context_length = get_model_context_length(model_name_or_path)
        except (OSError, RuntimeError, TypeError, ValueError):
            logger.warning(
                "Could not derive the model context window for %s; leaving "
                "--max-model-len=%s unchanged.",
                model_name_or_path,
                current,
                exc_info=True,
            )
            return args
        if context_length is None:
            return args

        target_value = min(current_value, context_length)
        if target_value < current_value:
            logger.warning(
                "Clamping vLLM --max-model-len from %d to model context window %d for %s.",
                current_value,
                context_length,
                model_name_or_path,
            )
        return set_unique_argument_value(args, "--max-model-len", str(target_value))

    @classmethod
    def apply_model_runtime_constraints(
        cls, config: dict, model_name_or_path: str | None
    ) -> dict:
        try:
            cfg = Config.model_validate(config)
        except ValidationError:
            logger.debug(
                "Skipping vLLM model runtime constraints for partial config",
                exc_info=True,
            )
            return config
        for component_type in (SubComponentType.PREFILL, SubComponentType.DECODE):
            try:
                worker_service = get_worker_component_from_config(
                    cfg, backend="vllm", sub_component_type=component_type
                )
                main_container = get_main_container(worker_service)
                command = main_container.command or []
                raw_args = main_container.args or []
                if (
                    len(command) >= 2
                    and command[0] in ("/bin/sh", "sh")
                    and command[1] == "-c"
                    and len(raw_args) == 1
                ):
                    logger.debug(
                        "Skipping vLLM model runtime constraints for shell-form component %s",
                        worker_service.name,
                    )
                    continue
                args = validate_and_get_worker_args(worker_service, backend="vllm")
                args = break_arguments(args)
            except (KeyError, ValueError):
                logger.debug(
                    "Skipping vLLM model runtime constraints for partial worker config",
                    exc_info=True,
                )
                continue
            args = cls._apply_model_context_window_ceiling(args, model_name_or_path)
            args = cls._apply_mamba_cache_align_token_floor(args, model_name_or_path)
            main_container.args = args
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
            cfg, backend="vllm", sub_component_type=component_type
        )

        # Set up resources
        setup_worker_component_resources(worker_service, tp_size)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="vllm")
        args = break_arguments(args)

        # Remove --tp alias if present, use --tensor-parallel-size as canonical form
        args = remove_valued_arguments(args, "--tp")
        args = set_argument_value(args, "--tensor-parallel-size", str(tp_size))

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

        Mirrors set_config_tp_size's shape: read the searched Candidate values,
        write the corresponding vLLM flags, and leave everything else in the
        worker's existing args untouched.
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_component_from_config(
            cfg, backend="vllm", sub_component_type=component_type
        )

        args = validate_and_get_worker_args(worker_service, backend="vllm")
        args = break_arguments(args)

        args = set_argument_value(args, "--block-size", str(block_size))
        args = set_argument_value(
            args, "--gpu-memory-utilization", str(memory_fraction)
        )

        # --enable-prefix-caching / --no-enable-prefix-caching are bare
        # boolean flags, not "--key value" pairs, so remove_valued_arguments
        # (which deletes two tokens) does not apply here -- confirmed by a
        # regression test where a trailing boolean flag with no following
        # token silently failed to be removed. Matches the plain list.remove
        # pattern convert_config already uses for this same flag pair.
        if "--enable-prefix-caching" in args:
            args.remove("--enable-prefix-caching")
        if "--no-enable-prefix-caching" in args:
            args.remove("--no-enable-prefix-caching")
        args = append_argument(
            args,
            "--enable-prefix-caching"
            if prefix_caching
            else "--no-enable-prefix-caching",
        )

        get_main_container(worker_service).args = args

        return cfg.model_dump()

    @classmethod
    def set_config_tep_size(
        cls,
        config: dict,
        tep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ):
        """
        Set Tensor Expert Parallelism (TEP) for vLLM MoE models.

        vLLM derives expert parallelism size automatically:
        expert_parallel_size = tensor_parallel_size * data_parallel_size

        For TEP: TP=tep_size, DP=1 → EP size = tep_size
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_component_from_config(
            cfg, backend="vllm", sub_component_type=component_type
        )

        # Set up resources with multinode configuration
        setup_worker_component_resources(worker_service, tep_size, num_gpus_per_node)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="vllm")
        args = break_arguments(args)

        # Remove aliases, use canonical forms
        args = remove_valued_arguments(args, "--tp")
        args = set_argument_value(args, "--tensor-parallel-size", str(tep_size))
        args = remove_valued_arguments(args, "--dp")
        args = set_argument_value(args, "--data-parallel-size", "1")

        # Remove hybrid load balancing flags - not compatible with DP=1
        args = remove_valued_arguments(args, "--data-parallel-size-local")
        if "--data-parallel-hybrid-lb" in args:
            args.remove("--data-parallel-hybrid-lb")

        # Enable expert parallel for MoE
        if "--enable-expert-parallel" not in args:
            args = append_argument(args, "--enable-expert-parallel")

        get_main_container(worker_service).args = args
        return cfg.model_dump()

    @classmethod
    def set_config_dep_size(
        cls,
        config: dict,
        dep_size: int,
        num_gpus_per_node: int,
        component_type: SubComponentType = SubComponentType.DECODE,
    ):
        """
        Set Data Expert Parallelism (DEP) for vLLM MoE models.

        vLLM derives expert parallelism size automatically:
        expert_parallel_size = tensor_parallel_size * data_parallel_size

        For DEP: TP=1, DP=dep_size → EP size = dep_size
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_component_from_config(
            cfg, backend="vllm", sub_component_type=component_type
        )

        # Set up resources with multinode configuration
        setup_worker_component_resources(worker_service, dep_size, num_gpus_per_node)

        # Get and validate args
        args = validate_and_get_worker_args(worker_service, backend="vllm")
        args = break_arguments(args)

        # Remove aliases, use canonical forms
        args = remove_valued_arguments(args, "--tp")
        args = set_argument_value(args, "--tensor-parallel-size", "1")
        args = remove_valued_arguments(args, "--dp")
        args = set_argument_value(args, "--data-parallel-size", str(dep_size))

        # Handle hybrid load balancing for multinode DEP
        # If dep_size > num_gpus_per_node, we need multinode and can use hybrid-lb
        if dep_size > num_gpus_per_node and "--data-parallel-hybrid-lb" in args:
            # Set local DP size to GPUs per node for hybrid load balancing
            args = set_argument_value(
                args, "--data-parallel-size-local", str(num_gpus_per_node)
            )
        else:
            # Remove hybrid-lb flags if not needed or not multinode
            args = remove_valued_arguments(args, "--data-parallel-size-local")
            if "--data-parallel-hybrid-lb" in args:
                args.remove("--data-parallel-hybrid-lb")

        # Enable expert parallel for MoE
        if "--enable-expert-parallel" not in args:
            args = append_argument(args, "--enable-expert-parallel")

        get_main_container(worker_service).args = args
        return cfg.model_dump()

    @classmethod
    def get_model_name(cls, config: dict) -> Tuple[str, str]:
        cfg = Config.model_validate(config)
        worker_service = get_worker_component_from_config(cfg, backend="vllm")
        args = validate_and_get_worker_args(worker_service, backend="vllm")
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
                    if "Maximum concurrency for" in line:
                        try:
                            line = line.strip().split("Maximum concurrency for ")[1]
                            token_count = int(
                                line.split(" tokens per request: ")[0].replace(",", "")
                            )
                            concurrency = float(
                                line.split(" tokens per request: ")[1][:-1]
                            )

                            # Log shows per-rank KV cache; multiply by attention_dp_size for total
                            kv_cache_per_rank = int(token_count * concurrency)
                            total_kv_cache = kv_cache_per_rank * attention_dp_size
                            logger.info(
                                f"Found KV cache: {kv_cache_per_rank} per rank x {attention_dp_size} = {total_kv_cache} total"
                            )
                            return total_kv_cache
                        except Exception as e:
                            logger.warning(
                                f"Failed to parse KV cache size from line: {line}. Error: {e}"
                            )
        except FileNotFoundError:
            logger.warning(f"Log file not found: {dynamo_log_fn}")
        except Exception as e:
            logger.warning(f"Failed to read log file {dynamo_log_fn}: {e}")
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
        vLLM uses --max-num-seqs to limit concurrency and
        --max-num-batched-tokens to cap total tokens per step.

        In vLLM, --max-num-batched-tokens controls per-GPU buffer allocation
        during memory profiling. For DEP (DP > 1), we must use the base token
        limit per GPU, not the multiplied total, to avoid OOM during profiling.
        """
        cfg = Config.model_validate(config)
        worker_service = get_worker_component_from_config(
            cfg, backend="vllm", sub_component_type=component_type
        )
        args = validate_and_get_worker_args(worker_service, backend="vllm")
        args = break_arguments(args)

        # Get DP size from args (check both --dp and --data-parallel-size aliases)
        dp_size = 1
        for i, arg in enumerate(args):
            if arg in ("--dp", "--data-parallel-size") and i + 1 < len(args):
                dp_size = int(args[i + 1])
                break

        # For DEP (DP > 1), compute per-GPU token limit to avoid OOM
        per_gpu_max_tokens = (
            max_num_tokens // dp_size if dp_size > 1 else max_num_tokens
        )

        args = set_argument_value(args, "--max-num-seqs", str(max_batch_size))
        args = set_argument_value(
            args, "--max-num-batched-tokens", str(per_gpu_max_tokens)
        )

        get_main_container(worker_service).args = args
        return cfg.model_dump()
