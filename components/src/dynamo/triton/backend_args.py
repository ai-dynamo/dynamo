# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dynamo Triton wrapper configuration ArgGroup."""

import argparse
import os
from typing import Callable, Optional

from tritonserver import LogFormat as TritonLogFormat
from tritonserver import RateLimitMode as TritonRateLimitMode

from dynamo.common.configuration.utils import add_argument
from dynamo.triton.args import Config, DynamoArgGroup


def _enum_arg(enum_cls, flag: str) -> Callable[[str], object]:
    """Build an argparse ``type`` that maps a member name onto a triton_runtime enum."""

    def convert(value: str) -> object:
        try:
            return enum_cls.__members__[value]
        except KeyError:
            choices = ", ".join(enum_cls.__members__)
            raise argparse.ArgumentTypeError(
                f"invalid {flag} {value!r} (choose from {choices})"
            )

    return convert


def _bool_arg(value: str) -> bool:
    """Parse a boolean CLI value the way the triton_runtime CLI does: a required
    argument that is case-insensitive true/on/1 or false/off/0 (Option::ArgBool)."""
    normalized = value.strip().lower()
    if normalized in ("true", "on", "1"):
        return True
    if normalized in ("false", "off", "0"):
        return False

    raise argparse.ArgumentTypeError(
        f"invalid boolean value {value!r} (choose from true/false, on/off, 1/0)"
    )


class ThreeArgsDictAction(argparse.Action):
    """Merge repeated '<name>,<setting>=<value>' flags into the
    {name: {setting: value}} form triton_runtime.Options expects
    (backend_configuration, host_policies, metrics_configuration, cache_config).
    """

    def __call__(self, parser, namespace, value, option_string=None):
        name, comma, rest = value.partition(",")
        setting, eq, setting_value = rest.partition("=")
        if not (comma and eq) or not name or not setting or not setting_value:
            raise argparse.ArgumentError(
                self, f"expected '<name>,<setting>=<value>', got {value!r}"
            )

        merged = getattr(namespace, self.dest, None) or {}
        if setting in merged.get(name, {}):
            raise argparse.ArgumentError(
                self,
                f"duplicate setting '{name},{setting}' "
                f"(already set to {merged[name][setting]!r})",
            )

        merged.setdefault(name, {})[setting] = setting_value
        setattr(namespace, self.dest, merged)


class TwoArgsDictAction(argparse.Action):
    """Merge repeated '<key>:<value>' flags into the {int: int} form
    triton_runtime.Options.cuda_memory_pool_sizes expects.
    """

    def __call__(self, parser, namespace, value, option_string=None):
        key_str, colon, value_str = value.partition(":")
        if not colon:
            raise argparse.ArgumentError(
                self, f"expected '<key>:<value>', got {value!r}"
            )

        try:
            key, int_value = int(key_str), int(value_str)
        except ValueError:
            raise argparse.ArgumentError(
                self, f"expected integer '<key>:<value>', got {value!r}"
            )

        if key < 0 or int_value < 0:
            raise argparse.ArgumentError(
                self, f"key and value must be non-negative, got {value!r}"
            )

        merged = getattr(namespace, self.dest, None) or {}
        if key in merged:
            raise argparse.ArgumentError(
                self, f"duplicate key {key} (already set to {merged[key]})"
            )

        merged[key] = int_value
        setattr(namespace, self.dest, merged)


class DynamoTritonArgGroup(DynamoArgGroup):
    """Triton Runtime configuration options."""

    name = "dynamo-triton"

    @staticmethod
    def add_arguments(parser: argparse.ArgumentParser) -> None:
        """Add Dynamo Triton arguments to parser."""
        if not isinstance(parser, argparse.ArgumentParser):
            raise TypeError("parser must be an instance of argparse.ArgumentParser")

        DynamoArgGroup.add_arguments(parser)

        # -- Triton Runtime Model Options --------------------
        repo_group = parser.add_argument_group("Triton Runtime Model Options")
        add_argument(
            repo_group,
            flag_name="--model-repository",
            env_var="DYN_TRITON_MODEL_REPOSITORY",
            default="/models",
            help="Model repository directory. "
            "Required (validated below); the worker is model-agnostic and does not assume a bundled repo.",
        )
        add_argument(
            repo_group,
            flag_name="--backend-directory",
            env_var="DYN_TRITON_BACKEND_DIRECTORY",
            default=None,
            help="Triton backend directory ; unset defers to the default (/opt/tritonserver/backends). "
            "A provided path must exist.",
        )
        add_argument(
            repo_group,
            flag_name="--repoagent-directory",
            env_var="DYN_TRITON_REPO_AGENT_DIRECTORY",
            default=None,
            dest="repo_agent_directory",
            help="Repository agent directory.",
        )
        add_argument(
            repo_group,
            flag_name="--strict-model-config",
            env_var="DYN_TRITON_STRICT_MODEL_CONFIG",
            default=None,
            arg_type=_bool_arg,
            help="Enable strict model configuration.",
        )
        add_argument(
            repo_group,
            flag_name="--backend-config",
            env_var="DYN_TRITON_BACKEND_CONFIG",
            default=None,
            dest="backend_configuration",
            action=ThreeArgsDictAction,
            help="Backend configuration, e.g. 'tensorrt,plugins=/a.so;/b.so'. "
            "May be repeated.",
        )

        # -- Triton Runtime Resource & Lifecycle Options --------------------------
        res_group = parser.add_argument_group(
            "Triton Runtime Resource & Lifecycle Options"
        )
        add_argument(
            res_group,
            flag_name="--id",
            env_var="DYN_TRITON_RUNTIME_ID",
            default="triton",
            dest="server_id",
            help="Textual Identifier.",
        )
        add_argument(
            res_group,
            flag_name="--exit-on-error",
            env_var="DYN_TRITON_EXIT_ON_ERROR",
            default=None,
            arg_type=_bool_arg,
            help="Exit on initialization error.",
        )
        add_argument(
            res_group,
            flag_name="--strict-readiness",
            env_var="DYN_TRITON_STRICT_READINESS",
            default=None,
            arg_type=_bool_arg,
            help="Enable strict readiness handling.",
        )
        add_argument(
            res_group,
            flag_name="--exit-timeout-secs",
            env_var="DYN_TRITON_EXIT_TIMEOUT",
            default=None,
            dest="exit_timeout",
            arg_type=int,
            help="Exit timeout in seconds.",
        )
        add_argument(
            res_group,
            flag_name="--buffer-manager-thread-count",
            env_var="DYN_TRITON_BUFFER_MANAGER_THREAD_COUNT",
            default=None,
            arg_type=int,
            help="Buffer manager thread count.",
        )
        add_argument(
            res_group,
            flag_name="--model-load-thread-count",
            env_var="DYN_TRITON_MODEL_LOAD_THREAD_COUNT",
            default=None,
            arg_type=int,
            help="Concurrent model load thread count.",
        )
        add_argument(
            res_group,
            flag_name="--model-load-retry-count",
            env_var="DYN_TRITON_MODEL_LOAD_RETRY_COUNT",
            default=None,
            arg_type=int,
            help="Model load retry count.",
        )
        add_argument(
            res_group,
            flag_name="--model-namespacing",
            env_var="DYN_TRITON_MODEL_NAMESPACING",
            default=None,
            arg_type=_bool_arg,
            help="Enable model namespacing.",
        )
        add_argument(
            res_group,
            flag_name="--enable-peer-access",
            env_var="DYN_TRITON_ENABLE_PEER_ACCESS",
            default=None,
            arg_type=_bool_arg,
            help="Enable GPU peer access.",
        )
        add_argument(
            res_group,
            flag_name="--pinned-memory-pool-byte-size",
            env_var="DYN_TRITON_PINNED_MEMORY_POOL_SIZE",
            default=None,
            dest="pinned_memory_pool_size",
            arg_type=int,
            help="Pinned memory pool size in bytes.",
        )
        add_argument(
            res_group,
            flag_name="--cuda-memory-pool-byte-size",
            env_var="DYN_TRITON_CUDA_MEMORY_POOL_SIZE",
            default=None,
            dest="cuda_memory_pool_sizes",
            action=TwoArgsDictAction,
            help="Per-device CUDA memory pool size as '<device>:<bytes>'. May be repeated.",
        )
        add_argument(
            res_group,
            flag_name="--min-supported-compute-capability",
            env_var="DYN_TRITON_MIN_SUPPORTED_COMPUTE_CAPABILITY",
            default=None,
            arg_type=float,
            help="Minimum required CUDA compute capability.",
        )
        add_argument(
            res_group,
            flag_name="--rate-limit",
            env_var="DYN_TRITON_RATE_LIMITER_MODE",
            default=None,
            dest="rate_limiter_mode",
            arg_type=_enum_arg(TritonRateLimitMode, "--rate-limit"),
            help="Rate limiter mode.",
        )

        # -- Triton Runtime Logging & Metrics Options ------------------------------
        log_group = parser.add_argument_group(
            "Triton Runtime Logging & Metrics Options"
        )
        add_argument(
            log_group,
            flag_name="--log-verbose",
            env_var="DYN_TRITON_LOG_VERBOSE",
            default=None,
            arg_type=int,
            help="Verbose logging level; 0 disables.",
        )
        add_argument(
            log_group,
            flag_name="--log-file",
            env_var="DYN_TRITON_LOG_FILE",
            default=None,
            help="Log file path; logs go to stdout if unset.",
        )
        add_argument(
            log_group,
            flag_name="--log-info",
            env_var="DYN_TRITON_LOG_INFO",
            default=True,
            arg_type=_bool_arg,
            help="Enable INFO logging.",
        )
        add_argument(
            log_group,
            flag_name="--log-warning",
            env_var="DYN_TRITON_LOG_WARN",
            default=True,
            dest="log_warn",
            arg_type=_bool_arg,
            help="Enable WARNING logging.",
        )
        add_argument(
            log_group,
            flag_name="--log-error",
            env_var="DYN_TRITON_LOG_ERROR",
            default=True,
            arg_type=_bool_arg,
            help="Enable ERROR logging.",
        )
        add_argument(
            log_group,
            flag_name="--log-format",
            env_var="DYN_TRITON_LOG_FORMAT",
            default=None,
            arg_type=_enum_arg(TritonLogFormat, "--log-format"),
            help="Log message format.",
        )
        add_argument(
            log_group,
            flag_name="--allow-metrics",
            env_var="DYN_TRITON_METRICS",
            default=True,
            dest="metrics",
            arg_type=_bool_arg,
            help="Enable metric collection.",
        )
        add_argument(
            log_group,
            flag_name="--allow-gpu-metrics",
            env_var="DYN_TRITON_GPU_METRICS",
            default=None,
            dest="gpu_metrics",
            arg_type=_bool_arg,
            help="Enable GPU metric collection.",
        )
        add_argument(
            log_group,
            flag_name="--allow-cpu-metrics",
            env_var="DYN_TRITON_CPU_METRICS",
            default=None,
            dest="cpu_metrics",
            arg_type=_bool_arg,
            help="Enable CPU metric collection.",
        )
        add_argument(
            log_group,
            flag_name="--metrics-interval-ms",
            env_var="DYN_TRITON_METRICS_INTERVAL",
            default=None,
            dest="metrics_interval",
            arg_type=int,
            help="Metric collection interval in ms.",
        )
        add_argument(
            log_group,
            flag_name="--metrics-config",
            env_var="DYN_TRITON_METRICS_CONFIG",
            default=None,
            dest="metrics_configuration",
            action=ThreeArgsDictAction,
            help="Metrics configuration as '<name>,<setting>=<value>'. "
            "May be repeated.",
        )

        # -- Triton Runtime Additional Options -----------------------------
        cache_group = parser.add_argument_group("Triton Runtime Additional Options")
        add_argument(
            cache_group,
            flag_name="--cache-config",
            env_var="DYN_TRITON_CACHE_CONFIG",
            default=None,
            action=ThreeArgsDictAction,
            help="Response cache configuration as '<name>,<setting>=<value>'. "
            "May be repeated.",
        )
        add_argument(
            cache_group,
            flag_name="--cache-directory",
            env_var="DYN_TRITON_CACHE_DIRECTORY",
            default=None,
            help="Cache provider directory.",
        )
        add_argument(
            cache_group,
            flag_name="--host-policy",
            env_var="DYN_TRITON_HOST_POLICY",
            default=None,
            dest="host_policies",
            action=ThreeArgsDictAction,
            help="Host policy as '<name>,<setting>=<value>'. May be repeated.",
        )


class DynamoTritonConfig(Config):
    """Configuration for Dynamo Triton Runtime specific options."""

    model_repository: str
    backend_directory: Optional[str]
    repo_agent_directory: Optional[str]
    strict_model_config: Optional[bool]
    backend_configuration: Optional[dict]

    server_id: Optional[str]
    exit_on_error: Optional[bool]
    strict_readiness: Optional[bool]
    exit_timeout: Optional[int]
    buffer_manager_thread_count: Optional[int]
    model_load_thread_count: Optional[int]
    model_load_retry_count: Optional[int]
    model_namespacing: Optional[bool]
    enable_peer_access: Optional[bool]
    pinned_memory_pool_size: Optional[int]
    cuda_memory_pool_sizes: Optional[dict]
    min_supported_compute_capability: Optional[float]
    rate_limiter_mode: Optional[TritonRateLimitMode]

    log_verbose: Optional[int]
    log_file: Optional[str]
    log_info: Optional[bool]
    log_warn: Optional[bool]
    log_error: Optional[bool]
    log_format: Optional[TritonLogFormat]
    metrics: Optional[bool]
    gpu_metrics: Optional[bool]
    cpu_metrics: Optional[bool]
    metrics_interval: Optional[int]
    metrics_configuration: Optional[dict]

    cache_config: Optional[dict]
    cache_directory: Optional[str]
    host_policies: Optional[dict]

    def validate(self) -> None:
        if hasattr(super(), "validate"):
            super().validate()

        if not self.model_repository:
            raise ValueError(
                "--model-repository is required (or set DYN_TRITON_MODEL_REPOSITORY); "
                "the worker is model-agnostic and does not assume a bundled repo."
            )

        if self.backend_directory is not None and not os.path.isdir(
            self.backend_directory
        ):
            raise ValueError(
                f"--backend-directory '{self.backend_directory}' does not exist or is "
                "not a directory."
            )

        # GPU/CPU metrics are only collected when metrics are enabled, so mask
        # them off when --allow-metrics is explicitly disabled. Mirrors Triton's
        # CLI (allow_gpu_metrics &= allow_metrics). Unset (None) values are left
        # alone so to_server_options keeps deferring to the binding defaults.
        if self.metrics is False:
            if self.gpu_metrics:
                self.gpu_metrics = False
            if self.cpu_metrics:
                self.cpu_metrics = False

    def to_server_options(self) -> dict:
        """Render the triton_runtime.Server/Options kwargs this config maps to,
        mirroring the Triton Python binding's Options fields
        (triton_runtime._api._server.Options).
        """
        opts: dict = {
            name: getattr(self, name) for name in DynamoTritonConfig.__annotations__
        }
        # Drop unset (None) ones for the binding's own defaults to apply.
        return {key: value for key, value in opts.items() if value is not None}


def parse_args(argv: Optional[list[str]] = None) -> Config:
    """Parse command-line arguments for the Dynamo Triton Runtime.

    Args:
        argv: Command-line arguments.

    Returns:
        Config: Parsed configuration object.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Triton Runtime for Dynamo. Triton Runtime options mirror "
            "Triton Options; unset options use the binding's defaults."
        )
    )

    DynamoTritonArgGroup.add_arguments(parser)
    config = DynamoTritonConfig.from_cli_args(parser.parse_args(argv))
    config.validate()
    return config
