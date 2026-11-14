# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import pathlib
from collections.abc import Mapping
from typing import Any, Dict, Optional

from dynamo.common.config_dump import register_encoder
from dynamo.common.configuration.arg_group import ArgGroup
from dynamo.common.configuration.groups.ais_perf_args import (
    AisPerfArgGroup,
    AisPerfConfigBase,
)
from dynamo.common.configuration.groups.kv_router_args import (
    CONDITIONAL_DISAGG_POLICY_CHOICES,
    KvRouterArgGroup,
    KvRouterConfigBase,
)
from dynamo.common.configuration.groups.router_args import (
    RouterArgGroup,
    RouterConfigBase,
)
from dynamo.common.configuration.utils import (
    ArgEnv,
    add_argument,
    add_negatable_bool_argument,
    env_or_default,
    parse_bool,
)

from . import __version__

_U32_MAX = 2**32 - 1
_MAX_SESSION_AFFINITY_TTL_SECS = 31_536_000


# --dyn-chat-processor values Dynamo implements; any other value names a custom
# chat processor, resolved by dynamo.frontend.main.
BUILTIN_CHAT_PROCESSORS = ("dynamo", "vllm", "sglang")


def _parse_chat_template_kwargs(value: str) -> Dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--default-chat-template-kwargs must be a JSON object")
    return parsed


def _parse_chat_processor_config(value: str) -> Dict[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("--dyn-chat-processor-config must be a JSON object")
    return parsed


def _mountinfo_path(value: str) -> str:
    for escape, character in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        value = value.replace(escape, character)
    return value


def _cpu_quota_count() -> Optional[int]:
    memberships: dict[str, pathlib.PurePosixPath] = {}
    try:
        for line in pathlib.Path("/proc/self/cgroup").read_text().splitlines():
            parts = line.split(":", 2)
            if len(parts) != 3 or not parts[2].startswith("/"):
                continue
            if parts[0] == "0" and not parts[1]:
                memberships["v2"] = pathlib.PurePosixPath(parts[2])
            elif "cpu" in parts[1].split(","):
                memberships["v1"] = pathlib.PurePosixPath(parts[2])

        mount_lines = pathlib.Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        mount_lines = []

    ambiguous_mount = False
    for line in mount_lines:
        if " - " not in line:
            continue
        mount, filesystem = line.split(" - ", 1)
        fields = mount.split()
        fs_fields = filesystem.split()
        if len(fields) < 5 or len(fs_fields) < 3:
            continue
        if fs_fields[0] == "cgroup2":
            version = "v2"
        elif fs_fields[0] == "cgroup" and "cpu" in fs_fields[2].split(","):
            version = "v1"
        else:
            continue
        membership = memberships.get(version)
        if membership is None:
            continue

        mount_root = pathlib.PurePosixPath(_mountinfo_path(fields[3]))
        if (
            ".." in mount_root.parts
            or ".." in membership.parts
            or not membership.is_relative_to(mount_root)
        ):
            # An inherited mount may not expose the process's cgroup.
            ambiguous_mount = True
            continue
        relative = membership.relative_to(mount_root)
        root = pathlib.Path(_mountinfo_path(fields[4]))
        found, quota_count = _quota_in_cgroup_hierarchy(root / relative, root, version)
        if found:
            return quota_count

    if ambiguous_mount:
        return 1  # Unknown effective quota: keep automatic workers disabled.

    for root, version in (
        (pathlib.Path("/sys/fs/cgroup"), "v2"),
        (pathlib.Path("/sys/fs/cgroup/cpu"), "v1"),
        (pathlib.Path("/sys/fs/cgroup/cpu,cpuacct"), "v1"),
    ):
        found, quota_count = _quota_in_cgroup_hierarchy(root, root, version)
        if found:
            return quota_count
    return None


def _quota_in_cgroup_hierarchy(
    current: pathlib.Path, root: pathlib.Path, version: str
) -> tuple[bool, Optional[int]]:
    found = False
    limit: Optional[int] = None
    while current.is_relative_to(root):
        try:
            if version == "v2":
                quota_text, period_text = (current / "cpu.max").read_text().split()
                found = True
                if quota_text == "max":
                    quota = -1
                else:
                    quota = int(quota_text)
                period = int(period_text)
            else:
                quota = int((current / "cpu.cfs_quota_us").read_text())
                period = int((current / "cpu.cfs_period_us").read_text())
                found = True
            if quota > 0 and period > 0:
                count = max(1, quota // period)
                limit = count if limit is None else min(limit, count)
        except (OSError, ValueError, ZeroDivisionError):
            pass
        if current == root:
            break
        current = current.parent
    return found, limit


def _default_sglang_preprocess_workers() -> int:
    try:
        cpu_count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cpu_count = os.cpu_count() or 1

    quota_count = _cpu_quota_count()
    if quota_count is not None:
        cpu_count = min(cpu_count, quota_count)
    return min(2, max(0, cpu_count - 1))


def validate_model_name(value: str) -> str:
    """Validate that model-name is a non-empty string."""
    if not value or not isinstance(value, str) or len(value.strip()) == 0:
        raise argparse.ArgumentTypeError(
            f"model-name must be a non-empty string, got: {value}"
        )
    return value.strip()


def validate_model_path(value: str) -> str:
    """Validate that model-path is a valid directory on disk."""
    if not os.path.isdir(value):
        raise argparse.ArgumentTypeError(
            f"model-path must be a valid directory on disk, got: {value}"
        )
    return value


class FrontendConfig(RouterConfigBase, KvRouterConfigBase, AisPerfConfigBase):
    """Configuration for the Dynamo frontend."""

    interactive: bool
    kv_cache_block_size: Optional[int]
    http_host: str
    http_port: int
    batch_gateway_url: Optional[str] = None
    tls_cert_path: Optional[pathlib.Path]
    tls_key_path: Optional[pathlib.Path]
    tls_client_ca_cert_path: Optional[pathlib.Path]
    tcp_tls_cert_path: Optional[str] = None
    tcp_tls_key_path: Optional[str] = None
    tcp_tls_ca_cert_path: Optional[str] = None
    tcp_tls_client_cert_path: Optional[str] = None
    tcp_tls_client_key_path: Optional[str] = None
    tcp_tls_client_ca_cert_path: Optional[str] = None
    nats_tls_ca_cert_path: Optional[str] = None
    nats_tls_insecure: bool = False
    nats_tls_client_cert_path: Optional[str] = None
    nats_tls_client_key_path: Optional[str] = None

    namespace: Optional[str] = None
    namespace_prefix: Optional[str] = None

    migration_limit: int
    migration_max_seq_len: Optional[int]
    model_name: Optional[str]
    model_path: Optional[str]
    metrics_prefix: Optional[str] = None

    kserve_grpc_server: bool
    grpc_metrics_port: int
    dump_config_to: Optional[str]

    discovery_backend: str
    request_plane: str
    response_plane: Optional[str] = None
    event_plane: Optional[str] = None
    chat_processor: str
    chat_processor_config: Optional[Dict[str, Any]] = None
    enable_anthropic_api: bool
    enable_systemone_api: bool
    strip_anthropic_preamble: bool
    reject_media: bool
    reject_media_status_code: int
    debug_perf: bool
    enable_streaming_tool_dispatch: bool
    enable_streaming_reasoning_dispatch: bool
    reasoning_field_name: str
    exclude_tools_when_tool_choice_none: bool
    default_chat_template_kwargs: Optional[Dict[str, Any]] = None
    preprocess_workers: Optional[int]
    tokenizer_backend: str
    tokenizer_fallback: bool
    trust_remote_code: bool
    frontend_route_extensions: list[str]

    _VALID_TOKENIZER_BACKENDS = {"default", "fastokens", "basetenkenizer"}

    def validate_default_chat_template_kwargs(self) -> None:
        # Request validation rejects a nested chat_template on every path, so
        # this default would fail every request.
        if "chat_template" in (self.default_chat_template_kwargs or {}):
            raise ValueError(
                "--default-chat-template-kwargs cannot set chat_template. Use "
                "--custom-jinja-template on the worker to replace the chat template."
            )

    def validate(self) -> None:
        if (
            self.chat_processor_config is not None
            and self.chat_processor in BUILTIN_CHAT_PROCESSORS
        ):
            raise ValueError(
                "--dyn-chat-processor-config only applies to a custom "
                f"--dyn-chat-processor, not '{self.chat_processor}'"
            )
        if self.preprocess_workers is None:
            self.preprocess_workers = (
                _default_sglang_preprocess_workers()
                if self.chat_processor == "sglang"
                else 0
            )
        elif self.preprocess_workers < 0:
            raise ValueError("--dyn-preprocess-workers must be >= 0")

        if self.load_aware:
            self.router_mode = "kv"
        self.apply_router_config()

        # Fail at startup on an invalid DYN_MM_* value, not on every request.
        # Imported here: dynamo.common.http imports this configuration package.
        from dynamo.common.http.url_validator import UrlValidationPolicy

        UrlValidationPolicy.from_env()

        if bool(self.tls_cert_path) ^ bool(self.tls_key_path):  # ^ is XOR
            raise ValueError(
                "--tls-cert-path and --tls-key-path must be provided together"
            )
        if self.tls_client_ca_cert_path and not (
            self.tls_cert_path and self.tls_key_path
        ):
            raise ValueError(
                "--tls-client-ca-cert-path requires --tls-cert-path and --tls-key-path"
            )
        if self.frontend_route_extensions and (
            self.interactive or self.kserve_grpc_server
        ):
            mode_flag = "--interactive" if self.interactive else "--kserve-grpc-server"
            raise ValueError(
                "--frontend-route-extension is only supported by the HTTP frontend, "
                f"so it cannot be combined with {mode_flag}"
            )
        if self.batch_gateway_url is not None and (
            self.interactive or self.kserve_grpc_server
        ):
            mode_flag = "--interactive" if self.interactive else "--kserve-grpc-server"
            raise ValueError(
                "--batch-gateway-url is only supported by the HTTP frontend, "
                f"so it cannot be combined with {mode_flag}"
            )
        if self.migration_limit < 0 or self.migration_limit > _U32_MAX:
            raise ValueError(
                f"--migration-limit must be between 0 and {_U32_MAX} (0=disabled)"
            )
        if self.migration_max_seq_len is not None and (
            self.migration_max_seq_len < 1 or self.migration_max_seq_len > _U32_MAX
        ):
            raise ValueError(
                f"--migration-max-seq-len must be between 1 and {_U32_MAX}"
            )
        if self.min_initial_workers < 0:
            raise ValueError("--router-min-initial-workers must be >= 0")
        if not 400 <= self.reject_media_status_code <= 599:
            raise ValueError("--reject-media-status-code must be a 4xx or 5xx status")
        if self.session_affinity_ttl_secs is not None and not (
            1 <= self.session_affinity_ttl_secs <= _MAX_SESSION_AFFINITY_TTL_SECS
        ):
            raise ValueError(
                "--router-session-affinity-ttl-secs must be between 1 and "
                f"{_MAX_SESSION_AFFINITY_TTL_SECS}"
            )
        if self.session_affinity_max_entries < 1:
            raise ValueError("--router-session-affinity-max-entries must be >= 1")
        self.validate_default_chat_template_kwargs()
        if self.tokenizer_backend not in self._VALID_TOKENIZER_BACKENDS:
            raise ValueError(
                f"--tokenizer: invalid value '{self.tokenizer_backend}' "
                f"(choose from {sorted(self._VALID_TOKENIZER_BACKENDS)})"
            )
        if self.ais_perf_config is not None and self.router_prefill_load_model != "ais":
            raise ValueError(
                "--ais-perf-config requires --router-prefill-load-model=ais"
            )
        if self.router_prefill_load_model == "ais":
            if self.router_mode != "kv":
                raise ValueError(
                    "--router-prefill-load-model=ais requires --router-mode=kv"
                )
            if self.chat_processor != "dynamo":
                raise ValueError(
                    "--router-prefill-load-model=ais currently requires "
                    "--dyn-chat-processor=dynamo"
                )
            self.ais_perf_kwargs()
            if not self.router_track_prefill_tokens:
                raise ValueError(
                    "--router-prefill-load-model=ais requires "
                    "--router-track-prefill-tokens"
                )
        if self.serve_indexer:
            if self.router_mode != "kv":
                raise ValueError("--serve-indexer requires --router-mode=kv")
            if self.use_remote_indexer:
                raise ValueError(
                    "--serve-indexer and --use-remote-indexer are mutually exclusive"
                )
        if self.conditional_disagg_policy not in CONDITIONAL_DISAGG_POLICY_CHOICES:
            raise ValueError(
                "--router-conditional-disagg-config policy must be one of "
                + ", ".join(
                    f"'{choice}'" for choice in CONDITIONAL_DISAGG_POLICY_CHOICES
                )
            )
        if self.conditional_disagg_eff_isl_threshold < 0:
            raise ValueError(
                "--router-conditional-disagg-config eff_isl_threshold must be >= 0"
            )
        if not 0.0 <= self.conditional_disagg_eff_isl_ratio_threshold <= 1.0:
            raise ValueError(
                "--router-conditional-disagg-config eff_isl_ratio_threshold must be in [0.0, 1.0]"
            )
        if (
            self.conditional_disagg_prefill_busy_threshold is not None
            and self.conditional_disagg_prefill_busy_threshold < 0
        ):
            raise ValueError(
                "--router-conditional-disagg-config prefill_busy_threshold must be >= 0"
            )
        if (
            self.conditional_disagg_decode_busy_threshold is not None
            and self.conditional_disagg_decode_busy_threshold < 0
        ):
            raise ValueError(
                "--router-conditional-disagg-config decode_busy_threshold must be >= 0"
            )
        if self.conditional_disagg_enabled and self.router_mode != "kv":
            raise ValueError("--router-conditional-disagg requires --router-mode=kv")
        if self.conditional_disagg_enabled and not self.use_kv_events:
            raise ValueError("--router-conditional-disagg requires --router-kv-events")
        self.validate_rejection_thresholds()
        # Direct mode ignores the thresholds.
        if self.router_mode != "direct":
            self.log_rejection_thresholds()


@register_encoder(FrontendConfig)
def _preprocess_for_encode_config(config: FrontendConfig) -> Dict[str, Any]:
    """Convert FrontendConfig object to dictionary for encoding."""
    return config.__dict__


class FrontendArgGroup(ArgGroup):
    """Frontend configuration parameters.

    Args:
        env: Another process's env (see ``ArgEnv``). Nothing then reads the
            local filesystem: ``--model-path`` is not checked, and the
            AISimulate perf-model arguments, whose values name files, are not
            registered.
    """

    def __init__(self, *, env: Optional[ArgEnv] = None) -> None:
        self.env = env

    def add_arguments(self, parser) -> None:
        parser.add_argument(
            "--version", action="version", version=f"Dynamo Frontend {__version__}"
        )

        g = parser.add_argument_group("Dynamo Frontend Options")

        # Interactive needs -i short option; use raw add_argument with BooleanOptionalAction
        g.add_argument(
            "-i",
            "--interactive",
            dest="interactive",
            action=argparse.BooleanOptionalAction,
            default=env_or_default("DYN_INTERACTIVE", False, env=self.env),
            help="Interactive text chat.\nenv var: DYN_INTERACTIVE",
        )

        add_argument(
            g,
            flag_name="--namespace",
            env_var="DYN_NAMESPACE",
            default=None,
            help=(
                "Dynamo namespace for model discovery scoping. Use for exact namespace matching. "
                "If --namespace-prefix is also specified, prefix takes precedence."
            ),
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--kv-cache-block-size",
            env_var="DYN_KV_CACHE_BLOCK_SIZE",
            default=None,
            help="KV cache block size (u32).",
            arg_type=int,
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--http-host",
            env_var="DYN_HTTP_HOST",
            default="0.0.0.0",
            help="HTTP host for the engine (str).",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--http-port",
            env_var="DYN_HTTP_PORT",
            default=8000,
            help="HTTP port for the engine (u16).",
            arg_type=int,
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--batch-gateway-url",
            env_var="DYN_BATCH_GATEWAY_URL",
            default=None,
            help="Internal Batch Gateway URL for proxying /v1/files and /v1/batches.",
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--serve-indexer",
            env_var="DYN_SERVE_INDEXER",
            default=False,
            help="Serve this frontend's local KV indexers over the request plane.",
            dest="serve_indexer",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--tls-cert-path",
            env_var="DYN_TLS_CERT_PATH",
            default=None,
            help="TLS certificate path, PEM format.",
            arg_type=pathlib.Path,
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--tls-key-path",
            env_var="DYN_TLS_KEY_PATH",
            default=None,
            help="TLS certificate key path, PEM format.",
            arg_type=pathlib.Path,
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--tls-client-ca-cert-path",
            env_var="DYN_TLS_CLIENT_CA_CERT_PATH",
            default=None,
            help="Client CA certificate path for mutual TLS, PEM format.",
            arg_type=pathlib.Path,
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--tcp-tls-cert-path",
            env_var="DYN_TCP_TLS_CERT_PATH",
            default=None,
            help="Path to PEM certificate for the TCP server.",
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--tcp-tls-key-path",
            env_var="DYN_TCP_TLS_KEY_PATH",
            default=None,
            help="Path to PEM private key for the TCP server certificate.",
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--tcp-tls-ca-cert-path",
            env_var="DYN_TCP_TLS_CA_CERT_PATH",
            default=None,
            help="Path to PEM CA certificate used to verify the TCP peer's certificate.",
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--tcp-tls-client-cert-path",
            env_var="DYN_TCP_TLS_CLIENT_CERT_PATH",
            default=None,
            help="Path to PEM client certificate presented to the TCP server for mTLS.",
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--tcp-tls-client-key-path",
            env_var="DYN_TCP_TLS_CLIENT_KEY_PATH",
            default=None,
            help="Path to PEM private key for the TCP client certificate (mTLS).",
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--tcp-tls-client-ca-cert-path",
            env_var="DYN_TCP_TLS_CLIENT_CA_CERT_PATH",
            default=None,
            help="Path to PEM CA certificate the TCP server uses to verify client "
            "certificates. When set, clients must present a trusted certificate (mTLS enforced).",
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--nats-tls-ca-cert-path",
            env_var="NATS_TLS_CA_CERT_PATH",
            default=None,
            help="Path to PEM CA certificate for verifying the NATS server.",
            env=self.env,
        )

        add_negatable_bool_argument(
            g,
            flag_name="--nats-tls-insecure",
            env_var="NATS_TLS_INSECURE",
            default=False,
            help="Disable NATS TLS certificate verification. For local development only.",
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--nats-tls-client-cert-path",
            env_var="NATS_TLS_CLIENT_CERT_PATH",
            default=None,
            help="Path to PEM client certificate presented to the NATS server for mTLS.",
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--nats-tls-client-key-path",
            env_var="NATS_TLS_CLIENT_KEY_PATH",
            default=None,
            help="Path to PEM private key for the NATS client certificate (mTLS).",
            env=self.env,
        )

        # Router options (shared with dynamo.router)
        RouterArgGroup(
            default_router_mode="round-robin",
            include_frontend_only=True,
            env=self.env,
        ).add_arguments(parser)

        # KV router options (shared with dynamo.router)
        KvRouterArgGroup(env=self.env).add_arguments(parser)
        if self.env is None:
            AisPerfArgGroup().add_arguments(parser)

        add_argument(
            g,
            flag_name="--namespace-prefix",
            env_var="DYN_NAMESPACE_PREFIX",
            default=None,
            help=(
                "Dynamo namespace prefix for model discovery scoping. Discovers models from "
                "namespaces starting with this prefix (e.g., 'ns' matches 'ns', 'ns-abc123', "
                "'ns-def456'). Takes precedence over --namespace if both are specified."
            ),
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--migration-limit",
            env_var="DYN_MIGRATION_LIMIT",
            default=0,
            help=(
                "Maximum number of times a request may be migrated to a different engine worker. "
                "When > 0, enables migration after worker disconnects, response timeouts, "
                "incomplete streams, and worker-local overload rejection. Also limits replacement "
                "attempts over the lifetime of an explicit-commit PCM realtime transcription "
                "session; unfinished audio is replayed to a compatible warm worker."
            ),
            arg_type=int,
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--migration-max-seq-len",
            env_var="DYN_MIGRATION_MAX_SEQ_LEN",
            default=None,
            help=(
                "Maximum sequence length (prompt + generated tokens) for migration state tracking. "
                "Once the accumulated token count exceeds this limit, the request becomes "
                "non-migratable. Prevents unbounded memory growth from caching long sequences. "
                "Default: no limit."
            ),
            arg_type=int,
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--model-name",
            env_var="DYN_MODEL_NAME",
            default=None,
            help="Model name as a string (e.g., 'Llama-3.2-1B-Instruct')",
            arg_type=validate_model_name,
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--model-path",
            env_var="DYN_MODEL_PATH",
            default=None,
            help="Path to model directory on disk (e.g., /tmp/model_cache/llama3.2_1B/)",
            arg_type=validate_model_path if self.env is None else str,
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--metrics-prefix",
            env_var="DYN_METRICS_PREFIX",
            default=None,
            help=(
                "Prefix for Dynamo frontend metrics. If unset, uses DYN_METRICS_PREFIX env var "
                "or 'dynamo_frontend'."
            ),
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--kserve-grpc-server",
            env_var="DYN_KSERVE_GRPC_SERVER",
            default=False,
            help="Start KServe gRPC server.",
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--grpc-metrics-port",
            env_var="DYN_GRPC_METRICS_PORT",
            default=8788,
            help=(
                "HTTP metrics port for gRPC service (u16). Only used with --kserve-grpc-server. "
                "Defaults to 8788."
            ),
            arg_type=int,
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--dump-config-to",
            env_var="DYN_DUMP_CONFIG_TO",
            default=None,
            help="Dump config to the specified file path.",
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--frontend-route-extension",
            env_var="DYN_FRONTEND_ROUTE_EXTENSIONS",
            default=[],
            dest="frontend_route_extensions",
            action="append",
            help=(
                "Trusted frontend route extension: a name registered under the "
                "'dynamo.frontend.routes' entry-point group, or a 'module:function' "
                "path. May be repeated. DYN_FRONTEND_ROUTE_EXTENSIONS accepts "
                "whitespace-separated values."
            ),
            env=self.env,
        )

        add_argument(
            g,
            flag_name="--discovery-backend",
            env_var="DYN_DISCOVERY_BACKEND",
            default="etcd",
            help=(
                "Discovery backend: kubernetes (K8s API), etcd (distributed KV), file (local filesystem), "
                "mem (in-memory). Etcd uses the ETCD_* env vars (e.g. ETCD_ENDPOINTS) for connection details. "
                "File uses root dir from env var DYN_FILE_KV or defaults to $TMPDIR/dynamo_store_kv."
            ),
            choices=["kubernetes", "etcd", "file", "mem"],
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--request-plane",
            env_var="DYN_REQUEST_PLANE",
            default="tcp",
            help=(
                "Determines how requests are distributed from routers to workers. "
                "'tcp' is fastest [nats|tcp]"
            ),
            choices=["nats", "tcp"],
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--response-plane",
            env_var="DYN_RESPONSE_PLANE",
            default=None,
            help=(
                "Select the response transport. Frontend and workers must match. "
                "When unset, the runtime uses tcp. velo has no TLS, so it falls "
                "back to tcp while any DYN_TCP_TLS_* setting is present."
            ),
            choices=["tcp", "quic", "velo"],
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--event-plane",
            env_var="DYN_EVENT_PLANE",
            default=None,
            help="Determines how events are published [nats|zmq]. If unset, "
            "defaults to 'zmq' for all discovery backends. Set to 'nats' to use a "
            "NATS-based event plane.",
            choices=["nats", "zmq"],
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-anthropic-api",
            env_var="DYN_ENABLE_ANTHROPIC_API",
            default=False,
            help=(
                "[EXPERIMENTAL] Enable Anthropic Messages API endpoint (/v1/messages). "
                "This feature is experimental and may change."
            ),
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-systemone-api",
            env_var="DYN_ENABLE_SYSTEMONE_API",
            default=False,
            help=(
                "[EXPERIMENTAL] Enable typed decision APIs (/v1/decisions and "
                "/v1/systemone) for qualified aggregate SGLang models."
            ),
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--strip-anthropic-preamble",
            env_var="DYN_STRIP_ANTHROPIC_PREAMBLE",
            default=True,
            help=(
                "Strip the Claude Code billing preamble (x-anthropic-billing-header) "
                "from the system prompt. Saves tokens and improves prompt caching."
            ),
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--reject-media",
            env_var="DYN_REJECT_MEDIA",
            default=False,
            help=(
                "Refuse chat requests that carry image, video, or audio content in "
                "any message, including history, before routing. For frontends that "
                "serve text-only models: a streaming client gets an HTTP error "
                "instead of a 200 followed by an error event. The status is set by "
                "--reject-media-status-code."
            ),
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--reject-media-status-code",
            env_var="DYN_REJECT_MEDIA_STATUS_CODE",
            default=415,
            help=(
                "HTTP status (4xx or 5xx) for a request refused by --reject-media. "
                "Set it to a status an upstream gateway fails over on, such as 429, "
                "to send these requests to a multimodal model."
            ),
            arg_type=int,
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-streaming-tool-dispatch",
            env_var="DYN_ENABLE_STREAMING_TOOL_DISPATCH",
            default=False,
            help=(
                "[EXPERIMENTAL] Enable streaming tool call dispatch. Emits "
                "'event: tool_call_dispatch' SSE events on /v1/chat/completions "
                "for each complete tool call before finish_reason arrives. "
                "Can be combined with --enable-streaming-reasoning-dispatch."
            ),
            env=self.env,
        )
        add_negatable_bool_argument(
            g,
            flag_name="--enable-streaming-reasoning-dispatch",
            env_var="DYN_ENABLE_STREAMING_REASONING_DISPATCH",
            default=False,
            help=(
                "[EXPERIMENTAL] Enable streaming reasoning dispatch. Emits a "
                "single 'event: reasoning_dispatch' SSE event on /v1/chat/completions "
                "with the complete reasoning block once thinking ends. "
                "Can be combined with --enable-streaming-tool-dispatch."
            ),
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--reasoning-field-name",
            env_var="DYN_REASONING_FIELD_NAME",
            default="reasoning_content",
            help=(
                "OpenAI-compatible response field used for emitted reasoning content."
            ),
            choices=["reasoning_content", "reasoning"],
            env=self.env,
        )
        # NOTE: This flag also exists in DynamoRuntimeArgGroup (runtime_args.py).
        # Both definitions are needed: runtime_args controls the Rust-native
        # chat template path (oai.rs), while this one controls the Python
        # frontend processors (vllm_processor / sglang_processor) which parse
        # arguments independently via FrontendConfig.
        add_negatable_bool_argument(
            g,
            flag_name="--exclude-tools-when-tool-choice-none",
            env_var="DYN_EXCLUDE_TOOLS_WHEN_TOOL_CHOICE_NONE",
            default=True,
            help=(
                "Exclude tool definitions from the chat template when "
                "tool_choice='none'. Prevents models from generating raw XML "
                "tool calls in the content field."
            ),
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--default-chat-template-kwargs",
            env_var="DYN_DEFAULT_CHAT_TEMPLATE_KWARGS",
            default=None,
            arg_type=_parse_chat_template_kwargs,
            help=(
                "JSON object of chat template kwargs that apply to every chat request "
                "this frontend serves, for every model. Each request's own "
                'chat_template_kwargs win per key, except null and "auto" values, '
                "and a request thinking control "
                "replaces the thinking keys. Set this on the frontend, not on the "
                "engine: the frontend renders the chat template, so the engine flag "
                "has no effect behind Dynamo."
            ),
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--dyn-chat-processor",
            env_var="DYN_CHAT_PROCESSOR",
            default="dynamo",
            dest="chat_processor",
            help=(
                "[EXPERIMENTAL] Chat pre/post processor backend. 'dynamo' uses the Rust "
                "preprocessor. 'vllm' uses local vLLM for pre and post processing. "
                "'sglang' uses SGLang APIs for chat template rendering, tool call "
                "parsing, and reasoning parsing. Any other value loads a custom chat "
                "processor: a name registered in the 'dynamo.frontend.chat_processors' "
                "entry-point group, or a 'module:function' path."
            ),
            env=self.env,
        )
        add_argument(
            g,
            flag_name="--dyn-chat-processor-config",
            env_var="DYN_CHAT_PROCESSOR_CONFIG",
            default=None,
            dest="chat_processor_config",
            arg_type=_parse_chat_processor_config,
            help=(
                "JSON object passed to a custom --dyn-chat-processor as its settings. "
                "Not accepted with the built-in processors."
            ),
            env=self.env,
        )

        add_negatable_bool_argument(
            g,
            flag_name="--dyn-debug-perf",
            env_var="DYN_DEBUG_PERF",
            default=False,
            dest="debug_perf",
            help=(
                "[EXPERIMENTAL] Enable performance instrumentation for diagnosing preprocessing bottlenecks. "
                "Logs per-function timing, request concurrency, and hot-path section durations. "
                "Supported with '--dyn-chat-processor vllm' and '--dyn-chat-processor sglang'."
            ),
            env=self.env,
        )

        g.add_argument(
            "--dyn-preprocess-workers",
            dest="preprocess_workers",
            type=int,
            default=env_or_default(
                "DYN_PREPROCESS_WORKERS", None, value_type=int, env=self.env
            ),
            help=(
                "[EXPERIMENTAL] SGLang preprocessing worker processes per model. "
                "When unset, selects up to 2 workers per model from visible CPU capacity, "
                "leaving one CPU out of the per-model calculation. "
                "Multiple models each create a pool. "
                "0 runs preprocessing on the main event loop. "
                "Nonzero values are supported only with '--dyn-chat-processor sglang'.\n"
                "env var: DYN_PREPROCESS_WORKERS | default: auto for SGLang, 0 otherwise"
            ),
        )

        add_argument(
            g,
            flag_name="--tokenizer",
            env_var="DYN_TOKENIZER",
            default="default",
            dest="tokenizer_backend",
            help=(
                "Tokenizer backend for BPE models: 'default' (HuggingFace tokenizers library), "
                "'fastokens' (fastokens crate for high-performance BPE encoding), or "
                "'basetenkenizer' (Baseten Tokenizer for native encoding and decoding). "
                "Has no effect on TikToken models."
            ),
            choices=["default", "fastokens", "basetenkenizer"],
            env=self.env,
        )

        add_negatable_bool_argument(
            g,
            flag_name="--tokenizer-fallback",
            env_var="DYN_TOKENIZER_FALLBACK",
            default=True,
            help=(
                "Automatic fallback to HuggingFace is deprecated and will be "
                "disabled by default in a future release. The current behavior "
                "falls back when the selected fastokens or basetenkenizer backend "
                "cannot load the model tokenizer. Use "
                "--no-tokenizer-fallback to fail model initialization instead. "
                "In dynamic mode, discovery retries the load while the frontend "
                "continues running."
            ),
            env_value_type=parse_bool,
            env=self.env,
        )

        add_negatable_bool_argument(
            g,
            flag_name="--trust-remote-code",
            env_var="DYN_TRUST_REMOTE_CODE",
            default=False,
            help=(
                "Trust remote code when loading the tokenizer. Required for models "
                "that ship custom tokenizer code (e.g. Qwen, Falcon)."
            ),
            env=self.env,
        )


def build_frontend_parser(
    env: Mapping[str, str],
    parser: Optional[argparse.ArgumentParser] = None,
) -> tuple[argparse.ArgumentParser, dict[str, str]]:
    """The argument parser of a frontend that runs with ``env``.

    For a process that parses another frontend's command line (the planner
    reads one from the DynamoGraphDeployment spec): env defaults come from
    ``env`` only, never ``os.environ``, and nothing logs, warns or reads the
    local filesystem (see ``FrontendArgGroup``). Registers the arguments on
    ``parser`` when given, else on a new ``ArgumentParser(add_help=False)``.
    Returns the parser and the env var -> dest map of every argument with an
    env var. Raises what the frontend raises at startup on invalid env
    (``ValueError``, ``TypeError`` or ``argparse.ArgumentTypeError``).
    """

    arg_env = ArgEnv(environ=dict(env))
    if parser is None:
        parser = argparse.ArgumentParser(add_help=False)
    FrontendArgGroup(env=arg_env).add_arguments(parser)
    return parser, arg_env.env_dests
