# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""DeepInfer engine and topology options, shared by the head's Dynamo worker
(``args.py``) and the non-head node launcher (``node.py``).

One DeepInfer engine has ``dp_size`` ranks over ``nnodes`` nodes,
``ranks_per_node`` on each; global rank ``g`` lives on node
``g // ranks_per_node``. This module must not import ``dynamo._core`` or
``dynamo.common.backend``: ``node.py`` parses the same flags without the Dynamo
runtime.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import shutil
import socket
import subprocess
from collections.abc import Iterable, MutableMapping, Sequence
from typing import Any, Optional

from dynamo.common.constants import DisaggregationMode
from dynamo.deepinfer.rank import RankSpec

ENGINE_TYPES = {
    DisaggregationMode.AGGREGATED: "NORMAL",
    DisaggregationMode.PREFILL: "PROMPT",
    DisaggregationMode.DECODE: "COMPLETION",
}
# DeepInfer's own boolean env parsing (deepinfer/environ.py: _bool).
_DEEPINFER_FALSE = ("0", "", "false", "False")
DEFAULT_MASTER_PORT = 23333
DEFAULT_CONTROL_PORT = 23400
# DeepInfer's gloo store listens on MASTER_PORT - 50 + engine_group_rank
# (deepinfer/utils.py: get_distributed_init_method).
STORE_PORT_OFFSET = -50

logger = logging.getLogger(__name__)


class DeepInferOptions:
    """DeepInfer settings of one engine, the same on every node."""

    disaggregation_mode: DisaggregationMode = DisaggregationMode.AGGREGATED
    model: str
    served_model_name: Optional[str] = None
    # DP ranks over all nodes.
    dp_size: int
    ep_size: Optional[int] = None
    max_model_len: Optional[int] = None
    max_num_seqs: int = 256
    max_num_batched_tokens: Optional[int] = None
    gpu_memory_utilization: float = 0.9
    lookahead_tokens: int = 1
    prefill_attempt_interval: int = 8
    kv_cache_manager: str = "none"
    kv_transfer: str = "none"
    kv_events_port_base: int = 5557
    kv_lease_s: float = 120.0
    prefix_snapshot_interval: int = 32768
    prefix_snapshot_slots: int = 256
    # None: on host with the host tier, else on GPU.
    prefix_snapshot_device: Optional[str] = None
    prefix_free_watermark: float = 0.05
    prefix_host_bytes: int = 0
    prefix_ssd_dir: Optional[str] = None
    prefix_ssd_bytes: int = 0
    prefix_ssd_wait_ms: float = 50.0
    rank0_in_process: bool = True
    startup_timeout_s: float = 3600.0
    engine_overrides: dict[str, Any]
    deepinfer_backend: str = "deepinfer"
    nnodes: int = 1
    node_rank: int = 0
    master_addr: str = "127.0.0.1"
    master_port: int = DEFAULT_MASTER_PORT
    control_port: int = DEFAULT_CONTROL_PORT
    node_timeout_s: float = 30.0
    advertise_addr: Optional[str] = None

    @classmethod
    def from_namespace(cls, ns: argparse.Namespace) -> DeepInferOptions:
        """Build from :func:`add_deepinfer_arguments`' namespace (``node.py``;
        the head goes through ``DeepInferArgs.from_cli_args``)."""
        obj = cls.__new__(cls)
        for base in reversed(cls.__mro__):
            for name in getattr(base, "__annotations__", {}):
                if name in vars(base):
                    setattr(obj, name, getattr(base, name))
        for key, value in vars(ns).items():
            setattr(obj, key, value)
        finalize_options(obj, ns)
        return obj

    @property
    def engine_type(self) -> str:
        return ENGINE_TYPES[self.disaggregation_mode]

    @property
    def ranks_per_node(self) -> int:
        return self.dp_size // self.nnodes

    @property
    def local_ranks(self) -> range:
        """Global ranks on this node."""
        first = self.node_rank * self.ranks_per_node
        return range(first, first + self.ranks_per_node)

    def engine_id(self) -> str:
        if self.nnodes == 1:
            return f"{socket.gethostname()}-{os.getpid()}"
        # Every node must derive the same id without talking to the head.
        return f"{self.master_addr}-{self.master_port}"

    def engine_args(self, rank: int) -> dict[str, Any]:
        """deepinfer ``EngineArgs`` kwargs for one (global) rank."""
        args: dict[str, Any] = {
            "model": self.model,
            "served_model_name": self.served_model_name or self.model,
            "trust_remote_code": True,
            "chunked_prefill": True,
            "data_parallel_size": self.dp_size,
            "expert_parallel_size": self.ep_size or self.dp_size,
            "engine_rank": rank,
            "max_num_seqs": self.max_num_seqs,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "lookahead_tokens": self.lookahead_tokens,
        }
        if self.max_model_len is not None:
            args["max_model_len"] = self.max_model_len
        if self.max_num_batched_tokens is not None:
            args["max_num_batched_tokens"] = self.max_num_batched_tokens
        if self.engine_type == "COMPLETION":
            # DeepInfer refuses a decode-only engine that recomputes preemptions.
            args["stop_preempted_request"] = True
        args.update(self.engine_overrides)
        return args

    def rank_specs(
        self, engine_id: str, ranks: Optional[Iterable[int]] = None
    ) -> list[RankSpec]:
        """Specs of ``ranks`` (default: every rank of the engine). A rank's KV
        events go to ``--kv-events-port-base`` plus its index on its node."""
        cache_options: dict[str, Any] = {}
        if self.kv_cache_manager == "prefix":
            cache_options = {
                "snapshot_interval": self.prefix_snapshot_interval,
                "snapshot_slots": self.prefix_snapshot_slots,
                "snapshot_device": self.prefix_snapshot_device,
                "free_watermark": self.prefix_free_watermark,
                "host_bytes": self.prefix_host_bytes,
                "ssd_dir": self.prefix_ssd_dir,
                "ssd_bytes": self.prefix_ssd_bytes,
                "ssd_share": 1 / self.ranks_per_node,
                "ssd_wait_ms": self.prefix_ssd_wait_ms,
            }
        events = self.kv_cache_manager != "none"
        per_node = self.ranks_per_node
        return [
            RankSpec(
                rank=rank,
                dp_size=self.dp_size,
                engine_args=self.engine_args(rank),
                engine_type=self.engine_type,
                backend=self.deepinfer_backend,
                engine_id=engine_id,
                cache_manager=self.kv_cache_manager,
                cache_options=(
                    {**cache_options, "ssd_rank": rank} if cache_options else {}
                ),
                kv_transfer=self.kv_transfer,
                kv_events_endpoint=(
                    f"tcp://*:{self.kv_events_port_base + rank % per_node}"
                    if events
                    else None
                ),
                kv_lease_s=self.kv_lease_s,
            )
            for rank in (range(self.dp_size) if ranks is None else ranks)
        ]

    def kv_event_endpoint(self, rank: int) -> str:
        """Where the head subscribes to a rank on its own node."""
        port = self.kv_events_port_base + rank % self.ranks_per_node
        return f"tcp://127.0.0.1:{port}"


def add_deepinfer_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("DeepInfer")
    g.add_argument("--model", required=True, help="Checkpoint directory.")
    g.add_argument("--served-model-name", default=None)
    g.add_argument(
        "--disaggregation-mode",
        choices=[m.value for m in ENGINE_TYPES],
        default=DisaggregationMode.AGGREGATED.value,
    )
    g.add_argument(
        "--dp-size",
        type=int,
        default=None,
        help="DP ranks over all nodes (default: local GPUs x --nnodes).",
    )
    g.add_argument("--ep-size", type=int, default=None, help="Default: --dp-size.")
    g.add_argument("--max-model-len", type=int, default=None)
    g.add_argument("--max-num-seqs", type=int, default=256)
    g.add_argument("--max-num-batched-tokens", type=int, default=None)
    g.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    g.add_argument("--lookahead-tokens", type=int, default=1)
    g.add_argument(
        "--prefill-attempt-interval",
        type=int,
        default=8,
        help="DEEPINFER_PREFILL_ATTEMPT_INTERVAL default for aggregated workers.",
    )
    g.add_argument("--kv-cache-manager", choices=["none", "prefix"], default="none")
    g.add_argument(
        "--kv-transfer",
        choices=["none", "nixl", "local"],
        default=None,
        help="Default: nixl for prefill/decode workers, none otherwise.",
    )
    g.add_argument(
        "--kv-events-port-base",
        type=int,
        default=5557,
        help="This node's ranks publish KV events on this port plus their index on the node.",
    )
    g.add_argument("--kv-lease-s", type=float, default=120.0)
    g.add_argument("--prefix-snapshot-interval", type=int, default=32768)
    g.add_argument("--prefix-snapshot-slots", type=int, default=256)
    g.add_argument(
        "--prefix-snapshot-device",
        choices=["gpu", "host"],
        default=None,
        help="Where prefix snapshots live (default: host with --prefix-host-bytes, else gpu).",
    )
    g.add_argument("--prefix-free-watermark", type=float, default=0.05)
    g.add_argument(
        "--prefix-host-bytes",
        type=parse_size,
        default=0,
        help="Per rank: pinned host memory for prefix blocks evicted from the GPU "
        "(and the snapshots on host), e.g. 80GiB; 0 turns the host tier off. "
        "Clamped at startup to keep 10%% of the host's memory available.",
    )
    g.add_argument(
        "--prefix-ssd-dir",
        default=None,
        help="Directory of the SSD tier's files (one per rank, recreated at every "
        "start); needs --prefix-ssd-bytes.",
    )
    g.add_argument(
        "--prefix-ssd-bytes",
        type=parse_size,
        default=0,
        help="Per rank: file space for prefix blocks evicted from host memory, "
        "e.g. 1TiB; 0 turns the SSD tier off. Needs --prefix-host-bytes and "
        "--prefix-ssd-dir. Clamped at startup to what the snapshots can serve and "
        "to this rank's share (1 / ranks per node) of the directory's free space.",
    )
    g.add_argument(
        "--prefix-ssd-wait-ms",
        type=float,
        default=50.0,
        help="Longest a request waits for the SSD read of its cached prefix (less "
        "for shorter prefixes; the read also starts when the request is queued). "
        "0 turns waiting and that prefetch off: the request recomputes the prefix. "
        "Ignored without the SSD tier.",
    )
    g.add_argument(
        "--rank0-in-process",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run rank 0 inside the Dynamo process (default) or as a child.",
    )
    g.add_argument("--startup-timeout-s", type=float, default=3600.0)
    g.add_argument(
        "--engine-arg",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra deepinfer EngineArgs field (VALUE parsed as JSON when possible).",
    )
    g.add_argument(
        "--deepinfer-backend",
        choices=["deepinfer", "fake"],
        default="deepinfer",
        help=argparse.SUPPRESS,
    )

    m = parser.add_argument_group(
        "DeepInfer multi-node",
        "One engine over several nodes: python -m dynamo.deepinfer on the head "
        "(--node-rank 0), python -m dynamo.deepinfer.node on the others.",
    )
    m.add_argument("--nnodes", type=int, default=1, help="Nodes the engine spans.")
    m.add_argument(
        "--node-rank",
        type=int,
        default=0,
        help="This node's index; 0 is the head, which runs the Dynamo worker.",
    )
    m.add_argument(
        "--master-addr",
        default=None,
        help="Head node address for DeepInfer's rendezvous and the lockstep "
        "controller (default: MASTER_ADDR, else 127.0.0.1 on one node).",
    )
    m.add_argument(
        "--master-port",
        type=int,
        default=None,
        help="DeepInfer MASTER_PORT; its gloo store listens on MASTER_PORT-50 "
        f"(default: MASTER_PORT, else {DEFAULT_MASTER_PORT}).",
    )
    m.add_argument(
        "--control-port",
        type=int,
        default=DEFAULT_CONTROL_PORT,
        help="TCP port on the head where ranks on other nodes reach the controller.",
    )
    m.add_argument(
        "--node-timeout-s",
        type=float,
        default=30.0,
        help="A node whose connection stays silent this long is lost (ZMTP heartbeats).",
    )
    m.add_argument(
        "--advertise-addr",
        default=None,
        help="Non-head nodes: address the head uses for this node's KV event "
        "publishers (default: the local address on the route to --master-addr).",
    )


def finalize_options(opts: Any, ns: argparse.Namespace) -> None:
    """Derived defaults and validation, after the namespace was copied onto ``opts``."""
    opts.disaggregation_mode = DisaggregationMode(ns.disaggregation_mode)
    opts.engine_overrides = parse_engine_overrides(ns.engine_arg)
    if ns.kv_transfer is None:
        opts.kv_transfer = (
            "none"
            if opts.disaggregation_mode == DisaggregationMode.AGGREGATED
            else "nixl"
        )
    opts.served_model_name = ns.served_model_name or ns.model
    # Resolved once here: every rank on every node must use the same values.
    master_addr = ns.master_addr or os.environ.get("MASTER_ADDR")
    if not master_addr:
        if opts.nnodes > 1:
            raise ValueError("--nnodes > 1 needs --master-addr (or MASTER_ADDR)")
        master_addr = "127.0.0.1"
    opts.master_addr = master_addr
    opts.master_port = ns.master_port or int(
        os.environ.get("MASTER_PORT") or DEFAULT_MASTER_PORT
    )
    opts.dp_size = ns.dp_size or local_gpu_count() * opts.nnodes
    validate_options(opts)


def validate_options(opts: Any) -> None:
    if opts.dp_size < 1:
        raise ValueError("--dp-size must be >= 1")
    if opts.prefix_host_bytes and opts.kv_cache_manager != "prefix":
        raise ValueError("--prefix-host-bytes needs --kv-cache-manager prefix")
    if opts.prefix_ssd_wait_ms < 0:
        raise ValueError("--prefix-ssd-wait-ms must be >= 0")
    if opts.prefix_ssd_dir and not opts.prefix_ssd_bytes:
        raise ValueError("--prefix-ssd-dir needs --prefix-ssd-bytes")
    if opts.prefix_ssd_bytes:
        if opts.kv_cache_manager != "prefix":
            raise ValueError("--prefix-ssd-bytes needs --kv-cache-manager prefix")
        if opts.prefix_host_bytes <= 0:
            raise ValueError("--prefix-ssd-bytes needs --prefix-host-bytes")
        if not opts.prefix_ssd_dir:
            raise ValueError("--prefix-ssd-bytes needs --prefix-ssd-dir")
    if opts.nnodes < 1 or not 0 <= opts.node_rank < opts.nnodes:
        raise ValueError(
            f"--node-rank {opts.node_rank} is outside 0..{opts.nnodes - 1} (--nnodes)"
        )
    if opts.dp_size % opts.nnodes:
        raise ValueError(
            f"--dp-size {opts.dp_size} is not a multiple of --nnodes {opts.nnodes}"
        )
    disagg = opts.disaggregation_mode != DisaggregationMode.AGGREGATED
    if disagg and opts.kv_transfer == "none":
        raise ValueError("prefill/decode workers need --kv-transfer nixl or local")
    if not disagg and opts.kv_transfer != "none":
        raise ValueError("--kv-transfer applies to prefill/decode workers only")
    if opts.nnodes > 1:
        ports = {
            "MASTER_PORT-50 (DeepInfer's store)": opts.master_port + STORE_PORT_OFFSET,
            "--control-port": opts.control_port,
        }
        if opts.kv_cache_manager != "none":
            base = opts.kv_events_port_base
            ports.update(
                {
                    f"KV events port {p}": p
                    for p in range(base, base + opts.ranks_per_node)
                }
            )
        if len(set(ports.values())) != len(ports):
            raise ValueError(f"DeepInfer ports collide: {ports}")


def deepinfer_env(opts: Any) -> dict[str, str]:
    """``DEEPINFER_*``/UCX defaults the launcher applies with ``setdefault``
    (the pod's own environment wins) before anything imports DeepInfer, plus
    the resolved ``MASTER_ADDR``/``MASTER_PORT``."""
    env = {
        "DEEPINFER_GLOO_TIMEOUT": "600",
        "DEEPINFER_USE_MULTIMODAL_POOL": "0",
        "MASTER_ADDR": opts.master_addr,
        "MASTER_PORT": str(opts.master_port),
    }
    if opts.engine_type == "NORMAL":
        # Lets newly arrived prompts join sooner between decode steps.
        env["DEEPINFER_PREFILL_ATTEMPT_INTERVAL"] = str(opts.prefill_attempt_interval)
    if opts.kv_cache_manager == "prefix":
        env["DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE"] = "compact"
        env["DEEPINFER_YOCO_PREFILL_KEEP_TOKENS"] = "140"
    if opts.kv_transfer == "nixl":
        env.update(
            {
                "UCX_TLS": "cuda_copy,cuda_ipc,tcp,rc",
                "UCX_CUDA_IPC_ENABLE_MNNVL": "y",
                "UCX_MEMTYPE_CACHE": "n",
                "UCX_RCACHE_MAX_UNRELEASED": "1024",
            }
        )
    return env


def apply_env_defaults(
    opts: Any, environ: Optional[MutableMapping[str, str]] = None
) -> None:
    environ = os.environ if environ is None else environ
    for key, value in deepinfer_env(opts).items():
        if key in ("MASTER_ADDR", "MASTER_PORT"):
            # Already resolved from the flags, then the environment.
            environ[key] = value
        else:
            environ.setdefault(key, value)
    if opts.kv_cache_manager == "prefix":
        if environ["DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE"] != "compact":
            raise ValueError(
                "--kv-cache-manager prefix needs DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact"
            )
        extra_state = environ.get("DEEPINFER_PREFILL_STORE_EXTRA_STATE", "0")
        if extra_state not in _DEEPINFER_FALSE:
            raise ValueError(
                "--kv-cache-manager prefix is incompatible with "
                f"DEEPINFER_PREFILL_STORE_EXTRA_STATE={extra_state!r}"
            )
    if opts.kv_transfer == "nixl" and "UCX_NET_DEVICES" not in environ:
        _pin_ucx_rail(environ)


def _pin_ucx_rail(environ: MutableMapping[str, str]) -> None:
    """Put UCX on the rail the pod pinned NCCL to. GKE's rdma-0..3 are
    isolated rails (mlx5_0 cannot reach mlx5_1), and a decode rank may read
    from any prefill rank, so every rank must use the same NIC."""
    hca = environ.get("NCCL_IB_HCA", "").lstrip("=")
    if hca and not hca.startswith("^") and "," not in hca:
        environ["UCX_NET_DEVICES"] = hca if ":" in hca else f"{hca}:1"
    elif len(glob.glob("/dev/infiniband/uverbs*")) > 1:
        logger.warning(
            "Several RDMA devices and no UCX_NET_DEVICES: on isolated rails "
            "(GKE rdma-0..3) set UCX_NET_DEVICES=mlx5_0:1 or NCCL_IB_HCA==mlx5_0"
        )


def local_gpu_count() -> int:
    """GPUs this process may use, without importing torch."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        return len([d for d in visible.split(",") if d.strip()])
    smi = shutil.which("nvidia-smi")
    if smi is not None:
        try:
            out = subprocess.run(
                [smi, "-L"], capture_output=True, text=True, timeout=30, check=True
            ).stdout
            count = sum(line.startswith("GPU ") for line in out.splitlines())
            if count:
                return count
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        return max(1, len(os.listdir("/proc/driver/nvidia/gpus")))
    except OSError:
        return 1


def visible_devices(n: int) -> list[str]:
    """CUDA device ids for this node's first ``n`` ranks: positions in
    ``CUDA_VISIBLE_DEVICES`` (two engines on one node each see their own
    GPUs), else 0..n-1."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    ids = (
        [d.strip() for d in visible.split(",") if d.strip()]
        if visible
        else [str(i) for i in range(n)]
    )
    if len(ids) < n:
        raise ValueError(
            f"{n} DeepInfer ranks on this node need {n} GPUs; "
            f"CUDA_VISIBLE_DEVICES={visible}"
        )
    return ids[:n]


def route_address(peer: str, port: int) -> str:
    """This host's address on the route to ``peer`` (no packet is sent)."""
    try:
        family, _, _, _, addr = socket.getaddrinfo(peer, port, type=socket.SOCK_DGRAM)[
            0
        ]
        with socket.socket(family, socket.SOCK_DGRAM) as s:
            s.connect(addr)
            return s.getsockname()[0]
    except OSError:
        return socket.gethostname()


_SIZE_UNITS = {"": 1, "k": 1 << 10, "m": 1 << 20, "g": 1 << 30, "t": 1 << 40}


def parse_size(text: str) -> int:
    """Bytes: an integer, or a number with K/M/G/T (binary, optional "iB"/"B")."""
    value = text.strip().lower().removesuffix("b").removesuffix("i")
    unit = value[-1:] if value[-1:].isalpha() else ""
    try:
        return int(float(value[: len(value) - len(unit)]) * _SIZE_UNITS[unit])
    except (KeyError, ValueError):
        raise argparse.ArgumentTypeError(f"not a size: {text!r}") from None


def parse_engine_overrides(items: Sequence[str]) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for item in items:
        key, sep, raw = item.partition("=")
        if not sep or not key:
            raise ValueError(f"--engine-arg expects KEY=VALUE, got {item!r}")
        try:
            overrides[key] = json.loads(raw)
        except json.JSONDecodeError:
            overrides[key] = raw
    return overrides
