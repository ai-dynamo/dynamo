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
import json
import logging
import os
import shutil
import socket
import subprocess
from collections.abc import Iterable, MutableMapping, Sequence
from typing import Any, Optional

from dynamo.deepinfer.rank import RankSpec

DEFAULT_MASTER_PORT = 23333
DEFAULT_CONTROL_PORT = 23400
# DeepInfer's gloo store listens on MASTER_PORT - 50 + engine_group_rank
# (deepinfer/utils.py: get_distributed_init_method).
STORE_PORT_OFFSET = -50

logger = logging.getLogger(__name__)


class DeepInferOptions:
    """DeepInfer settings of one engine, the same on every node."""

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
        """DeepInfer's engine type: aggregated prefill and decode."""
        return "NORMAL"

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
        args.update(self.engine_overrides)
        return args

    def rank_specs(
        self, engine_id: str, ranks: Optional[Iterable[int]] = None
    ) -> list[RankSpec]:
        """Specs of ``ranks`` (default: every rank of the engine)."""
        return [
            RankSpec(
                rank=rank,
                dp_size=self.dp_size,
                engine_args=self.engine_args(rank),
                engine_type=self.engine_type,
                backend=self.deepinfer_backend,
                engine_id=engine_id,
            )
            for rank in (range(self.dp_size) if ranks is None else ranks)
        ]


def add_deepinfer_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("DeepInfer")
    g.add_argument("--model", required=True, help="Checkpoint directory.")
    g.add_argument("--served-model-name", default=None)
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
        help="DEEPINFER_PREFILL_ATTEMPT_INTERVAL default.",
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


def finalize_options(opts: Any, ns: argparse.Namespace) -> None:
    """Derived defaults and validation, after the namespace was copied onto ``opts``."""
    opts.engine_overrides = parse_engine_overrides(ns.engine_arg)
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
    if opts.nnodes < 1 or not 0 <= opts.node_rank < opts.nnodes:
        raise ValueError(
            f"--node-rank {opts.node_rank} is outside 0..{opts.nnodes - 1} (--nnodes)"
        )
    if opts.dp_size % opts.nnodes:
        raise ValueError(
            f"--dp-size {opts.dp_size} is not a multiple of --nnodes {opts.nnodes}"
        )
    if opts.nnodes > 1:
        ports = {
            "MASTER_PORT-50 (DeepInfer's store)": opts.master_port + STORE_PORT_OFFSET,
            "--control-port": opts.control_port,
        }
        if len(set(ports.values())) != len(ports):
            raise ValueError(f"DeepInfer ports collide: {ports}")


def deepinfer_env(opts: Any) -> dict[str, str]:
    """``DEEPINFER_*`` defaults the launcher applies with ``setdefault`` (the
    pod's own environment wins) before anything imports DeepInfer, plus the
    resolved ``MASTER_ADDR``/``MASTER_PORT``."""
    return {
        "DEEPINFER_GLOO_TIMEOUT": "600",
        "DEEPINFER_USE_MULTIMODAL_POOL": "0",
        # Lets newly arrived prompts join sooner between decode steps.
        "DEEPINFER_PREFILL_ATTEMPT_INTERVAL": str(opts.prefill_attempt_interval),
        "MASTER_ADDR": opts.master_addr,
        "MASTER_PORT": str(opts.master_port),
    }


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
