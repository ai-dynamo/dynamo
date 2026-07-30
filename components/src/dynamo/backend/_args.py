# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Argparse helpers for turning CLI flags into a :class:`WorkerConfig`.

A backend's ``from_args`` declares only its engine-specific flags;
``add_worker_args`` contributes the runtime/discovery flags every Dynamo
backend shares, and ``build_worker_config`` assembles the ``WorkerConfig``
from the parsed namespace. This is the ergonomic path for a self-contained
backend. Feature-rich backends (vLLM, SGLang, TRT-LLM, TokenSpeed) instead
use the fuller ``dynamo.common.configuration`` ArgGroup machinery, which
`WorkerConfig.from_runtime_config` also feeds.
"""

from __future__ import annotations

import argparse
from typing import Optional

from dynamo.common.constants import DisaggregationMode

from ._worker import WorkerConfig


def add_worker_args(
    parser: argparse.ArgumentParser,
    *,
    default_component: str = "backend",
    default_endpoint_types: str = "chat,completions",
) -> argparse.ArgumentParser:
    """Add the runtime/discovery flags every Dynamo backend needs.

    Contributes ``--namespace``, ``--component``, ``--endpoint``,
    ``--endpoint-types``, ``--discovery-backend``, ``--request-plane``, and
    ``--event-plane`` to ``parser``. These are modality-agnostic, so they fit an
    aggregated LLM backend and a raw-media backend alike. Disaggregation is
    LLM-specific — a backend that supports prefill/decode/encode declares its own
    ``--disaggregation-mode`` / ``--route-to-encoder`` (``build_worker_config``
    picks them up from the namespace). Returns ``parser`` for chaining.
    """
    g = parser.add_argument_group("Dynamo runtime")
    g.add_argument(
        "--namespace", default="dynamo", help="Dynamo namespace to register under."
    )
    g.add_argument(
        "--component", default=default_component, help="Dynamo component name."
    )
    g.add_argument("--endpoint", default="generate", help="Dynamo endpoint name.")
    g.add_argument(
        "--endpoint-types",
        default=default_endpoint_types,
        help="Comma-separated endpoint types to serve (e.g. 'chat,completions').",
    )
    g.add_argument(
        "--discovery-backend",
        default="etcd",
        choices=["kubernetes", "etcd", "file", "mem"],
        help="Discovery backend used for model registration.",
    )
    g.add_argument(
        "--request-plane",
        default="tcp",
        choices=["tcp", "nats"],
        help="Transport for router-to-worker request streaming ('tcp' is fastest).",
    )
    g.add_argument(
        "--event-plane",
        default=None,
        choices=["nats", "zmq"],
        help="Transport for runtime events. Defaults to 'zmq' when unset.",
    )
    return parser


def build_worker_config(
    args: argparse.Namespace,
    *,
    model_name: str,
    served_model_name: Optional[str] = None,
    **overrides,
) -> WorkerConfig:
    """Build a :class:`WorkerConfig` from a namespace produced by
    :func:`add_worker_args`.

    ``model_name`` is engine-specific and not a runtime flag, so it is passed
    explicitly. ``served_model_name`` defaults to ``model_name``. Any keyword in
    ``overrides`` sets the matching ``WorkerConfig`` field, taking precedence
    over the parsed flags — pass ``enable_kv_routing=...`` and similar here.
    """
    if "disaggregation_mode" not in overrides:
        mode = getattr(args, "disaggregation_mode", None)
        if isinstance(mode, str):
            overrides["disaggregation_mode"] = DisaggregationMode(mode)
    return WorkerConfig.from_runtime_config(
        args,
        model_name=model_name,
        served_model_name=served_model_name or model_name,
        **overrides,
    )
