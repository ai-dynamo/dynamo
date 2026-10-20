# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Template-invariant bootstrap for growth-only, one-GPU SGLang allocations.

Grove supplies stable allocation identity; it does not interpret engine arguments.
This isolated compatibility launcher derives initial participants and append
joiners from the declared initial size and Grove slot. Initial participants form
one complete world; each later allocation starts a single-rank bare joiner and
never registers an independent Dynamo endpoint. The legacy protocol requires
one append allocation per resize. Replacement, recovery and packed ranks are
not supported. Upstream automatic bootstrap can replace this translation once
its lifecycle contract is available.
"""

import argparse
import os
import sys
from collections.abc import Mapping, Sequence
from urllib.parse import urlsplit


def resolve_command(
    arguments: Sequence[str], environment: Mapping[str, str], executable: str
) -> list[str]:
    """Resolve a native Grove slot to an exec-form command without mutating inputs."""
    # Interpret only the geometry owned by this explicit, narrow launch contract.
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--tp-size", "--tensor-parallel-size", "--tp", type=int)
    parser.add_argument("--dp-size", "--data-parallel-size", "--dp", type=int)
    parser.add_argument("--ep-size", "--expert-parallel-size", "--ep", type=int)
    parser.add_argument(
        "--pp-size", "--pipeline-parallel-size", "--pp", type=int, default=1
    )
    parser.add_argument(
        "--attn-cp-size", "--attention-context-parallel-size", type=int, default=1
    )
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument(
        "--moe-dp-size", "--moe-data-parallel-size", type=int, default=1
    )
    parser.add_argument("--moe-dense-tp-size", type=int, default=1)
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--elastic-ep-initial-size", type=int)
    parser.add_argument("--max-ep-size", type=int)
    parser.add_argument("--elastic-ep-join-mode")
    parser.add_argument("--elastic-ep-join-rank-offset", type=int)
    parser.add_argument("--dist-init-addr", required=True)
    parser.add_argument("--enable-dp-attention", action="store_true")
    parser.add_argument("--enable-dp-lm-head", action="store_true")
    geometry, remaining = parser.parse_known_args(list(arguments))
    initial_size = (
        geometry.elastic_ep_initial_size
        if geometry.elastic_ep_initial_size is not None
        else geometry.tp_size
    )
    maximum_size = geometry.max_ep_size
    ep_size = geometry.ep_size if geometry.ep_size is not None else geometry.tp_size
    if (
        initial_size is None
        or initial_size <= 1
        or maximum_size is None
        or maximum_size <= initial_size
        or geometry.tp_size != initial_size
        or geometry.dp_size != initial_size
        or ep_size != initial_size
        or geometry.pp_size != 1
        or geometry.attn_cp_size != 1
        or geometry.nnodes != initial_size
        or geometry.moe_dp_size != 1
        or geometry.moe_dense_tp_size != 1
        or not geometry.enable_dp_attention
        or not geometry.enable_dp_lm_head
    ):
        raise ValueError(
            "Grove bootstrap requires a multi-rank initial world with "
            "TP=DP=EP=nnodes=initial size, a larger maximum, and "
            "one-GPU-per-pod DP attention with local dense TP"
        )
    if (
        geometry.node_rank is not None
        or geometry.elastic_ep_join_mode is not None
        or geometry.elastic_ep_join_rank_offset is not None
    ):
        raise ValueError("Grove bootstrap owns node rank and joining arguments")

    # Native injected identity, rather than a second rank flag, determines the role.
    slot = environment["GROVE_PCLQ_POD_INDEX"]
    try:
        index = int(slot)
    except ValueError as error:
        raise ValueError("Grove allocation index must be an integer") from error
    if str(index) != slot or not 0 <= index < maximum_size:
        raise ValueError(
            "Grove allocation index must be canonical and below max EP size"
        )
    clique = environment["GROVE_PCLQ_NAME"]
    service = environment["GROVE_HEADLESS_SERVICE"]
    if not clique or not service:
        raise ValueError("Grove clique name and headless service are required")
    address = urlsplit("//" + geometry.dist_init_addr)
    port = address.port
    if not address.hostname or port is None or port <= 0:
        raise ValueError("--dist-init-addr must declare a valid rendezvous port")
    rendezvous = f"{clique}-0.{service}:{port}"

    # Initial participants share global geometry; the append joiner owns one local rank.
    initial_participant = index < initial_size
    width = str(initial_size) if initial_participant else "1"
    common = [
        "--tp",
        width,
        "--dp",
        width,
        "--ep",
        width,
        "--pp-size",
        "1",
        "--attn-cp-size",
        "1",
        "--moe-dense-tp-size",
        "1",
        "--moe-dp-size",
        "1",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--elastic-ep-initial-size",
        str(initial_size),
        "--max-ep-size",
        str(maximum_size),
        "--dist-init-addr",
        rendezvous,
    ]
    if initial_participant:
        return [
            executable,
            "-m",
            "dynamo.sglang",
            *remaining,
            *common,
            "--nnodes",
            str(initial_size),
            "--node-rank",
            str(index),
        ]

    # SGLang's bare-joiner convention uses nnodes=2/node-rank=1 independently
    # of the world size. It does not register a new Dynamo endpoint.
    return [
        executable,
        "-m",
        "sglang.launch_server",
        *remaining,
        *common,
        "--nnodes",
        "2",
        "--node-rank",
        "1",
        "--elastic-ep-join-mode",
        "scale",
        "--elastic-ep-join-rank-offset",
        str(index),
    ]


def main() -> None:
    """Replace the launcher with the chosen engine process so Kubernetes owns its lifecycle."""
    command = resolve_command(sys.argv[1:], os.environ, sys.executable)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
