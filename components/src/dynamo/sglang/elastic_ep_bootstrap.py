# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Template-invariant bootstrap for the growth-only, one-GPU SGLang proof.

Grove supplies stable allocation identity; it does not interpret engine arguments.
This isolated compatibility launcher supports only EP1 -> EP2 on the merged
SGLang scale path. Allocation zero serves through Dynamo, while allocation one
starts a bare engine joiner and never registers an independent Dynamo endpoint.
Replacement, recovery, packed ranks, and multi-allocation initial formation are
not supported. Upstream automatic bootstrap can replace the argument translation
once its lifecycle contract is available.
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
    parser.add_argument(
        "--ep-size", "--expert-parallel-size", "--ep", type=int, default=1
    )
    parser.add_argument(
        "--pp-size", "--pipeline-parallel-size", "--pp", type=int, default=1
    )
    parser.add_argument(
        "--attn-cp-size", "--attention-context-parallel-size", type=int, default=1
    )
    parser.add_argument("--nnodes", type=int, default=1)
    parser.add_argument("--node-rank", type=int)
    parser.add_argument("--elastic-ep-initial-size", type=int)
    parser.add_argument("--max-ep-size", type=int)
    parser.add_argument("--elastic-ep-join-mode")
    parser.add_argument("--elastic-ep-join-rank-offset", type=int)
    parser.add_argument("--dist-init-addr", required=True)
    parser.add_argument("--enable-dp-attention", action="store_true")
    parser.add_argument("--enable-dp-lm-head", action="store_true")
    geometry, remaining = parser.parse_known_args(list(arguments))
    if (
        geometry.tp_size != 1
        or geometry.dp_size != 1
        or geometry.ep_size != 1
        or geometry.pp_size != 1
        or geometry.attn_cp_size != 1
        or geometry.nnodes != 1
        or geometry.elastic_ep_initial_size != 1
        or geometry.max_ep_size != 2
        or not geometry.enable_dp_attention
        or not geometry.enable_dp_lm_head
    ):
        raise ValueError(
            "Grove bootstrap supports only the declared one-GPU EP1 -> EP2 profile"
        )
    if (
        geometry.node_rank is not None
        or geometry.elastic_ep_join_mode is not None
        or geometry.elastic_ep_join_rank_offset is not None
    ):
        raise ValueError("Grove bootstrap owns node rank and joining arguments")

    # Native injected identity, rather than a second rank flag, determines the role.
    index = environment["GROVE_PCLQ_POD_INDEX"]
    if index not in ("0", "1"):
        raise ValueError("Grove bootstrap requires allocation index 0 or 1")
    clique = environment["GROVE_PCLQ_NAME"]
    service = environment["GROVE_HEADLESS_SERVICE"]
    if not clique or not service:
        raise ValueError("Grove clique name and headless service are required")
    address = urlsplit("//" + geometry.dist_init_addr)
    port = address.port
    if not address.hostname or port is None or port <= 0:
        raise ValueError("--dist-init-addr must declare a valid rendezvous port")
    rendezvous = f"{clique}-0.{service}:{port}"

    # Both roles preserve the storage layout and communication options from one template.
    common = [
        "--tp",
        "1",
        "--dp",
        "1",
        "--ep",
        "1",
        "--pp",
        "1",
        "--attn-cp-size",
        "1",
        "--enable-dp-attention",
        "--enable-dp-lm-head",
        "--elastic-ep-initial-size",
        "1",
        "--max-ep-size",
        "2",
        "--dist-init-addr",
        rendezvous,
    ]
    if index == "0":
        return [executable, "-m", "dynamo.sglang", *remaining, *common, "--nnodes", "1"]

    # A scale joiner uses SGLang's native non-primary launch, not Dynamo discovery.
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
        "1",
    ]


def main() -> None:
    """Replace the launcher with the chosen engine process so Kubernetes owns its lifecycle."""
    command = resolve_command(sys.argv[1:], os.environ, sys.executable)
    os.execv(command[0], command)


if __name__ == "__main__":
    main()
