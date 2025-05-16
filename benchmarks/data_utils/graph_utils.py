# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import networkx as nx
import numpy as np

from benchmarks.data_utils.protocols import CACHE_END, END_NODE, SUPER_ROOT
from benchmarks.data_utils.sampler import get_cdf


def _merge_chains(G: nx.DiGraph) -> nx.DiGraph:
    """Make the graph radix-like (meaning all unary paths are contracted).
    In addition, keep track of the contracted lengths.

    Args:
        G (networkx.DiGraph): A directed graph representing a prefix tree structure.

    Returns:
        networkx.DiGraph: The modified graph with unary paths contracted.
    """
    for visited in sorted(np.unique([G.nodes[node]["visited"] for node in G.nodes()])):
        sub_nodes = [node for node in G.nodes() if G.nodes[node]["visited"] == visited]
        subgraph = G.subgraph(sub_nodes)
        if len(subgraph) == 1:
            continue

        chain_nodes = [
            node
            for node in subgraph.nodes()
            if G.in_degree(node) == 1 and G.out_degree(node) == 1
        ]
        if not chain_nodes:
            continue
        chain_nodes = sorted(chain_nodes)

        nodes_rm = []
        for node in chain_nodes:
            node_pred = list(G.predecessors(node))[0]
            # find the parent node source
            if G.nodes[node_pred]["visited"] == visited and node_pred != SUPER_ROOT:
                continue
            weight = G[node_pred][node]["weight"]

            end_node = node
            chain_len = 1
            succ = list(G.successors(end_node))

            # find the end of the chain
            while succ and G.nodes[succ[0]]["visited"] == visited:
                nodes_rm.append(end_node)
                end_node = succ[0]
                chain_len += 1
                succ = list(G.successors(end_node))

            G.add_edge(node_pred, end_node, weight=weight)
            G.nodes[end_node]["length"] = chain_len

        G.remove_nodes_from(nodes_rm)

    for node in G.nodes():
        if "length" not in G.nodes[node]:
            G.nodes[node]["length"] = 1

    return G


def _remove_leaves(G: nx.DiGraph) -> tuple[nx.DiGraph, list[int]]:
    leaves = {
        node: G.nodes[node]["length"]
        for node in G.nodes()
        if G.nodes[node]["visited"] == 1
    }
    leaves_id = list(leaves.keys())
    leaves_len = list(leaves.values())
    G.remove_nodes_from(leaves_id)
    return G, leaves_len


def _precompute_transition_cdfs(G: nx.DiGraph) -> nx.DiGraph:
    for node in G.nodes():
        out_edges = list(G.out_edges(node))
        weights = [G[edge[0]][edge[1]]["weight"] for edge in out_edges] + [
            G.nodes[node]["to_leaf"],
            G.nodes[node]["end"],
        ]
        G.nodes[node]["out_cdf"] = get_cdf(weights)
        G.nodes[node]["out_nodes"] = [edge[1] for edge in out_edges] + [
            CACHE_END,
            END_NODE,
        ]

    return G


def _validate_graph(G: nx.DiGraph) -> bool:
    for node in G.nodes():
        # Skip nodes without parents or children
        if G.in_degree(node) == 0 or G.out_degree(node) == 0:
            continue

        # Get incoming edge weight (should only be one parent)
        parent = list(G.predecessors(node))[0]
        in_weight = G[parent][node]["weight"]

        # Sum outgoing edge weights
        out_weights = [G[node][child]["weight"] for child in G.successors(node)]
        out_weights += [G.nodes[node]["to_leaf"], G.nodes[node]["end"]]

        # Compare weights (using np.isclose for float comparison)
        if not in_weight == sum(out_weights):
            raise ValueError(
                f"Weight mismatch at node {node}: "
                f"incoming weight {in_weight} != sum of outgoing weights {sum(out_weights)}"
            )

    return True
