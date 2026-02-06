# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from prefix_data_generator.sampler import get_cdf

# Protocol-level constants for synthetic data graph structure
SUPER_ROOT = -1  # Dummy node preceding all real nodes; not an actual data root
CACHE_END = -2  # Special node indicating end of a path
END_NODE = -3  # Special node indicating to skip leaf sampling


def _verify_tree(G: nx.DiGraph) -> None:
    invalid_nodes = [(node, d) for node, d in G.in_degree() if d > 1]
    if invalid_nodes:
        print("ERROR: The following nodes have multiple parents (in-degree > 1):")
        for node, in_degree in invalid_nodes:
            parents = list(G.predecessors(node))
            print(f"  Node {node}: in-degree={in_degree}, parents={parents}")
        raise ValueError(
            "Graph is not a valid tree: nodes with multiple parents detected"
        )


def _mark_visited(G: nx.DiGraph) -> None:
    # visits to leaf nodes (non-core branches) are considered as ended
    for node in G.nodes():
        if "to_leaf" not in G.nodes[node]:
            G.nodes[node]["to_leaf"] = 0
        if G.nodes[node]["visited"] <= 1:
            continue
        for child in G.successors(node):
            if G.nodes[child]["visited"] == 1:
                G.nodes[node]["to_leaf"] += 1


def _merge_chains(G: nx.DiGraph) -> nx.DiGraph:
    """
    Make the graph radix-like (meaning all unary paths are contracted).

    This function transforms a prefix tree into a radix tree structure by contracting
    unary paths (chains of nodes with exactly one predecessor and one successor).
    The resulting radix tree is significantly more compact than the original prefix tree,
    as it eliminates redundant intermediate nodes while preserving the structural
    information needed for path sampling.

    This compression is particularly beneficial for efficient path sampling during data
    synthesis. In addition, keep track of the contracted lengths in the 'length' attribute
    of each node to preserve the original path information.

    Args:
        G (networkx.DiGraph): A directed graph representing a prefix tree structure.

    Returns:
        networkx.DiGraph: The resulting radix tree with unary paths contracted.
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

            G.add_edge(
                node_pred, end_node, weight=weight
            )  # may overwrite the edge (should be harmless)
            G.nodes[end_node]["length"] = chain_len

        G.remove_nodes_from(nodes_rm)

    for node in G.nodes():
        if "length" not in G.nodes[node]:
            G.nodes[node]["length"] = 1

    return G


def _remove_leaves(G: nx.DiGraph) -> tuple[nx.DiGraph, list[int]]:
    """
    Remove all nodes that are only visited once from the tree.

    This function removes nodes representing unique user prompts (nodes with visited=1)
    from the radix tree, leaving only the "core radix tree" structure that contains
    commonly traversed paths. The removed nodes typically represent leaf paths that
    were accessed only once and don't contribute to the core structural patterns.

    Args:
        G (networkx.DiGraph): A directed graph representing a radix tree structure.

    Returns:
        tuple[networkx.DiGraph, list[int]]: A tuple containing:
            - The modified graph with unique nodes removed
            - A list of lengths of the removed leaf nodes
    """
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
