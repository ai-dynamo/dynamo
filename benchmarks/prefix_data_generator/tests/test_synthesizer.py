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

import json
import os
import random
import tempfile
import unittest

from prefix_data_generator.synthesizer import Synthesizer


# Helper function to create and dump data
def dump_record(handle, hash_ids, block_size=512):
    input_length = block_size * len(hash_ids)
    output_length = random.randint(50, 250)

    data = {
        "timestamp": 1000,
        "hash_ids": hash_ids,
        "input_length": input_length,
        "output_length": output_length,
    }
    json.dump(data, handle)
    handle.write("\n")


def check_attributes(
    graph,
    node,
    expected_children,
    expected_visited=None,
    expected_length=None,
    expected_to_leaf=None,
):
    # Check children
    actual_children = list(graph.successors(node))
    assert sorted(actual_children) == sorted(
        expected_children
    ), f"Node {node} has children {actual_children}, expected {expected_children}"

    # Check 'visited' attribute if expected
    if expected_visited is not None:
        assert (
            graph.nodes[node].get("visited") == expected_visited
        ), f"Node {node} has 'visited' value {graph.nodes[node].get('visited')}, expected {expected_visited}"

    # Check 'length' attribute if expected
    if expected_length is not None:
        assert (
            graph.nodes[node].get("length") == expected_length
        ), f"Node {node} has 'length' value {graph.nodes[node].get('length')}, expected {expected_length}"

    # Check 'to_leaf' attribute if expected
    if expected_to_leaf is not None:
        assert (
            graph.nodes[node].get("to_leaf") == expected_to_leaf
        ), f"Node {node} has 'to_leaf' value {graph.nodes[node].get('to_leaf')}, expected {expected_to_leaf}"

    return True


def test_graph_structure():
    # Create a temporary JSONL file with the specified data
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        dump_record(tmp, [0, 1])
        dump_record(tmp, [0, 1, 2, 3, 4])
        dump_record(tmp, [0, 1, 2, 3, 4, 5, 6])
        dump_record(tmp, [7, 8])
        dump_record(tmp, [7, 8, 9, 10])
        dump_record(tmp, [11, 12])

    # Create the Synthesizer with the temporary file
    synthesizer = Synthesizer(tmp.name, block_size=512)
    G = synthesizer.G

    # Verify the graph structure
    check_attributes(G, -1, [1, 8], 6, None, 1)
    check_attributes(G, 1, [4], 3, 2, 0)
    check_attributes(G, 4, [], 2, 3, 1)
    check_attributes(G, 8, [], 2, 2, 1)

    # Clean up
    os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
