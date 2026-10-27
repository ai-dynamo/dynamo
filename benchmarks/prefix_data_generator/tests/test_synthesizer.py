# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from prefix_data_generator.synthesizer import Synthesizer

pytestmark = [pytest.mark.post_merge, pytest.mark.gpu_0, pytest.mark.unit]


@pytest.fixture
def make_trace(tmp_path):
    def write(paths, *, input_lengths=None, output_lengths=None, timestamps=None):
        records = [
            {
                "hash_ids": path,
                "input_length": (
                    input_lengths[index]
                    if input_lengths is not None
                    else 64 * len(path)
                ),
                "output_length": (
                    output_lengths[index] if output_lengths is not None else 16
                ),
                "timestamp": timestamps[index] if timestamps is not None else index,
            }
            for index, path in enumerate(paths)
        ]
        trace = tmp_path / "trace.jsonl"
        trace.write_text("".join(json.dumps(record) + "\n" for record in records))
        return str(trace)

    return write


def test_graph_preserves_branching_and_early_termination(make_trace):
    trace = make_trace(
        [
            [0, 1],
            [0, 1, 2, 3, 4],
            [0, 1, 2, 3, 4, 5, 6],
            [7, 8],
            [7, 8, 9, 10],
            [11, 12],
        ]
    )
    synthesizer = Synthesizer(trace, block_size=64)
    graph = synthesizer.G
    assert set(graph.successors(-1)) == {1, 8}
    assert list(graph.successors(1)) == [4]
    assert graph.nodes[1]["length"] == 2
    assert graph.nodes[4]["length"] == 3
    assert graph.nodes[8]["length"] == 2
    assert graph.nodes[-1]["to_leaf"] == 1
    assert graph.nodes[4]["to_leaf"] == 1
    assert graph.nodes[8]["to_leaf"] == 1

    requests = synthesizer.synthesize_requests(128)
    assert {request["context_len"] for request in requests} == {0, 128, 320}
    assert any(request["unique_user_prompt_len"] == 0 for request in requests)
    assert any(request["unique_user_prompt_len"] > 0 for request in requests)


@pytest.mark.parametrize("factor,core_blocks", [(0.5, 1), (1.5, 3)])
def test_prefix_scaling_preserves_shared_and_disjoint_paths(
    make_trace, factor, core_blocks
):
    trace = make_trace([[10, 11, 12], [10, 11, 13], [20, 21, 22], [20, 21, 23]])
    requests = Synthesizer(
        trace, block_size=64, prefix_len_multiplier=factor, seed=0
    ).synthesize_requests(128)

    prefixes = {tuple(request["hash_ids"][:core_blocks]) for request in requests}
    assert len(prefixes) == 2
    left, right = prefixes
    assert set(left).isdisjoint(right)
    assert all(len(set(prefix)) == core_blocks for prefix in prefixes)
    core_ids = set(left) | set(right)
    leaf_ids = [request["hash_ids"][-1] for request in requests]
    assert len(set(leaf_ids)) == len(requests)
    assert core_ids.isdisjoint(leaf_ids)
    assert all(request["context_len"] == core_blocks * 64 for request in requests)
    assert all(
        request["input_length"] == (core_blocks + 1) * 64 for request in requests
    )
    assert all(len(request["hash_ids"]) == core_blocks + 1 for request in requests)


def test_prompt_scaling_changes_only_unique_suffixes(make_trace):
    trace = make_trace([[0, 1, 2, 3], [0, 1, 4, 5]])
    requests = Synthesizer(
        trace, block_size=64, prompt_len_multiplier=0.5
    ).synthesize_requests(32)
    assert len({tuple(request["hash_ids"][:2]) for request in requests}) == 1
    assert all(request["context_len"] == 128 for request in requests)
    assert all(request["unique_user_prompt_len"] == 64 for request in requests)
    assert all(len(request["hash_ids"]) == 3 for request in requests)


def test_single_request_trace_generates_unique_partial_prompts(make_trace):
    trace = make_trace([[91, 92]], input_lengths=[101])
    requests = Synthesizer(trace, block_size=64).synthesize_requests(8)
    assert all(request["input_length"] == 101 for request in requests)
    assert all(request["context_len"] == 0 for request in requests)
    assert all(request["unique_user_prompt_len"] == 101 for request in requests)
    assert (
        len({hash_id for request in requests for hash_id in request["hash_ids"]}) == 16
    )
    assert {request["timestamp"] for request in requests} == {0}


def test_repeated_terminal_paths_preserve_their_partial_lengths(make_trace):
    trace = make_trace([[5], [5], [9, 10], [9, 10]], input_lengths=[37, 37, 101, 101])
    requests = Synthesizer(trace, block_size=64).synthesize_requests(64)
    assert {request["input_length"] for request in requests} == {37, 101}
    for request in requests:
        expected_length = 37 if len(request["hash_ids"]) == 1 else 101
        assert request["input_length"] == expected_length
        assert request["context_len"] == expected_length
        assert request["unique_user_prompt_len"] == 0


def test_full_terminal_paths_keep_normalized_hash_ids(make_trace):
    trace = make_trace([[5, 6], [5, 6]], timestamps=[1000, 1000])
    requests = Synthesizer(trace, block_size=64).synthesize_requests(2)
    assert [request["hash_ids"] for request in requests] == [[0, 1], [0, 1]]
    assert all(request["input_length"] == 128 for request in requests)


def test_copy_namespaces_are_stable_and_disjoint_across_batches(make_trace):
    trace = make_trace([[0, 1], [0, 2]])
    synthesizer = Synthesizer(trace, block_size=64, prefix_root_multiplier=3, seed=0)
    first = synthesizer.synthesize_requests(64)
    second = synthesizer.synthesize_requests(64)
    first_roots = {request["hash_ids"][0] for request in first}
    second_roots = {request["hash_ids"][0] for request in second}
    assert len(first_roots) == 3
    assert first_roots == second_roots
    leaves = [request["hash_ids"][1] for request in first + second]
    assert len(set(leaves)) == len(leaves)
    assert first_roots.isdisjoint(leaves)


def test_seed_reproduces_complete_batches_without_coupling_samplers(make_trace):
    trace = make_trace([[0, 1], [0, 2, 3]], output_lengths=[10, 20])
    first = Synthesizer(trace, block_size=64, prefix_root_multiplier=2, seed=37)
    second = Synthesizer(trace, block_size=64, prefix_root_multiplier=2, seed=37)
    first_batch = first.synthesize_requests(128)

    def jsonl_bytes(rows):
        return "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows).encode()

    assert jsonl_bytes(first_batch) == jsonl_bytes(second.synthesize_requests(128))
    assert jsonl_bytes(first.synthesize_requests(16)) == jsonl_bytes(
        second.synthesize_requests(16)
    )
    assert {
        (request["input_length"], request["output_length"]) for request in first_batch
    } == {(128, 10), (128, 20), (192, 10), (192, 20)}


def test_fractional_timestamps_and_speedup_preserve_intervals(make_trace):
    trace = make_trace([[0], [0], [0]], timestamps=[1000, 1000.25, 1000.5])
    requests = Synthesizer(trace, block_size=64, speedup_ratio=2).synthesize_requests(4)
    assert [request["timestamp"] for request in requests] == [0, 0.125, 0.25, 0.375]
