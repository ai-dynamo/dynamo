# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for SglangPostProcessor."""


from dynamo.sglang.postprocessor import SglangPostProcessor


class TestSglangPostProcessorBasic:
    """Tests without SGLang parsers (no tool/reasoning parsing)."""

    def test_basic_text_delta(self):
        pp = SglangPostProcessor()
        chunk = pp.process_chunk(
            {
                "text": "Hello",
                "output_ids": [1],
                "meta_info": {"finish_reason": None},
                "index": 0,
            },
            is_final=False,
        )
        assert chunk is not None
        assert chunk["text"] == "Hello"
        assert chunk["index"] == 0
        assert "finish_reason" not in chunk

    def test_cumulative_text_delta(self):
        pp = SglangPostProcessor()
        pp.process_chunk(
            {
                "text": "Hello",
                "output_ids": [1],
                "meta_info": {"finish_reason": None},
                "index": 0,
            },
            is_final=False,
        )
        chunk = pp.process_chunk(
            {
                "text": "Hello world",
                "output_ids": [2],
                "meta_info": {"finish_reason": None},
                "index": 0,
            },
            is_final=False,
        )
        assert chunk is not None
        assert chunk["text"] == " world"

    def test_final_chunk_with_usage(self):
        pp = SglangPostProcessor()
        chunk = pp.process_chunk(
            {
                "text": "Done",
                "output_ids": [1],
                "meta_info": {
                    "finish_reason": {"type": "stop", "matched": None},
                    "prompt_tokens": 10,
                    "completion_tokens": 5,
                },
                "index": 0,
            },
            is_final=True,
        )
        assert chunk is not None
        assert chunk["finish_reason"] == "stop"
        assert chunk["completion_usage"]["prompt_tokens"] == 10
        assert chunk["completion_usage"]["completion_tokens"] == 5
        assert chunk["completion_usage"]["total_tokens"] == 15

    def test_empty_delta_skipped(self):
        pp = SglangPostProcessor()
        pp.process_chunk(
            {
                "text": "Hello",
                "output_ids": [1],
                "meta_info": {"finish_reason": None},
                "index": 0,
            },
            is_final=False,
        )
        chunk = pp.process_chunk(
            {
                "text": "Hello",
                "output_ids": [],
                "meta_info": {"finish_reason": None},
                "index": 0,
            },
            is_final=False,
        )
        assert chunk is None

    def test_reset_clears_state(self):
        pp = SglangPostProcessor()
        pp.process_chunk(
            {
                "text": "Hello",
                "output_ids": [1],
                "meta_info": {"finish_reason": None},
                "index": 0,
            },
            is_final=False,
        )
        pp.reset()
        chunk = pp.process_chunk(
            {
                "text": "Hello",
                "output_ids": [1],
                "meta_info": {"finish_reason": None},
                "index": 0,
            },
            is_final=False,
        )
        assert chunk is not None
        assert chunk["text"] == "Hello"

    def test_final_chunk_emitted_even_without_content(self):
        pp = SglangPostProcessor()
        pp.process_chunk(
            {
                "text": "Hello",
                "output_ids": [1],
                "meta_info": {"finish_reason": None},
                "index": 0,
            },
            is_final=False,
        )
        # Final chunk with same cumulative text (empty delta)
        chunk = pp.process_chunk(
            {
                "text": "Hello",
                "output_ids": [],
                "meta_info": {
                    "finish_reason": {"type": "length", "matched": None},
                    "prompt_tokens": 5,
                    "completion_tokens": 1,
                },
                "index": 0,
            },
            is_final=True,
        )
        assert chunk is not None
        assert chunk["finish_reason"] == "length"

    def test_multi_index_tracking(self):
        pp = SglangPostProcessor()
        c0 = pp.process_chunk(
            {
                "text": "A",
                "output_ids": [1],
                "meta_info": {"finish_reason": None},
                "index": 0,
            },
            is_final=False,
        )
        c1 = pp.process_chunk(
            {
                "text": "B",
                "output_ids": [2],
                "meta_info": {"finish_reason": None},
                "index": 1,
            },
            is_final=False,
        )
        assert c0["text"] == "A"
        assert c0["index"] == 0
        assert c1["text"] == "B"
        assert c1["index"] == 1
