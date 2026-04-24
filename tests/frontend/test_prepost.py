#  SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#  SPDX-License-Identifier: Apache-2.0

"""Unit test for StreamingPostProcessor with Qwen3 reasoning + Hermes tool calling."""

# mypy seems to be running both sides of the HAS_VLLM if statement
# mypy: ignore-errors

import json

import pytest

from .common import check_module_available

HAS_VLLM = check_module_available("vllm")
if HAS_VLLM:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
        ChatCompletionToolsParam,
    )
    from vllm.entrypoints.openai.engine.protocol import FunctionDefinition
    from vllm.outputs import CompletionOutput
    from vllm.reasoning.qwen3_reasoning_parser import Qwen3ReasoningParser
    from vllm.sampling_params import SamplingParams
    from vllm.tool_parsers.hermes_tool_parser import Hermes2ProToolParser

    from dynamo.frontend.prepost import StreamingPostProcessor
else:
    # Fake some types so that `pre-commit` passes
    class CompletionOutput:
        def __init__(*args, **kwargs):
            pass


pytestmark = [
    pytest.mark.vllm,
    pytest.mark.gpu_0,  # "Hardware"
    pytest.mark.pre_merge,  # "Lifecyle"
    pytest.mark.unit,  # "Test Type"
    pytest.mark.skipif(not HAS_VLLM, reason="requires vllm"),
]


# ---------------------------------------------------------------------------
# Mock tokenizer mimicking CachedQwen2TokenizerFast for Qwen3-0.6B
# ---------------------------------------------------------------------------
class MockQwen3Tokenizer:
    """Minimal tokenizer mock with the tokens needed for this test."""

    def __init__(self):
        self._vocab = {
            "<|endoftext|>": 151643,
            "<|im_start|>": 151644,
            "<|im_end|>": 151645,
            "<|object_ref_start|>": 151646,
            "<|object_ref_end|>": 151647,
            "<|box_start|>": 151648,
            "<|box_end|>": 151649,
            "<|quad_start|>": 151650,
            "<|quad_end|>": 151651,
            "<|vision_start|>": 151652,
            "<|vision_end|>": 151653,
            "<|vision_pad|>": 151654,
            "<|image_pad|>": 151655,
            "<|video_pad|>": 151656,
            "<tool_call>": 151657,
            "</tool_call>": 151658,
            "<tool_response>": 151665,
            "</tool_response>": 151666,
            "<think>": 151667,
            "</think>": 151668,
        }
        self._id_to_token = {v: k for k, v in self._vocab.items()}
        self.all_special_tokens = [
            "<|endoftext|>",
            "<|im_start|>",
            "<|im_end|>",
            "<|object_ref_start|>",
            "<|object_ref_end|>",
            "<|box_start|>",
            "<|box_end|>",
            "<|quad_start|>",
            "<|quad_end|>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|vision_pad|>",
            "<|image_pad|>",
            "<|video_pad|>",
        ]

    def get_vocab(self):
        return dict(self._vocab)

    def encode(self, text, add_special_tokens=False):
        if text in self._vocab:
            return [self._vocab[text]]
        raise ValueError(f"Cannot encode unknown text: {text!r}")

    def decode(self, token_ids):
        return "".join(self._id_to_token.get(tid, f"<unk:{tid}>") for tid in token_ids)


# ---------------------------------------------------------------------------
# Test data: stream_interval=1 (one token per output)
# ---------------------------------------------------------------------------
OUTPUTS_INTERVAL_1 = [
    CompletionOutput(
        index=0,
        text="<think>",
        token_ids=[151667],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="\n",
        token_ids=[198],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="Okay",
        token_ids=[32313],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=",",
        token_ids=[11],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" the",
        token_ids=[279],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" user",
        token_ids=[1196],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" is",
        token_ids=[374],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" asking",
        token_ids=[10161],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" for",
        token_ids=[369],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" the",
        token_ids=[279],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" titles",
        token_ids=[15311],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" of",
        token_ids=[315],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" some",
        token_ids=[1045],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" James",
        token_ids=[7801],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" Joyce",
        token_ids=[53626],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" books",
        token_ids=[6467],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" and",
        token_ids=[323],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" wants",
        token_ids=[6801],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" me",
        token_ids=[752],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" to",
        token_ids=[311],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" use",
        token_ids=[990],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" the",
        token_ids=[279],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" provided",
        token_ids=[3897],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" tool",
        token_ids=[5392],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=".\n",
        token_ids=[624],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="</think>",
        token_ids=[151668],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="\n\n",
        token_ids=[271],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="<tool_call>",
        token_ids=[151657],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="\n",
        token_ids=[198],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='{"',
        token_ids=[4913],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="name",
        token_ids=[606],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='":',
        token_ids=[788],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=' "',
        token_ids=[330],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="search",
        token_ids=[1836],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="_g",
        token_ids=[1889],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="utenberg",
        token_ids=[44433],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="_books",
        token_ids=[73084],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='",',
        token_ids=[497],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=' "',
        token_ids=[330],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="arguments",
        token_ids=[16370],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='":',
        token_ids=[788],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=' {"',
        token_ids=[5212],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="search",
        token_ids=[1836],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="_terms",
        token_ids=[37498],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='":',
        token_ids=[788],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=' ["',
        token_ids=[4383],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="James",
        token_ids=[28084],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" Joyce",
        token_ids=[53626],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='",',
        token_ids=[497],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=' "',
        token_ids=[330],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="Project",
        token_ids=[7849],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" Gutenberg",
        token_ids=[51586],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='"]',
        token_ids=[1341],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="}}\n",
        token_ids=[11248],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="</tool_call>",
        token_ids=[151658],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="",
        token_ids=[151645],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    ),
]

# ---------------------------------------------------------------------------
# Test data: stream_interval=20 (multiple tokens per output)
# The critical difference: </think>, \n\n, <tool_call>, and the start of the
# JSON tool-call body can all arrive in a single CompletionOutput chunk.
# ---------------------------------------------------------------------------
OUTPUTS_INTERVAL_20 = [
    CompletionOutput(
        index=0,
        text="<think>",
        token_ids=[151667],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="\nOkay, the user is asking for the titles of some James Joyce books and wants me to use",
        token_ids=[
            198,
            32313,
            11,
            279,
            1196,
            374,
            10161,
            369,
            279,
            15311,
            315,
            1045,
            7801,
            53626,
            6467,
            323,
            6801,
            752,
            311,
            990,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" the provided tool. Let me check the available functions. There's a search_gutenberg_books function that",
        token_ids=[
            279,
            3897,
            5392,
            13,
            6771,
            752,
            1779,
            279,
            2500,
            5746,
            13,
            2619,
            594,
            264,
            2711,
            1889,
            44433,
            73084,
            729,
            429,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=' takes an array of search terms. The user mentioned "James Joyce books," so I need to use',
        token_ids=[
            4990,
            458,
            1334,
            315,
            2711,
            3793,
            13,
            576,
            1196,
            9733,
            330,
            28084,
            53626,
            6467,
            1335,
            773,
            358,
            1184,
            311,
            990,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" the search terms related to that. I should make sure to list the relevant terms. Let me think",
        token_ids=[
            279,
            2711,
            3793,
            5435,
            311,
            429,
            13,
            358,
            1265,
            1281,
            2704,
            311,
            1140,
            279,
            9760,
            3793,
            13,
            6771,
            752,
            1744,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='... "James Joyce" and "Project Gutenberg" might be the keywords here. So I\'ll structure',
        token_ids=[
            1112,
            330,
            28084,
            53626,
            1,
            323,
            330,
            7849,
            51586,
            1,
            2578,
            387,
            279,
            20844,
            1588,
            13,
            2055,
            358,
            3278,
            5944,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=' the search terms as ["James Joyce", "Project Gutenberg"] to find the books. That should cover',
        token_ids=[
            279,
            2711,
            3793,
            438,
            4383,
            28084,
            53626,
            497,
            330,
            7849,
            51586,
            1341,
            311,
            1477,
            279,
            6467,
            13,
            2938,
            1265,
            3421,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=' the user\'s request.\n</think>\n\n<tool_call>\n{"name": "search_gutenberg_books", "arguments',
        token_ids=[
            279,
            1196,
            594,
            1681,
            624,
            151668,
            271,
            151657,
            198,
            4913,
            606,
            788,
            330,
            1836,
            1889,
            44433,
            73084,
            497,
            330,
            16370,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text='": {"search_terms": ["James Joyce", "Project Gutenberg"]}}\n</tool_call>',
        token_ids=[
            788,
            5212,
            1836,
            37498,
            788,
            4383,
            28084,
            53626,
            497,
            330,
            7849,
            51586,
            1341,
            11248,
            151658,
            151645,
        ],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    ),
]

# ---------------------------------------------------------------------------
# Test data: stream_interval=20, reasoning + plain content (no tool calls).
# The critical difference from OUTPUTS_INTERVAL_20: the last chunk contains
# </think>, the response content, AND finish_reason=stop all in one
# CompletionOutput.  There is no <tool_call> markup at all.
# ---------------------------------------------------------------------------
OUTPUTS_NO_TOOL_CALL = [
    CompletionOutput(
        index=0,
        text="<think>",
        token_ids=[151667],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="\nOkay, I need to find out the capital of Tuvalu. Let me start by recalling what",
        token_ids=[
            198,
            32313,
            11,
            358,
            1184,
            311,
            1477,
            700,
            279,
            6722,
            315,
            28649,
            25510,
            13,
            6771,
            752,
            1191,
            553,
            88646,
            1128,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" I know. Tuvalu is a small island nation in the Pacific Ocean. I remember studying geography in",
        token_ids=[
            358,
            1414,
            13,
            28649,
            25510,
            374,
            264,
            2613,
            12922,
            6995,
            304,
            279,
            16462,
            21575,
            13,
            358,
            6099,
            20956,
            53142,
            304,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" school, so probably there's some information there.\n\nWait, Tuvalu's capital is probably called H",
        token_ids=[
            2906,
            11,
            773,
            4658,
            1052,
            594,
            1045,
            1995,
            1052,
            382,
            14190,
            11,
            28649,
            25510,
            594,
            6722,
            374,
            4658,
            2598,
            472,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="aka at the bottom of the list. But let me think again. When I was learning about islands",
        token_ids=[
            13334,
            518,
            279,
            5622,
            315,
            279,
            1140,
            13,
            1988,
            1077,
            752,
            1744,
            1549,
            13,
            3197,
            358,
            572,
            6832,
            911,
            29000,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=", I remember that some countries have capital cities named after animals or other things. Haka sounds familiar",
        token_ids=[
            11,
            358,
            6099,
            429,
            1045,
            5837,
            614,
            6722,
            9720,
            6941,
            1283,
            9898,
            476,
            1008,
            2513,
            13,
            472,
            13334,
            10362,
            11285,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=' from some pictures or maybe the name "Haka" relates to the island. \n\nI should check',
        token_ids=[
            504,
            1045,
            9185,
            476,
            7196,
            279,
            829,
            330,
            39,
            13334,
            1,
            35616,
            311,
            279,
            12922,
            13,
            4710,
            40,
            1265,
            1779,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" if there's another name for the capital. Maybe there's another city too. But looking at the",
        token_ids=[
            421,
            1052,
            594,
            2441,
            829,
            369,
            279,
            6722,
            13,
            10696,
            1052,
            594,
            2441,
            3283,
            2238,
            13,
            1988,
            3330,
            518,
            279,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" options, the capital is definitely Haka. I don't think there's another one like that.",
        token_ids=[
            2606,
            11,
            279,
            6722,
            374,
            8491,
            472,
            13334,
            13,
            358,
            1513,
            944,
            1744,
            1052,
            594,
            2441,
            825,
            1075,
            429,
            13,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" Let me make sure there's no other possible answer in the list that I'm missing. The user",
        token_ids=[
            6771,
            752,
            1281,
            2704,
            1052,
            594,
            902,
            1008,
            3204,
            4226,
            304,
            279,
            1140,
            429,
            358,
            2776,
            7402,
            13,
            576,
            1196,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=" provided the options, and the correct one is Haka. So I'm confident that's it.\n",
        token_ids=[
            3897,
            279,
            2606,
            11,
            323,
            279,
            4396,
            825,
            374,
            472,
            13334,
            13,
            2055,
            358,
            2776,
            16506,
            429,
            594,
            432,
            624,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text="</think>\n\nThe capital of Tuvalu is **Haka**.",
        token_ids=[
            151668,
            271,
            785,
            6722,
            315,
            28649,
            25510,
            374,
            3070,
            39,
            13334,
            334,
            13,
            151645,
        ],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    ),
]

PROMPT_TOKEN_IDS = [
    151644,
    8948,
    198,
    2,
    13852,
    271,
    2610,
    1231,
    1618,
    825,
    476,
    803,
    5746,
    311,
    7789,
    448,
    279,
    1196,
    3239,
    382,
    2610,
    525,
    3897,
    448,
    729,
    32628,
    2878,
    366,
    15918,
    1472,
    15918,
    29,
    11874,
    9492,
    510,
    27,
    15918,
    397,
    4913,
    1313,
    788,
    330,
    1688,
    497,
    330,
    1688,
    788,
    5212,
    606,
    788,
    330,
    1836,
    1889,
    44433,
    73084,
    497,
    330,
    4684,
    788,
    330,
    5890,
    369,
    6467,
    304,
    279,
    5787,
    51586,
    6733,
    497,
    330,
    13786,
    788,
    5212,
    1313,
    788,
    330,
    1700,
    497,
    330,
    13193,
    788,
    5212,
    1836,
    37498,
    788,
    5212,
    1313,
    788,
    330,
    1653,
    497,
    330,
    3615,
    788,
    5212,
    1313,
    788,
    330,
    917,
    14345,
    330,
    4684,
    788,
    330,
    852,
    315,
    2711,
    3793,
    311,
    1477,
    6467,
    9207,
    2137,
    330,
    6279,
    788,
    4383,
    1836,
    37498,
    1341,
    3417,
    532,
    522,
    15918,
    1339,
    2461,
    1817,
    729,
    1618,
    11,
    470,
    264,
    2951,
    1633,
    448,
    729,
    829,
    323,
    5977,
    2878,
    220,
    151657,
    151658,
    11874,
    9492,
    510,
    151657,
    198,
    4913,
    606,
    788,
    366,
    1688,
    11494,
    8066,
    330,
    16370,
    788,
    366,
    2116,
    56080,
    40432,
    31296,
    151658,
    151645,
    198,
    151644,
    872,
    198,
    3838,
    525,
    279,
    15311,
    315,
    1045,
    7801,
    53626,
    6467,
    30,
    5443,
    279,
    5392,
    311,
    2711,
    13,
    151645,
    198,
    151644,
    77091,
    198,
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tokenizer():
    return MockQwen3Tokenizer()


@pytest.fixture
def request_for_sampling():
    """Construct a ChatCompletionRequest matching the test spec."""
    return ChatCompletionRequest.model_construct(
        messages=[
            {
                "content": "What are the titles of some James Joyce books? "
                "Use the tool to search.",
                "role": "user",
            }
        ],
        model="Qwen/Qwen3-0.6B",
        tools=[
            ChatCompletionToolsParam(
                type="function",
                function=FunctionDefinition(
                    name="search_gutenberg_books",
                    description="Search for books in the Project Gutenberg library",
                    parameters={
                        "type": "object",
                        "properties": {
                            "search_terms": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of search terms to find books",
                            }
                        },
                        "required": ["search_terms"],
                    },
                ),
            )
        ],
        tool_choice="auto",
        include_reasoning=True,
        stream=False,
        n=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        temperature=None,
        top_p=None,
        skip_special_tokens=False,
        chat_template_kwargs=None,
        reasoning_effort=None,
        parallel_tool_calls=True,
    )


@pytest.fixture
def sampling_params():
    return SamplingParams(
        n=1,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        repetition_penalty=1.0,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        seed=None,
        stop=[],
        stop_token_ids=[],
        include_stop_str_in_output=False,
        ignore_eos=False,
        max_tokens=100000,
        min_tokens=0,
        logprobs=None,
        prompt_logprobs=None,
        skip_special_tokens=False,
        spaces_between_special_tokens=True,
    )


@pytest.fixture
def processor(tokenizer, request_for_sampling, sampling_params):
    tool_parser = Hermes2ProToolParser(tokenizer)
    return StreamingPostProcessor(
        tokenizer=tokenizer,
        request_for_sampling=request_for_sampling,
        sampling_params=sampling_params,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        tool_parser=tool_parser,
        reasoning_parser_class=Qwen3ReasoningParser,
        chat_template_kwargs={"reasoning_effort": None},
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _collect_results(processor, outputs):
    """Run all outputs through process_output and collect non-None results."""
    results = []
    for output in outputs:
        result = processor.process_output(output)
        if result is not None:
            results.append(result)
    return results


def _collect_reasoning(results):
    """Extract and join all reasoning_content from results."""
    parts = []
    for r in results:
        rc = r.get("delta", {}).get("reasoning_content")
        if rc is not None:
            parts.append(rc)
    return "".join(parts)


def _collect_tool_calls(results):
    """Merge all streamed tool_call deltas into complete tool calls.

    Returns a list of dicts, each with 'id', 'type', 'function' (with 'name'
    and 'arguments').
    """
    merged: dict[int, dict] = {}
    for r in results:
        tc_list = r.get("delta", {}).get("tool_calls")
        if not tc_list:
            continue
        for tc in tc_list:
            idx = tc["index"]
            if idx not in merged:
                merged[idx] = {
                    "id": tc.get("id"),
                    "type": tc.get("type"),
                    "function": {
                        "name": tc.get("function", {}).get("name"),
                        "arguments": tc.get("function", {}).get("arguments", ""),
                    },
                }
            else:
                existing = merged[idx]
                if tc.get("id") and not existing["id"]:
                    existing["id"] = tc["id"]
                if tc.get("type") and not existing["type"]:
                    existing["type"] = tc["type"]
                fn = tc.get("function", {})
                if fn.get("name") and not existing["function"]["name"]:
                    existing["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    existing["function"]["arguments"] += fn["arguments"]
    return [merged[k] for k in sorted(merged)]


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
@pytest.mark.vllm
def test_stream_interval_1(processor):
    """stream_interval=1: one token per chunk. Baseline that works."""
    results = _collect_results(processor, OUTPUTS_INTERVAL_1)
    reasoning = _collect_reasoning(results)
    tool_calls = _collect_tool_calls(results)

    expected_reasoning = (
        "\nOkay, the user is asking for the titles of some James Joyce"
        " books and wants me to use the provided tool.\n"
    )
    assert reasoning == expected_reasoning

    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["function"]["name"] == "search_gutenberg_books"
    assert json.loads(tc["function"]["arguments"]) == {
        "search_terms": ["James Joyce", "Project Gutenberg"],
    }
    assert tc["id"] is not None and tc["id"].startswith("chatcmpl-tool-")
    assert tc["type"] == "function"

    # finish_reason remapped "stop" → "tool_calls" per openai-openapi
    # ChatCompletion finish_reason enum.
    # See https://github.com/ai-dynamo/dynamo/issues/8636
    finish_reasons = [r["finish_reason"] for r in results if r.get("finish_reason")]
    assert finish_reasons == ["tool_calls"]

    seen_content = False
    for r in results:
        delta = r.get("delta", {})
        if delta.get("content") is not None:
            seen_content = True
        if seen_content:
            assert (
                delta.get("reasoning_content") is None
            ), "reasoning_content appeared after regular content started"

    for r in results:
        delta = r.get("delta", {})
        if delta:
            assert delta.get("role") == "assistant"


@pytest.mark.vllm
def test_stream_interval_20(tokenizer, request_for_sampling, sampling_params):
    """stream_interval=20: multiple tokens per chunk.

    When </think>, <tool_call>, and the start of the JSON body arrive in a
    single CompletionOutput, the tool parser must still extract the tool call
    correctly instead of leaking raw tool-call markup into ``content``.
    """
    # Fresh processor — the tool parser is stateful.
    tool_parser = Hermes2ProToolParser(tokenizer)
    proc = StreamingPostProcessor(
        tokenizer=tokenizer,
        request_for_sampling=request_for_sampling,
        sampling_params=sampling_params,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        tool_parser=tool_parser,
        reasoning_parser_class=Qwen3ReasoningParser,
        chat_template_kwargs={"reasoning_effort": None},
    )

    results = _collect_results(proc, OUTPUTS_INTERVAL_20)
    reasoning = _collect_reasoning(results)
    tool_calls = _collect_tool_calls(results)

    # -- reasoning_content should contain the full think block ---------------
    assert "the user is asking for the titles of some James Joyce books" in reasoning
    assert "the user's request.\n" in reasoning

    # -- tool calls must be parsed, not leaked as content -------------------
    assert len(tool_calls) == 1, (
        f"Expected 1 tool call but got {len(tool_calls)}. "
        "Tool-call markup was likely emitted as plain content instead."
    )
    tc = tool_calls[0]
    assert tc["function"]["name"] == "search_gutenberg_books"
    assert json.loads(tc["function"]["arguments"]) == {
        "search_terms": ["James Joyce", "Project Gutenberg"],
    }
    assert tc["id"] is not None and tc["id"].startswith("chatcmpl-tool-")
    assert tc["type"] == "function"

    # -- no <tool_call> markup should appear in content ---------------------
    all_content = "".join(r.get("delta", {}).get("content", "") for r in results)
    assert (
        "<tool_call>" not in all_content
    ), f"Raw <tool_call> markup leaked into content: {all_content!r}"
    assert "</tool_call>" not in all_content

    # -- finish reason: remaps "stop" → "tool_calls" per openai-openapi
    finish_reasons = [r["finish_reason"] for r in results if r.get("finish_reason")]
    assert finish_reasons == ["tool_calls"]


@pytest.mark.vllm
def test_stream_interval_20_reasoning_and_tool_finish_same_chunk(
    tokenizer, request_for_sampling, sampling_params
):
    """Regression: final chunk contains reasoning end + tool call + finish.

    When </think>, <tool_call>... </tool_call>, and finish_reason=stop arrive
    in one CompletionOutput, the tool call must still be emitted.
    """
    tool_parser = Hermes2ProToolParser(tokenizer)
    proc = StreamingPostProcessor(
        tokenizer=tokenizer,
        request_for_sampling=request_for_sampling,
        sampling_params=sampling_params,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        tool_parser=tool_parser,
        reasoning_parser_class=Qwen3ReasoningParser,
        chat_template_kwargs={"reasoning_effort": None},
    )

    penultimate = OUTPUTS_INTERVAL_20[-2]
    final = OUTPUTS_INTERVAL_20[-1]
    merged_final = CompletionOutput(
        index=0,
        text=(penultimate.text or "") + (final.text or ""),
        token_ids=list(penultimate.token_ids) + list(final.token_ids),
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    )
    outputs = [*OUTPUTS_INTERVAL_20[:-2], merged_final]

    results = _collect_results(proc, outputs)
    reasoning = _collect_reasoning(results)
    tool_calls = _collect_tool_calls(results)

    assert "the user's request.\n" in reasoning
    assert len(tool_calls) == 1
    tc = tool_calls[0]
    assert tc["function"]["name"] == "search_gutenberg_books"
    assert json.loads(tc["function"]["arguments"]) == {
        "search_terms": ["James Joyce", "Project Gutenberg"],
    }

    all_content = "".join(r.get("delta", {}).get("content", "") for r in results)
    assert "<tool_call>" not in all_content
    assert "</tool_call>" not in all_content

    # finish_reason remapped "stop" → "tool_calls" per OpenAI spec.
    # See https://github.com/ai-dynamo/dynamo/issues/8636
    finish_reasons = [r["finish_reason"] for r in results if r.get("finish_reason")]
    assert finish_reasons == ["tool_calls"]


@pytest.mark.vllm
def test_stream_terminal_single_chunk(tokenizer, request_for_sampling, sampling_params):
    """Regression: everything arrives in a single CompletionOutput.

    The closing </think>, the full <tool_call>…</tool_call>, and
    finish_reason="stop" are all packed into one chunk.  This exercises
    the terminal single-chunk buffer-drain path in the post-processor.
    """
    tool_parser = Hermes2ProToolParser(tokenizer)
    proc = StreamingPostProcessor(
        tokenizer=tokenizer,
        request_for_sampling=request_for_sampling,
        sampling_params=sampling_params,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        tool_parser=tool_parser,
        reasoning_parser_class=Qwen3ReasoningParser,
        chat_template_kwargs={"reasoning_effort": None},
    )

    # Build a single chunk that contains *all* text and token IDs from the
    # OUTPUTS_INTERVAL_20 sequence, with finish_reason="stop".
    all_text = "".join(o.text or "" for o in OUTPUTS_INTERVAL_20)
    all_token_ids = [tid for o in OUTPUTS_INTERVAL_20 for tid in o.token_ids]
    single_chunk = CompletionOutput(
        index=0,
        text=all_text,
        token_ids=all_token_ids,
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    )

    results = _collect_results(proc, [single_chunk])
    reasoning = _collect_reasoning(results)
    tool_calls = _collect_tool_calls(results)

    # -- reasoning_content should contain the full think block ---------------
    assert "the user is asking for the titles of some James Joyce books" in reasoning
    assert "the user's request.\n" in reasoning

    # -- tool calls must be parsed, not leaked as content -------------------
    assert len(tool_calls) == 1, (
        f"Expected 1 tool call but got {len(tool_calls)}. "
        "Tool-call markup was likely emitted as plain content instead."
    )
    tc = tool_calls[0]
    assert tc["function"]["name"] == "search_gutenberg_books"
    assert json.loads(tc["function"]["arguments"]) == {
        "search_terms": ["James Joyce", "Project Gutenberg"],
    }

    # -- no <tool_call> markup should appear in content ---------------------
    all_content = "".join(r.get("delta", {}).get("content", "") for r in results)
    assert (
        "<tool_call>" not in all_content
    ), f"Raw <tool_call> markup leaked into content: {all_content!r}"
    assert "</tool_call>" not in all_content

    # -- finish reason: remaps "stop" → "tool_calls" per openai-openapi
    finish_reasons = [r["finish_reason"] for r in results if r.get("finish_reason")]
    assert finish_reasons == ["tool_calls"]


@pytest.mark.vllm
def test_no_tool_call(tokenizer, request_for_sampling, sampling_params):
    """Reasoning + plain content, no tool calls.

    When </think> and the actual response content arrive in the same chunk
    (with finish_reason=stop), the content must still be emitted.  This
    reproduces a regression where the post-reasoning content was
    unconditionally buffered for tool-call extraction and never emitted
    when no tool call was present.
    """
    tool_parser = Hermes2ProToolParser(tokenizer)
    proc = StreamingPostProcessor(
        tokenizer=tokenizer,
        request_for_sampling=request_for_sampling,
        sampling_params=sampling_params,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        tool_parser=tool_parser,
        reasoning_parser_class=Qwen3ReasoningParser,
        chat_template_kwargs={"reasoning_effort": None},
    )

    results = _collect_results(proc, OUTPUTS_NO_TOOL_CALL)
    reasoning = _collect_reasoning(results)

    # -- reasoning should contain the think block ----------------------------
    assert "I need to find out the capital of Tuvalu" in reasoning
    assert "confident that's it.\n" in reasoning

    # -- content must include the actual response ----------------------------
    all_content = "".join(r.get("delta", {}).get("content", "") for r in results)
    assert (
        "The capital of Tuvalu is **Haka**." in all_content
    ), f"Post-reasoning content was lost. Got content: {all_content!r}"

    # -- no tool calls should be present ------------------------------------
    tool_calls = _collect_tool_calls(results)
    assert len(tool_calls) == 0, f"Expected 0 tool calls but got {len(tool_calls)}"

    # -- finish reason: no tool calls ⇒ "stop" is passed through unchanged.
    finish_reasons = [r["finish_reason"] for r in results if r.get("finish_reason")]
    assert finish_reasons == ["stop"]


# ---------------------------------------------------------------------------
# vllm chat processor + hermes tool parser + Qwen3 reasoning drops the second
# tool call in streaming mode on parallel-tool prompts. Observed symptom: 0
# tool_calls reach the client, tool-call markup is routed into
# reasoning_content, finish_reason="stop".
# See https://github.com/ai-dynamo/dynamo/issues/8636
#
# Token IDs for the tool-call JSON bodies are arbitrary non-special ints
# — the hermes parser operates on the text attribute, not decoded tokens.
# ---------------------------------------------------------------------------
_FAKE_TC_BODY_1 = tuple(range(1000, 1020))
_FAKE_TC_BODY_2 = tuple(range(2000, 2020))
_FAKE_TC_BODY_3 = tuple(range(3000, 3020))

OUTPUTS_PARALLEL_NO_THINK = [
    CompletionOutput(
        index=0,
        text=(
            '<tool_call>\n{"name": "search_gutenberg_books", "arguments":'
            ' {"search_terms": ["James Joyce"]}}\n</tool_call>\n'
            '<tool_call>\n{"name": "search_gutenberg_books", "arguments"'
        ),
        token_ids=[
            151657,
            198,
            *_FAKE_TC_BODY_1,
            198,
            151658,
            198,
            151657,
            198,
            *_FAKE_TC_BODY_2,
        ],
        cumulative_logprob=None,
        logprobs=None,
    ),
    CompletionOutput(
        index=0,
        text=': {"search_terms": ["Charles Dickens"]}}\n</tool_call>',
        token_ids=[
            *_FAKE_TC_BODY_3,
            198,
            151658,
            151645,
        ],
        cumulative_logprob=None,
        logprobs=None,
        finish_reason="stop",
    ),
]


@pytest.mark.vllm
def test_streaming_parallel_tool_calls_no_think(
    tokenizer, request_for_sampling, sampling_params
):
    """Regression: parallel tools with thinking disabled.

    See https://github.com/ai-dynamo/dynamo/issues/8636

    Shape: `enable_thinking=False` ⇒ <think>…</think> lives in the
    prompt, the generated output has neither. In streaming mode the
    reasoning parser stays engaged for every chunk (is_reasoning_end_streaming
    never fires because </think> never arrives), so the tool-call markup is
    mis-classified as reasoning_content and 0 tool_calls reach the client.

    Pre-fix: this test FAILS — 0 tool_calls extracted, all tool-call markup
    routed into reasoning_content, finish_reason="stop".
    """
    tool_parser = Hermes2ProToolParser(tokenizer)
    proc = StreamingPostProcessor(
        tokenizer=tokenizer,
        request_for_sampling=request_for_sampling,
        sampling_params=sampling_params,
        prompt_token_ids=PROMPT_TOKEN_IDS,
        tool_parser=tool_parser,
        reasoning_parser_class=Qwen3ReasoningParser,
        chat_template_kwargs={"enable_thinking": False},
    )

    results = _collect_results(proc, OUTPUTS_PARALLEL_NO_THINK)
    tool_calls = _collect_tool_calls(results)

    # -- both tool calls reach the client -----------------------------------
    assert len(tool_calls) == 2, (
        f"Expected 2 tool calls (parallel-tool regression, see #8636); got "
        f"{len(tool_calls)}. Results: {results!r}"
    )
    args = [json.loads(tc["function"]["arguments"]) for tc in tool_calls]
    assert args[0] == {"search_terms": ["James Joyce"]}
    assert args[1] == {"search_terms": ["Charles Dickens"]}
    assert tool_calls[0]["id"] != tool_calls[1]["id"]

    # -- no tool-call markup leaks into content or reasoning_content --------
    all_content = "".join(r.get("delta", {}).get("content", "") for r in results)
    assert "<tool_call>" not in all_content
    assert "</tool_call>" not in all_content
    all_reasoning = _collect_reasoning(results)
    assert (
        "search_gutenberg_books" not in all_reasoning
    ), f"Tool-call content leaked into reasoning_content: {all_reasoning!r}"

    # -- finish_reason must be remapped to "tool_calls" (acc #1 of #8636).
    # openai-openapi ChatCompletion finish_reason enum is {stop, length,
    # tool_calls, content_filter, function_call}; vLLM emits "stop" at
    # <|im_end|>, so the frontend remaps when tool calls were produced.
    finish_reasons = [r["finish_reason"] for r in results if r.get("finish_reason")]
    assert finish_reasons == [
        "tool_calls"
    ], f"Expected finish_reason=['tool_calls']; got {finish_reasons}"
