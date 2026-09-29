# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build replay fixtures from agent trajectories such as nvidia/Open-SWE-Traces.

A trajectory keeps only the parsed fields of each assistant turn
(`reasoning_content`, `content`, `tool_calls`), not the raw text the teacher
model emitted. The raw completion is rebuilt with the teacher's own chat
template: the text an assistant turn adds to the rendered conversation, after
the generation prompt and up to end-of-turn. Tokenized with the teacher's
tokenizer, it becomes the mocker's scripted output; the parsed fields become
the expected frontend response.

Because only the parsed fields are used, trajectories of one family can be
rendered as another family's output (`build(..., source_key=...)`): the target
supplies template, tokenizer and parsers, the source supplies the conversation.

Output directory:
    model/              tokenizer + chat template files, usable as --model-path
    trajectories.jsonl  conversation, tools and chat-template args per trajectory
    cases.jsonl         one request per line (turn x variant) with expectations
    replay.jsonl        mocker rows {request_id, output_length, output_token_ids}
    build_stats.json    kept / skipped turns and why

This module needs `transformers`, `tokenizers` and `huggingface_hub`; the
runner and the CI test only read the files it writes.
"""

from __future__ import annotations

import collections
import copy
import importlib.util
import json
import logging
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, Iterable, Iterator

import tokenizers
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

JsonDict = dict[str, Any]

DATASET = "nvidia/Open-SWE-Traces"
_ROWS_URL = "https://datasets-server.huggingface.co/rows"
_ROWS_PER_PAGE = 5
_MODEL_FILES = [
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "video_preprocessor_config.json",
    "encoding/encoding_dsv4.py",
    # Kimi K3: tiktoken tokenizer and reference segment encoder, no chat template.
    "tiktoken.model",
    "tokenization_kimi.py",
    "encoding_k3.py",
]
# A truncation point needs this much text on both sides to be meaningful.
_MIN_SPAN_CHARS = 40
# Reasoning-effort levels in increasing order, for mapping between families.
_EFFORT_ORDER = ("low", "medium", "high", "xhigh", "max")
_STOP_WORD = re.compile(r"[A-Za-z]{7,}")


@dataclass(frozen=True)
class Teacher:
    """A trajectory-generating model and how the frontend must serve it."""

    dataset_name: str  # metadata.teacher_model.name in the dataset
    repo: str  # HF repo with the tokenizer and chat template; no weights needed
    config: str  # dataset config and split holding its trajectories
    split: str
    renderer: str  # "jinja" (HF chat template), "dsv4" or "k3" (reference encoders)
    eos: str  # token text that closes an assistant turn
    tool_start: str  # opens the tool-call block in a raw completion
    tool_call_parser: str
    reasoning_parser: str
    markup: tuple[str, ...]  # raw syntax that must never reach parsed fields
    efforts: tuple[str, ...] = ()  # reasoning_effort values its template accepts
    think_end: str = "</think>"


TEACHERS: dict[str, Teacher] = {
    "qwen3.8": Teacher(
        dataset_name="Qwen3.8-27B",
        repo="Qwen/Qwen3.8-27B",
        config="v1.2",
        split="minisweagent",
        renderer="jinja",
        eos="<|im_end|>",
        tool_start="<tool_call>",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        markup=("<think>", "</think>", "<tool_call>", "<function=", "<parameter="),
        efforts=("low", "medium", "xhigh"),
    ),
    "deepseek-v4": Teacher(
        dataset_name="DeepSeek-V4-Flash",
        repo="deepseek-ai/DeepSeek-V4-Flash",
        config="v1.1",
        split="openhands",
        renderer="dsv4",
        eos="<｜end▁of▁sentence｜>",
        tool_start="<｜DSML｜tool_calls>",
        tool_call_parser="deepseek_v4",
        reasoning_parser="deepseek_v4",
        markup=("<think>", "</think>", "｜DSML｜"),
        efforts=("high", "max"),
    ),
    "minimax-m2.5": Teacher(
        dataset_name="MiniMax-M2.5",
        repo="MiniMaxAI/MiniMax-M2.5",
        config="v1.0",
        split="openhands",
        renderer="jinja",
        eos="[e~[",
        tool_start="<minimax:tool_call>",
        tool_call_parser="minimax_m2",
        reasoning_parser="minimax_m2",
        markup=("<think>", "</think>", "<minimax:tool_call>", "<invoke name="),
    ),
    # Open-SWE-Traces has no Kimi K3 trajectories, so K3 is a render target only
    # (`--source-teacher`). Its output is XTML: think / response / tools channels
    # opened and closed with <|open|>, <|close|> and <|sep|> tokens.
    "kimi-k3": Teacher(
        dataset_name="Kimi-K3",
        repo="moonshotai/Kimi-K3",
        config="",
        split="",
        renderer="k3",
        eos="<|end_of_msg|>",
        tool_start="<|open|>tools",
        tool_call_parser="kimi_k3",
        reasoning_parser="kimi_k3",
        markup=("<|open|>", "<|close|>", "<|sep|>"),
        efforts=("low", "high", "max"),
        think_end="<|close|>think",
    ),
    # Small Qwen3.5-family tokenizer for the committed CI fixture; it shares the
    # vocabulary and tool syntax of the Qwen3.8 teacher.
    "qwen3.5-0.8b": Teacher(
        dataset_name="handwritten",
        repo="Qwen/Qwen3.5-0.8B",
        config="",
        split="",
        renderer="jinja",
        eos="<|im_end|>",
        tool_start="<tool_call>",
        tool_call_parser="qwen3_coder",
        reasoning_parser="qwen3",
        markup=("<think>", "</think>", "<tool_call>", "<function=", "<parameter="),
    ),
}


class SkipTurn(Exception):
    """An assistant turn that cannot be replayed faithfully; the reason is counted."""


def download_model_files(teacher: Teacher, dest: Path) -> Path:
    """Fetch the tokenizer and chat template (no weights) into `dest`."""
    snapshot_download(
        repo_id=teacher.repo, local_dir=str(dest), allow_patterns=_MODEL_FILES
    )
    return dest


def fetch_rows(teacher: Teacher, offset: int, length: int) -> list[JsonDict]:
    """Read dataset rows through the Hugging Face datasets-server API."""
    query = urllib.parse.urlencode(
        {
            "dataset": DATASET,
            "config": teacher.config,
            "split": teacher.split,
            "offset": offset,
            "length": length,
        }
    )
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{_ROWS_URL}?{query}", timeout=180) as resp:
                return json.load(resp)["rows"]
        except (urllib.error.URLError, TimeoutError) as err:
            if attempt == 5:
                raise
            logger.warning("rows fetch at offset %d failed (%s); retrying", offset, err)
            time.sleep(2**attempt)
    raise AssertionError("unreachable")


def iter_dataset_rows(
    teacher: Teacher, offset: int, stats: collections.Counter
) -> Iterator[JsonDict]:
    """Yield complete rows produced by `teacher`, starting at `offset`."""
    while True:
        page = fetch_rows(teacher, offset, _ROWS_PER_PAGE)
        if not page:
            return
        for entry in page:
            row = entry["row"]
            if row["metadata"]["teacher_model"]["name"] != teacher.dataset_name:
                stats["rows_other_teacher"] += 1
            elif entry.get("truncated_cells"):
                stats["rows_truncated_by_api"] += 1
            else:
                yield row
        offset += len(page)


def normalize_messages(raw_messages: Iterable[JsonDict]) -> list[JsonDict]:
    """Convert trace messages into a valid OpenAI chat conversation.

    Trace tool messages carry no `tool_call_id`; they answer the preceding
    assistant's calls in order, so ids are assigned from that queue.
    """
    messages: list[JsonDict] = []
    pending_ids: collections.deque[str] = collections.deque()
    for raw in raw_messages:
        role = raw["role"]
        content = raw.get("content") or ""
        if role == "assistant":
            message: JsonDict = {"role": "assistant", "content": content}
            if raw.get("reasoning_content"):
                message["reasoning_content"] = raw["reasoning_content"]
            calls = raw.get("tool_calls") or []
            if calls:
                message["tool_calls"] = [
                    {
                        "id": call["id"],
                        "type": "function",
                        "function": {
                            "name": call["function"]["name"],
                            "arguments": call["function"]["arguments"],
                        },
                    }
                    for call in calls
                ]
            pending_ids = collections.deque(call["id"] for call in calls)
            messages.append(message)
        elif role == "tool":
            if not pending_ids:
                raise SkipTurn("tool-message-without-call")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": pending_ids.popleft(),
                    "content": content,
                }
            )
        else:
            messages.append({"role": role, "content": content})
    return messages


def chat_template_args(teacher: Teacher, teacher_meta: JsonDict) -> JsonDict:
    """Template kwargs for `teacher` that reproduce the trajectory's thinking setup."""
    thinking = bool(teacher_meta.get("enable_thinking"))
    if teacher.renderer == "dsv4":
        args: JsonDict = {"thinking_mode": "thinking" if thinking else "chat"}
    else:
        args = {"enable_thinking": thinking}
    effort = _map_effort(teacher_meta.get("reasoning_effort"), teacher.efforts)
    if effort:
        args["reasoning_effort"] = effort
    return args


def _map_effort(effort: str | None, supported: tuple[str, ...]) -> str | None:
    """The supported level closest to `effort`; templates reject unknown levels."""
    if not effort or not supported or effort not in _EFFORT_ORDER:
        return None
    if effort in supported:
        return effort
    rank = _EFFORT_ORDER.index(effort)
    return min(supported, key=lambda level: abs(_EFFORT_ORDER.index(level) - rank))


def _with_object_arguments(message: JsonDict) -> JsonDict:
    """HF templates iterate `arguments|items`, so they need a mapping, not a string."""
    if not message.get("tool_calls"):
        return message
    message = copy.deepcopy(message)
    for call in message["tool_calls"]:
        call["function"]["arguments"] = json.loads(call["function"]["arguments"])
    return message


def _load_module(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise FileNotFoundError(path)
    module = importlib.util.module_from_spec(spec)
    # Dataclasses in the module resolve their types through sys.modules.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Renderer:
    """Renders conversations and tokenizes completions the way the teacher did."""

    def __init__(self, teacher: Teacher, model_dir: Path):
        self.teacher = teacher
        self._dsv4: ModuleType | None = None
        self._k3: ModuleType | None = None
        self._kimi: Any = None
        self._hf: PreTrainedTokenizerFast | None = None
        self._template = ""
        if teacher.renderer == "k3":
            # Kimi K3 encodes typed segments: literal text never becomes a control
            # token, so ids come from its own segment encoder, not from re-tokenizing.
            self._k3 = _load_module(model_dir / "encoding_k3.py")
            self._kimi = AutoTokenizer.from_pretrained(
                str(model_dir), trust_remote_code=True
            )
            self.eos_id: int = self._kimi.convert_tokens_to_ids(teacher.eos)
            return
        self.tokenizer = tokenizers.Tokenizer.from_file(
            str(model_dir / "tokenizer.json")
        )
        eos_id = self.tokenizer.token_to_id(teacher.eos)
        if eos_id is None:
            raise ValueError(f"{teacher.eos!r} is not a single token in {teacher.repo}")
        self.eos_id = eos_id
        if teacher.renderer == "dsv4":
            self._dsv4 = _load_module(model_dir / "encoding" / "encoding_dsv4.py")
            return
        config = json.loads((model_dir / "tokenizer_config.json").read_text())
        template_file = model_dir / "chat_template.jinja"
        self._template = (
            template_file.read_text()
            if template_file.exists()
            else config["chat_template"]
        )
        special = {
            name: _token_text(config.get(name))
            for name in ("bos_token", "eos_token")
            if config.get(name)
        }
        self._hf = PreTrainedTokenizerFast(
            tokenizer_file=str(model_dir / "tokenizer.json"), **special
        )

    def render(
        self,
        messages: list[JsonDict],
        tools: list[JsonDict],
        template_args: JsonDict,
        add_generation_prompt: bool,
    ) -> str:
        if self._dsv4 is not None:
            conversation = copy.deepcopy(messages)
            if tools:
                conversation[0]["tools"] = tools
            # encode_messages appends the generation prompt after a user/tool turn.
            return self._dsv4.encode_messages(
                conversation,
                thinking_mode=template_args.get("thinking_mode", "thinking"),
                reasoning_effort=template_args.get("reasoning_effort"),
            )
        assert self._hf is not None
        rendered = self._hf.apply_chat_template(
            [_with_object_arguments(message) for message in messages],
            tools=tools or None,
            chat_template=self._template,
            add_generation_prompt=add_generation_prompt,
            tokenize=False,
            **template_args,
        )
        assert isinstance(rendered, str)
        return rendered

    def completion(
        self,
        messages: list[JsonDict],
        index: int,
        tools: list[JsonDict],
        template_args: JsonDict,
    ) -> str:
        """Raw text of assistant turn `index`: what follows the generation prompt."""
        prompt = self.render(messages[:index], tools, template_args, True)
        full = self.render(messages[: index + 1], tools, template_args, False)
        if not full.startswith(prompt):
            raise SkipTurn("template-prefix-mismatch")
        completion = full[len(prompt) :]
        end = completion.find(self.teacher.eos)
        if end < 0:
            raise SkipTurn("template-no-eos")
        return completion[: end + len(self.teacher.eos)]

    def turn_tokens(
        self,
        messages: list[JsonDict],
        index: int,
        tools: list[JsonDict],
        template_args: JsonDict,
    ) -> tuple[str, list[int], list[int]]:
        """Assistant turn `index` as (text, token ids ending in EOS, char offset per token)."""
        if self._k3 is not None:
            return self._k3_turn_tokens(messages, index, tools, template_args)
        completion = self.completion(messages, index, tools, template_args)
        encoding = self.tokenizer.encode(completion, add_special_tokens=False)
        ids: list[int] = encoding.ids
        if not ids or ids[-1] != self.eos_id:
            raise SkipTurn("eos-not-a-single-final-token")
        if self.tokenizer.decode(ids, skip_special_tokens=False) != completion:
            raise SkipTurn("tokenizer-roundtrip-mismatch")
        return completion, ids, [start for start, _ in encoding.offsets]

    def _k3_turn_tokens(
        self,
        messages: list[JsonDict],
        index: int,
        tools: list[JsonDict],
        template_args: JsonDict,
    ) -> tuple[str, list[int], list[int]]:
        assert self._k3 is not None
        options: JsonDict = {"thinking": template_args.get("enable_thinking", True)}
        if template_args.get("reasoning_effort"):
            options["thinking_effort"] = template_args["reasoning_effort"]

        def encode(conversation: list[JsonDict], generation_prompt: bool) -> list[int]:
            segments = self._k3.build_chat_segments(
                copy.deepcopy(conversation),
                tools or None,
                add_generation_prompt=generation_prompt,
                **options,
            )
            return self._kimi._encode_chat_segments(segments)

        prompt = encode(messages[:index], True)
        full = encode(messages[: index + 1], False)
        if full[: len(prompt)] != prompt:
            raise SkipTurn("template-prefix-mismatch")
        rest = full[len(prompt) :]
        if self.eos_id not in rest:
            raise SkipTurn("template-no-eos")
        ids = rest[: rest.index(self.eos_id) + 1]
        text, offsets = self._kimi.model.decode_with_offsets(ids)
        return text, ids, offsets

    def decode(self, ids: list[int], skip_special_tokens: bool) -> str:
        if self._kimi is not None:
            return self._kimi.decode(ids, skip_special_tokens=skip_special_tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)


def _token_text(value: Any) -> str:
    return value["content"] if isinstance(value, dict) else str(value)


def expected_turn(message: JsonDict) -> JsonDict:
    """What the frontend should return for a trace assistant turn."""
    calls = []
    for call in message.get("tool_calls", []):
        try:
            arguments = json.loads(call["function"]["arguments"])
        except json.JSONDecodeError as err:
            raise SkipTurn("trace-arguments-not-json") from err
        calls.append({"name": call["function"]["name"], "arguments": arguments})
    return {
        "reasoning": message.get("reasoning_content", ""),
        "content": message.get("content", ""),
        "tool_calls": calls,
        "finish_reason": "tool_calls" if calls else "stop",
    }


def _cut_points(teacher: Teacher, completion: str, expected: JsonDict) -> JsonDict:
    """Character offsets that end a truncated script inside each region of the turn."""
    body_end = len(completion) - len(teacher.eos)
    think_end = completion.find(teacher.think_end) if expected["reasoning"] else -1
    content_start = think_end + len(teacher.think_end) if think_end >= 0 else 0
    tool_at = completion.find(teacher.tool_start, content_start)
    content_end = tool_at if tool_at >= 0 else body_end
    cuts: JsonDict = {}
    if think_end > 2 * _MIN_SPAN_CHARS:
        cuts["trunc_reasoning"] = think_end // 2
    if (
        expected["content"].strip()
        and content_end - content_start > 2 * _MIN_SPAN_CHARS
    ):
        cuts["trunc_content"] = (content_start + content_end) // 2
    if tool_at >= 0 and body_end - tool_at > 2 * _MIN_SPAN_CHARS:
        cuts["trunc_tool_args"] = (tool_at + body_end) // 2
    return cuts


def _stop_word(
    completion: str, reasoning: str, think_end: int
) -> tuple[str, int] | None:
    """A word from the middle of the reasoning whose first occurrence is in it."""
    words = _STOP_WORD.findall(reasoning)
    for word in words[len(words) // 2 :]:
        position = completion.find(word)
        if _MIN_SPAN_CHARS < position < think_end:
            return word, position
    return None


@dataclass
class FixtureWriter:
    """Accumulates trajectories and replay cases, then writes the fixture files."""

    teacher: Teacher
    renderer: Renderer
    variant_every: int

    def __post_init__(self) -> None:
        self.trajectories: list[JsonDict] = []
        self.cases: list[JsonDict] = []
        self.replay_rows: list[JsonDict] = []
        self.stats: collections.Counter = collections.Counter()

    def add_row(self, row: JsonDict) -> None:
        trajectory_id = row["trajectory_id"]
        try:
            messages = normalize_messages(row["messages"])
        except SkipTurn as skip:
            self.stats[f"trajectory_skipped:{skip}"] += 1
            return
        tools = [
            json.loads(tool) if isinstance(tool, str) else tool for tool in row["tools"]
        ]
        template_args = chat_template_args(
            self.teacher, row["metadata"]["teacher_model"]
        )
        source = row["metadata"]["teacher_model"]["name"]
        self.stats[f"source:{source}"] += 1
        self.trajectories.append(
            {
                "trajectory": trajectory_id,
                "instance_id": row.get("instance_id"),
                "teacher": self.teacher.dataset_name,
                "source": source,
                "messages": messages,
                "tools": tools,
                "chat_template_args": template_args,
            }
        )
        self.stats["trajectories"] += 1
        kept = 0
        for index, message in enumerate(messages):
            if message["role"] != "assistant":
                continue
            try:
                self._add_turn(
                    trajectory_id, messages, index, tools, template_args, kept
                )
            except SkipTurn as skip:
                self.stats[f"turn_skipped:{skip}"] += 1
                continue
            kept += 1
        self.stats["turns"] += kept

    def _add_turn(
        self,
        trajectory_id: str,
        messages: list[JsonDict],
        index: int,
        tools: list[JsonDict],
        template_args: JsonDict,
        kept: int,
    ) -> None:
        teacher = self.teacher
        expected = expected_turn(messages[index])
        completion, ids, offsets = self.renderer.turn_tokens(
            messages, index, tools, template_args
        )

        base = f"{trajectory_id}:{index}"
        self._add_case(base, "full", trajectory_id, index, ids, expected, eos=True)
        self.stats["tool_calls"] += len(expected["tool_calls"])
        self.stats["parallel_tool_call_turns"] += len(expected["tool_calls"]) > 1
        if kept % self.variant_every:
            return
        for variant, cut in _cut_points(teacher, completion, expected).items():
            end = next(
                (i for i, start in enumerate(offsets) if start >= cut), len(ids) - 1
            )
            if 0 < end < len(ids) - 1:
                self._add_case(
                    base, variant, trajectory_id, index, ids[:end], expected, eos=False
                )
        think_end = completion.find(teacher.think_end)
        stop = _stop_word(completion, expected["reasoning"], think_end)
        if stop is not None:
            word, position = stop
            self.cases.append(
                {
                    "key": f"{base}:stop_in_reasoning",
                    "replay_key": f"{base}:full",
                    "trajectory": trajectory_id,
                    "message_index": index,
                    "variant": "stop_in_reasoning",
                    "script_len": len(ids),
                    "stop": [word],
                    "raw_text": completion[:position],
                    "expected": {
                        "reasoning": completion[:position],
                        "content": "",
                        "tool_calls": [],
                        "finish_reason": "stop",
                    },
                }
            )
            self.stats["case:stop_in_reasoning"] += 1

    def _add_case(
        self,
        base: str,
        variant: str,
        trajectory_id: str,
        index: int,
        ids: list[int],
        expected: JsonDict,
        eos: bool,
    ) -> None:
        key = f"{base}:{variant}"
        visible = ids[:-1] if eos else ids
        self.replay_rows.append(
            {"request_id": key, "output_length": len(ids), "output_token_ids": ids}
        )
        self.cases.append(
            {
                "key": key,
                "replay_key": key,
                "trajectory": trajectory_id,
                "message_index": index,
                "variant": variant,
                "script_len": len(ids),
                "eos_terminated": eos,
                # What the frontend returns with no parsers configured.
                "raw_text": self.renderer.decode(visible, skip_special_tokens=True),
                "expected": expected
                if eos
                else {**expected, "finish_reason": "length"},
            }
        )
        self.stats[f"case:{variant}"] += 1

    def write(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        _write_jsonl(out_dir / "trajectories.jsonl", self.trajectories)
        _write_jsonl(out_dir / "cases.jsonl", self.cases)
        _write_jsonl(out_dir / "replay.jsonl", self.replay_rows)
        manifest = {
            "teacher": self.teacher.dataset_name,
            "repo": self.teacher.repo,
            "tool_call_parser": self.teacher.tool_call_parser,
            "reasoning_parser": self.teacher.reasoning_parser,
            "markup": list(self.teacher.markup),
            "stats": dict(sorted(self.stats.items())),
        }
        (out_dir / "build_stats.json").write_text(json.dumps(manifest, indent=2) + "\n")


def _write_jsonl(path: Path, records: Iterable[JsonDict]) -> None:
    with path.open("w", encoding="utf-8") as out:
        for record in records:
            out.write(json.dumps(record, ensure_ascii=False) + "\n")


def build(
    teacher_key: str,
    out_dir: Path,
    *,
    offset: int = 0,
    min_turns: int = 1000,
    max_trajectories: int = 1000,
    variant_every: int = 4,
    rows_file: Path | None = None,
    source_key: str | None = None,
) -> JsonDict:
    """Build fixtures from dataset rows (or `rows_file`) until `min_turns` turns are kept.

    Rows come from `source_key`'s slice of the dataset (default: the teacher's own)
    and are rendered as `teacher_key`'s output.
    """
    teacher = TEACHERS[teacher_key]
    model_dir = download_model_files(teacher, out_dir / "model")
    writer = FixtureWriter(teacher, Renderer(teacher, model_dir), variant_every)
    if rows_file is not None:
        rows: Iterable[JsonDict] = json.loads(rows_file.read_text())
    else:
        source = TEACHERS[source_key or teacher_key]
        if not source.config:
            raise ValueError(
                f"{source.dataset_name} has no dataset slice; pass source_key"
            )
        rows = iter_dataset_rows(source, offset, writer.stats)
    for row in rows:
        writer.add_row(row)
        logger.info(
            "%s: %d trajectories, %d turns kept",
            teacher_key,
            writer.stats["trajectories"],
            writer.stats["turns"],
        )
        if (
            writer.stats["turns"] >= min_turns
            or writer.stats["trajectories"] >= max_trajectories
        ):
            break
    writer.write(out_dir)
    return dict(writer.stats)
