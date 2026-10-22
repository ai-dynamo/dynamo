# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned strategy/reducer CPU fixtures; no installed backend or GPU execution."""

import ast
import asyncio
import contextlib
import hashlib
import json
import math
import os
import random
import subprocess
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest

from dynamo.vllm import decision_reads as reads


class SamplingParams(SimpleNamespace):
    def __init__(self, **kwargs):
        super().__init__(
            **{
                "max_tokens": 1,
                "temperature": 1,
                "logprob_token_ids": None,
                "extra_args": None,
                **kwargs,
            }
        )


class RendererValidationError(ValueError):
    """CPU stand-in for the pinned renderer's typed validation failure."""


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, *, add_special_tokens=False):
        specials = {"<|channel>thought\n<channel|>": [8001, 8002], "<turn|>": [8003]}
        if text in specials:
            return specials[text]
        labels = {c: 1000 + ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"}
        labels.update({f" {c}": 2000 + ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"})
        labels.update(
            {s: 3000 + i for i, s in enumerate(("yes", "Yes", "YES", "no", "No", "NO"))}
        )
        return [labels[text]] if text in labels else [4096 + ord(c) for c in text]

    def decode(self, tokens):
        return "".join(chr(token - 4096) for token in tokens)

    def get_added_vocab(self):
        return {"start": 9000, "assistant": 9001, "thought": 8002}

    def apply_chat_template(self, messages, **kwargs):
        content = "\n".join(message["content"] for message in messages)
        return [9000, *self.encode(content), 9001]


class OnlineRenderer:
    def __init__(self, *, model_config, renderer, **kwargs):
        self.renderer = renderer
        self.model_config = model_config

    async def preprocess_chat(self, request, messages, **kwargs):
        self.renderer.rendered.append(messages)
        ids = self.renderer.get_tokenizer().apply_chat_template(messages)
        return None, (
            {
                "type": "token",
                "prompt_token_ids": ids,
                "cache_salt": request.cache_salt,
            },
        )


class Engine:
    def __init__(self, diffusion=False):
        self.model_config = SimpleNamespace(
            architecture="DiffusionGemmaForBlockDiffusion"
            if diffusion
            else "Lfm2VlForConditionalGeneration",
            runner_type="generate",
            is_encoder_decoder=False,
            is_diffusion=diffusion,
            model_impl="auto",
            model_class_overrides={},
            convert_type="none",
            logprobs_mode="raw_logprobs",
            max_model_len=8192,
            is_hybrid=False,
            hf_config=SimpleNamespace(canvas_length=32),
            get_vocab_size=lambda: 32768,
        )
        self.vllm_config = SimpleNamespace(
            model_config=self.model_config,
            parallel_config=SimpleNamespace(data_parallel_size=1),
            scheduler_config=SimpleNamespace(max_num_seqs=4),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            lora_config=None,
            speculative_config=None,
            kv_transfer_config=None,
            ec_transfer_config=None,
            diffusion_config=SimpleNamespace(canvas_length=32) if diffusion else None,
        )
        tokenizer = Tokenizer()
        self.rendered = []
        self.renderer = SimpleNamespace(
            get_tokenizer=lambda: tokenizer, rendered=self.rendered
        )
        self.calls, self.aborted, self.closed, self.admissions = [], [], [], []
        self.active, self.peak_active = 0, 0
        self.cached_tokens = None
        self.block = None
        self.started = asyncio.Event()
        self.failure = None
        self.diffusion = diffusion

    async def get_supported_tasks(self):
        return ("generate",)

    def check_admission(self, count):
        self.admissions.append(count)

    async def abort(self, request_id):
        self.aborted.append(request_id)

    async def generate(self, prompt, params, request_id, **kwargs):
        self.calls.append((prompt, params, request_id, kwargs))
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        self.started.set()
        try:
            if self.block is not None:
                await self.block.wait()
            ids = params.logprob_token_ids or []
            logprobs = {
                token: SimpleNamespace(
                    logprob=(-1.0 if i == 0 else -2.0 if i == 1 else -4.0)
                )
                for i, token in enumerate(ids)
            }
            if self.failure == "nonfinite":
                logprobs[ids[0]].logprob = float("nan")
            completion = SimpleNamespace(
                index=0,
                token_ids=[] if self.diffusion else [ids[0] if ids else 1],
                logprobs=None if self.failure == "missing_logprobs" else [logprobs],
                finish_reason="length",
            )
            yield SimpleNamespace(
                request_id="wrong" if self.failure == "wrong_id" else request_id,
                prompt_token_ids=prompt["prompt_token_ids"],
                outputs=[completion],
                finished=self.failure != "unfinished",
                num_cached_tokens=self.cached_tokens,
                num_cache_creation_tokens=None,
            )
        finally:
            self.active -= 1
            self.closed.append(request_id)


@pytest.fixture
def native_reads(monkeypatch):
    source_repo = os.environ.get("DYNAMO_VLLM_SOURCE_REPO")
    if source_repo is None:
        pytest.skip("Set DYNAMO_VLLM_SOURCE_REPO for hash-verified native CPU fixtures")

    def source(path):
        result = subprocess.run(
            [
                "git",
                f"--git-dir={source_repo}",
                "show",
                f"{reads.PINNED_VLLM_COMMIT}:{path}",
            ],
            check=True,
            capture_output=True,
            timeout=10,
            env={**os.environ, "GIT_NO_LAZY_FETCH": "1"},
        )
        expected = dict(reads._PINNED_SOURCE_BLOBS)[path]
        assert (
            hashlib.sha1(
                f"blob {len(result.stdout)}\0".encode() + result.stdout
            ).hexdigest()
            == expected
        )
        return result.stdout.decode()

    def module(label):
        result = ModuleType(f"_native_read_{label}_{uuid4().hex}")
        monkeypatch.setitem(sys.modules, result.__name__, result)
        return result

    questions = module("questions")
    exec(  # noqa: S102 -- Executes hash-verified upstream fixtures, never request data.
        compile(
            source("vllm/entrypoints/openai/decisions/question_types.py"),
            questions.__name__,
            "exec",
        ),
        questions.__dict__,
    )
    strategies = module("strategies")
    namespace = strategies.__dict__
    namespace.update(
        {
            key: value
            for key, value in vars(questions).items()
            if not key.startswith("__")
        }
    )
    namespace.update(
        asyncio=asyncio,
        math=math,
        json=json,
        hashlib=hashlib,
        random=random,
        dataclass=dataclass,
        cached_property=cached_property,
        ABC=ABC,
        abstractmethod=abstractmethod,
        OpenAIBaseModel=SimpleNamespace,
        SamplingParams=SamplingParams,
        ChatParams=SimpleNamespace,
        TokenizeParams=SimpleNamespace,
        merge_kwargs=lambda first, second: {**(first or {}), **(second or {})},
        tokens_input=lambda ids, **kwargs: {
            "type": "token",
            "prompt_token_ids": ids,
            **kwargs,
        },
        extract_prompt_components=lambda config, value: SimpleNamespace(
            token_ids=value["prompt_token_ids"]
        ),
        contextlib=contextlib,
        FIRST_COMPLETED=asyncio.FIRST_COMPLETED,
        MIN_WARM_PREFIX_TOKENS=128,
        GenerationError=RuntimeError,
    )
    selected = {
        "common_prefix_len",
        "ReadContext",
        "DecisionLimits",
        "QuestionRead",
        "ReadPromptRequest",
        "state_text",
        "ReadStrategy",
        "prompt_tail",
        "reply_label_ids",
        "type_label_ids",
        "NextTokenStrategy",
        "LiquidStrategy",
        "DiffusionGemmaCanvasStrategy",
        "LabelRead",
        "next_token_label_reads",
        "merge_async_iterators",
    }
    nodes = []
    for path in (
        "vllm/utils/async_utils.py",
        "vllm/entrypoints/generate/label_reads.py",
        "vllm/entrypoints/openai/decisions/strategies.py",
    ):
        nodes.extend(
            n
            for n in ast.parse(source(path)).body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
            and n.name in selected
        )
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 -- Executes hash-verified upstream fixtures, never request data.
        compile(ast.fix_missing_locations(tree), strategies.__name__, "exec"), namespace
    )
    api = SimpleNamespace(
        overload_errors=(),
        strategies=strategies,
        questions=questions,
        online_renderer=OnlineRenderer,
        validation_error=RendererValidationError,
    )
    monkeypatch.setattr(reads, "_load_native", lambda _: api)
    return api
