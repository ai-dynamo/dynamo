# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned strategy/reducer CPU fixtures; no installed backend or GPU execution."""

import ast
import asyncio
import contextlib
import hashlib
import json
import logging
import math
import random
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property, partial
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest
from jinja2 import TemplateError

from dynamo.vllm import decision_native_lifecycle as native_lifecycle
from dynamo.vllm.tests.decision_head_test_support import NativeApi
from dynamo.vllm.tests.decision_pinned_source import (
    load_pinned_engine_errors,
    read_pinned_source,
    source_repository,
)
from dynamo.vllm.tests.decision_read_test_support import (
    MAX_LOGPROB_TOKEN_IDS,
    OnlineRenderer,
    SamplingParams,
    extract_prompt_components,
    tokens_input,
)


@pytest.fixture
def pinned_source():
    repository = source_repository()
    return partial(read_pinned_source, repository)


@pytest.fixture
def native_module(monkeypatch):
    def create(label):
        module = ModuleType(f"_pinned_native_{label}_{uuid4().hex}")
        monkeypatch.setitem(sys.modules, module.__name__, module)
        return module

    return create


@pytest.fixture
def pinned_questions(pinned_source, native_module):
    module = native_module("questions")
    exec(
        compile(
            pinned_source("vllm/entrypoints/openai/decisions/question_types.py"),
            module.__name__,
            "exec",
        ),
        module.__dict__,
    )
    return module


@pytest.fixture
def pinned_errors(pinned_source, native_module):
    module = native_module("exceptions")
    exec(  # noqa: S102 -- Executes a hash-verified, stdlib-only upstream module.
        compile(pinned_source("vllm/exceptions.py"), module.__name__, "exec"),
        module.__dict__,
    )
    return module


@pytest.fixture
def pinned_engine_errors():
    return load_pinned_engine_errors()


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(native_lifecycle, "load_native", lambda: NativeApi())


@pytest.fixture
def native_reads(
    monkeypatch,
    pinned_source,
    native_module,
    pinned_questions,
    pinned_errors,
    pinned_engine_errors,
):
    source = pinned_source
    module = native_module
    questions = pinned_questions
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
        partial=partial,
        logger=logging.getLogger(__name__),
        MAX_LOGPROB_TOKEN_IDS=MAX_LOGPROB_TOKEN_IDS,
        ABC=ABC,
        abstractmethod=abstractmethod,
        OpenAIBaseModel=SimpleNamespace,
        SamplingParams=SamplingParams,
        ChatParams=SimpleNamespace,
        TokenizeParams=SimpleNamespace,
        merge_kwargs=lambda first, second: {**(first or {}), **(second or {})},
        tokens_input=tokens_input,
        extract_prompt_components=extract_prompt_components,
        contextlib=contextlib,
        FIRST_COMPLETED=asyncio.FIRST_COMPLETED,
        GenerationError=pinned_errors.GenerationError,
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
            if (
                isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                and n.name in selected
            )
            or (
                isinstance(n, ast.Assign)
                and any(
                    isinstance(target, ast.Name)
                    and target.id == "MIN_WARM_PREFIX_TOKENS"
                    for target in n.targets
                )
            )
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
        validation_error=pinned_errors.VLLMValidationError,
        template_error=TemplateError,
        engine_generate_error=pinned_engine_errors.EngineGenerateError,
        engine_dead_error=pinned_engine_errors.EngineDeadError,
        retryable_request_error=pinned_errors.RetryableRequestError,
    )
    monkeypatch.setattr(native_lifecycle, "load_native", lambda: api)
    return api


@pytest.fixture
def pinned_preparation(monkeypatch, pinned_source, native_module, pinned_questions):
    question_module = pinned_questions
    tree = ast.parse(pinned_source("vllm/entrypoints/openai/decisions/strategies.py"))
    selected = {
        "state_text",
        "DecisionLimits",
        "D3Strategy",
        "JointSchemaStrategy",
        "ClefFlashStrategy",
    }
    definitions = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in selected
    ]

    class ReadStrategy:
        def __init__(self, context):
            self.context = context

    strategy_module = native_module("head_strategies")
    namespace = strategy_module.__dict__
    namespace.update(
        {
            "ReadStrategy": ReadStrategy,
            "json": json,
            "dataclass": dataclass,
            "LABELS": question_module.LABELS,
            "StructuredDecisionError": question_module.StructuredDecisionError,
        }
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    exec(  # noqa: S102 -- Executes hash-verified upstream fixtures, never request data.
        compile(ast.fix_missing_locations(module), "pinned-head-strategies", "exec"),
        namespace,
    )
    api = NativeApi()
    api.questions = question_module
    api.strategies = SimpleNamespace(
        **{name: namespace[name] for name in selected if name != "state_text"},
        ReadContext=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setattr(native_lifecycle, "load_native", lambda: api)
