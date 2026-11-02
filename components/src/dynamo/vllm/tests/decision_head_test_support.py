# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU contracts for the optional, unregistered native head executor."""

import asyncio
import math
from contextlib import asynccontextmanager
from dataclasses import dataclass
from types import SimpleNamespace

from jinja2 import TemplateError

from dynamo.vllm import decision_heads as heads
from dynamo.vllm import decision_native_registry as native_registry


@dataclass(frozen=True)
class Option:
    name: str
    description: str | None = None


@dataclass(frozen=True)
class Question:
    id: str
    type: object
    instructions: str
    options: tuple
    labels: tuple


class DecisionContext(list):
    """Minimal runtime context with a genuine asynchronous cancellation waker."""

    async def async_killed_or_stopped(self):
        await asyncio.Event().wait()


class Tensor:
    def __init__(self, values):
        self.values = values
        self.ndim = 1

    def tolist(self):
        return list(self.values)


class Tokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())

    def apply_chat_template(self, messages, **kwargs):
        return "\n".join(message["content"] for message in messages)


class D3Strategy:
    def __init__(self, context):
        self.context = context
        self.codes = ["A", "B", "C"]
        self.temperature = 2.0
        self.tokenizer = context.online_renderer.renderer.get_tokenizer()
        self.max_model_len = context.engine_client.model_config.max_model_len

    def limits(self):
        return SimpleNamespace(max_questions=64, max_options=len(self.codes))

    def render_state(self, state):
        return "(empty)" if state is None or state == "" else str(state)

    def _options(self, question):
        if question.type.name == "noul":
            return ["No / false", "Yes / true"], [1, 0]
        return [option.name for option in question.options], list(
            range(len(question.options))
        )

    def _prompt(self, state, question, media):
        assert media == []
        names, _ = self._options(question)
        return "\n".join([state, question.instructions, *names])

    def _tokens(self, text):
        return self.tokenizer.encode(text, add_special_tokens=False)


class JointStrategy(D3Strategy):
    def limits(self):
        return SimpleNamespace(max_questions=64, max_options=26)

    def encode(self, questions, state):
        orders = [
            (
                sorted(range(len(q.options)), key=lambda i: q.options[i].name)
                if q.type.name == "choice"
                else list(range(len(q.options)))
            )
            for q in questions
        ]
        prompt = self._tokens(state + "|" + "|".join(q.id for q in questions))
        schema = {
            "question_types": [0 if q.type.name == "noul" else 1 for q in questions],
            "question_spans": [[-2, -1]] * len(questions),
            "option_spans": [[-2, -1]] * sum(len(q.options) for q in questions),
            "option_counts": [len(q.options) for q in questions],
        }
        return prompt, schema, orders


class NativeApi:
    template_error = TemplateError
    overload_errors = ()
    engine_generate_error = ()
    engine_dead_error = ()
    retryable_request_error = ()
    validation_error = ValueError
    strategies = SimpleNamespace(
        D3Strategy=D3Strategy,
        ClefFlashStrategy=JointStrategy,
        JointSchemaStrategy=JointStrategy,
        ReadContext=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    questions = SimpleNamespace(
        Option=Option,
        Question=Question,
        StructuredDecisionError=ValueError,
        get_question_type=lambda name: SimpleNamespace(name=name),
        label_softmax=lambda values: [
            math.exp(value - max(values))
            / sum(math.exp(item - max(values)) for item in values)
            for value in values
        ],
    )

    @staticmethod
    def pooling_params(**kwargs):
        return SimpleNamespace(**kwargs)

    @staticmethod
    def tokens_input(token_ids, *, cache_salt):
        return {"prompt_token_ids": list(token_ids), "cache_salt": cache_salt}

    @staticmethod
    def online_renderer(**kwargs):
        return SimpleNamespace(**kwargs)


class Engine:
    def __init__(self, architecture="D3ForSequenceClassification"):
        self.model_config = SimpleNamespace(
            architecture=architecture,
            runner_type="pooling",
            max_model_len=8192,
            is_encoder_decoder=False,
            is_diffusion=False,
            model_impl="auto",
            model_class_overrides={},
            convert_type="none",
        )
        self.vllm_config = SimpleNamespace(
            model_config=self.model_config,
            parallel_config=SimpleNamespace(data_parallel_size=1),
            scheduler_config=SimpleNamespace(
                enable_chunked_prefill=False,
                max_num_seqs=8,
                async_scheduling=False,
                scheduler_cls=None,
            ),
            cache_config=SimpleNamespace(enable_prefix_caching=False),
            lora_config=None,
            speculative_config=None,
            kv_transfer_config=None,
            ec_transfer_config=None,
        )
        self.renderer = SimpleNamespace(
            get_tokenizer=lambda: Tokenizer(),
            default_cmpl_tok_params=SimpleNamespace(
                apply_post_tokenization=lambda tokenizer, prompt: prompt
            ),
        )
        self.calls = []
        self.aborted = []
        self.admissions = []
        self.values = [0.0, 2.0, 99.0]
        self.mutate_output = lambda output: output
        self.block = None
        self.started = asyncio.Event()
        self.active = 0
        self.peak_active = 0

    async def get_supported_tasks(self):
        return ("classify", "token_classify")

    def check_admission(self, count):
        self.admissions.append(count)

    async def abort(self, request_id):
        self.aborted.append(request_id)

    async def encode(self, **kwargs):
        self.calls.append(kwargs)
        self.active += 1
        self.peak_active = max(self.peak_active, self.active)
        self.started.set()
        try:
            if self.block is not None:
                await self.block.wait()
            output = SimpleNamespace(
                request_id=kwargs["request_id"],
                prompt_token_ids=kwargs["prompt"]["prompt_token_ids"],
                num_cached_tokens=1,
                finished=True,
                error=None,
                outputs=SimpleNamespace(data=Tensor(self.values)),
            )
            yield self.mutate_output(output)
        finally:
            self.active -= 1


@asynccontextmanager
async def monitor(context, request_id):
    if isinstance(context, list):
        context.append(request_id)
    yield


def question(ordinal=0, kind="Choice", values=("z", "a")):
    return {
        "ordinal": ordinal,
        "id": "client-name-not-instructions",
        "kind": kind,
        "instructions": "Choose.",
        "candidates": [
            {"value": value, "description": None, "label": None} for value in values
        ],
    }


def wire(profile=None, questions=None):
    return {
        "decision_head": {
            "protocol_version": 1,
            "profile": profile or native_registry.D3_PROFILE,
            "model": "test/model",
            "input": "state",
            "questions": questions or [question()],
            "temperature": 1.0,
            "cache_salt": "isolated",
            "max_input_tokens": 16384,
            "max_tokenization_work": 65536,
            "max_encoding_bytes": 65536,
        }
    }


async def adapter(engine, profile=None, **kwargs):
    return await heads.HeadExecutionAdapter.create(
        engine,
        native_registry.PROFILES[profile or native_registry.D3_PROFILE],
        abort_monitor=monitor,
        **kwargs,
    )


async def execute(instance, request, **kwargs):
    return await instance.execute(
        request,
        DecisionContext(),
        request_id="parent",
        cancelled=lambda: False,
        **kwargs,
    )
