# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned strategy/reducer CPU fixtures; no installed backend or GPU execution."""

import asyncio
from types import SimpleNamespace

MAX_LOGPROB_TOKEN_IDS = 128


class SamplingParams(SimpleNamespace):
    """Only the source-pinned sampling fields exercised by CPU native reads."""

    def __init__(
        self, *, max_tokens=16, temperature=1.0, logprob_token_ids=None, extra_args=None
    ):
        super().__init__(
            max_tokens=max_tokens,
            temperature=temperature,
            logprob_token_ids=logprob_token_ids,
            extra_args=extra_args,
        )


def tokens_input(prompt_token_ids, *, prompt=None, cache_salt=None):
    inputs = {"type": "token", "prompt_token_ids": prompt_token_ids}
    if prompt is not None:
        inputs["prompt"] = prompt
    if cache_salt is not None:
        inputs["cache_salt"] = cache_salt
    return inputs


def extract_prompt_components(model_config, prompt):
    """Support only the decoder-only token inputs used by the CPU fixtures."""
    if model_config.is_encoder_decoder or prompt["type"] != "token":
        raise ValueError("CPU native fixtures require decoder-only token input")
    return SimpleNamespace(
        text=prompt.get("prompt"),
        token_ids=prompt.get("prompt_token_ids"),
        embeds=prompt.get("prompt_embeds"),
    )


class Tokenizer:
    pad_token_id = 0

    def encode(self, text, *, add_special_tokens=False):
        specials = {"<|channel>thought\n<channel|>": [8001, 8002], "<turn|>": [8003]}
        if text in specials:
            return specials[text]
        if text == "\n":
            return [4106, 4106]  # Disable the optional joint-canvas path.
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
    """Signature-faithful renderer; text/token behavior remains a CPU stand-in."""

    def __init__(
        self,
        model_config,
        renderer,
        *,
        request_logger,
        chat_template,
        chat_template_content_format,
        trust_request_chat_template=False,
        trust_request_mm_kwargs=False,
        enable_auto_tools=False,
        exclude_tools_when_tool_choice_none=False,
        tool_parser=None,
        reasoning_parser=None,
        tool_strict_level=None,
        default_chat_template_kwargs=None,
        reasoning_effort_map=None,
        log_error_stack=False,
    ):
        self.renderer = renderer
        self.model_config = model_config

    async def preprocess_chat(
        self,
        request,
        messages,
        default_template,
        default_template_content_format,
        default_template_kwargs,
        tool_dicts=None,
        parser=None,
        *,
        skip_mm_cache=False,
    ):
        self.renderer.rendered.append(messages)
        ids = self.renderer.get_tokenizer().apply_chat_template(messages)
        return [], [tokens_input(ids, cache_salt=request.cache_salt)]


class Engine:
    def __init__(self, diffusion=False):
        self.model_config = SimpleNamespace(
            architecture=(
                "DiffusionGemmaForBlockDiffusion"
                if diffusion
                else "Lfm2VlForConditionalGeneration"
            ),
            runner_type="generate",
            is_encoder_decoder=False,
            is_diffusion=diffusion,
            supports_multimodal_inputs=False,
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
            scheduler_config=SimpleNamespace(
                max_num_seqs=4,
                async_scheduling=False,
                scheduler_cls=(
                    "vllm.v1.core.sched.diffusion_scheduler.DiffusionScheduler"
                    if diffusion
                    else None
                ),
            ),
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
        self.cached_tokens = 0
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
