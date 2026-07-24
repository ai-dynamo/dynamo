# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Squeeze-Evolve evolutionary orchestrator: native port for Dynamo.

Native re-implementation of the Squeeze-Evolve diversity loop
(https://arxiv.org/abs/2604.07725) as a Dynamo component. The orchestrator owns
one ``KvRouter`` (and one tokenizer) per model tier and runs the evolutionary
loop per chat request: a population is initialized on the most expensive tier,
then each loop selects candidate groups, scores them by answer diversity, routes
each group to the cheapest capable tier, recombines, and updates the population.

No external ``squeeze_evolve`` package, no Pydantic, no operator registry, no
storage/checkpoint/metrics machinery.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from typing import Any, Optional

from dynamo.squeeze_evolve.operators import (
    assign_routes,
    compute_thresholds,
    extract_answer,
    group_diversity,
    make_aggregate_prompt,
    select_uniform,
    update_replace,
)

try:  # Rust bindings: absent in binding-free unit-test envs (tests inject fakes).
    from dynamo.llm import KvRouter
except ImportError:  # pragma: no cover
    KvRouter = None  # type: ignore[assignment,misc]

try:  # core dependency, but fail gracefully if the wheel is missing.
    from transformers import AutoTokenizer
except ImportError:  # pragma: no cover
    AutoTokenizer = None  # type: ignore[assignment,misc]

logger = logging.getLogger(__name__)

# Recombination answer format requested of the models (math/boxed).
_ANSWER_FORMAT = "\\boxed{}"


# ---------------------------------------------------------------------------
# Config dataclasses (plain — no Pydantic)
# ---------------------------------------------------------------------------


@dataclass
class Tier:
    """One model tier = a dynamo:// worker endpoint + its model/tokenizer + sampling."""

    endpoint: str  # 3-part "namespace.component.endpoint"
    model: str  # HF repo id; served model name + tokenizer source
    temperature: float = 1.0
    top_p: float = 1.0
    max_tokens: int = 8192
    block_size: int = 0  # 0 -> use the component default block size
    tokenizer: Optional[str] = None  # defaults to `model`
    trust_remote_code: bool = False  # opt-in per tier; runs the model repo's code


def parse_tiers(raw: str) -> list[Tier]:
    """Parse the ``--tiers`` JSON array (cheapest first) into Tier objects.

    Raises ``ValueError`` on malformed input (argparse surfaces it as a usage error).
    """
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--tiers is not valid JSON: {exc}") from exc
    if not isinstance(data, list) or not data:
        raise ValueError("--tiers must be a non-empty JSON array")
    tiers: list[Tier] = []
    for i, entry in enumerate(data):
        if (
            not isinstance(entry, dict)
            or "endpoint" not in entry
            or "model" not in entry
        ):
            raise ValueError(
                f"--tiers[{i}] must be an object with 'endpoint' and 'model'"
            )
        parts = str(entry["endpoint"]).split(".")
        if len(parts) != 3 or any(not part for part in parts):
            raise ValueError(
                f"--tiers[{i}].endpoint must be 'namespace.component.endpoint', "
                f"got {entry['endpoint']!r}"
            )
        try:
            tiers.append(Tier(**entry))
        except TypeError as exc:
            raise ValueError(f"--tiers[{i}]: {exc}") from exc
    return tiers


@dataclass
class SqueezeEvolveConfig:
    """Plain algorithm config consumed by the orchestrator."""

    k: int = 4
    population: int = 16
    groups: int = 16
    loops: int = 5
    confidence_percentiles: list[float] = field(default_factory=lambda: [50.0])
    task: str = "math"
    seed: Optional[int] = None
    tier_concurrency: int = 32


# ---------------------------------------------------------------------------
# Per-tier transport (one KvRouter + tokenizer per tier)
# ---------------------------------------------------------------------------


class TierTransport:
    """Generates over the Dynamo runtime for one model tier.

    Tokenizes a prompt with the tier's tokenizer (token-in / token-out worker
    path), dispatches via the tier's ``KvRouter``, and detokenizes the streamed
    output. ``dynamo.llm`` / ``transformers`` imports are lazy so the orchestrator
    module imports without the bindings (tests inject fakes instead).
    """

    def __init__(
        self,
        runtime: Any,
        tier: Tier,
        kv_router_config: Any,
        aic_perf_config: Any,
        default_block_size: int,
        concurrency: int = 32,
    ) -> None:
        self._runtime = runtime
        self._tier = tier
        self._kv_router_config = kv_router_config
        self._aic_perf_config = aic_perf_config
        self._block_size = tier.block_size or default_block_size
        self._semaphore = asyncio.Semaphore(concurrency)
        self._router: Any = None
        self._tokenizer: Any = None
        self._eos_ids: Optional[list[int]] = None

    def _get_router(self) -> Any:
        if self._router is None:
            if KvRouter is None:
                raise RuntimeError(
                    "dynamo.llm bindings are unavailable; build them with "
                    "`maturin develop` before serving."
                )
            endpoint = self._runtime.endpoint(self._tier.endpoint)
            self._router = KvRouter(
                endpoint=endpoint,
                block_size=self._block_size,
                kv_router_config=self._kv_router_config,
                aic_perf_config=self._aic_perf_config,
            )
        return self._router

    def _get_tokenizer(self) -> Any:
        if self._tokenizer is None:
            if AutoTokenizer is None:
                raise RuntimeError("transformers is required to tokenize tier prompts.")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._tier.tokenizer or self._tier.model,
                trust_remote_code=self._tier.trust_remote_code,
            )
        return self._tokenizer

    def _get_eos_token_ids(self, tokenizer: Any) -> list[int]:
        """EOS token id(s) from the tier tokenizer so generation stops at turn end."""
        if self._eos_ids is None:
            ids: list[int] = []
            eos = getattr(tokenizer, "eos_token_id", None)
            if isinstance(eos, int):
                ids.append(eos)
            elif isinstance(eos, (list, tuple)):
                ids.extend(int(x) for x in eos if isinstance(x, int))
            self._eos_ids = ids
        return self._eos_ids

    async def generate(self, messages: list[dict[str, str]]) -> str:
        """Generate one completion for a chat message list on this tier."""
        tokenizer = self._get_tokenizer()
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        # transformers >=5 returns a BatchEncoding (dict-like); <5 returns a
        # plain list[int]. Pull out input_ids so token_ids is always list[int].
        if hasattr(encoded, "input_ids"):
            encoded = encoded.input_ids
        token_ids = [int(t) for t in encoded]
        # Schema mirrors lib/llm/src/protocols/common/preprocessor.rs.
        request = {
            "model": self._tier.model,
            "token_ids": token_ids,
            "stop_conditions": {"max_tokens": self._tier.max_tokens},
            "sampling_options": {
                "temperature": self._tier.temperature,
                "top_p": self._tier.top_p,
            },
            "output_options": {},
            "eos_token_ids": self._get_eos_token_ids(tokenizer),
            "annotations": [],
        }
        router = self._get_router()
        out_tokens: list[int] = []
        async with self._semaphore:
            async for chunk in await router.generate_from_request(request):
                if not isinstance(chunk, dict):
                    continue
                finish = chunk.get("finish_reason")
                if chunk.get("status") == "error" or (
                    isinstance(finish, str) and finish.startswith("error")
                ):
                    raise RuntimeError(
                        f"tier {self._tier.endpoint} generation failed: "
                        f"{chunk.get('err_msg') or finish or 'unknown error'}"
                    )
                out_tokens.extend(chunk.get("token_ids") or [])
        return tokenizer.decode(out_tokens, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class SqueezeEvolveOrchestrator:
    """Runs the Squeeze-Evolve diversity loop for one problem per chat request."""

    def __init__(
        self,
        *,
        cfg: SqueezeEvolveConfig,
        tiers: Optional[list[Tier]] = None,
        runtime: Any = None,
        kv_router_config: Any = None,
        aic_perf_config: Any = None,
        default_block_size: int = 64,
        transports: Optional[list[Any]] = None,
    ) -> None:
        self._cfg = cfg
        # tiers[0] = cheapest, tiers[-1] = most expensive (Loop 0 generator).
        if transports is not None:
            if not transports:
                raise ValueError("transports must be a non-empty list")
            self._transports = transports  # injected (tests)
        else:
            if not tiers:
                raise ValueError("tiers required when transports are not injected")
            self._transports = [
                TierTransport(
                    runtime,
                    tier,
                    kv_router_config,
                    aic_perf_config,
                    default_block_size,
                    concurrency=cfg.tier_concurrency,
                )
                for tier in tiers
            ]
        self._percentiles = list(cfg.confidence_percentiles)

    async def run(self, messages: list[dict[str, str]]) -> str:
        """Run the evolutionary loop for one chat request; return the final answer."""
        cfg = self._cfg
        # Request-local RNG: a process-global seed would let concurrent requests
        # overwrite each other's selection sequence. seed=None -> nondeterministic.
        rng = random.Random(cfg.seed)

        n_tiers = len(self._transports)
        # Plain-text problem statement (rendered from the chat history) that each
        # recombination prompt embeds.
        question_text = "\n\n".join(
            m["content"] for m in messages if isinstance(m, dict) and m.get("content")
        )

        # -- Loop 0: most expensive tier generates the initial population --------
        candidates = list(
            await asyncio.gather(
                *(
                    self._transports[-1].generate(messages)
                    for _ in range(cfg.population)
                )
            )
        )
        logger.info("Loop 0: %d candidates from tier %d", len(candidates), n_tiers - 1)

        # -- Loops 1..T-1: score -> select -> route -> recombine -> update -------
        for t in range(1, cfg.loops):
            indices = select_uniform(candidates, cfg.k, cfg.groups, rng)
            # Routing fitness is NEGATIVE answer diversity: a group whose candidates
            # agree (low diversity = easy) scores high and routes to the cheap tier;
            # one that disagrees (high diversity = hard) scores low and routes to the
            # expensive tier. (assign_routes sends low fitness to the expensive end,
            # same as the confidence signal, where high confidence = easy = cheap.)
            gf = [
                -group_diversity([extract_answer(candidates[i], cfg.task) for i in grp])
                for grp in indices
            ]
            thresholds = (
                compute_thresholds(gf, self._percentiles) if n_tiers > 1 else []
            )
            routes = assign_routes(gf, thresholds, n_tiers)

            # Bucket each group's recombination prompt by its routed tier,
            # remembering original order so we can un-interleave the responses.
            tier_prompts: dict[int, list[str]] = {i: [] for i in range(n_tiers)}
            order: list[tuple[int, int]] = []
            for g_idx, grp in enumerate(indices):
                group_cands = [candidates[i] for i in grp]
                tier = routes[g_idx]
                order.append((tier, len(tier_prompts[tier])))
                tier_prompts[tier].append(
                    make_aggregate_prompt(question_text, group_cands, _ANSWER_FORMAT)
                )

            async def _agg(i: int, prompts: list[str]) -> list[str]:
                if not prompts:
                    return []
                return list(
                    await asyncio.gather(
                        *(
                            self._transports[i].generate(
                                [{"role": "user", "content": p}]
                            )
                            for p in prompts
                        )
                    )
                )

            tier_resps = await asyncio.gather(
                *(_agg(i, tier_prompts[i]) for i in range(n_tiers))
            )

            cursors = {i: 0 for i in range(n_tiers)}
            merged: list[str] = []
            for tier, _ in order:
                merged.append(tier_resps[tier][cursors[tier]])
                cursors[tier] += 1
            candidates = update_replace(candidates, merged)

            counts = {i: routes.count(i) for i in range(n_tiers)}
            logger.info("Loop %d: routes=%s", t, counts)

        # -- Final answer: first candidate of the evolved population -------------
        return candidates[0] if candidates else ""


__all__ = [
    "SqueezeEvolveConfig",
    "SqueezeEvolveOrchestrator",
    "Tier",
    "TierTransport",
    "parse_tiers",
]
