# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TokTier 0.3.0 CPU adapter. Tokenizer artifacts must be staged before startup.

Each session owns its own TokTier tokenizer/store: upstream's per-tokenizer store
lock must not serialize unrelated Dynamo sessions. No persistent store is opened.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from contextlib import AbstractContextManager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock
from typing import Any

from .delta_tokenizer import (
    DeltaBudget,
    DeltaConfig,
    DeltaTokenizer,
    TokenizerIdentity,
    UnsupportedTransition,
)

# TokTier is an optional extra. Disabled frontends never load it.
try:
    from tokenizers import Tokenizer as HfCoreTokenizer
    from toktier import Config, Session, Tokenizer, ToktierError, load
    from toktier.artifacts import ArtifactManifest
    from toktier.artifacts.tables import ARTIFACT_MANIFEST
    from toktier.integrations.sglang import check_core_identity
    from transformers import PreTrainedTokenizerFast
except ImportError:
    _AVAILABLE = False
else:
    _AVAILABLE = True

logger = logging.getLogger(__name__)
_CERTIFIED_ORACLES = frozenset({"0.22.2", "0.23.2", "0.23.3"})
_BUDGET_LOCK = Lock()
_PREPARE_LOCK = Lock()
_PROCESS_BUDGET: DeltaBudget | None = None


def _process_budget(config: DeltaConfig) -> DeltaBudget:
    global _PROCESS_BUDGET
    with _BUDGET_LOCK:
        if _PROCESS_BUDGET is None:
            _PROCESS_BUDGET = DeltaBudget(config.max_entries, config.max_bytes)
        if (_PROCESS_BUDGET.max_entries, _PROCESS_BUDGET.max_bytes) != (
            config.max_entries,
            config.max_bytes,
        ):
            raise ValueError("delta tokenizer process budget is already configured")
        return _PROCESS_BUDGET


def _certified(report: dict[str, Any]) -> bool:
    certification = report.get("certification")
    probe = report.get("probe")
    if not isinstance(certification, dict) or not isinstance(probe, dict):
        return False
    oracle = probe.get("oracle_version")
    return (
        certification.get("effective_verdict") == "certified"
        and certification.get("state") in ("certified", "certified_source")
        and report.get("supported_untested") == []
        and report.get("experimental_waivers") == []
        and report.get("planned_backend") == "fast_cpu"
        and isinstance(oracle, str)
        and oracle in _CERTIFIED_ORACLES
    )


class _Session:
    def __init__(self, family: str, reference: Any, cache_bytes: int) -> None:
        assert _AVAILABLE
        self._seeded = False
        self._owned: tuple[
            Tokenizer, AbstractContextManager[Session], Session
        ] | None = None
        tokenizer = load(
            family,
            device="cpu",
            policy="certified",
            config=Config(offline=True, disable_gpu=True),
            cache_budget_bytes=cache_bytes,
        )
        try:
            if (
                not _certified(tokenizer.explain())
                or check_core_identity(reference, tokenizer)["status"] != "equivalent"
            ):
                raise UnsupportedTransition
            context = tokenizer.session()
            self._owned = tokenizer, context, context.__enter__()
        except BaseException:
            tokenizer.close()
            raise

    def append(self, text: str) -> tuple[int, list[int]]:
        if self._owned is None:
            raise UnsupportedTransition
        tokenizer, _context, session = self._owned
        try:
            patch = session.append_patch(text)
            report = tokenizer.explain()
            repair = report["session_repair"]
            if not isinstance(repair, dict):
                raise UnsupportedTransition
            if not _certified(report):
                raise UnsupportedTransition
            if self._seeded:
                # A cold seed can use the certified HF oracle; the manager
                # compares it to Dynamo's full reference before publication.
                last = repair["last"]
                if (
                    repair["status"] != "active"
                    or last is None
                    or last["reason"] is not None
                    or report["backend"] != "fast_cpu"
                ):
                    raise UnsupportedTransition
            self._seeded = True
            return patch.replace_from, list(patch.replacement_ids)
        except (ToktierError, KeyError, TypeError) as exc:
            raise UnsupportedTransition from exc

    def close(self) -> None:
        owned, self._owned = self._owned, None
        if owned is not None:
            tokenizer, context, _session = owned
            try:
                context.__exit__(None, None, None)
            finally:
                tokenizer.close()


def _create_delta_tokenizer(
    reference: Any,
    local_dir: str,
    model: str,
    config: DeltaConfig,
) -> DeltaTokenizer | None:
    """Admit an immutable snapshot of a supported standard HF fast tokenizer.

    Unknown package versions, custom loaders and unstaged artifacts stay on the
    original frontend path. Construction never downloads or changes dependencies.
    """
    if not _AVAILABLE:
        logger.warning(
            "delta tokenizer unavailable: install ai-dynamo[delta-tokenizer]"
        )
        return None
    try:
        if (
            version("toktier") != "0.3.0"
            or version("tokenizers") not in _CERTIFIED_ORACLES
        ):
            logger.warning("delta tokenizer unavailable: uncertified package version")
            return None
        from_module = type(reference).__module__
        if not from_module.startswith("transformers."):
            return None
        # Keep the standard HF encode contract; custom wrappers can change IDs
        # without changing the backend JSON and are not admitted.
        if (
            not isinstance(reference, PreTrainedTokenizerFast)
            or type(reference).encode is not PreTrainedTokenizerFast.encode
            or type(reference).__call__ is not PreTrainedTokenizerFast.__call__
        ):
            return None
        if not isinstance(reference.backend_tokenizer, HfCoreTokenizer):
            return None
        serialized = reference.backend_tokenizer.to_str()
        document = json.loads(serialized)
        if document["truncation"] is not None or document["padding"] is not None:
            return None
        if (
            reference.split_special_tokens
            or reference.backend_tokenizer.encode_special_tokens
        ):
            return None
        artifact = hashlib.sha256(
            Path(local_dir, "tokenizer.json").read_bytes()
        ).hexdigest()
        manifest = ArtifactManifest.load(ARTIFACT_MANIFEST)
        family = next(
            (
                name
                for name in manifest.families()
                if manifest.get(name).file("tokenizer.json").sha256 == artifact
            ),
            None,
        )
        if family is None:
            return None
        # This private clone is the configured reference captured at model-load
        # time. No caller can mutate it through the frontend tokenizer object.
        snapshot = copy.deepcopy(reference)
        identity = TokenizerIdentity(
            model=model,
            revision=manifest.get(family).revision,
            artifact=artifact,
            loader=f"{from_module}.{type(reference).__qualname__}:"
            f"{version('transformers')}:{version('tokenizers')}",
            configuration=hashlib.sha256(serialized.encode()).hexdigest(),
            backend="default",
        )
        # Account for loaded model tables as well as Python/native prompt state.
        # This is conservative approximate accounting, not an RSS limit.
        base_bytes = 16 * len(serialized.encode())
        if 2 * base_bytes + 2048 >= config.max_bytes:
            return None

        def factory() -> _Session:
            try:
                return _Session(family, reference, config.max_bytes - base_bytes)
            except (ToktierError, OSError, ValueError) as exc:
                raise UnsupportedTransition from exc

        # Verify availability/identity once, before admitting any request. The
        # probe owns no transcript. Real sessions use independent stores.
        probe = factory()
        probe.close()
        return DeltaTokenizer(
            lambda text: snapshot.encode(text, add_special_tokens=False),
            identity,
            factory,
            config,
            base_bytes=base_bytes,
            budget=_process_budget(config),
            background_seed=True,
            fixed_bytes=base_bytes,
            reference_after_close=lambda text: reference.encode(
                text, add_special_tokens=False
            ),
            min_prompt_bytes=4096,
        )
    except (
        PackageNotFoundError,
        ToktierError,
        OSError,
        ValueError,
        UnsupportedTransition,
    ):
        logger.warning("delta tokenizer unavailable: artifact or loader not certified")
        return None


def create_delta_tokenizer(
    reference: Any,
    local_dir: str,
    model: str,
    config: DeltaConfig,
) -> DeltaTokenizer | None:
    # Serializes model preparation only, never prompt tokenization. Frontend
    # discovery offloads this to a worker thread rather than blocking its loop.
    with _PREPARE_LOCK:
        return _create_delta_tokenizer(reference, local_dir, model, config)
