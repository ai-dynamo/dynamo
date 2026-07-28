# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evolutionary operators for the Squeeze-Evolve diversity loop.

Native port of the operator set from Squeeze-Evolve (https://arxiv.org/abs/2604.07725):
selection, group fitness (diversity), percentile routing thresholds, route
assignment, the aggregate recombination prompt, population update, and math
answer extraction. Plain functions — no registry, no plugin system, no Pydantic.
"""

from __future__ import annotations

import random
import re
from typing import Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Answer extraction (math)
# ---------------------------------------------------------------------------


def strip_think_blocks(text: str) -> str:
    """Remove ``<think>...</think>`` chain-of-thought wrappers."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


def _extract_boxed_content(text: str) -> str | None:
    """Return the content of the last balanced ``\\boxed{...}``, or None."""
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    idx = start + len(marker)
    depth = 1
    chars: list[str] = []
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
        chars.append(char)
        idx += 1
    return None


def _extract_tagged_answer(text: str) -> str | None:
    """Return the content of an ``<answer>...</answer>`` tag, or None."""
    match = re.search(
        r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else None


def _extract_final_answer_line(text: str) -> str | None:
    """Return the ``final answer:`` value, else the last non-empty line."""
    match = re.search(r"final answer\s*:\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else None


def _normalize_math_answer(text: str) -> str:
    """Canonicalize a math answer (strip LaTeX/`$`/commas, int-normalize)."""
    stripped = text.strip()
    stripped = stripped.replace("\\(", "").replace("\\)", "")
    stripped = stripped.replace("\\[", "").replace("\\]", "")
    if stripped.startswith("$") and stripped.endswith("$") and len(stripped) >= 2:
        stripped = stripped[1:-1]
    stripped = stripped.strip().replace(",", "")
    stripped = re.sub(r"\s+", "", stripped).strip(".")
    if stripped.startswith("\\text{") and stripped.endswith("}"):
        stripped = stripped[6:-1].strip()
    if re.fullmatch(r"[-+]?\d+", stripped):
        return str(int(stripped))
    return stripped.lower()


def extract_boxed_math_answer(candidate: str) -> str:
    """Extract a normalized final answer from a candidate solution."""
    text = strip_think_blocks(candidate or "")
    boxed = _extract_boxed_content(text)
    if boxed is not None:
        return _normalize_math_answer(boxed)
    tagged = _extract_tagged_answer(text)
    if tagged is not None:
        return _normalize_math_answer(tagged)
    final_line = _extract_final_answer_line(text)
    if final_line is None:
        return ""
    inline_boxed = _extract_boxed_content(final_line)
    if inline_boxed is not None:
        return _normalize_math_answer(inline_boxed)
    number_match = re.search(r"[-+]?\d+", final_line.replace(",", ""))
    if number_match:
        return str(int(number_match.group(0)))
    return _normalize_math_answer(final_line)


def extract_answer(candidate: str, task: str) -> str:
    """Task-aware answer extraction used for the diversity fitness signal."""
    if task in ("math", "gpqa_diamond"):
        return extract_boxed_math_answer(candidate)
    answer = extract_boxed_math_answer(candidate)
    return answer if answer else strip_think_blocks(candidate or "")


# ---------------------------------------------------------------------------
# Selection / fitness / routing
# ---------------------------------------------------------------------------


def select_uniform(
    candidates: Sequence[str], k: int, m: int, rng: random.Random | None = None
) -> list[list[int]]:
    """Return ``m`` uniform-random index subsets of size ``k`` from ``candidates``.

    ``rng`` is a request-local ``random.Random`` for per-request isolation and
    reproducibility; it defaults to the shared ``random`` module.
    """
    n = len(candidates)
    sampler = rng or random
    return [sampler.sample(range(n), k) for _ in range(m)]


def group_diversity(answers: Sequence[str]) -> float:
    """Answer diversity: count of unique final answers (higher = more disagreement)."""
    return float(len(set(answers)))


def compute_thresholds(
    fitnesses: Sequence[float], percentiles: Sequence[float]
) -> list[float]:
    """Per-problem routing thresholds from ``percentiles`` (sorted ascending)."""
    results: list[float] = []
    for p in sorted(percentiles):
        if p >= 100:
            results.append(float(max(fitnesses)) + 1.0)
        elif p <= 0:
            results.append(float(min(fitnesses)) - 1.0)
        else:
            results.append(float(np.percentile(fitnesses, p)))
    return results


def assign_routes(
    fitnesses: Sequence[float], thresholds: list[float], n_tiers: int
) -> list[int]:
    """Map each group's fitness to a tier index (0=cheapest .. n_tiers-1=most expensive).

    Low fitness (hard/uncertain) routes to the most expensive tier; high fitness
    (easy/consensus) to the cheapest. With N-1 ascending thresholds, a fitness at
    or below ``thresholds[i]`` goes to tier ``n_tiers-1-i``; anything above all
    thresholds goes to tier 0.
    """
    routes: list[int] = []
    for f in fitnesses:
        tier = 0
        for i, t in enumerate(thresholds):
            if f <= t:
                tier = n_tiers - 1 - i
                break
        routes.append(tier)
    return routes


# ---------------------------------------------------------------------------
# Recombination prompt / population update / lite aggregation
# ---------------------------------------------------------------------------


def make_aggregate_prompt(query: str, candidates: list[str], answer_format: str) -> str:
    """Build a task-aware aggregate recombination prompt for a group of candidates."""
    if not candidates:
        return query

    parts: list[str] = []
    if len(candidates) == 1:
        parts.append(
            "You are given a problem and a candidate solution. The candidate may "
            "be incomplete or contain errors. Refine this trajectory and produce "
            "an improved, higher-quality solution. If it is entirely wrong, "
            "attempt a new strategy. End with the final result in "
            f"{answer_format}.\n"
        )
    else:
        parts.append(
            "You are given a problem and several candidate solutions. Some "
            "candidates may be incorrect or contain errors. Aggregate the useful "
            "ideas and produce a single, high-quality solution. Reason carefully; "
            "if candidates disagree, choose the correct path. If all are "
            f"incorrect, attempt a different strategy. End with the final result "
            f"in {answer_format}.\n"
        )

    parts.append("Problem:\n")
    parts.append(query.strip() + "\n")
    if len(candidates) == 1:
        parts.append("Candidate solution (may contain mistakes):\n")
        parts.append(f"---- Candidate ----\n{(candidates[0] or '').strip()}\n")
        parts.append(
            "Now refine the candidate into an improved solution. Provide clear "
            f"reasoning and end with the final answer in {answer_format}."
        )
    else:
        parts.append("Candidate solutions (may contain mistakes):\n")
        for i, ans in enumerate(candidates, 1):
            parts.append(f"---- Solution {i} ----\n{(ans or '').strip()}\n")
        parts.append(
            "Now write a single improved solution. Provide clear reasoning and "
            f"end with the final answer in {answer_format}."
        )
    return "\n".join(parts)


def update_replace(old: list[str], new: list[str]) -> list[str]:
    """Population update: replace the old population with the new candidates."""
    return new


__all__ = [
    "assign_routes",
    "compute_thresholds",
    "extract_answer",
    "extract_boxed_math_answer",
    "group_diversity",
    "make_aggregate_prompt",
    "select_uniform",
    "strip_think_blocks",
    "update_replace",
]
