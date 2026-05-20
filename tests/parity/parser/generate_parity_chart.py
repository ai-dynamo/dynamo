#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate the parity table (matrix of cell markers) from the YAML fixtures.

================================================================================
EXAMPLE OUTPUT (truncated; illustrative, NOT a snapshot of current fixtures
— run the script for the real table):

    | model          | parser     | 1 | 2.a | 2.b | 2.c | ... | 9 | 10 |
    |---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
    | **Top-N models** |   |   |   |   |   |   |   |   |
    | Kimi K2.6      | kimi_k2    | = | =   | =   | VS  | ... | = | =  |
    | gpt-oss        | harmony †  | S | S   | n/a | S?  | ... | = | S  |
    | **Others** |   |   |   |   |   |   |   |   |
    | Mistral series | mistral    | S | S   | n/a | VS  | ... | = | S  |

================================================================================

Reads every `tests/parity/parser/fixtures/<family>/PARSER.*.yaml` and emits
the table referenced in `tests/parity/README.md`.

Cell markers (per peer, vllm + sglang):
  =     peer block is `*d_<case>` anchor ref to dynamo (matches)
  V/S   peer is a concrete inline block AND has `reason:` (intentional —
        rendered same color as =, since the divergence is accounted for)
  V?/S? peer is a concrete inline block AND has no `reason:` yet
        (research-needed; we observed it but haven't classified it)
  V!/S! peer has `error: <substring>` (expected to crash)
  VS, V?S, VS!, etc. — combinations
  n/a   peer marked `unavailable`, or family/case doesn't apply
  —     no fixture entry exists for this family/case yet

Footnote markers `†` (no vLLM peer) and `§` (no SGLang peer) are auto-derived
from `expected.<impl>.unavailable` across each family's cases.

Run:
    # Markdown table to stdout
    python3 tests/parity/parser/generate_parity_chart.py \
        > tests/parity/parser/PARITY.md

    # HTML table with clickable YAML links + hover tooltips. Write next
    # to this script so `<a href="fixtures/<family>/PARSER.batch.N.yaml">`
    # resolves when opened in a browser.
    python3 tests/parity/parser/generate_parity_chart.py --html \
        > tests/parity/parser/PARITY.html

PARITY.{md,html} are for local viewing only; don't check them in.
"""
from __future__ import annotations

import argparse
import copy
import datetime
import html as html_lib
import json
import re
import subprocess
import zoneinfo
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests/parity/parser/fixtures"
PARSER_CASES_MD = REPO_ROOT / "lib/parsers/PARSER_CASES.md"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
SCRIPT_DIR = Path(__file__).resolve().parent

RUST_TOOL_CALLING_DIR = REPO_ROOT / "lib/parsers/src/tool_calling"


def _make_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(SCRIPT_DIR),
        trim_blocks=False,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )


def _commit_sha() -> str | None:
    """HEAD SHA at table-generation time, or None if not in a git tree."""
    try:
        out = (
            subprocess.check_output(
                ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
        return out or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def _peer_versions() -> dict[str, str]:
    """Extract pinned vllm / sglang versions from pyproject.toml.

    Matches a line like `"vllm[flashinfer,runai,otel]==X.Y.Z",` (TOML is
    not parsed — the regex is sufficient and avoids a tomllib import on
    older Pythons running this script outside a Python 3.11+ env)."""
    out: dict[str, str] = {}
    if not PYPROJECT_TOML.exists():
        return out
    text = PYPROJECT_TOML.read_text()
    for name in ("vllm", "sglang"):
        m = re.search(rf'"{name}(?:\[[^\]]*\])?==([0-9][^"]*)"', text)
        if m:
            out[name] = m.group(1)
    return out


def _build_family_inheritance(
    refs: dict[str, tuple[str, int]],
) -> dict[str, dict]:
    """Derive each family's parser-inheritance map from config.rs + parsers.rs.

    Detects:
      • `ParserConfig::<Variant>(...)` — top-level backend variant
      • `JsonParserType::<Sub>`         — Json sub-dispatch (Basic / DeepseekV3 / DeepseekV31)
      • `Self::<factory>(...)`          — private factories (e.g. `deepseek_dsml`)
      • `map.insert("alias", ToolCallConfig::<family>())` — aliases (parsers.rs)

    Backend file is derived from the resolved (variant, sub_variant) tuple.
    Returns `{family: {variant, sub_variant, factory, backend_file,
    base_label, shared_with, aliases, filed_under_xml_misleading}}`.
    """
    cfg = (RUST_TOOL_CALLING_DIR / "config.rs").read_text()
    pars_path = RUST_TOOL_CALLING_DIR / "parsers.rs"
    pars = pars_path.read_text() if pars_path.exists() else ""

    # Extract all ctor bodies (pub fn + fn) — captures private factories too.
    ctor_pat = re.compile(
        r"^\s*(?:pub )?fn (\w+)\([^)]*\)\s*->\s*Self\s*\{", re.MULTILINE
    )
    bodies: dict[str, str] = {}
    for m in ctor_pat.finditer(cfg):
        start = m.end()
        depth, i = 1, start
        while i < len(cfg) and depth > 0:
            c = cfg[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
            i += 1
        bodies[m.group(1)] = cfg[start : i - 1]

    def _classify(body: str) -> tuple[str | None, str | None, str | None]:
        vm = re.search(r"ParserConfig::(\w+)\b", body)
        variant = vm.group(1) if vm else None
        sub = None
        if variant == "Json":
            sm = re.search(r"JsonParserType::(\w+)\b", body)
            sub = sm.group(1) if sm else "Basic"
        fm = re.search(r"Self::(\w+)\(([^)]*)\)", body)
        factory = f"{fm.group(1)}({fm.group(2).strip()})" if fm else None
        return variant, sub, factory

    backend_file = {
        ("Json", "Basic"): "json/base_json_parser.rs",
        ("Json", "DeepseekV3"): "json/deepseek_v3_parser.rs",
        ("Json", "DeepseekV31"): "json/deepseek_v3_1_parser.rs",
        ("Xml", None): "xml/parser.rs",
        ("Pythonic", None): "pythonic/pythonic_parser.rs",
        ("Harmony", None): "harmony/harmony_parser.rs",
        ("Dsml", None): "dsml/parser.rs",
        ("Glm47", None): "xml/glm47_parser.rs",
        ("KimiK2", None): "xml/kimi_k2_parser.rs",
        ("Gemma4", None): "gemma4/parser.rs",
    }
    base_label = {
        ("Json", "Basic"): "base_json_parser (JsonParserType::Basic)",
        ("Json", "DeepseekV3"): "deepseek_v3_parser (JsonParserType::DeepseekV3)",
        ("Json", "DeepseekV31"): "deepseek_v3_1_parser (JsonParserType::DeepseekV31)",
        ("Xml", None): "xml::parser (shared XML base)",
        ("Pythonic", None): "pythonic::parser (standalone)",
        (
            "Harmony",
            None,
        ): "harmony::parser (standalone; partial reuse of base_json's try_repair_truncated_json)",
        ("Dsml", None): "dsml::parser (shared via deepseek_dsml() factory)",
        ("Glm47", None): "glm47_parser (standalone; filed under xml/)",
        ("KimiK2", None): "kimi_k2_parser (standalone; filed under xml/)",
        ("Gemma4", None): "gemma4::parser (standalone)",
    }

    out: dict[str, dict] = {}
    for family in refs:
        body = bodies.get(family)
        if body is None:
            continue
        variant, sub, factory = _classify(body)
        if variant is None and factory:
            # Resolve through factory (e.g. deepseek_dsml)
            fbody = bodies.get(factory.split("(")[0])
            if fbody:
                variant, sub, _ = _classify(fbody)
        key = (variant, sub) if variant == "Json" else (variant, None)
        out[family] = {
            "variant": variant,
            "sub_variant": sub,
            "factory": factory,
            "backend_file": backend_file.get(key, "unknown"),
            "base_label": base_label.get(key, f"{variant}"),
            "key": key,
            "aliases": [],
            "shared_with": [],
            "filed_under_xml_misleading": False,
        }

    # Aliases from parsers.rs (only ones where alias name != family name).
    alias_pat = re.compile(r'map\.insert\("([^"]+)",\s*ToolCallConfig::(\w+)\(\)\)')
    alias_to_target: dict[str, str] = {}
    for m in alias_pat.finditer(pars):
        alias, fam = m.group(1), m.group(2)
        if alias != fam and fam in out:
            out[fam]["aliases"].append(alias)
            alias_to_target[alias] = fam

    # shared_with — other families with the same (variant, sub_variant).
    by_key: dict[tuple, list[str]] = {}
    for fam, info in out.items():
        by_key.setdefault(info["key"], []).append(fam)
    for fam, info in out.items():
        info["shared_with"] = [s for s in by_key[info["key"]] if s != fam]
        info["filed_under_xml_misleading"] = (
            info["backend_file"].startswith("xml/") and info["variant"] != "Xml"
        )

    # Synthesize entries for alias-only families (e.g. nemotron_nano, qwen25).
    # These are in `refs` (registered in parsers.rs) but have no ctor of their
    # own — the alias `map.insert("nemotron_nano", ToolCallConfig::qwen3_coder())`
    # routes to the target's config. The alias gets the target's full
    # inheritance tree, plus `alias_of` so the tooltip can mark itself as a
    # leaf under the target rather than as the target itself.
    for alias, target in alias_to_target.items():
        if alias in out or target not in out:
            continue
        tgt = out[target]
        out[alias] = {
            **tgt,
            "alias_of": target,
        }

    return out


def _build_family_to_rust_ref() -> dict[str, tuple[str, int]]:
    """Scan the Rust source for each family's anchor point.

    Two patterns:
      `config.rs` :  `pub fn <family>() -> Self`               (parser config ctor)
      `parsers.rs`:  `map.insert("<family>", ToolCallConfig::...);`  (aliases)

    Config-ctor wins when the same family appears in both (the ctor is
    the canonical definition; the registration is just plumbing). Aliases
    (e.g. `nemotron_nano`, `qwen25`) only appear in `parsers.rs`.
    Returns `{family: (filename, line)}`; line is 1-indexed.
    """
    out: dict[str, tuple[str, int]] = {}

    config_rs = RUST_TOOL_CALLING_DIR / "config.rs"
    if config_rs.exists():
        pat = re.compile(r"^\s*pub fn (\w+)\(\)\s*->\s*Self\b")
        for lineno, line in enumerate(config_rs.read_text().splitlines(), 1):
            m = pat.match(line)
            if m:
                out[m.group(1)] = ("config.rs", lineno)

    parsers_rs = RUST_TOOL_CALLING_DIR / "parsers.rs"
    if parsers_rs.exists():
        pat = re.compile(r'^\s*map\.insert\("([^"]+)",\s*ToolCallConfig::')
        for lineno, line in enumerate(parsers_rs.read_text().splitlines(), 1):
            m = pat.match(line)
            if m and m.group(1) not in out:
                out[m.group(1)] = ("parsers.rs", lineno)

    return out


# Curated featured-model order. The only hand-maintained list left in the
# script: it picks WHICH families lead the table and IN WHAT ORDER. The
# human-readable label for each (e.g. "DeepSeek V4") is sourced from the
# fixture YAML's `model_label:` field, so the label travels with the
# family definition, not with this script.
#
# Source of truth: the tracked Top-N model mapping used by the parity
# planning notes. When that list changes, this list and the corresponding
# `model_label:` fields under `fixtures/<family>/PARSER.batch.yaml` must
# be updated together.
# Alphabetical-by-family-id within the list (matches the table row order).
TOP_N_FAMILIES = [
    "deepseek_v4",
    "gemma4",
    "glm47",
    "harmony",
    "kimi_k2",
    "minimax_m2",
    "qwen3_coder",
]

SUB_CASE_GROUPS = [
    ("Core", ("1", "3", "9", "9.a", "9.b")),
    ("Multi-call", ("2.a", "2.b", "2.c", "2.d", "10")),
    (
        "Malformed / recovery",
        ("4.a", "4.b", "4.c", "4.d", "4.e", "5.a", "5.b", "5.c", "5.d", "5.e"),
    ),
    (
        "Args",
        (
            "6.a",
            "6.b",
            "6.c",
            "7.a",
            "7.b",
            "7.c",
            "7.d",
            "7.e",
        ),
    ),
    ("Text interleaving", ("8.a", "8.b", "8.c", "8.d")),
    ("Unknown tools", ("13", "13.a", "13.c")),
    (
        "String contents",
        ("30", "30.a", "30.b", "30.c", "31", "31.a", "31.b"),
    ),
]

SPLIT_PARENT_SUBCASES = {
    # Once a taxonomy bucket has leaf cases, the matrix should render only the
    # leaves. Existing parent fixtures still carry useful expectations/reasons
    # for parser families that have not been rewritten to leaf IDs yet.
    "9": ("9.a",),
    "30": ("30.a", "30.b", "30.c"),
    "31": ("31.a", "31.b"),
    "13": ("13.a",),
}

_SUB_CASE_GROUP_KEY_BY_LABEL = {
    label: re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
    for label, _subs in SUB_CASE_GROUPS
}

_SUB_CASE_DISPLAY_ORDER = {
    sub: (group_idx, sub_idx)
    for group_idx, (_label, subs) in enumerate(SUB_CASE_GROUPS)
    for sub_idx, sub in enumerate(subs)
}

_SUB_CASE_GROUP_INDEX_BY_SUB = {
    sub: group_idx
    for group_idx, (_label, subs) in enumerate(SUB_CASE_GROUPS)
    for sub in subs
}

_SUB_CASE_GROUP_BY_SUB = {sub: label for label, subs in SUB_CASE_GROUPS for sub in subs}

_SUB_CASE_GROUP_KEY_BY_SUB = {
    sub: _SUB_CASE_GROUP_KEY_BY_LABEL[label]
    for label, subs in SUB_CASE_GROUPS
    for sub in subs
}


def _natural_sub_sort_key(sub: str) -> tuple[int, str]:
    """`8.a` → (8, 'a'); `9` → (9, '')."""
    parts = sub.split(".")
    return (int(parts[0]), parts[1] if len(parts) > 1 else "")


def _sub_sort_key(sub: str) -> tuple[int, int, int, str]:
    """Sort known cases by semantic display group, future cases naturally last."""
    display_order = _SUB_CASE_DISPLAY_ORDER.get(sub)
    if display_order is not None:
        group_idx, sub_idx = display_order
        return (0, group_idx, sub_idx, "")
    num, suffix = _natural_sub_sort_key(sub)
    return (1, num, 0, suffix)


def _subcase_band_class(sub: str) -> str:
    group_idx = _SUB_CASE_GROUP_INDEX_BY_SUB.get(sub, len(SUB_CASE_GROUPS))
    return f"case-band-{group_idx % 2}"


def _subcase_group_key(sub: str) -> str:
    return _SUB_CASE_GROUP_KEY_BY_SUB.get(sub, "other")


def _discover_sub_cases(cases: dict) -> list[str]:
    """Union of sub-case IDs across all loaded fixtures, in stable order."""
    return sorted({sub for _fam, sub in cases.keys()}, key=_sub_sort_key)


def _normalize_split_parent_cases(cases: dict) -> dict:
    """Render split taxonomy buckets as leaf cases only.

    Some older fixture files still define parent buckets such as
    `PARSER.batch.30`, while newer files define leaf buckets such as
    `PARSER.batch.30.a`. For display, parent+leaf duplication is confusing:
    once any leaf exists for a parent bucket, the table should show only the
    leaf columns. Parent entries are copied into missing leaf cells so their
    existing expectations or n/a reasons remain visible until the YAML itself
    is migrated.
    """
    all_subs = {sub for _fam, sub in cases.keys()}
    active_split_parents = {
        parent
        for parent, children in SPLIT_PARENT_SUBCASES.items()
        if any(child in all_subs for child in children)
    }
    if not active_split_parents:
        return cases

    normalized = dict(cases)
    families = {fam for fam, _sub in cases.keys()}
    for family in families:
        for parent in active_split_parents:
            parent_key = (family, parent)
            parent_case = normalized.get(parent_key)
            if parent_case is None:
                continue
            for child in SPLIT_PARENT_SUBCASES[parent]:
                child_key = (family, child)
                if child_key in normalized:
                    continue
                child_case = copy.deepcopy(parent_case)
                child_case["__case_id"] = f"PARSER.batch.{child}"
                child_case["__synthetic_from_case_id"] = parent_case.get("__case_id")
                normalized[child_key] = child_case
            del normalized[parent_key]
    return normalized


def _derive_no_peer_sets(cases: dict) -> tuple[set[str], set[str]]:
    """Families where every case marks the engine `unavailable`.

    Used to render the † (no vLLM peer) and § (no SGLang peer) footnote
    markers next to a family's name. A family qualifies when every case
    in every fixture file under that family has
    `expected.<impl>.unavailable: <reason>` recorded — i.e. the wrapper
    rejected the family for that parser in `capture_parser_outputs.py`.
    """
    by_family: dict[str, list[dict]] = {}
    for (fam, _sub), case in cases.items():
        by_family.setdefault(fam, []).append(case)

    def all_unavail(fam_cases: list[dict], impl: str) -> bool:
        expected_cases = [c for c in fam_cases if isinstance(c.get("expected"), dict)]
        if not expected_cases:
            return False
        for c in expected_cases:
            block = c.get("expected", {}).get(impl)
            if not isinstance(block, dict) or "unavailable" not in block:
                return False
        return True

    no_vllm = {fam for fam, cs in by_family.items() if all_unavail(cs, "vllm")}
    no_sglang = {fam for fam, cs in by_family.items() if all_unavail(cs, "sglang")}
    return no_vllm, no_sglang


def family_suffix(fam: str, no_vllm: set[str], no_sglang: set[str]) -> str:
    suff = ""
    if fam in no_vllm:
        suff += "†"
    if fam in no_sglang:
        suff += "§"
    return suff


def load_all_cases() -> tuple[dict[tuple[str, str], dict], dict[str, str]]:
    """Load every fixture YAML.

    Returns `(cases, labels)`:
      cases  — `{(family, sub_case_id): case_data}`; each case dict gets
               `__fixture_path` (relative to this script) and `__case_id`
               annotations for the HTML renderer.
      labels — `{family: model_label}` collected from the fixtures' doc-level
               `model_label:` field. Falls back to the family ID if a fixture
               doesn't declare one.
    """
    cases: dict[tuple[str, str], dict] = {}
    labels: dict[str, str] = {}
    script_dir = Path(__file__).resolve().parent
    for fp in sorted(FIXTURES.glob("*/PARSER.*.yaml")):
        doc = yaml.safe_load(fp.read_text())
        family = doc["family"]
        rel = str(fp.relative_to(script_dir))
        if "model_label" in doc:
            labels.setdefault(family, doc["model_label"])
        for cid, case in doc["cases"].items():
            sub = cid.replace("PARSER.batch.", "")
            case["__family"] = family
            case["__fixture_path"] = rel
            case["__case_id"] = cid
            cases[(family, sub)] = case
    return _normalize_split_parent_cases(cases), labels


def _build_display_groups(
    cases: dict, labels: dict[str, str]
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return `(top_n, others)` as `[(label, family), ...]` lists.

    Top-N: families listed in `TOP_N_FAMILIES`, in that exact order.
    Others: every YAML-discovered family not in TOP_N, sorted by label.
    Missing labels fall back to the family ID.
    """
    families = {fam for fam, _ in cases.keys()}

    def label_of(fam: str) -> str:
        return labels.get(fam, fam)

    top_n = [(label_of(f), f) for f in TOP_N_FAMILIES if f in families]
    other_fams = sorted(
        families - set(TOP_N_FAMILIES), key=lambda f: label_of(f).lower()
    )
    others = [(label_of(f), f) for f in other_fams]
    return top_n, others


def peer_status(case: dict, dyn: dict, impl: str) -> tuple[str, bool]:
    """Returns (kind, is_unknown).

    kind:
      'na'      — peer key missing from `expected:` (block not recorded)
      'match'   — peer is anchor ref to dynamo, or value-equal to dynamo
      'unavail' — peer block is `{unavailable: <msg>}`
      'err'     — peer block is `{error: <substring>}`
      'div'     — peer block is a concrete divergent {calls, normal_text}
    is_unknown is True iff kind == 'div' AND block has no `reason:`.
    """
    block = case.get("expected", {}).get(impl)
    if block is None:
        return ("na", False)
    if block is dyn:
        return ("match", False)
    if not isinstance(block, dict):
        return ("na", False)
    if "unavailable" in block:
        return ("unavail", False)
    if "error" in block:
        return ("err", False)
    if "calls" in block or "normal_text" in block:
        # Value-equal to dynamo (non-anchor)? Treat as match.
        n_block = {
            "calls": block.get("calls") or [],
            "normal_text": block.get("normal_text") or "",
        }
        n_dyn = {
            "calls": dyn.get("calls") or [],
            "normal_text": dyn.get("normal_text") or "",
        }
        if n_block == n_dyn:
            return ("match", False)
        return ("div", "reason" not in block)
    return ("na", False)


def cell_for(case: dict | None) -> str:
    if case is None:
        return "—"
    dyn = case.get("expected", {}).get("dynamo")
    if not isinstance(dyn, dict):
        return "n/a"
    v_kind, v_unknown = peer_status(case, dyn, "vllm")
    s_kind, s_unknown = peer_status(case, dyn, "sglang")

    parts: list[str] = []
    if v_kind == "div":
        parts.append("V?" if v_unknown else "V")
    elif v_kind == "err":
        parts.append("V!")
    if s_kind == "div":
        parts.append("S?" if s_unknown else "S")
    elif s_kind == "err":
        parts.append("S!")

    # `reason:` on the `expected.dynamo` block flags Dynamo's own output as
    # leaking tool call markup only when Dynamo also leaves residual
    # `normal_text`. Dynamo can have non-leak reasons for dropped malformed
    # markup, so don't mark those as `↯`.
    if isinstance(dyn, dict) and dyn.get("reason") and bool(dyn.get("normal_text")):
        if parts:
            return "↯" + "".join(parts)
        return "↯"

    if parts:
        return "".join(parts)
    if v_kind == "unavail" and s_kind == "unavail":
        return "n/a"
    return "="


def render_row(
    model: str,
    family: str,
    cases: dict,
    sub_cases: list[str],
    no_vllm: set[str],
    no_sglang: set[str],
) -> str:
    cells = [cell_for(cases.get((family, sub))) for sub in sub_cases]
    suff = family_suffix(family, no_vllm, no_sglang)
    return f"| {model} | {family}{suff} | " + " | ".join(cells) + " |"


_LEGEND_MD = (
    "**Legend:** "
    "`=` full parity (Dynamo, vLLM, and SGLang produce the same results) · "
    "`V`/`S` divergence (V = vLLM, S = SGLang; intentional, has `reason:`) · "
    "`?` research-needed suffix (e.g. V?, S? — diverges with no `reason:` yet) · "
    "`↯` Dynamo leaks tool call markup into `normal_text` "
    "(`expected.dynamo.reason:` carries the explanation) · "
    "`!` expected-error suffix (e.g. V!, S! — engine crashes by design) · "
    "`n/a` not applicable · "
    "`—` missing fixture coverage · "
    "`†` (parser column) = no vLLM peer parser for this family · "
    "`§` (parser column) = no SGLang peer parser for this family."
)


def render_markdown(
    cases: dict,
    sub_cases: list[str],
    no_vllm: set[str],
    no_sglang: set[str],
    top_n: list[tuple[str, str]],
    others: list[tuple[str, str]],
) -> str:
    header = "| model | parser | " + " | ".join(sub_cases) + " |"
    sep = "|---|---|" + ":-:|" * len(sub_cases)
    lines = [header, sep]
    lines.append("| **Top-N models** |   |" + "   |" * len(sub_cases))
    for model, fam in top_n:
        lines.append(render_row(model, fam, cases, sub_cases, no_vllm, no_sglang))
    lines.append("| **Others** |   |" + "   |" * len(sub_cases))
    for model, fam in others:
        lines.append(render_row(model, fam, cases, sub_cases, no_vllm, no_sglang))
    lines.append("")
    lines.append(_LEGEND_MD)
    return "\n".join(lines)


_IMPL_DISPLAY = {"dynamo": "Dynamo", "vllm": "vLLM", "sglang": "SGLang"}
_URL_RE = re.compile(r"https?://[^\s<>'\"]+")
_TRAILING_URL_PUNCTUATION = ".,;:)"


def _linkify_text_html(text: str) -> str:
    """Escape plain text and turn embedded URLs into anchors."""
    parts: list[str] = []
    last = 0
    for match in _URL_RE.finditer(text):
        raw_url = match.group(0)
        url = raw_url.rstrip(_TRAILING_URL_PUNCTUATION)
        if not url:
            continue
        parts.append(html_lib.escape(text[last : match.start()]))
        href = html_lib.escape(url, quote=True)
        parts.append(
            f'<a href="{href}" target="_blank" rel="noopener noreferrer">'
            f"{html_lib.escape(url)}</a>"
        )
        parts.append(html_lib.escape(raw_url[len(url) :]))
        last = match.end()
    parts.append(html_lib.escape(text[last:]))
    return "".join(parts)


def _format_output_block_html(block, family: str | None = None) -> str:
    """HTML rendering of an `expected.<impl>` block for tooltips.
    Applies _colorize_xml to `normal_text` so raw model output the engine
    failed to parse shows the same tag coloring as the input."""
    if not isinstance(block, dict):
        return html_lib.escape("(no expectation)")
    if block.get("unavailable"):
        return html_lib.escape(f"unavailable: {block['unavailable']}")
    if "error" in block:
        return html_lib.escape(f"error matching {block['error']!r}")
    nt = block.get("normal_text", "") or ""
    calls = block.get("calls") or []
    if calls:
        rendered = ", ".join(
            f"{c.get('name', '?')}({json.dumps(c.get('arguments', {}), ensure_ascii=False)})"
            for c in calls
        )
        calls_line = html_lib.escape(f"calls=[{rendered}]")
    else:
        calls_line = "calls=[]"
    nt_line = f"normal_text='{_colorize_markup(nt, family)}'"
    return f"{nt_line}\n{calls_line}"


# Matches both `<...>` and the Mistral-style `[NAME]`/`[/NAME]` form.
# Brackets only match ALL-CAPS-underscore names so JSON arrays
# (e.g. `[{...}]`, `[1, 2]`) don't get false-matched as tags.
_TAG_RE = re.compile(r"<[^<>]+>|\[/?[A-Z][A-Z0-9_]*\]")

# Two coloring schemes share one cycling palette:
#   * Paired open/close: every matched pair gets a *fresh* palette index, so
#     two `<tool_call>...</tool_call>` instances in the same input render as
#     two different colors.
#   * Singletons (e.g. harmony's `<|channel|>` section markers): stable
#     by-role color so all `<|channel|>` tokens share one class everywhere.
_PAIRED_PALETTE_SIZE = 8
_color_seq = 0
_singleton_classes: dict[str, str] = {}


def _next_color_class() -> str:
    global _color_seq
    cls = f"tt-c{_color_seq % _PAIRED_PALETTE_SIZE}"
    _color_seq += 1
    return cls


def _singleton_class_for(name: str) -> str:
    cls = _singleton_classes.get(name)
    if cls is None:
        cls = _next_color_class()
        _singleton_classes[name] = cls
    return cls


_PIPES = ("|", "｜")  # ASCII and FULLWIDTH VERTICAL LINE (U+FF5C)
_BEGIN_SUFFIXES = (
    "_begin",
    "▁begin",
)  # ASCII underscore and LOWER ONE EIGHTH BLOCK (U+2581)
_END_SUFFIXES = ("_end", "▁end")

# Harmony (gpt-oss) tokens are linear state-machine delimiters. Turn-boundary
# tokens form pseudo-pairs: `<|start|>` opens a turn; `<|end|>`, `<|return|>`,
# or `<|call|>` closes it. Each closer flavor gets its own paired color so a
# tool-call turn (start→call) is visually distinct from a normal turn (start→end)
# or a final turn (start→return). Section markers stay as same-colored singletons.
_HARMONY_TURN_OPEN = "start"
_HARMONY_TURN_CLOSE = frozenset({"end", "return", "call"})
_HARMONY_SECTION_MARKERS = frozenset({"channel", "constrain", "message"})
_HARMONY_TOKEN_RE = re.compile(r"<\|([A-Za-z_]+)\|>")
_HARMONY_SEGMENT_CLASS = {
    "start": "tt-h-start",
    "channel": "tt-h-channel",
    "constrain": "tt-h-constrain",
    "message": "tt-h-message",
    "end": "tt-h-stop",
    "return": "tt-h-stop",
    "call": "tt-h-call",
}


def _colorize_harmony(text: str) -> str:
    """Color Harmony's linear token segments as `<|token|>related-text`.

    Harmony is not paired XML. Tokens such as `<|channel|>` and `<|message|>`
    introduce the header/content span that follows, so each special token is
    grouped with text up to the next Harmony token.
    """
    pieces: list[str] = []
    last = 0
    for m in _HARMONY_TOKEN_RE.finditer(text):
        if m.start() > last:
            pieces.append(html_lib.escape(text[last : m.start()]))
        token_name = m.group(1)
        if token_name in _HARMONY_TURN_CLOSE:
            end = m.end()
        else:
            next_m = _HARMONY_TOKEN_RE.search(text, m.end())
            end = next_m.start() if next_m else len(text)
        cls = _HARMONY_SEGMENT_CLASS.get(token_name, "tt-h-other")
        token = html_lib.escape(m.group(0))
        related = html_lib.escape(text[m.end() : end])
        pieces.append(
            f'<span class="tt-h {cls}"><span class="tt-h-token">{token}</span>{related}</span>'
        )
        last = end
    if last < len(text):
        pieces.append(html_lib.escape(text[last:]))
    return "".join(pieces)


def _colorize_markup(text: str, family: str | None = None) -> str:
    if family == "harmony" and _HARMONY_TOKEN_RE.search(text):
        return _colorize_harmony(text)
    return _colorize_xml(text)


def _strip_suffix(s: str, suffixes: tuple[str, ...]) -> str | None:
    for suf in suffixes:
        if s.endswith(suf) and len(s) > len(suf):
            return s[: -len(suf)]
    return None


def _tag_kind_and_name(inner: str) -> tuple[str | None, str, str | None]:
    """Classify `<...>` tag inner text into an open/close/singleton kind.

    Returns (kind, pair_id, color_override):
      * kind: 'open' | 'close' | 'singleton' | None
      * pair_id: name used to match open against close on the stack
      * color_override: when set, the paired span uses this name for the
        color class instead of pair_id (lets `<|start|>...<|call|>` color
        differently from `<|start|>...<|end|>` despite sharing pair_id).
    """

    def _name_of(s: str) -> str:
        # Split on whitespace, slash, `>`, OR `=` so that
        # `<function=book_flight>` pairs with `</function>` and
        # `<parameter=destination>` pairs with `</parameter>`.
        if not s:
            return ""
        return re.split(r"[\s/>=]", s, 1)[0].rstrip("|")

    if inner.startswith("/"):
        return ("close", _name_of(inner[1:]), None)
    starts_pipe = inner[:1] in _PIPES
    ends_pipe = inner[-1:] in _PIPES
    if starts_pipe and ends_pipe and len(inner) >= 2:
        middle = inner[1:-1]
        stripped = _strip_suffix(middle, _BEGIN_SUFFIXES)
        if stripped is not None:
            return ("open", stripped, None)
        stripped = _strip_suffix(middle, _END_SUFFIXES)
        if stripped is not None:
            return ("close", stripped, None)
        if middle == _HARMONY_TURN_OPEN:
            return ("open", "__harmony_turn", None)
        if middle in _HARMONY_TURN_CLOSE:
            # Color the pair by which closer flavor was used.
            return ("close", "__harmony_turn", f"__harmony_pair_{middle}")
        if middle in _HARMONY_SECTION_MARKERS:
            return ("singleton", "__harmony_section", None)
        return (None, "", None)
    if starts_pipe and inner[:1] == "|":
        return ("open", _name_of(inner[1:]), None)
    if ends_pipe and inner[-1:] == "|":
        return ("close", _name_of(inner[:-1]), None)
    return ("open", _name_of(inner), None)


def _colorize_xml(text: str) -> str:
    """HTML-escape `text` and wrap each `<...>` token in a span.
    Paired open/close (stack match by tag name) → class 'tt-paired'.
    Unmatched close, or open that never closes → class 'tt-orphan'.

    Pairs standard XML (`<X>...</X>`) AND alt pipe-marker conventions:
      `<|X>...<X|>`          (boundary ASCII pipes)
      `<|X_begin|>...<|X_end|>`  (both-side pipes with _begin/_end suffix)
    Lenient pop-through: a close looks down the stack for the nearest
    name-match; anything un-closed above it is marked orphan. Lets
    no-close singletons (e.g. `<|tool_call_argument_begin|>`) localize
    their orphan-ness without poisoning the surrounding pairs.
    """
    pieces: list[str] = []
    stack: list[tuple[str, int]] = []
    last = 0
    for m in _TAG_RE.finditer(text):
        if m.start() > last:
            pieces.append(html_lib.escape(text[last : m.start()]))
        tok = m.group(0)
        kind, pair_id, color_override = _tag_kind_and_name(tok[1:-1])
        esc = html_lib.escape(tok)
        if kind is None:
            pieces.append(f'<span class="tt-orphan">{esc}</span>')
        elif kind == "singleton":
            cls = _singleton_class_for(pair_id)
            pieces.append(f'<span class="{cls}">{esc}</span>')
        elif kind == "close":
            match_at = -1
            for i in range(len(stack) - 1, -1, -1):
                if stack[i][0] == pair_id:
                    match_at = i
                    break
            if match_at >= 0:
                for _, unmatched_idx in stack[match_at + 1 :]:
                    pieces[
                        unmatched_idx
                    ] = f'<span class="tt-orphan">{pieces[unmatched_idx]}</span>'
                open_idx = stack[match_at][1]
                # Per-instance color: every matched pair gets a fresh palette
                # index, so two `<tool_call>...</tool_call>` (or two
                # `<|start|>...<|call|>`) blocks in the same input render as
                # different colors. (color_override unused on the paired
                # path — kept on the singleton-flavor branch only.)
                _ = color_override
                cls = _next_color_class()
                pieces[open_idx] = f'<span class="{cls}">{pieces[open_idx]}</span>'
                pieces.append(f'<span class="{cls}">{esc}</span>')
                del stack[match_at:]
            else:
                pieces.append(f'<span class="tt-orphan">{esc}</span>')
        else:
            pieces.append(esc)
            stack.append((pair_id, len(pieces) - 1))
        last = m.end()
    for _, idx in stack:
        pieces[idx] = f'<span class="tt-orphan">{pieces[idx]}</span>'
    if last < len(text):
        pieces.append(html_lib.escape(text[last:]))
    return "".join(pieces)


def _build_tooltip_html(case: dict, dyn) -> str:
    """Rich HTML hover tooltip: head, input (colorized), per-engine output,
    divergence reasons. Returns the full `<div class="ttip">...</div>`."""
    case_id = case.get("__case_id", "")
    desc = case.get("description") or ""
    head = f"{case_id} — {desc}" if (case_id and desc) else (case_id or desc)

    parts: list[str] = ['<div class="ttip">']
    if head:
        parts.append(f'<div class="ttip-head">{html_lib.escape(head)}</div>')

    ref = case.get("ref")
    if isinstance(ref, str) and ref:
        parts.append('<div class="ttip-section">Ref:</div>')
        parts.append(f'<pre class="ttip-pre">{_linkify_text_html(ref)}</pre>')

    model_text = case.get("model_text")
    if isinstance(model_text, str) and model_text:
        family = case.get("__family")
        parts.append('<div class="ttip-section">Input:</div>')
        parts.append(
            f"<pre class=\"ttip-pre\">input_text='{_colorize_markup(model_text, family)}'</pre>"
        )

    expected = case.get("expected") or {}

    def _norm(b):
        return {
            "calls": b.get("calls") or [],
            "normal_text": b.get("normal_text") or "",
        }

    n_dyn = _norm(dyn) if isinstance(dyn, dict) else None
    all_parity = isinstance(dyn, dict) and all(
        isinstance(expected.get(i), dict)
        and not expected[i].get("unavailable")
        and "error" not in expected[i]
        and _norm(expected[i]) == n_dyn
        for i in ("dynamo", "vllm", "sglang")
    )

    if all_parity:
        parts.append('<div class="ttip-section">All engines parity:</div>')
        parts.append(
            f'<pre class="ttip-pre">{_format_output_block_html(dyn, case.get("__family"))}</pre>'
        )
    else:
        for impl in ("dynamo", "vllm", "sglang"):
            block = expected.get(impl)
            parts.append(f'<div class="ttip-section">{_IMPL_DISPLAY[impl]}:</div>')
            parts.append(
                f'<pre class="ttip-pre">{_format_output_block_html(block, case.get("__family"))}</pre>'
            )

    reasons = _tooltip_for(case, dyn) if isinstance(dyn, dict) else ""
    if reasons:
        parts.append('<div class="ttip-section">Divergence reason:</div>')
        parts.append(f'<pre class="ttip-pre">{_linkify_text_html(reasons)}</pre>')

    dyn_leak = (
        dyn.get("reason")
        if isinstance(dyn, dict) and bool(dyn.get("normal_text"))
        else None
    )
    if dyn_leak:
        parts.append('<div class="ttip-section">↯ Dynamo tool call leaks:</div>')
        parts.append(f'<pre class="ttip-pre">{_linkify_text_html(str(dyn_leak))}</pre>')

    parts.append("</div>")
    return "".join(parts)


def _tooltip_for(case: dict, dyn: dict) -> str:
    """Build the hover-tooltip text for a divergent cell.

    Each non-matching, non-unavailable peer contributes one line:
      vllm: <reason>                        # `reason:` field present
      vllm: UNKNOWN — divergent ...         # divergent, no reason
      vllm: expected error matching '...'   # `error:` field present
    """
    parts: list[str] = []
    n_dyn = {
        "calls": dyn.get("calls") or [],
        "normal_text": dyn.get("normal_text") or "",
    }
    for impl in ("vllm", "sglang"):
        block = case.get("expected", {}).get(impl)
        if not isinstance(block, dict) or block is dyn:
            continue
        if "unavailable" in block:
            continue
        name = _IMPL_DISPLAY.get(impl, impl)
        if "error" in block:
            parts.append(f"{name}: expected error matching {block['error']!r}")
            continue
        # Don't rely on PyYAML preserving anchor identity (the `block is dyn`
        # check above is the fast path; value equality is the safety net).
        n_block = {
            "calls": block.get("calls") or [],
            "normal_text": block.get("normal_text") or "",
        }
        if n_block == n_dyn:
            continue
        if "reason" in block:
            parts.append(f"{name}: {block['reason']}")
        elif "calls" in block or "normal_text" in block:
            parts.append(f"{name}: (research-needed — no `reason:` field yet)")
    return "\n".join(parts)


def _cell_class(text: str) -> str:
    if text == "—":
        return "missing"
    if text == "n/a":
        return "na"
    if "!" in text:
        return "err"
    if "↯" in text:
        return "leak"
    if "?" in text:
        return "research"
    if text == "=":
        return "ok"
    # V / S / VS with a documented `reason:` — accepted divergence, but
    # NOT parity, so don't color it green.
    return "documented"


def _build_na_tooltip_html(case: dict) -> str:
    """Tooltip for an n/a stub case (only `reason:` in YAML, no `expected:`
    block). Renders case id + description + the reason. Used when the cell
    is n/a because the scenario doesn't apply to the family's parser syntax."""
    case_id = case.get("__case_id", "")
    desc = case.get("description") or ""
    head = f"{case_id} — {desc}" if (case_id and desc) else (case_id or desc)
    reason = case.get("reason") or "n/a (no reason given)"
    parts = ['<div class="ttip">']
    if head:
        parts.append(f'<div class="ttip-head">{html_lib.escape(head)}</div>')
    parts.append('<div class="ttip-section">Why n/a:</div>')
    parts.append(f'<pre class="ttip-pre">{_linkify_text_html(str(reason))}</pre>')
    parts.append("</div>")
    return "".join(parts)


def _build_missing_tooltip_html(family: str, sub: str) -> str:
    """Tooltip for an absent fixture entry.

    This is intentionally distinct from an explicit n/a stub. Missing means
    the table has no fixture data for this family/case; explicit n/a means a
    fixture author recorded why the case does not apply.
    """
    case_id = f"PARSER.batch.{sub}"
    parts = ['<div class="ttip">']
    parts.append(
        f'<div class="ttip-head">{html_lib.escape(case_id)} — '
        f"{html_lib.escape(family)}</div>"
    )
    parts.append('<div class="ttip-section">Missing fixture:</div>')
    parts.append(
        '<pre class="ttip-pre">No fixture entry exists for this family/case. '
        "If the case is intentionally not applicable, add an explicit n/a "
        "stub with description: and reason: so the table can explain it.</pre>"
    )
    parts.append("</div>")
    return "".join(parts)


def render_cell_html(case: dict | None, family: str, sub: str) -> str:
    text = cell_for(case)
    cls = _cell_class(text)
    band_cls = _subcase_band_class(sub)
    col_group = html_lib.escape(_subcase_group_key(sub))
    td_open = f'<td class="cell {cls} {band_cls}" data-col-hide-group="{col_group}">'
    if case is None:
        ttip = _build_missing_tooltip_html(family, sub)
        return f"{td_open}{text}{ttip}</td>"

    dyn = case.get("expected", {}).get("dynamo")
    if not isinstance(dyn, dict):
        # n/a stub: case has only `reason:` (no `expected:` block).
        fp = case.get("__fixture_path", "")
        ttip = _build_na_tooltip_html(case)
        if not fp:
            return f"{td_open}{text}{ttip}</td>"
        href = html_lib.escape(fp)
        return f'{td_open}<a href="{href}">{text}</a>{ttip}</td>'

    fp = case.get("__fixture_path", "")
    # Case id + description live in the rich CSS tooltip head — don't also
    # set `title=` on the link, or browsers stack a native tooltip on top.
    ttip = _build_tooltip_html(case, dyn)
    if not fp:
        return f"{td_open}{text}{ttip}</td>"
    href = html_lib.escape(fp)
    return f'{td_open}<a href="{href}">{text}</a>{ttip}</td>'


def _parser_inheritance_tooltip_html(
    family: str,
    info: dict,
    ctor_ref: tuple[str, int] | None,
    no_vllm: set[str] | None = None,
    no_sglang: set[str] | None = None,
) -> str:
    """Rich `.ttip` tooltip rendering the parser inheritance as an ASCII
    tree localized to the target family: variant header, siblings sharing
    the same backend, the target marked with `← THIS`, aliases nested under
    the target, and a warning when filed under `xml/` but bypassing the
    shared XML base. `ctor_ref` is unused here (was for older field-based
    layout) — kept for API stability with `_parser_cell_html`."""
    del ctor_ref

    variant = info["variant"] or "?"
    sub_variant = info["sub_variant"]
    backend_file = info["backend_file"]
    factory = info["factory"]
    alias_of = info.get("alias_of")  # set when this family is an alias-only entry

    # Header: ParserConfig::<Variant>[::<Sub>] → <backend file>  [(factory: <name>)]
    head_parts = [f"ParserConfig::{variant}"]
    if sub_variant:
        head_parts[-1] = f"ParserConfig::{variant}::{sub_variant}"
    bf_href = html_lib.escape(f"../../../lib/parsers/src/tool_calling/{backend_file}")
    bf_link = f'<a href="{bf_href}">{html_lib.escape(backend_file)}</a>'
    header_html = f"{html_lib.escape(head_parts[0])} → {bf_link}"
    if factory:
        factory_name = factory.split("(", 1)[0]
        header_html += html_lib.escape(f"  (factory: {factory_name})")

    # Body: anchor + siblings, alphabetical, target marked with ← THIS.
    # Anchor is the family whose entry "owns" this backend. If `family` is
    # an alias-only entry, the anchor is its target and the marker drops
    # to the alias's leaf row under that target.
    anchor = alias_of or family
    siblings = info["shared_with"]
    fam_list = sorted([anchor] + siblings)
    body_lines: list[str] = []
    for i, fam in enumerate(fam_list):
        is_last_fam = i == len(fam_list) - 1
        branch = "└── " if is_last_fam else "├── "
        # block="..." suffix only renders for the anchor row, since we only
        # have factory args for the current entry (not for siblings).
        suffix = ""
        if fam == anchor and info["factory"]:
            fm = re.search(r'\("([^"]+)"\)', info["factory"])
            if fm:
                suffix = f'  block="{fm.group(1)}"'
        # ← THIS goes on the target family's line only when the table cell
        # IS the target (not an alias).
        marker_html = (
            "  <strong>← THIS</strong>" if (fam == family and not alias_of) else ""
        )
        body_lines.append(
            f"{branch}{html_lib.escape(fam)}{html_lib.escape(suffix)}{marker_html}"
        )

        # Aliases nested under the anchor. If we're rendering the alias's
        # own tooltip, the alias's row carries the ← THIS marker.
        if fam == anchor and info["aliases"]:
            cont = "    " if is_last_fam else "│   "
            for j, alias in enumerate(info["aliases"]):
                alast = j == len(info["aliases"]) - 1
                ab = "└── " if alast else "├── "
                a_marker = (
                    "  <strong>← THIS</strong>"
                    if (alias_of and alias == family)
                    else ""
                )
                body_lines.append(
                    f"{cont}{ab}{html_lib.escape(alias)}  (alias){a_marker}"
                )

    # Peer-availability footnote (†/§) — embedded here so symbols don't need
    # their own native tooltip. Anchored on the table-cell family, so an
    # alias inherits its target's peer markers (they share the same fixture
    # YAMLs and the same `expected.<impl>.unavailable` flags).
    peer_notes: list[str] = []
    if no_vllm and family in no_vllm:
        peer_notes.append(
            '<span class="parser-suffix">†</span> no vLLM peer parser for this family'
        )
    if no_sglang and family in no_sglang:
        peer_notes.append(
            '<span class="parser-suffix">§</span> no SGLang peer parser for this family'
        )
    peer_html = ("\n\n" + "\n".join(peer_notes)) if peer_notes else ""

    # Misleading-location warning appended as a separate paragraph below the tree.
    warn_html = ""
    if info["filed_under_xml_misleading"]:
        warn_html = (
            "\n\n⚠ Filed under xml/ but does NOT use the shared xml::parser.\n"
            f"   Has its own ParserConfig::{html_lib.escape(variant)} variant."
        )

    tree_html = header_html + "\n" + "\n".join(body_lines) + peer_html + warn_html

    if alias_of:
        head_text = f"{family} — alias of {alias_of}; inherits {info['base_label']}"
    else:
        head_text = f"{family} — inherits {info['base_label']}"
    return (
        '<div class="ttip">'
        f'<div class="ttip-head">{html_lib.escape(head_text)}</div>'
        f'<pre class="ttip-pre">{tree_html}</pre>'
        "</div>"
    )


_SHARED_BACKEND_SHORT = {
    ("Json", "Basic"): "base_json",
    ("Xml", None): "xml",
    ("Dsml", None): "dsml",
}


def _parser_cell_html(
    family: str,
    refs: dict[str, tuple[str, int]],
    no_vllm: set[str],
    no_sglang: set[str],
    inheritance: dict[str, dict],
) -> str:
    suff = family_suffix(family, no_vllm, no_sglang)
    label = html_lib.escape(family)
    if suff:
        label += f'<span class="parser-suffix">{html_lib.escape(suff)}</span>'
    ref = refs.get(family)
    info = inheritance.get(family)
    ttip = (
        _parser_inheritance_tooltip_html(family, info, ref, no_vllm, no_sglang)
        if info
        else ""
    )

    # Shared-base suffix: only show when 2+ families share this backend, so
    # the column calls out the consolidation rows without adding noise to
    # standalone parsers (pythonic, gemma4, glm47, kimi_k2, harmony, ...).
    base_suffix = ""
    if info and info["shared_with"]:
        short = _SHARED_BACKEND_SHORT.get(info["key"])
        if short:
            base_suffix = f'<span class="parser-base">→ {html_lib.escape(short)}</span>'

    # Family-name link points to the **actual parser code** (backend_file from
    # the inheritance map), not to the config-ctor location in config.rs. The
    # ctor location is still referenced in the inheritance tooltip body when
    # useful (factory calls). For families with no inheritance info, fall back
    # to the refs entry (config.rs or parsers.rs).
    if info and info["backend_file"] != "unknown":
        href = f"../../../lib/parsers/src/tool_calling/{info['backend_file']}"
    elif ref is not None:
        href = f"../../../lib/parsers/src/tool_calling/{ref[0]}"
    else:
        return (
            f'<td class="parser" data-col-hide-group="parser">'
            f"{label}{base_suffix}{ttip}</td>"
        )
    return (
        f'<td class="parser" data-col-hide-group="parser">'
        f'<a href="{href}">{label}</a>{base_suffix}{ttip}</td>'
    )


def render_row_html(
    model: str,
    family: str,
    cases: dict,
    sub_cases: list[str],
    refs: dict[str, tuple[str, int]],
    no_vllm: set[str],
    no_sglang: set[str],
    inheritance: dict[str, dict],
) -> str:
    cells = [
        f'<tr><td class="model" data-col-hide-group="model">{html_lib.escape(model)}</td>',
        _column_placeholder_html("model"),
        _parser_cell_html(family, refs, no_vllm, no_sglang, inheritance),
        _column_placeholder_html("parser"),
    ]
    for run in _subcase_runs(sub_cases):
        cells.extend(
            render_cell_html(cases.get((family, sub)), family, sub) for sub in run
        )
        cells.append(_column_placeholder_html(_subcase_group_key(run[0])))
    cells.append("</tr>")
    return "".join(cells)


def _parse_subcase_descriptions() -> dict[str, str]:
    """Parse `lib/parsers/PARSER_CASES.md` for per-case descriptions.

    The Quick-reference section has one-liner bullets for top-level cases
    (`PARSER.batch.1` … `PARSER.batch.10`); the deeper per-case sections
    contain multi-line bullets for sub-cases (`2.a`, `4.c`, etc.). Both
    look like `- **`PARSER.batch.X`** <desc>`, where the bullet body may
    wrap across indented continuation lines. Returns
    `{"1": "...", "2.a": "...", ...}`.
    """
    if not PARSER_CASES_MD.exists():
        return {}
    pat = re.compile(r"\*\*`PARSER\.batch\.([0-9]+(?:\.[a-z])?)`\*\*\s+(.+)")
    out: dict[str, str] = {}
    lines = PARSER_CASES_MD.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        m = pat.search(lines[i])
        if not m:
            i += 1
            continue
        sub = m.group(1)
        body_parts = [m.group(2).strip()]
        # Join indented continuation lines until blank / next bullet / unindented.
        j = i + 1
        while j < len(lines):
            nxt = lines[j]
            if not nxt.strip():
                break
            if not nxt.startswith(" "):
                break
            if pat.search(nxt):
                break
            body_parts.append(nxt.strip())
            j += 1
        desc = " ".join(body_parts).rstrip(".")
        out.setdefault(sub, desc)
        i = j
    return out


def _subcase_header_html(sub: str, descriptions: dict[str, str]) -> str:
    desc = descriptions.get(sub) or descriptions.get(sub.split(".")[0]) or ""
    href = "../../../lib/parsers/PARSER_CASES.md"
    title = html_lib.escape(desc) if desc else ""
    band_cls = _subcase_band_class(sub)
    col_group = html_lib.escape(_subcase_group_key(sub))
    return (
        f'<th class="case-sub {band_cls}" data-col-hide-group="{col_group}">'
        f'<a href="{href}" title="{title}">{html_lib.escape(sub)}</a></th>'
    )


def _subcase_group_label(sub: str) -> str:
    return _SUB_CASE_GROUP_BY_SUB.get(sub, "Other")


def _subcase_runs(sub_cases: list[str]) -> list[list[str]]:
    runs: list[list[str]] = []
    start = 0
    while start < len(sub_cases):
        label = _subcase_group_label(sub_cases[start])
        end = start + 1
        while end < len(sub_cases) and _subcase_group_label(sub_cases[end]) == label:
            end += 1
        runs.append(sub_cases[start:end])
        start = end
    return runs


def _column_placeholder_html(key: str, tag: str = "td") -> str:
    key_attr = html_lib.escape(key)
    return (
        f'<{tag} class="col-placeholder col-hidden" '
        f'data-col-placeholder-group="{key_attr}"></{tag}>'
    )


def _column_control_header_html(
    key: str,
    label: str,
    *,
    default_visible: bool,
    css_class: str = "",
    colspan: int | None = None,
) -> str:
    key_attr = html_lib.escape(key)
    visible = "true" if default_visible else "false"
    classes = " ".join(part for part in ("column-control", css_class) if part)
    span_size = colspan if colspan is not None else 1
    if colspan is not None:
        span_attr = f'colspan="{colspan}" data-expanded-colspan="{colspan}"'
    else:
        span_attr = 'rowspan="2"'
    return (
        f'<th class="{html_lib.escape(classes)}" data-col-control-group="{key_attr}" '
        f"{span_attr}>"
        f'<button type="button" class="col-toggle" data-col-toggle="{key_attr}" '
        f'data-col-label="{html_lib.escape(label)}" data-col-span="{span_size}" '
        f'data-default-visible="{visible}" aria-pressed="{visible}" '
        f'aria-label="{"Collapse" if default_visible else "Expand"} '
        f'{html_lib.escape(label)} column">'
        '<span class="col-toggle-symbol" aria-hidden="true"></span>'
        f'<span class="col-toggle-label">{html_lib.escape(label)}</span>'
        "</button></th>"
    )


def _subcase_group_headers_html(sub_cases: list[str]) -> str:
    """Build semantic group headers spanning the displayed sub-case columns."""
    spans: list[str] = [
        _column_control_header_html("model", "Model", default_visible=False),
        _column_control_header_html("parser", "Parser", default_visible=True),
    ]
    for run in _subcase_runs(sub_cases):
        label = _subcase_group_label(run[0])
        band_cls = _subcase_band_class(run[0])
        col_group = html_lib.escape(_subcase_group_key(run[0]))
        spans.append(
            _column_control_header_html(
                col_group,
                label,
                default_visible=True,
                css_class=f"case-group {band_cls}",
                colspan=len(run),
            )
        )
    return "".join(spans)


def _subcase_headers_html(sub_cases: list[str], descriptions: dict[str, str]) -> str:
    headers: list[str] = []
    for run in _subcase_runs(sub_cases):
        headers.extend(_subcase_header_html(sub, descriptions) for sub in run)
        headers.append(_column_placeholder_html(_subcase_group_key(run[0]), tag="th"))
    return "".join(headers)


def _glossary_groups(
    descriptions: dict[str, str], sub_cases: list[str]
) -> list[dict[str, object]]:
    if not descriptions:
        return []
    return [
        {
            "label": _subcase_group_label(run[0]),
            "rows": [
                (
                    sub,
                    descriptions.get(sub) or descriptions.get(sub.split(".")[0]) or "",
                )
                for sub in run
            ],
        }
        for run in _subcase_runs(sub_cases)
    ]


def _peer_version_items(versions: dict[str, str]) -> list[tuple[str, str]]:
    return [(name, versions[name]) for name in ("vllm", "sglang") if name in versions]


def _compute_stats(
    cases: dict, sub_cases: list[str], families: list[str]
) -> dict[str, int]:
    """Aggregate cell outcomes across the (family × sub_case) grid."""
    s = {
        "families": len(families),
        "sub_cases": len(sub_cases),
        "slots": len(families) * len(sub_cases),
        "real": 0,
        "parity": 0,
        "documented": 0,
        "research": 0,
        "errors": 0,
        "na": 0,
        "missing": 0,
    }
    for fam in families:
        for sub in sub_cases:
            case = cases.get((fam, sub))
            text = cell_for(case)
            if text == "—":
                s["missing"] += 1
                continue
            if text == "n/a":
                s["na"] += 1
                continue
            s["real"] += 1
            if text == "=":
                s["parity"] += 1
            elif "!" in text:
                s["errors"] += 1
            elif "↯" in text:
                s["documented"] += 1
            elif "?" in text:
                s["research"] += 1
            else:
                s["documented"] += 1
    return s


def render_html(
    cases: dict,
    sub_cases: list[str],
    no_vllm: set[str],
    no_sglang: set[str],
    top_n: list[tuple[str, str]],
    others: list[tuple[str, str]],
    family_filter: str | None = None,
) -> str:
    descriptions = _parse_subcase_descriptions()
    refs = _build_family_to_rust_ref()
    inheritance = _build_family_inheritance(refs)

    group_headers = _subcase_group_headers_html(sub_cases)
    sub_headers = _subcase_headers_html(sub_cases, descriptions)
    n_cols = 2 + len(sub_cases)

    body_rows: list[str] = []
    if top_n:
        body_rows.append(
            f'<tr class="section"><td data-section-span colspan="{n_cols}">'
            "Top-N models</td></tr>"
        )
    for model, fam in top_n:
        body_rows.append(
            render_row_html(
                model, fam, cases, sub_cases, refs, no_vllm, no_sglang, inheritance
            )
        )
    if others:
        body_rows.append(
            f'<tr class="section"><td data-section-span colspan="{n_cols}">'
            "Others</td></tr>"
        )
    for model, fam in others:
        body_rows.append(
            render_row_html(
                model, fam, cases, sub_cases, refs, no_vllm, no_sglang, inheritance
            )
        )

    all_families = [fam for _, fam in top_n] + [fam for _, fam in others]
    stats = _compute_stats(cases, sub_cases, all_families)

    now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
    stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    title = (
        f"Dynamo {family_filter} Parser Parity Table"
        if family_filter
        else "Dynamo Parser Parity Table"
    )
    command = "python3 tests/parity/parser/generate_parity_chart.py --html"
    output = "tests/parity/parser/PARITY.html"
    if family_filter:
        command += f" --family {family_filter}"
        output = f"tests/parity/parser/PARITY.{family_filter}.html"

    sha = _commit_sha()

    return (
        _make_jinja_env()
        .get_template("parity_chart.html.j2")
        .render(
            title=title,
            stamp=stamp,
            sha=sha,
            short_sha=sha[:12] if sha else "",
            command=command,
            output=output,
            group_headers=group_headers,
            sub_headers=sub_headers,
            body_rows=body_rows,
            peer_versions=_peer_version_items(_peer_versions()),
            stats=stats,
            glossary_groups=_glossary_groups(descriptions, sub_cases),
        )
    )


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument(
        "--html",
        action="store_true",
        help="Emit HTML (clickable + tooltips) instead of Markdown.",
    )
    p.add_argument(
        "--family",
        help="Render only one parser family, e.g. harmony.",
    )
    args = p.parse_args()

    cases, labels = load_all_cases()
    if args.family:
        cases = {k: v for k, v in cases.items() if k[0] == args.family}
        labels = {k: v for k, v in labels.items() if k == args.family}
        if not cases:
            raise SystemExit(f"no parser fixtures found for family={args.family!r}")
    sub_cases = _discover_sub_cases(cases)
    no_vllm, no_sglang = _derive_no_peer_sets(cases)
    top_n, others = _build_display_groups(cases, labels)
    if args.html:
        print(
            render_html(
                cases,
                sub_cases,
                no_vllm,
                no_sglang,
                top_n,
                others,
                family_filter=args.family,
            )
        )
    else:
        print(render_markdown(cases, sub_cases, no_vllm, no_sglang, top_n, others))


if __name__ == "__main__":
    main()
