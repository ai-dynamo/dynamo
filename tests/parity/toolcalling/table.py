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

Reads every `tests/parity/toolcalling/fixtures/<family>/TOOLCALLING.batch*.yaml` and emits
the table referenced in `tests/parity/README.md`.

Cell markers (per peer, vllm + sglang):
  =     peer block is `*d_<case>` anchor ref to dynamo (matches)
  V/S   peer is a concrete inline block AND has `reason:` (intentional)
  V?/S? peer is a concrete inline block AND has no `reason:` yet
        (research-needed; we observed it but haven't classified it)
  V!/S! peer has `error: <substring>` (expected to crash)
  VS, V?S, VS!, etc. — combinations
  ·     Dynamo-only fixture; both peer blocks are `unavailable`
  n/a   family/case doesn't apply
  —     no fixture entry exists for this family/case yet

Footnote markers `†` (no vLLM peer) and `§` (no SGLang peer) are auto-derived
from `expected.<impl>.unavailable` across each family's cases.

Run:
    # Markdown table to stdout
    python3 tests/parity/generate_parity_table.py toolcalling \
        > tests/parity/toolcalling/PARITY.md
    python3 tests/parity/generate_parity_table.py toolcalling --mode stream \
        > tests/parity/toolcalling/PARITY.stream.md

    # HTML table with tabs, clickable YAML links, and hover tooltips. Write next
    # to this script so `<a href="fixtures/<family>/TOOLCALLING.batch.N.yaml">`
    # resolves when opened in a browser.
    python3 tests/parity/generate_parity_table.py toolcalling --html \
        > tests/parity/toolcalling/PARITY.html

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

from tests.parity.common import TOP_N_TOOL_CALLING_FAMILIES as TOP_N_FAMILIES
from tests.parity.common import (
    build_parity_tooltip_html,
    linkify_text_html,
    parity_cell_class,
)
from tests.parity.markup import colorize_markup, colorize_stream_deltas

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "tests/parity/toolcalling/fixtures"
TOOLCALLING_CASES_MD = REPO_ROOT / "lib/parsers/TOOLCALLING_CASES.md"
PYPROJECT_TOML = REPO_ROOT / "pyproject.toml"
TEMPLATE_DIR = REPO_ROOT / "tests/parity"

RUST_TOOL_CALLING_DIR = REPO_ROOT / "lib/parsers/src/tool_calling"

# Row-label / visibility overrides keyed by tool calling family; ‡ is explained
# by the legend note in parity_table.html.j2.
_TOOL_CALLING_LABEL_OVERRIDES = {
    "qwen3_coder": "Qwen 3 Coder / Nemotron V3‡",
}
# nemotron_nano: an alias for qwen3_coder, hide to avoid duplicate row
# nemotron_deci: for older v2 nemotron models, hide to avoid confusion with nemotron v3 models
_HIDDEN_TOOL_CALLING_FAMILIES = {"nemotron_deci", "nemotron_nano"}


def _model_label_html(model: str) -> str:
    """Escape a model label, styling any ‡ marker like the †/§ suffixes."""
    return html_lib.escape(model).replace("‡", '<span class="parser-suffix">‡</span>')


def _make_jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
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


BATCH_SUB_CASE_GROUPS = [
    ("Single-call", ("1.a", "1.b", "1.c", "1.d")),
    ("Core", ("1", "3", "9", "9.a", "9.b")),
    ("Multi-call", ("2.a", "2.b", "2.c", "2.d", "2.e", "10")),
    (
        "Malformed / recovery",
        (
            "4.a",
            "4.b",
            "4.c",
            "4.d",
            "4.e",
            "4.f",
            "5.a",
            "5.b",
            "5.c",
            "5.d",
            "5.e",
            "5.f",
            "5.g",
        ),
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
            "7.f",
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
    "1": ("1.a",),
    "9": ("9.a",),
    "30": ("30.a", "30.b", "30.c"),
    "31": ("31.a", "31.b"),
    "13": ("13.a",),
}

STREAM_SUB_CASE_GROUPS = [
    ("Single-call", ("1.a", "1.b")),
    ("Multi-call", ("2",)),
    ("Partial-token", ("3",)),
    ("Termination", ("4.a", "4.b", "4.c")),
]

SUB_CASE_GROUPS_BY_MODE = {
    "batch": BATCH_SUB_CASE_GROUPS,
    "stream": STREAM_SUB_CASE_GROUPS,
}

_SUB_CASE_GROUP_KEY_BY_LABEL_BY_MODE = {
    mode: {
        label: re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_")
        for label, _subs in groups
    }
    for mode, groups in SUB_CASE_GROUPS_BY_MODE.items()
}

_SUB_CASE_GROUP_KEY_BY_SUB_BY_MODE = {
    mode: {
        sub: _SUB_CASE_GROUP_KEY_BY_LABEL_BY_MODE[mode][label]
        for label, subs in groups
        for sub in subs
    }
    for mode, groups in SUB_CASE_GROUPS_BY_MODE.items()
}


def _display_order(mode: str) -> dict[str, tuple[int, int]]:
    return {
        sub: (group_idx, sub_idx)
        for group_idx, (_label, subs) in enumerate(SUB_CASE_GROUPS_BY_MODE[mode])
        for sub_idx, sub in enumerate(subs)
    }


def _group_index_by_sub(mode: str) -> dict[str, int]:
    return {
        sub: group_idx
        for group_idx, (_label, subs) in enumerate(SUB_CASE_GROUPS_BY_MODE[mode])
        for sub in subs
    }


def _group_by_sub(mode: str) -> dict[str, str]:
    return {sub: label for label, subs in SUB_CASE_GROUPS_BY_MODE[mode] for sub in subs}


def _natural_sub_sort_key(sub: str) -> tuple[int, str]:
    """`8.a` → (8, 'a'); `9` → (9, '')."""
    parts = sub.split(".")
    return (int(parts[0]), parts[1] if len(parts) > 1 else "")


def _sub_sort_key(mode: str, sub: str) -> tuple[int, int, int, str]:
    """Sort known cases by semantic display group, future cases naturally last."""
    display_order = _display_order(mode).get(sub)
    if display_order is not None:
        group_idx, sub_idx = display_order
        return (0, group_idx, sub_idx, "")
    num, suffix = _natural_sub_sort_key(sub)
    return (1, num, 0, suffix)


def _subcase_band_class(mode: str, sub: str) -> str:
    group_idx = _group_index_by_sub(mode).get(sub, len(SUB_CASE_GROUPS_BY_MODE[mode]))
    return f"case-band-{group_idx % 2}"


def _subcase_group_key(mode: str, sub: str) -> str:
    return _SUB_CASE_GROUP_KEY_BY_SUB_BY_MODE[mode].get(sub, "other")


def _discover_sub_cases(mode: str, cases: dict) -> list[str]:
    """Union of sub-case IDs across all loaded fixtures, in stable order."""
    return sorted(
        {sub for _fam, sub in cases.keys()}, key=lambda s: _sub_sort_key(mode, s)
    )


def _normalize_split_parent_cases(cases: dict) -> dict:
    """Render split taxonomy buckets as leaf cases only.

    Some older fixture files still define parent buckets such as
    `TOOLCALLING.batch.30`, while newer files define leaf buckets such as
    `TOOLCALLING.batch.30.a`. For display, parent+leaf duplication is confusing:
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
                child_case["__case_id"] = f"TOOLCALLING.batch.{child}"
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
    rejected the family for that parser in `capture_toolcalling_outputs.py`.
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


def load_all_cases(mode: str) -> tuple[dict[tuple[str, str], dict], dict[str, str]]:
    """Load every fixture YAML for one parser mode.

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
    for fp in sorted(FIXTURES.glob(f"*/TOOLCALLING.{mode}*.yaml")):
        doc = yaml.safe_load(fp.read_text())
        if doc.get("mode") != mode:
            continue
        family = doc["family"]
        rel = str(fp.relative_to(script_dir))
        if "model_label" in doc:
            labels.setdefault(family, doc["model_label"])
        for cid, case in doc["cases"].items():
            case["__family"] = family
            sub = cid.replace(f"TOOLCALLING.{mode}.", "")
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
    families = {
        fam for fam, _ in cases.keys() if fam not in _HIDDEN_TOOL_CALLING_FAMILIES
    }

    def label_of(fam: str) -> str:
        return _TOOL_CALLING_LABEL_OVERRIDES.get(fam, labels.get(fam, fam))

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


_TOOL_CALL_MARKUP_RE = re.compile(
    r"</?tool_call|</?tool_calls|<\|tool_call|<\|tool_calls|"
    r"<\|(?:channel|message|call|python_tag)\|>|"
    r"</?TOOLCALL|TOOL_CALLS|<｜(?:DSML｜)?(?:tool|tool▁call|tool▁calls)|"
    r"<｜DSML｜|</?minimax:tool_call|</?invoke|</?arg_key|</?arg_value"
)


def _dynamo_tool_call_leak(dyn: dict) -> str | None:
    normal_text = dyn.get("normal_text")
    if not dyn.get("reason") or not isinstance(normal_text, str):
        return None
    if not _TOOL_CALL_MARKUP_RE.search(normal_text):
        return None
    return str(dyn["reason"])


def _block_tool_call_leaks(block: dict) -> bool:
    normal_text = block.get("normal_text")
    return isinstance(normal_text, str) and bool(
        _TOOL_CALL_MARKUP_RE.search(normal_text)
    )


def _overview_status(case: dict | None, impl: str) -> str:
    if case is None or "expected" not in case:
        return "na"
    block = case.get("expected", {}).get(impl)
    if not isinstance(block, dict) or "unavailable" in block:
        return "na"
    if "error" in block or _block_tool_call_leaks(block):
        return "problem"
    return "ok"


def _overview_status_attrs(case: dict | None) -> str:
    return " ".join(
        f'data-status-{impl}="{_overview_status(case, impl)}"'
        for impl in ("dynamo", "vllm", "sglang")
    )


def _canonical_tool_output(block: object) -> dict | None:
    if not isinstance(block, dict) or "unavailable" in block or "error" in block:
        return None
    if "calls" not in block and "normal_text" not in block:
        return None
    return {
        "calls": block.get("calls") or [],
        "normal_text": block.get("normal_text") or "",
    }


def _selected_parity_marker(case: dict | None, impl: str) -> str | None:
    if case is None or "expected" not in case:
        return None
    expected = case.get("expected", {})
    outputs = {
        impl: _canonical_tool_output(expected.get(impl))
        for impl in ("dynamo", "vllm", "sglang")
    }
    if any(value is None for value in outputs.values()):
        return None
    if outputs["dynamo"] == outputs["vllm"] == outputs["sglang"]:
        return "="
    selected = outputs[impl]
    peers = (
        ("dynamo", "D"),
        ("vllm", "V"),
        ("sglang", "S"),
    )
    marker = "".join(
        letter for peer, letter in peers if peer != impl and outputs[peer] != selected
    )
    return marker or "="


def _selected_parity_suffix(case: dict | None, impl: str) -> str:
    if case is None or "expected" not in case:
        return ""
    block = case.get("expected", {}).get(impl)
    if isinstance(block, dict) and _block_tool_call_leaks(block):
        return "↯"
    return ""


def _parity_marker(case: dict | None, impl: str) -> str:
    marker = _selected_parity_marker(case, impl)
    if marker is None:
        return _parser_marker(case, impl)
    return _selected_parity_suffix(case, impl) + marker


def _parser_marker(case: dict | None, impl: str) -> str:
    if case is None:
        return "—"
    if "expected" not in case:
        return "n/a"
    expected = case.get("expected", {})
    block = expected.get(impl)
    if not isinstance(block, dict) or "unavailable" in block:
        return "n/a"
    if "error" in block:
        return "!"
    if _block_tool_call_leaks(block):
        return "↯"
    if impl == "dynamo":
        peers = (expected.get("vllm"), expected.get("sglang"))
        if all(isinstance(peer, dict) and "unavailable" in peer for peer in peers):
            return "·"
    return ""


def _parser_marker_attrs(case: dict | None) -> str:
    attrs = [
        f'data-marker-{impl}="{html_lib.escape(_parser_marker(case, impl))}"'
        for impl in ("dynamo", "vllm", "sglang")
    ]
    attrs.extend(
        f'data-marker-parity-{impl}="{html_lib.escape(_parity_marker(case, impl))}"'
        for impl in ("dynamo", "vllm", "sglang")
    )
    return " ".join(attrs)


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
    if isinstance(dyn, dict) and _dynamo_tool_call_leak(dyn):
        if v_kind == "unavail" and s_kind == "unavail":
            return "↯·"
        if parts:
            return "↯" + "".join(parts)
        return "↯"

    if parts:
        return "".join(parts)
    if v_kind == "unavail" and s_kind == "unavail":
        return "·"
    return "="


def render_row(
    model: str,
    family: str,
    cases: dict,
    sub_cases: list[str],
    no_vllm: set[str],
    no_sglang: set[str],
    inheritance: dict[str, dict],
) -> str:
    cells = [cell_for(cases.get((family, sub))) for sub in sub_cases]
    parser_label = _parser_label_markdown(family, no_vllm, no_sglang, inheritance)
    return f"| {model} | {parser_label} | " + " | ".join(cells) + " |"


_LEGEND_MD = (
    "**Legend:** "
    "`=` all captured peers match Dynamo · "
    "`·` Dynamo-only fixture (both peers unavailable) · "
    "`V`/`S` divergence (V = vLLM, S = SGLang; intentional, has `reason:`) · "
    "`?` research-needed suffix (e.g. V?, S? — diverges with no `reason:` yet) · "
    "`↯` Dynamo leaks tool call markup into `normal_text` "
    "(`expected.dynamo.reason:` carries the explanation) · "
    "`!` expected-error suffix (e.g. V!, S! — engine crashes by design) · "
    "`n/a` not applicable · "
    "`—` missing fixture coverage · "
    "`†` (tool calling parser column) = no vLLM peer parser for this family · "
    "`§` (tool calling parser column) = no SGLang peer parser for this family."
    "\n\n"
    "`‡` Nemotron V3 (Ultra) reuses the qwen3_coder tool calling parser; "
    "Nemotron V1 / V2 (DeciLM) is removed from the chart for being an older "
    "generation, but the nemotron_deci parser is still supported."
)


def render_markdown(
    cases: dict,
    sub_cases: list[str],
    no_vllm: set[str],
    no_sglang: set[str],
    top_n: list[tuple[str, str]],
    others: list[tuple[str, str]],
) -> str:
    inheritance = _build_family_inheritance(_build_family_to_rust_ref())
    header = "| model | Tool calling family | " + " | ".join(sub_cases) + " |"
    sep = "|---|---|" + ":-:|" * len(sub_cases)
    lines = [header, sep]
    lines.append("| **Top-N models** |   |" + "   |" * len(sub_cases))
    for model, fam in top_n:
        lines.append(
            render_row(model, fam, cases, sub_cases, no_vllm, no_sglang, inheritance)
        )
    lines.append("| **Others** |   |" + "   |" * len(sub_cases))
    for model, fam in others:
        lines.append(
            render_row(model, fam, cases, sub_cases, no_vllm, no_sglang, inheritance)
        )
    lines.append("")
    lines.append(_LEGEND_MD)
    return "\n".join(lines)


_IMPL_DISPLAY = {"dynamo": "Dynamo", "vllm": "vLLM", "sglang": "SGLang"}


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
    nt_line = f"normal_text='{colorize_markup(nt, family)}'"
    return f"{nt_line}\n{calls_line}"


def _build_tooltip_html(case: dict, dyn) -> str:
    """Rich HTML hover tooltip: head, input (colorized), per-engine output,
    divergence reasons. Returns the full `<div class="ttip">...</div>`."""
    case_id = case.get("__case_id", "")
    family = case.get("__family")
    desc = case.get("description") or ""
    if case_id and family:
        head = f"{case_id} — {family}"
    else:
        head = case_id or str(family or "")

    input_label = None
    input_html = None
    model_text = case.get("model_text")
    if isinstance(model_text, str) and model_text:
        input_label = "Input"
        input_html = f"input_text='{colorize_markup(model_text, family)}'"
    chunks = case.get("chunks")
    if isinstance(chunks, list) and chunks:
        chunk_lines = []
        chunk_html = colorize_stream_deltas(chunks, family)
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                continue
            suffix = ""
            if chunk.get("finish_reason"):
                suffix = " finish_reason=" + html_lib.escape(
                    str(chunk["finish_reason"])
                )
            chunk_lines.append(f"{i}: delta_text='{chunk_html[i]}'{suffix}")
        input_label = "Input chunks"
        input_html = "\n".join(chunk_lines)

    expected = case.get("expected") or {}

    def _norm(b):
        return {
            "calls": b.get("calls") or [],
            "normal_text": b.get("normal_text") or "",
        }

    n_dyn = _norm(dyn) if isinstance(dyn, dict) else None
    all_engines_parity = isinstance(dyn, dict) and all(
        isinstance(expected.get(i), dict)
        and not expected[i].get("unavailable")
        and "error" not in expected[i]
        and _norm(expected[i]) == n_dyn
        for i in ("dynamo", "vllm", "sglang")
    )

    output_sections: list[tuple[str, str]] = []
    if all_engines_parity:
        output_sections.append(
            ("All engines parity", _format_output_block_html(dyn, family))
        )
    else:
        for impl in ("dynamo", "vllm", "sglang"):
            block = expected.get(impl)
            output_sections.append(
                (_IMPL_DISPLAY[impl], _format_output_block_html(block, family))
            )

    reasons = _tooltip_for(case, dyn) if isinstance(dyn, dict) else ""

    dyn_leak = _dynamo_tool_call_leak(dyn) if isinstance(dyn, dict) else None
    return build_parity_tooltip_html(
        head=head,
        description=desc,
        input_label=input_label,
        input_html=input_html,
        output_sections=output_sections,
        divergent_reasons=reasons or None,
        leak_label="↯ Dynamo tool call leaks",
        leak_text=str(dyn_leak) if dyn_leak else None,
        refs=[("Ref", case.get("ref")), ("Spec ref", case.get("spec_ref"))],
    )


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


def _build_na_tooltip_html(case: dict) -> str:
    """Tooltip for an n/a stub case (only `reason:` in YAML, no `expected:`
    block). Renders case id + description + the reason. Used when the cell
    is n/a because the scenario doesn't apply to the family's parser syntax."""
    case_id = case.get("__case_id", "")
    desc = case.get("description") or ""
    head = f"{case_id} — {desc}" if (case_id and desc) else (case_id or desc)
    reason = case.get("reason") or "n/a (no reason given)"
    return build_parity_tooltip_html(
        head=head,
        extra_sections=[("Why not applicable", linkify_text_html(str(reason)))],
        refs=[("Ref", case.get("ref")), ("Spec ref", case.get("spec_ref"))],
    )


def _build_missing_tooltip_html(mode: str, family: str, sub: str) -> str:
    """Tooltip for an absent fixture entry.

    This is intentionally distinct from an explicit n/a stub. Missing means
    the table has no fixture data for this family/case; explicit n/a means a
    fixture author recorded why the case does not apply.
    """
    case_id = f"TOOLCALLING.{mode}.{sub}"
    return build_parity_tooltip_html(
        head=f"{case_id} — {family}",
        extra_sections=[
            (
                "Missing fixture",
                html_lib.escape(
                    "No fixture entry exists for this family/case. If the case "
                    "is intentionally not applicable, add an explicit n/a stub "
                    "with description: and reason: so the table can explain it."
                ),
            )
        ],
    )


def render_cell_html(case: dict | None, mode: str, family: str, sub: str) -> str:
    text = cell_for(case)
    cls = parity_cell_class(text)
    band_cls = _subcase_band_class(mode, sub)
    col_group = html_lib.escape(_subcase_group_key(mode, sub))
    status_attrs = _overview_status_attrs(case)
    marker_attrs = _parser_marker_attrs(case)
    td_open = (
        f'<td class="cell {cls} {band_cls}" data-col-hide-group="{col_group}" '
        f"{status_attrs} {marker_attrs}>"
    )
    if case is None:
        ttip = _build_missing_tooltip_html(mode, family, sub)
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
    """Rich `.ttip` tooltip for the tool calling parser column.

    Keep this field-list shape aligned with the reasoning parser column tooltip
    so both tables explain "effective parser/backend -> row family" the same
    way. `ctor_ref` is unused here (was for older field-based layout) — kept
    for API stability with `_parser_cell_html`.
    """
    del ctor_ref

    variant = info["variant"] or "?"
    sub_variant = info["sub_variant"]
    backend_file = info["backend_file"]
    factory = info["factory"]
    alias_of = info.get("alias_of")  # set when this family is an alias-only entry

    head_parts = [f"ParserConfig::{variant}"]
    if sub_variant:
        head_parts[-1] = f"ParserConfig::{variant}::{sub_variant}"
    bf_href = html_lib.escape(f"../../../lib/parsers/src/tool_calling/{backend_file}")
    bf_link = f'<a href="{bf_href}">{html_lib.escape(backend_file)}</a>'

    anchor = alias_of or family
    shared_family = sorted([anchor] + info["shared_with"])
    effective_backend = _shared_backend_short(info) or family

    implementation = f"{html_lib.escape(head_parts[0])} -> {bf_link}"
    if factory:
        factory_name = factory.split("(", 1)[0]
        implementation += html_lib.escape(f" (factory: {factory_name})")

    tooltip_lines = [
        "Tool calling parser family from fixture YAML.",
        f"Tool calling parser row: {html_lib.escape(family)}",
        f"Effective parser/backend: {html_lib.escape(effective_backend)}",
        f"Dynamo implementation: {implementation}",
    ]
    if info["shared_with"]:
        tooltip_lines.append(
            "Shared implementation family: " + html_lib.escape(", ".join(shared_family))
        )
    if alias_of:
        tooltip_lines.append(f"Alias of: {html_lib.escape(alias_of)}")
    if info["aliases"]:
        tooltip_lines.append(
            "Registered aliases: " + html_lib.escape(", ".join(info["aliases"]))
        )

    peer_notes: list[str] = []
    if no_vllm and family in no_vllm:
        peer_notes.append("no vLLM peer parser")
    if no_sglang and family in no_sglang:
        peer_notes.append("no SGLang peer parser")
    if peer_notes:
        tooltip_lines.append("Peer availability: " + ", ".join(peer_notes))

    if info["filed_under_xml_misleading"]:
        tooltip_lines.append(
            "Note: filed under xml/ but does not use the shared xml::parser; "
            f"it has its own ParserConfig::{html_lib.escape(variant)} variant."
        )
    tooltip_lines.extend(_tool_parser_tree_lines(family, info, effective_backend))

    if effective_backend == family:
        head_text = f"`{family}`"
    else:
        head_text = f"`{effective_backend}` (row: `{family}`)"
    return (
        '<div class="ttip">'
        f'<div class="ttip-head">{html_lib.escape(head_text)}</div>'
        f'<pre class="ttip-pre">{"".join(line + chr(10) for line in tooltip_lines).rstrip()}</pre>'
        "</div>"
    )


_SHARED_BACKEND_SHORT = {
    ("Json", "Basic"): "base_json",
    ("Xml", None): "xml",
    ("Dsml", None): "dsml",
}


def _tool_parser_tree_lines(
    family: str,
    info: dict,
    effective_backend: str,
) -> list[str]:
    alias_of = info.get("alias_of")
    anchor = alias_of or family
    aliases = info["aliases"]
    if not info["shared_with"] and not aliases and effective_backend == family:
        return []

    fam_list = sorted([anchor] + info["shared_with"])
    lines = ["", "Shared implementation tree:"]
    root_label = html_lib.escape(effective_backend)
    if effective_backend == family:
        root_label = f"<strong>{root_label}</strong>"
    lines.append(f"{root_label} (effective parser/backend)")

    for i, fam in enumerate(fam_list):
        is_last_fam = i == len(fam_list) - 1
        branch = "└── " if is_last_fam else "├── "
        fam_label = html_lib.escape(fam)
        if fam == family and not alias_of:
            fam_label = f"<strong>{fam_label}</strong>"
        lines.append(f"{branch}{fam_label}")

        if fam == anchor and aliases:
            cont = "    " if is_last_fam else "│   "
            for j, alias in enumerate(aliases):
                alast = j == len(aliases) - 1
                ab = "└── " if alast else "├── "
                alias_label = html_lib.escape(alias)
                if alias_of and alias == family:
                    alias_label = f"<strong>{alias_label}</strong>"
                lines.append(f"{cont}{ab}{alias_label} (alias)")

    return lines


def _shared_backend_short(info: dict | None) -> str | None:
    if info and info["shared_with"]:
        return _SHARED_BACKEND_SHORT.get(info["key"])
    return None


def _parser_label_markdown(
    family: str,
    no_vllm: set[str],
    no_sglang: set[str],
    inheritance: dict[str, dict],
) -> str:
    suff = family_suffix(family, no_vllm, no_sglang)
    short = _shared_backend_short(inheritance.get(family))
    if short:
        return f"{short} -> {family}{suff}"
    return f"{family}{suff}"


def _parser_cell_html(
    family: str,
    refs: dict[str, tuple[str, int]],
    no_vllm: set[str],
    no_sglang: set[str],
    inheritance: dict[str, dict],
) -> str:
    suff = family_suffix(family, no_vllm, no_sglang)
    row_label = html_lib.escape(family)
    if suff:
        row_label += f'<span class="parser-suffix">{html_lib.escape(suff)}</span>'
    ref = refs.get(family)
    info = inheritance.get(family)
    ttip = (
        _parser_inheritance_tooltip_html(family, info, ref, no_vllm, no_sglang)
        if info
        else ""
    )

    # Shared-backend rows should read as implementation -> fixture family,
    # e.g. `xml -> minimax_m2` and `xml -> qwen3_coder`. Standalone parsers
    # keep the public family name as the primary label.
    short = _shared_backend_short(info)
    if short:
        label = html_lib.escape(short)
        base_suffix = f'<span class="parser-base">→ {row_label}</span>'
    else:
        label = row_label
        base_suffix = ""

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
    mode: str,
    cases: dict,
    sub_cases: list[str],
    refs: dict[str, tuple[str, int]],
    no_vllm: set[str],
    no_sglang: set[str],
    inheritance: dict[str, dict],
) -> str:
    cells = [
        f'<tr><td class="model" data-col-hide-group="model">{_model_label_html(model)}</td>',
        _column_placeholder_html("model"),
        _parser_cell_html(family, refs, no_vllm, no_sglang, inheritance),
        _column_placeholder_html("parser"),
    ]
    for run in _subcase_runs(mode, sub_cases):
        cells.extend(
            render_cell_html(cases.get((family, sub)), mode, family, sub) for sub in run
        )
        cells.append(_column_placeholder_html(_subcase_group_key(mode, run[0])))
    cells.append("</tr>")
    return "".join(cells)


def _parse_subcase_descriptions(mode: str) -> dict[str, str]:
    """Parse `lib/parsers/TOOLCALLING_CASES.md` for per-case descriptions.

    The Quick-reference section has one-liner bullets for top-level cases
    (`TOOLCALLING.<mode>.1` …); the deeper per-case sections
    contain multi-line bullets for sub-cases (`2.a`, `4.c`, etc.). Both
    look like `- **`TOOLCALLING.<mode>.X`** <desc>`, where the bullet body may
    wrap across indented continuation lines. Returns
    `{"1": "...", "2.a": "...", ...}`.
    """
    if not TOOLCALLING_CASES_MD.exists():
        return {}
    pat = re.compile(
        rf"\*\*`TOOLCALLING\.{re.escape(mode)}" rf"\.([0-9]+(?:\.[a-z])?)`\*\*\s+(.+)"
    )
    out: dict[str, str] = {}
    lines = TOOLCALLING_CASES_MD.read_text(encoding="utf-8").splitlines()
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


def _subcase_header_html(mode: str, sub: str, descriptions: dict[str, str]) -> str:
    desc = descriptions.get(sub) or descriptions.get(sub.split(".")[0]) or ""
    href = "../../../lib/parsers/TOOLCALLING_CASES.md"
    title = html_lib.escape(desc) if desc else ""
    band_cls = _subcase_band_class(mode, sub)
    col_group = html_lib.escape(_subcase_group_key(mode, sub))
    return (
        f'<th class="case-sub {band_cls}" data-col-hide-group="{col_group}">'
        f'<a href="{href}" title="{title}">{html_lib.escape(sub)}</a></th>'
    )


def _subcase_group_label(mode: str, sub: str) -> str:
    return _group_by_sub(mode).get(sub, "Other")


def _subcase_runs(mode: str, sub_cases: list[str]) -> list[list[str]]:
    runs: list[list[str]] = []
    start = 0
    while start < len(sub_cases):
        label = _subcase_group_label(mode, sub_cases[start])
        end = start + 1
        while (
            end < len(sub_cases) and _subcase_group_label(mode, sub_cases[end]) == label
        ):
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


def _subcase_group_headers_html(mode: str, sub_cases: list[str]) -> str:
    """Build semantic group headers spanning the displayed sub-case columns."""
    spans: list[str] = [
        _column_control_header_html("model", "Model", default_visible=True),
        _column_control_header_html(
            "parser", "Tool calling family", default_visible=True
        ),
    ]
    for run in _subcase_runs(mode, sub_cases):
        label = _subcase_group_label(mode, run[0])
        band_cls = _subcase_band_class(mode, run[0])
        col_group = html_lib.escape(_subcase_group_key(mode, run[0]))
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


def _subcase_headers_html(
    mode: str, sub_cases: list[str], descriptions: dict[str, str]
) -> str:
    headers: list[str] = []
    for run in _subcase_runs(mode, sub_cases):
        headers.extend(_subcase_header_html(mode, sub, descriptions) for sub in run)
        headers.append(
            _column_placeholder_html(_subcase_group_key(mode, run[0]), tag="th")
        )
    return "".join(headers)


def _glossary_groups(
    mode: str, descriptions: dict[str, str], sub_cases: list[str]
) -> list[dict[str, object]]:
    if not descriptions:
        return []
    return [
        {
            "label": _subcase_group_label(mode, run[0]),
            "rows": [
                (
                    sub,
                    descriptions.get(sub) or descriptions.get(sub.split(".")[0]) or "",
                )
                for sub in run
            ],
        }
        for run in _subcase_runs(mode, sub_cases)
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
        "dynamo_only": 0,
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
            elif text in {"D", "·"}:
                s["dynamo_only"] += 1
            elif "!" in text:
                s["errors"] += 1
            elif "↯" in text:
                s["documented"] += 1
            elif "?" in text:
                s["research"] += 1
            else:
                s["documented"] += 1
    return s


def _mode_label(mode: str) -> str:
    if mode == "batch":
        return "TOOLCALLING.batch.*"
    if mode == "stream":
        return "TOOLCALLING.stream.*"
    return mode


def render_html_panel(
    mode: str,
    cases: dict,
    sub_cases: list[str],
    no_vllm: set[str],
    no_sglang: set[str],
    top_n: list[tuple[str, str]],
    others: list[tuple[str, str]],
    active: bool = False,
) -> dict[str, object]:
    """Render one tab panel for one parser fixture mode."""
    descriptions = _parse_subcase_descriptions(mode)
    refs = _build_family_to_rust_ref()
    inheritance = _build_family_inheritance(refs)

    group_headers = _subcase_group_headers_html(mode, sub_cases)
    sub_headers = _subcase_headers_html(mode, sub_cases, descriptions)
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
                model,
                fam,
                mode,
                cases,
                sub_cases,
                refs,
                no_vllm,
                no_sglang,
                inheritance,
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
                model,
                fam,
                mode,
                cases,
                sub_cases,
                refs,
                no_vllm,
                no_sglang,
                inheritance,
            )
        )

    all_families = [fam for _, fam in top_n] + [fam for _, fam in others]
    stats = _compute_stats(cases, sub_cases, all_families)

    panel_id = f"tab-{mode}"
    return {
        "id": panel_id,
        "mode": mode,
        "label": _mode_label(mode),
        "active": active,
        "group_headers": group_headers,
        "sub_headers": sub_headers,
        "body_rows": body_rows,
        "stats": stats,
        "glossary_groups": _glossary_groups(mode, descriptions, sub_cases),
    }


def _filter_family(
    cases: dict[tuple[str, str], dict],
    labels: dict[str, str],
    family_filter: str | None,
) -> tuple[dict[tuple[str, str], dict], dict[str, str]]:
    if family_filter is None:
        return cases, labels
    return (
        {k: v for k, v in cases.items() if k[0] == family_filter},
        {k: v for k, v in labels.items() if k == family_filter},
    )


def _load_html_panel(
    mode: str,
    active: bool = False,
    family_filter: str | None = None,
) -> tuple[str, dict[str, object], bool]:
    cases, labels = load_all_cases(mode)
    cases, labels = _filter_family(cases, labels, family_filter)
    has_cases = bool(cases)
    sub_cases = _discover_sub_cases(mode, cases)
    no_vllm, no_sglang = _derive_no_peer_sets(cases)
    top_n, others = _build_display_groups(cases, labels)
    return (
        mode,
        render_html_panel(
            mode, cases, sub_cases, no_vllm, no_sglang, top_n, others, active
        ),
        has_cases,
    )


def render_html(modes: list[str], family_filter: str | None = None) -> str:
    panels = [
        _load_html_panel(mode, active=(i == 0), family_filter=family_filter)
        for i, mode in enumerate(modes)
    ]
    if family_filter and not any(has_cases for _mode, _panel, has_cases in panels):
        raise SystemExit(f"no parser fixtures found for family={family_filter!r}")

    now = datetime.datetime.now(zoneinfo.ZoneInfo("America/Los_Angeles"))
    stamp = now.strftime("%Y-%m-%d %H:%M %Z")
    title = (
        f"Dynamo {family_filter} Tool Calling Parser - Parity Table"
        if family_filter
        else "Dynamo Tool Calling Parser - Parity Table"
    )
    command = "python3 tests/parity/generate_parity_table.py toolcalling --html"
    output = "tests/parity/toolcalling/PARITY.html"
    if family_filter:
        command += f" --family {family_filter}"
        output = f"tests/parity/toolcalling/PARITY.{family_filter}.html"

    sha = _commit_sha()

    if len(modes) == 1:
        command += f" --mode {modes[0]}"
        output = f"tests/parity/toolcalling/PARITY.{modes[0]}.html"
        if family_filter:
            output = f"tests/parity/toolcalling/PARITY.{family_filter}.{modes[0]}.html"

    tabs = []
    for i, (mode, _panel, _has_cases) in enumerate(panels):
        panel_id = f"tab-{mode}"
        active = " active" if i == 0 else ""
        selected = "true" if i == 0 else "false"
        tabs.append(
            f'<button class="tab-button{active}" id="{panel_id}-button" '
            f'type="button" role="tab" aria-selected="{selected}" '
            f'data-tab-target="{panel_id}">{html_lib.escape(_mode_label(mode))}</button>'
        )

    return (
        _make_jinja_env()
        .get_template("parity_table.html.j2")
        .render(
            title=title,
            stamp=stamp,
            sha=sha,
            short_sha=sha[:12] if sha else "",
            command=command,
            output=output,
            tabs=tabs,
            panels=[panel for _mode, panel, _has_cases in panels],
            peer_versions=_peer_version_items(_peer_versions()),
        )
    )


def main(argv: list[str] | None = None) -> None:
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
    p.add_argument(
        "--mode",
        choices=("all", "batch", "stream"),
        help=(
            "Fixture mode to render. For HTML, default is all parser modes as "
            "tabs. For Markdown, default is batch."
        ),
    )
    args = p.parse_args(argv)

    if args.html:
        modes = ["batch", "stream"] if args.mode in (None, "all") else [args.mode]
        print(render_html(modes, family_filter=args.family))
        return

    if args.mode == "all":
        p.error("--mode all is only supported with --html")
    mode = args.mode or "batch"

    cases, labels = load_all_cases(mode)
    cases, labels = _filter_family(cases, labels, args.family)
    if args.family and not cases:
        raise SystemExit(f"no parser fixtures found for family={args.family!r}")
    sub_cases = _discover_sub_cases(mode, cases)
    no_vllm, no_sglang = _derive_no_peer_sets(cases)
    top_n, others = _build_display_groups(cases, labels)
    print(render_markdown(cases, sub_cases, no_vllm, no_sglang, top_n, others))


if __name__ == "__main__":
    main()
