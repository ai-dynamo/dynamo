# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract between Dynamo's SGLang ``/generate`` join rules and installed SGLang.

The frontend joins SGLang's streamed chunks into one unary response, and joins
prefill's prompt-side outputs into decode's response, using the rules in
``lib/generate-formats/sglang/src/merge_rules.json``. Those rules copy SGLang code
listed in the header of ``lib/generate-formats/sglang/src/merge.rs``.

These tests fail when an SGLang upgrade adds, renames, or changes a field the
rules must know about. To fix a failure, read the upstream code named in
``merge.rs``, classify the field in ``merge_rules.json``, and rerun.

Only the coalescing test imports SGLang's tokenizer manager, which takes
several seconds; the others parse SGLang's source files without importing it.
"""

import ast
import copy
import importlib.util
import json
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.unit,
    pytest.mark.sglang,
    pytest.mark.gpu_0,
    pytest.mark.profiled_vram_gib(0),
    pytest.mark.pre_merge,
]

RULES_PATH = (
    Path(__file__).resolve().parents[5]
    / "lib/generate-formats/sglang/src/merge_rules.json"
)
CATEGORIES = ("append", "take_prefill", "overlay_base64_prefix", "take_last")
FIX = (
    f"Classify it in {RULES_PATH.name} (see lib/generate-formats/sglang/src/merge.rs)."
)


@pytest.fixture(scope="module")
def rules() -> dict:
    assert RULES_PATH.is_file(), f"merge rules not found at {RULES_PATH}"
    rules = json.loads(RULES_PATH.read_text())
    # Chunk joins by field class: deltas append, prompt values come from
    # prefill, routed-expert rows overlay, and everything else keeps the last
    # value. Nested pointers refine a field classified above them.
    joins = {"delta": "append", "prompt": "take_prefill"}
    joins["sequence_final"] = "overlay_base64_prefix"
    for category in CATEGORIES:
        rules[category] = []
    for path, field_class in rules["fields"].items():
        if path.count("/") <= 2:
            rules[joins.get(field_class, "take_last")].append(path)
    return rules


def _sglang_tree(relative: str) -> ast.Module:
    """Parse a file of the installed SGLang without importing SGLang."""
    spec = importlib.util.find_spec("sglang")
    assert spec is not None and spec.submodule_search_locations, "sglang not found"
    root = Path(next(iter(spec.submodule_search_locations)))
    return ast.parse((root / relative).read_text())


@pytest.fixture(scope="module")
def tokenizer_manager_tree() -> ast.Module:
    return _sglang_tree("srt/managers/tokenizer_manager.py")


@pytest.fixture(scope="module")
def io_struct_tree() -> ast.Module:
    return _sglang_tree("srt/managers/io_struct.py")


def _meta_keys(paths: list[str]) -> set[str]:
    prefix = "/meta_info/"
    return {path[len(prefix) :] for path in paths if path.startswith(prefix)}


def _written_response_keys(tree: ast.Module) -> tuple[set[str], set[str]]:
    """Literal ``meta_info`` and top-level keys the tokenizer manager writes."""
    meta: set[str] = set()
    top: set[str] = set()

    def is_meta(node: ast.AST) -> bool:
        return (isinstance(node, ast.Name) and node.id == "meta_info") or (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and node.slice.value == "meta_info"
        )

    def keys(node: ast.Dict) -> set[str]:
        return {
            key.value
            for key in node.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Subscript) and isinstance(
                    target.slice, ast.Constant
                ):
                    if is_meta(target.value):
                        meta.add(target.slice.value)
                    elif isinstance(target.value, ast.Name) and target.value.id in {
                        "out",
                        "out_dict",
                    }:
                        top.add(target.slice.value)
                if is_meta(target) and isinstance(node.value, ast.Dict):
                    meta |= keys(node.value)
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "update"
            and is_meta(node.func.value)
            and node.args
            and isinstance(node.args[0], ast.Dict)
        ):
            meta |= keys(node.args[0])
        elif isinstance(node, ast.Dict) and "meta_info" in keys(node):
            top |= keys(node)
        elif isinstance(node, ast.FunctionDef) and "top_logprobs_fields" in node.name:
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Constant)
                    and isinstance(inner.value, str)
                    and inner.value.startswith("input_top_logprobs_")
                ):
                    meta.add(inner.value)
                # Newer builders name keys f"{side}_top_logprobs_..." for both sides.
                elif (
                    isinstance(inner, ast.JoinedStr)
                    and len(inner.values) == 2
                    and isinstance(inner.values[0], ast.FormattedValue)
                    and isinstance(inner.values[1], ast.Constant)
                    and str(inner.values[1].value).startswith("_top_logprobs_")
                ):
                    suffix = inner.values[1].value
                    meta |= {f"input{suffix}", f"output{suffix}"}
    return meta, top - {"meta_info"}


def test_incremental_streaming_keys_are_appended(rules, tokenizer_manager_tree):
    name = "_INCREMENTAL_STREAMING_META_INFO_KEYS"
    values = [
        ast.literal_eval(node.value)
        for node in tokenizer_manager_tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == name for t in node.targets)
    ]
    assert values, f"tokenizer_manager.py no longer defines {name}. {FIX}"
    missing = set(values[0]) - _meta_keys(rules["append"])
    assert not missing, (
        f"SGLang streams these meta_info keys as deltas, but Dynamo keeps only "
        f"the last chunk's value: {sorted(missing)}. Add them to `append`. {FIX}"
    )


def test_sglang_coalescing_matches_dynamo_join(rules):
    """SGLang's own chunk join must equal Dynamo's rule: append the `append`
    fields, take every other field from the last chunk."""
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    appended = set(rules["append"])
    last_only = rules["take_prefill"] + rules["overlay_base64_prefix"]
    last_only += rules["take_last"]

    def chunk(index: int) -> dict:
        response: dict = {"meta_info": {}}
        for path in rules["append"]:
            *parents, key = path.strip("/").split("/")
            target = response
            for parent in parents:
                target = target.setdefault(parent, {})
            target[key] = f"t{index}" if key == "text" else [f"{key}-{index}"]
        for path in last_only:
            *parents, key = path.strip("/").split("/")
            target = response
            for parent in parents:
                target = target.setdefault(parent, {})
            target[key] = f"{key}-{index}"
        return response

    chunks = [chunk(index) for index in range(3)]
    coalesced = TokenizerManager._coalesce_streaming_chunks(
        None, copy.deepcopy(chunks), "contract-test", []
    )

    expected = copy.deepcopy(chunks[-1])
    for path in appended:
        *parents, key = path.strip("/").split("/")
        target = expected
        for parent in parents:
            target = target[parent]
        values = []
        for item in chunks:
            for parent in parents:
                item = item[parent]
            values.append(item[key])
        target[key] = "".join(values) if key == "text" else sum(values, [])
    assert coalesced == expected, (
        "SGLang's _coalesce_streaming_chunks no longer joins chunks the way "
        f"Dynamo's `append` rule does. {FIX}"
    )


def test_every_sglang_response_field_is_classified(rules, tokenizer_manager_tree):
    meta, top = _written_response_keys(tokenizer_manager_tree)
    classified = {path for category in CATEGORIES for path in rules[category]}
    written = {f"/meta_info/{key}" for key in meta} | {f"/{key}" for key in top}
    unclassified = written - classified
    assert not unclassified, (
        f"SGLang writes response fields that Dynamo's join rules do not know: "
        f"{sorted(unclassified)}. Decide whether each is streamed as a delta "
        f"(`append`), computed only by prefill (`take_prefill` or "
        f"`overlay_base64_prefix`), or a current full value (`take_last`). {FIX}"
    )


def test_request_flags_are_known(rules, io_struct_tree):
    classes = [
        node
        for node in io_struct_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "GenerateReqInput"
    ]
    assert classes, f"io_struct.py no longer defines GenerateReqInput. {FIX}"
    fields = {
        stmt.target.id
        for stmt in classes[0].body
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name)
    }
    groups = rules["prefill_request_flags"]
    gating = {flag for group in groups["before_first"] for flag in group}
    gating |= {flag for group in groups["before_last"] for flag in group}

    renamed = gating - fields
    assert not renamed, (
        f"Dynamo decides when to wait for prefill output from request fields "
        f"that GenerateReqInput no longer has: {sorted(renamed)}. {FIX}"
    )
    known = gating | set(rules["other_request_flags"])
    new = {field for field in fields if field.startswith("return_")} - known
    assert not new, (
        f"GenerateReqInput has new output flags {sorted(new)}. If one adds a "
        f"prompt-side output, add it to `prefill_request_flags`; otherwise add "
        f"it to `other_request_flags`. {FIX}"
    )
