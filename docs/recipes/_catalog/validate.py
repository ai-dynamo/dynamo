#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Validate the split Dynamo recipe & benchmark catalogs.

Checks, for BOTH catalogs:
  1. Every id in index.yaml (recipes + deferred_recipes / benchmarks) has a
     matching <id>.yaml file, and every <id>.yaml file is listed in the index
     (no orphan files, no dangling index entries).
  2. Each per-entry file's internal `id:` field matches its filename.
  3. No duplicate ids.
  4. Each entry validates against schema.json (uses the `jsonschema` package
     if importable; otherwise falls back to a required-top-level-keys check).
  5. Every `page:` path resolves to a real file under docs/.
  6. Every deploy / perf / benchmark asset path resolves in the repo tree.
  7. Cross-catalog referential integrity: recipe related_benchmarks ids exist
     in the benchmark index; benchmark related_recipes ids exist in the recipe
     index (active OR deferred); benchmark promotion_candidate.deferred_recipe_id
     refers to a known recipe.

Exit code is non-zero on any failure.

YAML parsing: prefers PyYAML if importable; otherwise uses a small built-in
parser that handles the subset of YAML these catalog files use (nested
mappings, block lists of scalars and of mappings, quoted/plain scalars, null,
bool, int, and PyYAML-style soft-wrapped long scalar values). Run mode is
reported in the output.

Runnable from the repo root as:
    python3 docs/recipes/_catalog/validate.py
"""

import json
import os
import sys

# --- Locate repo root relative to this script ---------------------------------
# This script lives at <repo>/docs/recipes/_catalog/validate.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
DOCS_DIR = os.path.join(REPO_ROOT, "docs")
RECIPES_CAT = os.path.join(DOCS_DIR, "recipes", "_catalog")
BENCH_CAT = os.path.join(DOCS_DIR, "benchmarks", "_catalog")

# --- Optional deps -------------------------------------------------------------
try:
    import yaml as _yaml  # type: ignore

    HAVE_YAML = True
except Exception:
    _yaml = None
    HAVE_YAML = False

try:
    import jsonschema as _jsonschema  # type: ignore

    HAVE_JSONSCHEMA = True
except Exception:
    _jsonschema = None
    HAVE_JSONSCHEMA = False


# --- Minimal YAML parser (fallback) -------------------------------------------
# Handles the constrained YAML these catalog files use. Not a general parser.
class MiniYAMLError(Exception):
    pass


def _strip_comment(line):
    # Remove a trailing comment that is not inside quotes. The catalog files
    # only ever use whole-line comments (starting with optional spaces + '#'),
    # so we only strip those; inline '#' inside values is left alone.
    return line


def _scalar(tok):
    tok = tok.strip()
    if tok == "" or tok == "~" or tok == "null":
        return None
    if tok in ("true", "True"):
        return True
    if tok in ("false", "False"):
        return False
    if tok.startswith("'") and tok.endswith("'") and len(tok) >= 2:
        return tok[1:-1].replace("''", "'")
    if tok.startswith('"') and tok.endswith('"') and len(tok) >= 2:
        return tok[1:-1]
    if tok == "[]":
        return []
    if tok == "{}":
        return {}
    try:
        return int(tok)
    except ValueError:
        pass
    try:
        return float(tok)
    except ValueError:
        pass
    return tok


def _split_kv(s):
    """Split a YAML node line into (key, rest) at the first key/value colon.

    In YAML a ':' only acts as a mapping separator when it is followed by a
    space OR is at end-of-line, and is not inside quotes. Returns (None, None)
    if the line is a plain scalar (no such colon)."""
    in_single = False
    in_double = False
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c == "'" and not in_double:
            in_single = not in_single
        elif c == '"' and not in_single:
            in_double = not in_double
        elif c == ":" and not in_single and not in_double:
            if i + 1 == n or s[i + 1] == " ":
                return s[:i], s[i + 1 :].strip()
        i += 1
    return None, None


def _mini_load(text):
    """Parse a constrained-YAML document into Python objects."""
    raw_lines = text.split("\n")
    # Build (indent, content) for non-blank, non-comment lines, but first fold
    # PyYAML-style soft-wrapped scalar continuations into their owner line.
    lines = []
    for ln in raw_lines:
        stripped = ln.strip()
        if stripped == "" or stripped.startswith("#"):
            continue
        indent = len(ln) - len(ln.lstrip(" "))
        lines.append([indent, ln.rstrip(), stripped])

    # Fold continuations: a line is a continuation of the previous logical line
    # when the previous line started a scalar value (key: value or "- value"
    # with content) and this line is more indented AND is not itself a
    # key/list-item start. PyYAML wraps long plain scalars by indenting the
    # continuation; the join uses a single space.
    folded = []
    for entry in lines:
        indent, full, stripped = entry
        if folded:
            p_indent, p_full, p_stripped = folded[-1]
            prev_is_container_key = p_stripped.endswith(":")
            looks_like_key = _looks_like_mapping_or_item(stripped)
            if (
                indent > p_indent
                and not prev_is_container_key
                and not looks_like_key
                and _has_value(p_stripped)
            ):
                # continuation -> join into previous
                joined = p_stripped + " " + stripped
                folded[-1] = [p_indent, p_full, joined]
                continue
        folded.append([indent, full, stripped])

    pos = [0]

    def parse_block(min_indent):
        # Decide between a mapping and a list at the current indent.
        if pos[0] >= len(folded):
            return None
        indent, _full, stripped = folded[pos[0]]
        if indent < min_indent:
            return None
        if stripped.startswith("- "):
            return parse_list(indent)
        if stripped == "-":
            return parse_list(indent)
        return parse_map(indent)

    def parse_map(indent):
        result = {}
        while pos[0] < len(folded):
            cur_indent, _full, stripped = folded[pos[0]]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise MiniYAMLError("unexpected indent in map: %r" % stripped)
            if stripped.startswith("- "):
                break
            key, rest = _split_kv(stripped)
            if key is None:
                raise MiniYAMLError("expected mapping key: %r" % stripped)
            key = _scalar(key)
            pos[0] += 1
            if rest == "":
                # nested block (map or list) or empty -> peek next
                child = None
                if pos[0] < len(folded):
                    nxt_indent = folded[pos[0]][0]
                    if nxt_indent > indent:
                        child = parse_block(indent + 1)
                    elif nxt_indent == indent and folded[pos[0]][2].startswith("- "):
                        # list items at SAME indent as key belong to this key
                        child = parse_list(indent)
                result[key] = child
            else:
                result[key] = _scalar(rest)
        return result

    def parse_list(indent):
        result = []
        while pos[0] < len(folded):
            cur_indent, _full, stripped = folded[pos[0]]
            if cur_indent < indent:
                break
            if cur_indent > indent:
                raise MiniYAMLError("unexpected indent in list: %r" % stripped)
            if not (stripped.startswith("- ") or stripped == "-"):
                break
            item_body = stripped[2:] if stripped.startswith("- ") else ""
            pos[0] += 1
            first_key, first_rest = _split_kv(item_body)
            if item_body == "":
                # complex item on following lines
                child = parse_block(indent + 1)
                result.append(child)
            elif first_key is not None:
                # inline first key of a mapping item: "- id: foo"
                # Re-inject as a mapping starting at this line's content indent.
                m = {}
                m[_scalar(first_key)] = _scalar(first_rest) if first_rest else None
                if first_rest == "":
                    if pos[0] < len(folded) and folded[pos[0]][0] > indent:
                        m[_scalar(first_key)] = parse_block(indent + 1)
                # remaining keys of this mapping are indented two past the dash
                child_indent = indent + 2
                rest_map = parse_map_continuation(child_indent)
                m.update(rest_map)
                result.append(m)
            else:
                result.append(_scalar(item_body))
        return result

    def parse_map_continuation(indent):
        result = {}
        while pos[0] < len(folded):
            cur_indent, _full, stripped = folded[pos[0]]
            if cur_indent != indent:
                break
            if stripped.startswith("- "):
                break
            key, rest = _split_kv(stripped)
            if key is None:
                break
            key = _scalar(key)
            pos[0] += 1
            if rest == "":
                child = None
                if pos[0] < len(folded):
                    nxt_indent = folded[pos[0]][0]
                    if nxt_indent > indent:
                        child = parse_block(indent + 1)
                    elif nxt_indent == indent and folded[pos[0]][2].startswith("- "):
                        child = parse_list(indent)
                result[key] = child
            else:
                result[key] = _scalar(rest)
        return result

    if not folded:
        return None
    return parse_block(folded[0][0])


def _is_quoted_scalar(s):
    s = s.strip()
    return (s.startswith("'") and s.endswith("'")) or (
        s.startswith('"') and s.endswith('"')
    )


def _looks_like_mapping_or_item(stripped):
    if stripped.startswith("- "):
        return True
    if stripped == "-":
        return True
    if _is_quoted_scalar(stripped):
        return False
    # "key: value" or "key:" (key/value colon present) -> mapping
    key, _rest = _split_kv(stripped)
    return key is not None


def _has_value(stripped):
    if _is_quoted_scalar(stripped):
        return True
    key, rest = _split_kv(stripped)
    if key is None:
        # bare list item value e.g. "- foo" or a plain scalar
        return True
    return rest != ""


def load_yaml(path):
    with open(path, "r") as f:
        text = f.read()
    if HAVE_YAML:
        return _yaml.safe_load(text)
    return _mini_load(text)


# --- Reporting -----------------------------------------------------------------
ERRORS = []
WARNINGS = []


def err(msg):
    ERRORS.append(msg)


def warn(msg):
    WARNINGS.append(msg)


# --- Schema validation ---------------------------------------------------------
def validate_against_schema(obj, schema, label):
    if HAVE_JSONSCHEMA:
        validator_cls = _jsonschema.validators.validator_for(schema)
        validator_cls.check_schema(schema)
        v = validator_cls(schema)
        for e in sorted(v.iter_errors(obj), key=lambda e: list(e.path)):
            loc = "/".join(str(p) for p in e.path) or "<root>"
            err("[%s] schema: %s at %s" % (label, e.message, loc))
    else:
        # Fallback: required top-level keys only.
        required = schema.get("required", [])
        for key in required:
            if key not in obj:
                err("[%s] missing required key: %s" % (label, key))
        # Honor the conditional `page` requirement for recipes.
        cond = schema.get("if")
        then = schema.get("then")
        if cond and then and "not" in cond:
            not_required = cond["not"].get("required", [])
            # if NONE of not_required keys present -> then.required applies
            if not any(k in obj for k in not_required):
                for key in then.get("required", []):
                    if key not in obj:
                        err("[%s] missing required key: %s" % (label, key))


# --- Path resolution -----------------------------------------------------------
def resolve_repo_path(rel):
    return os.path.join(REPO_ROOT, rel)


def resolve_docs_path(rel):
    return os.path.join(DOCS_DIR, rel)


# --- Catalog loaders -----------------------------------------------------------
def load_catalog(cat_dir, subdir, index_keys):
    """Load index + per-entry files for one catalog.

    Returns (index, entries) where index is the parsed index.yaml dict and
    entries is {id: parsed_object}.
    """
    index_path = os.path.join(cat_dir, "index.yaml")
    if not os.path.isfile(index_path):
        err("missing index: %s" % index_path)
        return None, {}
    index = load_yaml(index_path)
    entries = {}
    entries_dir = os.path.join(cat_dir, subdir)
    if not os.path.isdir(entries_dir):
        err("missing entries dir: %s" % entries_dir)
        return index, {}
    for fname in sorted(os.listdir(entries_dir)):
        if not fname.endswith(".yaml"):
            continue
        fid = fname[: -len(".yaml")]
        obj = load_yaml(os.path.join(entries_dir, fname))
        entries[fid] = obj
    return index, entries


def check_index_file_correspondence(index, entries, keys, label):
    indexed_ids = []
    for key in keys:
        ids = index.get(key) or []
        for i in ids:
            indexed_ids.append(i)
    # duplicate ids in index
    seen = set()
    for i in indexed_ids:
        if i in seen:
            err("[%s] duplicate id in index: %s" % (label, i))
        seen.add(i)
    indexed_set = set(indexed_ids)
    file_set = set(entries.keys())
    for i in sorted(indexed_set - file_set):
        err("[%s] index references id with no matching file: %s.yaml" % (label, i))
    for f in sorted(file_set - indexed_set):
        err("[%s] orphan entry file not listed in index: %s.yaml" % (label, f))
    return indexed_set


def check_internal_id(entries, label):
    for fid, obj in entries.items():
        if not isinstance(obj, dict):
            err("[%s] %s.yaml did not parse to a mapping" % (label, fid))
            continue
        internal = obj.get("id")
        if internal != fid:
            err(
                "[%s] internal id %r does not match filename %s.yaml"
                % (label, internal, fid)
            )


def check_page(obj, label):
    page = obj.get("page")
    if page is None:
        return
    p = resolve_docs_path(page)
    if not os.path.isfile(p):
        err("[%s] page path does not resolve: docs/%s" % (label, page))


def check_assets_recipe(obj, label):
    for t in obj.get("targets") or []:
        if not isinstance(t, dict):
            continue
        dep = t.get("deploy") or {}
        asset = dep.get("asset") if isinstance(dep, dict) else None
        if asset:
            if not os.path.isfile(resolve_repo_path(asset)):
                err(
                    "[%s] target %s deploy asset missing: %s"
                    % (label, t.get("id"), asset)
                )
        bench = t.get("benchmark")
        if isinstance(bench, dict):
            basset = bench.get("asset")
            if basset and not os.path.isfile(resolve_repo_path(basset)):
                err(
                    "[%s] target %s benchmark asset missing: %s"
                    % (label, t.get("id"), basset)
                )


def check_assets_benchmark(obj, label):
    for a in obj.get("arms") or []:
        if not isinstance(a, dict):
            continue
        for field in ("deploy", "perf"):
            path = a.get(field)
            if path and not os.path.isfile(resolve_repo_path(path)):
                err(
                    "[%s] arm %r %s asset missing: %s"
                    % (label, a.get("name"), field, path)
                )


# --- Main ----------------------------------------------------------------------
def main():
    yaml_mode = "PyYAML" if HAVE_YAML else "builtin-mini-parser"
    schema_mode = "jsonschema" if HAVE_JSONSCHEMA else "required-keys-only"
    print("=" * 72)
    print("Dynamo catalog validator")
    print("  repo root : %s" % REPO_ROOT)
    print("  YAML mode : %s" % yaml_mode)
    print("  schema    : %s" % schema_mode)
    print("=" * 72)

    # Load schemas
    rec_schema = json.load(open(os.path.join(RECIPES_CAT, "schema.json")))
    ben_schema = json.load(open(os.path.join(BENCH_CAT, "schema.json")))

    # Load catalogs
    rec_index, rec_entries = load_catalog(
        RECIPES_CAT, "recipes", ["recipes", "deferred_recipes"]
    )
    ben_index, ben_entries = load_catalog(BENCH_CAT, "benchmarks", ["benchmarks"])

    # 1. index<->file correspondence + 3. dup ids
    rec_ids = check_index_file_correspondence(
        rec_index or {}, rec_entries, ["recipes", "deferred_recipes"], "recipes"
    )
    ben_ids = check_index_file_correspondence(
        ben_index or {}, ben_entries, ["benchmarks"], "benchmarks"
    )

    # 2. internal id matches filename
    check_internal_id(rec_entries, "recipes")
    check_internal_id(ben_entries, "benchmarks")

    # 4. schema validation + 5/6 path checks
    for fid, obj in sorted(rec_entries.items()):
        if not isinstance(obj, dict):
            continue
        label = "recipe:%s" % fid
        validate_against_schema(obj, rec_schema, label)
        check_page(obj, label)
        check_assets_recipe(obj, label)

    for fid, obj in sorted(ben_entries.items()):
        if not isinstance(obj, dict):
            continue
        label = "benchmark:%s" % fid
        validate_against_schema(obj, ben_schema, label)
        check_page(obj, label)
        check_assets_benchmark(obj, label)

    # 7. cross-catalog referential integrity
    valid_recipe_targets = set(rec_entries.keys()) | rec_ids
    valid_benchmark_targets = set(ben_entries.keys()) | ben_ids
    for fid, obj in sorted(rec_entries.items()):
        if not isinstance(obj, dict):
            continue
        for rb in obj.get("related_benchmarks") or []:
            if rb not in valid_benchmark_targets:
                err(
                    "[recipe:%s] related_benchmarks references unknown benchmark id: %s"
                    % (fid, rb)
                )
    for fid, obj in sorted(ben_entries.items()):
        if not isinstance(obj, dict):
            continue
        for rr in obj.get("related_recipes") or []:
            if rr not in valid_recipe_targets:
                err(
                    "[benchmark:%s] related_recipes references unknown recipe id: %s"
                    % (fid, rr)
                )
        pc = obj.get("promotion_candidate")
        if isinstance(pc, dict):
            drid = pc.get("deferred_recipe_id")
            if drid and drid not in valid_recipe_targets:
                err(
                    "[benchmark:%s] promotion_candidate.deferred_recipe_id unknown: %s"
                    % (fid, drid)
                )

    # --- Report ---
    print()
    print(
        "recipes : %d entries (%d active, %d deferred)"
        % (
            len(rec_entries),
            len((rec_index or {}).get("recipes") or []),
            len((rec_index or {}).get("deferred_recipes") or []),
        )
    )
    print("benchmarks: %d entries" % len(ben_entries))
    print()
    if WARNINGS:
        print("WARNINGS (%d):" % len(WARNINGS))
        for w in WARNINGS:
            print("  - %s" % w)
        print()
    if ERRORS:
        print("FAILED with %d error(s):" % len(ERRORS))
        for e in ERRORS:
            print("  - %s" % e)
        return 1
    print("OK: all catalog checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
