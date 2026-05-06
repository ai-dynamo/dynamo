#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""License-policy validator.

Reads one or more `*-deps.csv` files emitted by the per-ecosystem NOTICES
generators and validates each row against `licenses.toml`. Exits non-zero
on any violation, with a per-violation message to stderr.

Wired into the `licenses` Dockerfile stage so non-zero exit fails the
buildx invocation, which fails the workflow. Same enforcement on PR,
post-merge, RC, and release — no parallel CI infrastructure to maintain.

SPDX expression handling:
  "MIT"                              -> allowed iff MIT is allowed
  "MIT OR Apache-2.0"                -> allowed iff EITHER token is allowed
  "MIT AND Apache-2.0"               -> allowed iff BOTH tokens are allowed
  "Apache-2.0 WITH LLVM-exception"   -> allowed iff Apache-2.0 is allowed
                                        (the exception modifier is a clarifier,
                                        not a separate license)
  "(MIT OR Apache-2.0) AND BSD-3-Clause"
                                     -> grouped; both halves must be allowed
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

logger = logging.getLogger("compliance.policy.validate")


@dataclass(frozen=True)
class Policy:
    allow: frozenset[str]
    deny: frozenset[str]
    unknown_action: str  # "allow" | "deny" | "warn"
    copyleft_action: str  # "allow" | "deny" | "warn"
    exceptions: tuple[dict, ...]


def load_policy(policy_path: Path) -> Policy:
    raw = tomllib.loads(policy_path.read_text(encoding="utf-8"))
    section = raw.get("licenses", {})
    return Policy(
        allow=frozenset(section.get("allow", [])),
        deny=frozenset(section.get("deny", [])),
        unknown_action=section.get("unknown", "deny"),
        copyleft_action=section.get("copyleft", "deny"),
        exceptions=tuple(raw.get("exceptions", [])),
    )


# ---- SPDX expression mini-parser -------------------------------------------------
#
# Recursive-descent parser for the subset of SPDX license-expression syntax
# we see in real CycloneDX / pip-licenses output. Handles AND, OR, WITH, and
# parenthesized groups. Returns a tree (a nested tuple).

_TOKEN_RE = re.compile(
    r"\s*(?:(\()|(\))|(AND|OR|WITH)\b|([A-Za-z0-9.+\-_]+))"
)


def _tokenize(expr: str) -> list[tuple[str, str]]:
    pos = 0
    tokens: list[tuple[str, str]] = []
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            if expr[pos].isspace():
                pos += 1
                continue
            raise ValueError(f"unexpected character at offset {pos} in {expr!r}")
        if m.group(1):
            tokens.append(("LPAREN", "("))
        elif m.group(2):
            tokens.append(("RPAREN", ")"))
        elif m.group(3):
            tokens.append(("OP", m.group(3)))
        elif m.group(4):
            tokens.append(("LIC", m.group(4)))
        pos = m.end()
    return tokens


class _Parser:
    """Pratt-style parser for SPDX expressions.

    Grammar (precedence: OR < AND < WITH; left-associative):
        or_expr  := and_expr ('OR' and_expr)*
        and_expr := with_expr ('AND' with_expr)*
        with_expr := atom ('WITH' LICENSE)?
        atom := LICENSE | '(' or_expr ')'

    Tree shape:
        ('lic', name)
        ('with', name, exception_name)
        ('and', left, right)
        ('or', left, right)
    """

    def __init__(self, tokens: list[tuple[str, str]]):
        self.tokens = tokens
        self.pos = 0

    def _peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def _consume(self) -> tuple[str, str]:
        tok = self.tokens[self.pos]
        self.pos += 1
        return tok

    def parse(self):
        result = self._or()
        if self.pos != len(self.tokens):
            raise ValueError(f"trailing tokens after expression: {self.tokens[self.pos :]}")
        return result

    def _or(self):
        left = self._and()
        while (tok := self._peek()) and tok == ("OP", "OR"):
            self._consume()
            right = self._and()
            left = ("or", left, right)
        return left

    def _and(self):
        left = self._with()
        while (tok := self._peek()) and tok == ("OP", "AND"):
            self._consume()
            right = self._with()
            left = ("and", left, right)
        return left

    def _with(self):
        atom = self._atom()
        if (tok := self._peek()) and tok == ("OP", "WITH"):
            self._consume()
            exc = self._consume()
            if exc[0] != "LIC":
                raise ValueError(f"expected exception identifier after WITH, got {exc}")
            return ("with", atom, exc[1])
        return atom

    def _atom(self):
        tok = self._consume()
        if tok[0] == "LIC":
            return ("lic", tok[1])
        if tok[0] == "LPAREN":
            inner = self._or()
            close = self._consume()
            if close != ("RPAREN", ")"):
                raise ValueError(f"expected ), got {close}")
            return inner
        raise ValueError(f"unexpected token {tok}")


def parse_spdx(expr: str):
    return _Parser(_tokenize(expr)).parse()


# ---- Evaluation against policy ---------------------------------------------------


def _node_is_allowed(node, allow_set: frozenset[str], deny_set: frozenset[str]) -> bool:
    """Return True iff the SPDX expression tree is satisfiable under the policy.

    AND nodes need both sides allowed. OR nodes pick the more-permissive branch.
    WITH nodes are allowed iff the base license is allowed (the exception
    modifier doesn't change the underlying obligation in our evaluation).

    Deny is a hard veto: a single denied token in an AND clause blocks the
    expression even if the others are fine.
    """
    kind = node[0]
    if kind == "lic":
        name = node[1]
        if name in deny_set:
            return False
        return name in allow_set
    if kind == "with":
        return _node_is_allowed(node[1], allow_set, deny_set)
    if kind == "and":
        return _node_is_allowed(node[1], allow_set, deny_set) and _node_is_allowed(
            node[2], allow_set, deny_set
        )
    if kind == "or":
        return _node_is_allowed(node[1], allow_set, deny_set) or _node_is_allowed(
            node[2], allow_set, deny_set
        )
    raise ValueError(f"unknown node kind: {node}")


def _exception_allow_set(
    policy: Policy, ecosystem: str, name: str, version: str
) -> frozenset[str]:
    """Return the set of SPDX IDs additionally allowed for this specific package.

    Match: ecosystem + name (required); version (optional — omit to match all).
    """
    extra: set[str] = set()
    for exc in policy.exceptions:
        if exc.get("type") != ecosystem:
            continue
        if exc.get("name") != name:
            continue
        exc_version = exc.get("version")
        if exc_version is not None and exc_version != version:
            continue
        extra.update(exc.get("allow", []))
    return frozenset(extra)


# ---- Per-row validation ----------------------------------------------------------


@dataclass(frozen=True)
class Violation:
    ecosystem: str
    name: str
    version: str
    spdx: str
    reason: str  # human-readable

    def __str__(self) -> str:
        return f"  [{self.ecosystem}] {self.name}@{self.version}: {self.reason} (SPDX: {self.spdx!r})"


def validate_row(
    policy: Policy, ecosystem: str, name: str, version: str, spdx: str
) -> Violation | None:
    if not spdx or spdx == "UNKNOWN":
        if policy.unknown_action == "deny":
            return Violation(
                ecosystem, name, version, spdx,
                "license is UNKNOWN and policy.licenses.unknown = 'deny'",
            )
        return None

    extra_allow = _exception_allow_set(policy, ecosystem, name, version)
    effective_allow = policy.allow | extra_allow

    try:
        tree = parse_spdx(spdx)
    except ValueError as exc:
        return Violation(
            ecosystem, name, version, spdx,
            f"could not parse SPDX expression: {exc}",
        )

    if _node_is_allowed(tree, effective_allow, policy.deny):
        return None

    # Diagnose the failure: list every leaf and its allow/deny status.
    leaves = _collect_leaves(tree)
    denied = [n for n in leaves if n in policy.deny]
    not_allowed = [n for n in leaves if n not in effective_allow and n not in policy.deny]
    if denied:
        reason = f"contains denied license(s): {sorted(set(denied))}"
    elif not_allowed:
        reason = (
            f"no allowed license in expression; not in allow list: "
            f"{sorted(set(not_allowed))}"
        )
    else:
        reason = "expression evaluates to denied (likely AND mixing allowed + denied)"
    return Violation(ecosystem, name, version, spdx, reason)


def _collect_leaves(node) -> list[str]:
    kind = node[0]
    if kind == "lic":
        return [node[1]]
    if kind == "with":
        return _collect_leaves(node[1])
    if kind in ("and", "or"):
        return _collect_leaves(node[1]) + _collect_leaves(node[2])
    return []


# ---- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="compliance.policy.validate",
        description="Validate per-ecosystem deps CSVs against licenses.toml",
    )
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path(__file__).resolve().parent / "licenses.toml",
        help="Path to licenses.toml (default: %(default)s)",
    )
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Path to a *-deps.csv (repeatable for multiple ecosystems)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    policy = load_policy(args.policy)
    logger.info(
        "Loaded policy from %s: %d allowed, %d denied, %d exceptions, "
        "unknown=%s copyleft=%s",
        args.policy,
        len(policy.allow),
        len(policy.deny),
        len(policy.exceptions),
        policy.unknown_action,
        policy.copyleft_action,
    )

    violations: list[Violation] = []
    rows_checked = 0
    for csv_path in args.input:
        if not csv_path.is_file():
            logger.error("Input CSV not found: %s", csv_path)
            return 2
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows_checked += 1
                v = validate_row(
                    policy,
                    row["ecosystem"],
                    row["name"],
                    row["version"],
                    row.get("spdx", "") or "UNKNOWN",
                )
                if v is not None:
                    violations.append(v)
        logger.debug("Checked %s", csv_path)

    if violations:
        print(
            f"License validation failed: {len(violations)} violation(s) "
            f"across {rows_checked} packages",
            file=sys.stderr,
        )
        for v in violations:
            print(str(v), file=sys.stderr)
        return 1

    print(
        f"License validation passed: {rows_checked} packages, no violations",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
