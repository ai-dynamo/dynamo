# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The harness's import rules, enforced by walking the AST rather than by review.

Three rules, each with a reason that costs something real if broken:

1. **Nothing imports ``dynamo.*``.** The harness version would weld to the
   runtime version, and pointing one suite at an older Dynamo release becomes
   impossible.
2. **Only the pytest plugin imports ``pytest``.** Otherwise the manifest and
   evidence layers cannot be used outside a test run — by a CLI, by a planner,
   or by a suite that only wants to read a bundle.
3. **Tier 0 is stdlib only.** ``plan`` and ``judge`` have to run on a laptop, in
   CI's bare-checkout tier, and inside a runner with no cluster credentials at
   all.

Rule 3 is scoped to the modules named here rather than to the package, so that
later tiers can take dependencies behind extras without weakening the core.
"""

import ast
import pathlib

import pytest

PACKAGE = pathlib.Path(__file__).resolve().parents[1] / "dynamo_test"

# Modules that must import nothing outside the standard library.
TIER_0 = {
    "__init__.py",
    "argv.py",
    "facts.py",
    "roles.py",
    "verbs.py",
    "catalog.py",
    "dialect.py",
    "evidence.py",
    "site.py",
    "stack.py",
    "bringup.py",
    "sut.py",
    "local.py",
}

# The one module permitted to import pytest, once it exists.
PYTEST_OWNER = "pytest_plugin.py"

STDLIB_OK = {
    "__future__",
    "abc",
    "argparse",
    "ast",
    "base64",
    "collections",
    "contextlib",
    "copy",
    "dataclasses",
    "datetime",
    "difflib",
    "enum",
    "functools",
    "hashlib",
    "io",
    "ipaddress",
    "itertools",
    "json",
    "logging",
    "math",
    "os",
    "pathlib",
    "re",
    "shlex",
    "shutil",
    "signal",
    "socket",
    "statistics",
    "string",
    "subprocess",
    "sys",
    "tempfile",
    "textwrap",
    "time",
    "types",
    "typing",
    "urllib",
    "uuid",
    "warnings",
    "weakref",
}


def _modules():
    return sorted(PACKAGE.rglob("*.py"))


def _imported_roots(path):
    """Root package name of every import in ``path``, with its line number."""
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import, within the package
                continue
            if node.module:
                yield node.module.split(".")[0], node.lineno


def test_the_package_has_modules_to_check():
    assert _modules(), "no modules found; the path in this test is wrong"


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_no_module_imports_the_dynamo_runtime(path):
    """Importing ``dynamo.*`` would pin the harness to one runtime version."""
    offenders = [
        f"{path.name}:{line} imports {root}"
        for root, line in _imported_roots(path)
        if root == "dynamo"
    ]
    assert offenders == []


@pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
def test_only_the_pytest_plugin_imports_pytest(path):
    if path.name == PYTEST_OWNER:
        return
    offenders = [
        f"{path.name}:{line}"
        for root, line in _imported_roots(path)
        if root == "pytest" or root.startswith("_pytest")
    ]
    assert offenders == [], (
        f"{offenders} imports pytest; that makes this module unusable outside a "
        f"test run. Move the pytest-facing part into {PYTEST_OWNER}."
    )


@pytest.mark.parametrize(
    "path", [p for p in _modules() if p.name in TIER_0], ids=lambda p: p.name
)
def test_tier_zero_is_standard_library_only(path):
    offenders = [
        f"{path.name}:{line} imports {root}"
        for root, line in _imported_roots(path)
        if root not in STDLIB_OK
    ]
    assert offenders == [], (
        f"{offenders}. Tier 0 must import stdlib only so that planning and "
        "judging run with no cluster, no engine, and no credentials."
    )


def test_tier_zero_modules_all_exist():
    """Guards against the rule above silently applying to nothing."""
    present = {p.name for p in _modules()}
    assert TIER_0 <= present, f"missing Tier 0 modules: {TIER_0 - present}"


PROVIDER_METHODS = {
    "address",
    "start",
    "stop",
    "restart",
    "replicas",
    "restart_count",
    "logs",
    "request",
    "all_logs",
    "shutdown",
    "collect_into",
}


def test_the_provider_protocol_declares_everything_a_provider_must_have():
    """`all_logs`, `shutdown` and `collect_into` existed in the only
    implementation and were load-bearing, but were never declared.

    An undeclared method that every caller depends on is a contract in practice
    and a surprise in principle: a second provider could omit one and only fail
    at teardown, which is exactly when the evidence it was meant to collect is
    the point.
    """
    import ast

    tree = ast.parse((PACKAGE / "sut.py").read_text())
    protocol = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "Provider"
    )
    declared = {n.name for n in protocol.body if isinstance(n, ast.FunctionDef)}
    assert declared == PROVIDER_METHODS
    assert len(declared) == 11


def test_the_local_provider_satisfies_the_whole_protocol():
    from dynamo_test.providers import LocalProvider

    missing = PROVIDER_METHODS - set(dir(LocalProvider))
    assert missing == set(), f"LocalProvider is missing {missing}"
