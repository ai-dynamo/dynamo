# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Focused tests for the Dynamo Python API docs generator.

The tests are hermetic. They exercise deterministic, static Python API
discovery through griffe and never import Dynamo, boot a runtime, or hit
the network. The generator's outputs (the typed TypeScript data module, the
Python landing page, and one MDX file per curated module) are re-generated
into a scratch workspace so a failing test can never write into the tracked
docs tree.

Test modules import from the canonical implementation modules directly
(``api_discovery`` for the model + griffe pipeline, ``api_rendering`` for
the TypeScript/MDX serialization) rather than through the thin
``gen_python_api`` CLI shell. Griffe discovery is expensive, so the
session-scoped ``all_modules`` fixture runs once per session and the
generator I/O tests monkeypatch ``gen_python_api.discover_all_modules``
to return that same list -- one full static-discovery pass per pytest run.

``conftest.py`` next to this file owns the ``sys.path`` shim that makes the
top-level imports of ``api_discovery`` / ``api_rendering`` /
``gen_python_api`` resolve, so this module keeps every import at the top
and stays clean under the repo's flake8 config.

Invocation from an isolated environment (bypasses the repo's Python
resolution, which is unrelated to this generator)::

    uv run --no-project --python 3.13 --with 'griffe==2.1.0' \\
        --with pytest --with pyyaml \\
        python3 -m pytest docs/fern/scripts/tests -v
"""

from __future__ import annotations

import doctest
import importlib.util
import re
import shutil
from pathlib import Path

import api_discovery
import api_rendering
import gen_python_api
import pytest
import yaml
from griffe import GriffeLoader

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]

REPO_ROOT = Path(__file__).resolve().parents[4]
FERN_ROOT = REPO_ROOT / "docs" / "fern"
COMPONENTS_DIR = FERN_ROOT / "components"
PY_PAGES_DIR = FERN_ROOT / "pages" / "reference" / "api" / "python"
PY_LANDING = PY_PAGES_DIR / "README.mdx"

CURATED_MODULE_NAMES = {
    "dynamo._core",
    "dynamo.runtime",
    "dynamo.llm",
    "dynamo.frontend",
    "dynamo.common",
    "dynamo.health_check",
    "dynamo.logits_processing",
    "dynamo.planner",
    "dynamo.router",
    "dynamo.mocker",
    "dynamo.nixl_connect",
}
SELECTED_DOCSTRING_SUMMARIES = {
    "dynamo.runtime.dynamo_endpoint": (
        "Decorator that can parse a request payload into a Pydantic model "
        "before the endpoint runs."
    ),
    "dynamo.planner.connectors.base.PlannerConnector": (
        "Deployment-control interface the planner uses to inspect and scale "
        "one deployment."
    ),
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def loader() -> GriffeLoader:
    return api_discovery.build_loader()


@pytest.fixture(scope="session")
def all_modules(loader: GriffeLoader) -> list[api_discovery.Module]:
    """Session-cached discovery: one full griffe pass per pytest session."""
    return [
        api_discovery.discover_module(loader, spec) for spec in api_discovery.MODULES
    ]


@pytest.fixture(scope="session")
def modules_by_name(
    all_modules: list[api_discovery.Module],
) -> dict[str, api_discovery.Module]:
    return {mod.name: mod for mod in all_modules}


@pytest.fixture()
def cached_discovery(
    all_modules: list[api_discovery.Module],
    monkeypatch: pytest.MonkeyPatch,
) -> list[api_discovery.Module]:
    """Route the CLI orchestrator's discovery boundary to the session cache.

    Without this shim, every test that calls :func:`gen_python_api.main`
    would trigger a fresh griffe walk of the two Dynamo source roots -- a
    multi-second cost per test. Patching the module-level
    ``discover_all_modules`` binding on ``gen_python_api`` keeps the code
    under test unchanged while sharing the discovery result across the
    whole pytest session."""
    monkeypatch.setattr(gen_python_api, "discover_all_modules", lambda: all_modules)
    return all_modules


@pytest.fixture()
def workspace(tmp_path: Path) -> Path:
    """Clone the docs/fern subtree the generator needs into ``tmp_path``.

    Only the two directories the generator reads/writes are copied so the
    workspace stays small; the Dynamo Python source tree is symlinked (not
    copied) so test I/O stays cheap."""
    ws = tmp_path / "repo"
    dst = ws / "docs" / "fern"
    dst.mkdir(parents=True)
    shutil.copytree(FERN_ROOT / "components", dst / "components")
    shutil.copytree(
        FERN_ROOT / "pages" / "reference" / "api",
        dst / "pages" / "reference" / "api",
    )
    (ws / "lib" / "bindings" / "python").mkdir(parents=True)
    (ws / "lib" / "bindings" / "python" / "src").symlink_to(
        REPO_ROOT / "lib" / "bindings" / "python" / "src"
    )
    (ws / "components").mkdir()
    (ws / "components" / "src").symlink_to(REPO_ROOT / "components" / "src")
    return ws


# ---------------------------------------------------------------------------
# Docstring bodies
# ---------------------------------------------------------------------------


DOCSTRING_FIXTURE = '''\
"""Fixture package exercising every docstring section the renderer maps."""


def documented(request_model: str, response_model: int) -> None:
    """Decorator that parses a request payload before the endpoint runs.

    Parsing applies only when ``request_model`` is a ``BaseModel`` subclass;
    otherwise the wrapper forwards the raw value untouched. Generic types
    such as Dict<K, V> and placeholders like {braces} appear in prose here
    and must not reach the page unescaped, while ``Mapping<K, V>`` sits in a
    reST literal and must survive verbatim.

    Args:
        request_model: Request class used to parse ``str`` payloads.
        response_model: Expected response class. Not enforced today.

    Returns:
        Callable: A decorator that wraps an async generator.

    Raises:
        ValueError: On the first ``__anext__()`` of the returned generator.

    Examples:
        >>> from dynamo.docfixture import documented
        >>> documented("Request", 1)

    Note:
        A clean return is not proof of the outcome.
    """


def summary_only() -> None:
    """Just one line and nothing else."""


def wrapped_summary() -> None:
    """Direct publisher for engines that own their own metrics, such
    as the adapter. The opening sentence wraps across source lines.

    A second paragraph belongs in the body.
    """
'''

FIXTURE_BODY_SENTENCE = "otherwise the wrapper forwards the raw value untouched"


@pytest.fixture()
def docstring_module(tmp_path: Path) -> api_discovery.Module:
    """Discover a synthetic package that carries every docstring section.

    Hermetic on purpose: the rendering contract for docstring bodies must
    hold regardless of which docstrings the Dynamo tree happens to carry
    today, so the fixture owns its own source rather than pinning a real
    symbol that a later branch may reword.
    """
    pkg = tmp_path / "components" / "src" / "dynamo" / "docfixture"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text(DOCSTRING_FIXTURE, encoding="utf-8")
    loader = api_discovery.build_loader(tmp_path)
    return api_discovery.discover_module(
        loader, ("dynamo.docfixture", "docfixture", "Fixture module.")
    )


def _fixture_symbol(module: api_discovery.Module, name: str) -> api_discovery.Symbol:
    symbol = next((s for s in module.symbols if s.name == name), None)
    assert symbol is not None, f"{name} missing from the fixture module"
    return symbol


def _strip_code_fences(text: str) -> str:
    """Drop fenced code blocks so prose-only assertions ignore code.

    Code fences legitimately carry raw ``<`` and ``{``; MDX does not parse
    JSX inside them, so escaping there would surface literal entities.
    """
    out: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if not in_fence:
            out.append(line)
    return "\n".join(out)


def test_symbol_keeps_the_summary_as_the_first_docstring_line(
    docstring_module: api_discovery.Module,
) -> None:
    """``summary`` still means "the one-line synopsis".

    The landing cards and the module index render it verbatim, so widening
    it to the full docstring would blow up those surfaces. Body content is
    additive.
    """
    symbol = _fixture_symbol(docstring_module, "documented")
    assert (
        symbol.summary
        == "Decorator that parses a request payload before the endpoint runs."
    )


def test_symbol_captures_docstring_sections_beyond_the_summary(
    docstring_module: api_discovery.Module,
) -> None:
    """Everything after the first line used to be discarded at discovery."""
    symbol = _fixture_symbol(docstring_module, "documented")
    kinds = [section.kind for section in symbol.docs]
    assert "text" in kinds, "prose body was dropped"
    assert "parameters" in kinds
    assert "returns" in kinds
    assert "raises" in kinds
    assert "examples" in kinds
    assert "admonition" in kinds


def test_captured_text_section_keeps_the_whole_lead_paragraph(
    docstring_module: api_discovery.Module,
) -> None:
    """Discovery keeps the opening paragraph intact.

    ``summary`` truncates at the first newline; dropping the remainder here
    would lose the tail of any summary sentence that wraps.
    """
    symbol = _fixture_symbol(docstring_module, "documented")
    body = next(s.text for s in symbol.docs if s.kind == "text")
    assert body.startswith("Decorator that parses")
    assert FIXTURE_BODY_SENTENCE in body


def test_summary_renders_exactly_once_in_the_accordion(
    docstring_module: api_discovery.Module,
) -> None:
    """The lead paragraph must not be repeated as body prose."""
    text = api_rendering.render_module_page(docstring_module)
    assert text.count("Decorator that parses a request payload") == 1


def test_wrapped_summary_sentence_renders_whole_and_unbroken(
    docstring_module: api_discovery.Module,
) -> None:
    """A summary sentence that wraps must not strand its own tail.

    ``summary`` stops at the first newline, so leading the accordion with it
    would print a clause ending in "such" and then open the body mid-sentence
    with "as the adapter".
    """
    symbol = _fixture_symbol(docstring_module, "wrapped_summary")
    assert symbol.summary.endswith("such"), "fixture no longer wraps its summary"

    text = api_rendering.render_module_page(docstring_module)
    assert (
        "Direct publisher for engines that own their own metrics, such as the "
        "adapter. The opening sentence wraps across source lines." in text
    )
    assert "as the adapter. The opening sentence" not in text.replace(
        "such as the adapter. The opening sentence", ""
    )
    assert "A second paragraph belongs in the body." in text


def test_summary_only_docstring_adds_no_extra_prose(
    docstring_module: api_discovery.Module,
) -> None:
    """A one-line docstring must not grow a duplicate prose block."""
    symbol = _fixture_symbol(docstring_module, "summary_only")
    assert symbol.summary == "Just one line and nothing else."

    text = api_rendering.render_module_page(docstring_module)
    assert text.count("Just one line and nothing else.") == 1


def test_module_page_renders_the_docstring_body_prose(
    docstring_module: api_discovery.Module,
) -> None:
    """The whole point: body text has to reach the rendered page."""
    text = api_rendering.render_module_page(docstring_module)
    assert FIXTURE_BODY_SENTENCE in text


def test_module_page_renders_args_as_param_fields(
    docstring_module: api_discovery.Module,
) -> None:
    """Args map onto the same native ParamField the Kubernetes page uses."""
    text = api_rendering.render_module_page(docstring_module)
    assert '<ParamField path="request_model" type="str">' in text
    assert '<ParamField path="response_model" type="int">' in text
    assert "Request class used to parse ``str`` payloads." in text


def test_module_page_renders_returns_and_raises(
    docstring_module: api_discovery.Module,
) -> None:
    """Returns and Raises land as labelled prose, not silently dropped."""
    text = api_rendering.render_module_page(docstring_module)
    assert "**Returns**" in text
    assert "A decorator that wraps an async generator." in text
    assert "**Raises**" in text
    assert "`ValueError`" in text
    assert "On the first ``__anext__()`` of the returned generator." in text


def test_module_page_renders_examples_as_a_python_fence(
    docstring_module: api_discovery.Module,
) -> None:
    """Doctest blocks get Fern's syntax highlighting and copy button."""
    text = api_rendering.render_module_page(docstring_module)
    assert "**Examples**" in text
    assert ">>> from dynamo.docfixture import documented" in text
    fenced = re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
    assert any(
        ">>> from dynamo.docfixture import documented" in block for block in fenced
    ), "the doctest example must sit inside a python code fence"


def test_module_page_renders_admonitions_as_native_callouts(
    docstring_module: api_discovery.Module,
) -> None:
    """Fern ships Note/Warning callouts; no bespoke component is needed."""
    text = api_rendering.render_module_page(docstring_module)
    assert "<Note>" in text
    assert "A clean return is not proof of the outcome." in text


def test_docstring_body_escapes_jsx_significant_characters(
    docstring_module: api_discovery.Module,
) -> None:
    """Docstring prose is JSX source once it lands in MDX.

    A bare ``<K,`` reads as an unclosed JSX tag and fails the Fern build,
    so angle brackets and braces outside code spans must be entities.
    """
    prose = _strip_code_fences(api_rendering.render_module_page(docstring_module))
    assert "Dict&lt;K, V&gt;" in prose
    assert "&#123;braces&#125;" in prose
    assert "Dict<K, V>" not in prose
    assert "{braces}" not in prose


def test_rest_literal_spans_are_not_entity_escaped(
    docstring_module: api_discovery.Module,
) -> None:
    """Double-backtick reST literals are Markdown code spans.

    MDX does not parse JSX inside a code span and does not decode entities
    there, so escaping one surfaces a literal ``&lt;`` to the reader.
    """
    text = api_rendering.render_module_page(docstring_module)
    assert "``Mapping<K, V>``" in text
    assert "Mapping&lt;K, V&gt;" not in text


# ---------------------------------------------------------------------------
# Authored docstrings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("qualname", "expected"),
    SELECTED_DOCSTRING_SUMMARIES.items(),
)
def test_selected_complex_apis_have_high_value_summaries(
    all_modules: list[api_discovery.Module],
    qualname: str,
    expected: str,
) -> None:
    symbols = {
        symbol.qualname: symbol for module in all_modules for symbol in module.symbols
    }

    assert qualname in symbols, f"{qualname} is no longer a discovered public symbol"
    assert symbols[qualname].summary == expected


def test_storage_doctests_execute() -> None:
    """Run the storage examples instead of string-matching the source.

    A documented output that does not match what the code returns is a lie the
    docs site publishes, and nothing else in this repo runs a doctest. The
    module imports only asyncio, typing, and fsspec -- no compiled ``_core`` --
    so it loads standalone here.
    """
    path = REPO_ROOT / "components" / "src" / "dynamo" / "common" / "storage.py"
    spec = importlib.util.spec_from_file_location("dynamo_storage_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    results = doctest.testmod(module, verbose=False)

    assert results.attempted > 0, "storage.py examples are no longer being collected"
    assert results.failed == 0


# ---------------------------------------------------------------------------
# Curated module list + discovery
# ---------------------------------------------------------------------------


def test_curated_modules_match_the_agreed_eleven() -> None:
    """The generator's MODULES tuple must cover exactly the eleven Python
    packages the branch's plan pins as public surface. Adding or dropping a
    module is a scope decision and must be reviewed as such."""
    names = {spec[0] for spec in api_discovery.MODULES}
    assert (
        names == CURATED_MODULE_NAMES
    ), f"missing: {CURATED_MODULE_NAMES - names} extra: {names - CURATED_MODULE_NAMES}"


def test_every_curated_module_discovers_at_least_one_symbol(
    all_modules: list[api_discovery.Module],
) -> None:
    """Griffe already reconciled the eleven curated names with the current
    source tree, so every module in the list must produce a non-empty page;
    an empty page is a signal that either the module has disappeared or the
    discovery filter is wrong."""
    empty = [m.name for m in all_modules if not m.symbols]
    assert not empty, f"empty modules with no discovered public surface: {empty}"


def test_discovered_symbols_are_deterministically_ordered(
    all_modules: list[api_discovery.Module],
) -> None:
    """Symbols on every module page are grouped Classes-then-Functions,
    alphabetical within each group, so page diffs stay reviewable.

    ``qualname`` is part of the assertion because name alone is not unique:
    two submodules can export the same symbol name, and only the qualname
    tiebreaker keeps their relative order stable across runs."""
    for module in all_modules:
        order = [(s.kind, s.name, s.qualname) for s in module.symbols]
        assert order == sorted(order), f"{module.name}: symbols not sorted"


def test_dynamo_runtime_re_exports_from_core_are_present(
    modules_by_name: dict[str, api_discovery.Module],
) -> None:
    """``dynamo.runtime`` re-exports ``Client`` / ``DistributedRuntime`` /
    ``Endpoint`` / ``Context`` / ``PyAsyncRequestStream`` from ``dynamo._core``.
    The runtime page must surface those aliases (with their canonical
    ``dynamo._core.<Name>`` qualname preserved for the source link)."""
    runtime = modules_by_name["dynamo.runtime"]
    by_name = {s.name: s for s in runtime.symbols}
    for expected in (
        "Client",
        "Context",
        "DistributedRuntime",
        "Endpoint",
        "PyAsyncRequestStream",
    ):
        assert expected in by_name, f"{expected} missing from dynamo.runtime page"
        assert by_name[expected].qualname.startswith(
            "dynamo._core."
        ), f"{expected}.qualname should preserve the canonical dynamo._core.* path"


def test_class_symbols_carry_their_public_methods(
    modules_by_name: dict[str, api_discovery.Module],
) -> None:
    """The compact page layout expands into per-symbol details that must
    include public methods for classes. ``DistributedRuntime`` (on
    ``dynamo._core``) has documented ``endpoint`` / ``shutdown`` methods,
    so the discovery pipeline must have populated ``Symbol.methods``."""
    core = modules_by_name["dynamo._core"]
    by_name = {s.name: s for s in core.symbols}
    dr = by_name.get("DistributedRuntime")
    assert dr is not None, "DistributedRuntime missing from dynamo._core"
    method_names = {m.name for m in dr.methods}
    assert {
        "endpoint",
        "shutdown",
    } <= method_names, f"DistributedRuntime methods should include endpoint + shutdown; got {method_names}"
    ep = next((m for m in dr.methods if m.name == "endpoint"), None)
    assert ep is not None and ep.signature.startswith(
        "endpoint("
    ), "endpoint method should carry a signature string"


def test_discover_all_modules_returns_the_full_curated_list(
    all_modules: list[api_discovery.Module],
) -> None:
    """The CLI orchestrator calls :func:`api_discovery.discover_all_modules`
    once per run; the helper must return one module per :data:`MODULES` spec
    so tests can monkeypatch it to a session cache without under-serving
    the generator."""
    assert [m.name for m in all_modules] == [spec[0] for spec in api_discovery.MODULES]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


_MODULE_NAMES = [spec[0] for spec in api_discovery.MODULES]


@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_render_module_page_is_native_mdx(
    modules_by_name: dict[str, api_discovery.Module],
    module_name: str,
) -> None:
    """Each per-module page carries SPDX frontmatter and builds its body from
    Fern's own components, so the symbols stay searchable and Fern derives the
    Markdown twin itself rather than a hand-maintained fallback."""
    module = modules_by_name[module_name]
    text = api_rendering.render_module_page(module)
    assert text.startswith("---\n# SPDX-FileCopyrightText:")
    assert "SPDX-License-Identifier: Apache-2.0" in text
    assert f"title: {module.name}" in text
    assert "ApiSurfaceBrowser" not in text
    assert "<llms-only>" not in text
    body_lines = text.split("---\n", 2)[-1].splitlines()
    assert not any(
        ln.strip().startswith("# ") for ln in body_lines
    ), "body must not contain an H1 (Fern renders the title from the nav)"


@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_render_module_page_wraps_every_symbol_in_an_accordion(
    modules_by_name: dict[str, api_discovery.Module],
    module_name: str,
) -> None:
    """Accordion content stays indexed for search while collapsed."""
    module = modules_by_name[module_name]
    text = api_rendering.render_module_page(module)
    assert text.count("<Accordion ") == len(module.symbols)


@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_render_module_page_gives_each_accordion_an_explicit_id(
    modules_by_name: dict[str, api_discovery.Module],
    module_name: str,
) -> None:
    """Accordion takes the deep-link id natively.

    Fern derives an id from the title when none is given, which collides for
    same-named symbols; passing a qualname-derived id keeps them distinct. An
    empty ``<a id>`` before the accordion would also do it, but renders as a
    link with no text -- a real accessibility violation.
    """
    module = modules_by_name[module_name]
    text = api_rendering.render_module_page(module)
    assert "<a id=" not in text
    for symbol in module.symbols:
        anchor = api_rendering.symbol_anchor(symbol)
        assert f'<Accordion id="{anchor}"' in text


@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_render_module_page_emits_copyable_import_fences(
    modules_by_name: dict[str, api_discovery.Module],
    module_name: str,
) -> None:
    """A Python code fence gives Fern's own copy button and syntax
    highlighting, replacing the bespoke copy-to-clipboard affordance."""
    module = modules_by_name[module_name]
    text = api_rendering.render_module_page(module)
    for symbol in module.symbols:
        assert api_rendering.import_statement(symbol) in text
    assert "```python" in text


@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_module_page_is_deterministic(
    modules_by_name: dict[str, api_discovery.Module],
    module_name: str,
) -> None:
    module = modules_by_name[module_name]
    a = api_rendering.render_module_page(module)
    b = api_rendering.render_module_page(module)
    assert a == b


@pytest.mark.parametrize("module_name", _MODULE_NAMES)
def test_module_page_has_no_duplicate_html_anchors(
    modules_by_name: dict[str, api_discovery.Module],
    module_name: str,
) -> None:
    """The compact layout emits one ``<details id="dynref-asb-<slug>">``
    per symbol; duplicate anchors would collide when a user deep-links
    into a symbol, so every id on a page must be unique."""
    module = modules_by_name[module_name]
    text = api_rendering.render_module_page(module)
    ids = re.findall(r'id="([^"]+)"', text)
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"{module.name}: duplicate anchors {duplicates}"


def test_render_python_landing_lists_every_curated_module(
    all_modules: list[api_discovery.Module],
) -> None:
    """The Python landing page indexes every curated module so users
    can browse the full surface without opening 11 sidebar entries."""
    text = api_rendering.render_landing_page(all_modules)
    assert text.startswith("---\n# SPDX-FileCopyrightText:")
    assert "title: Python API" in text
    for module_name in CURATED_MODULE_NAMES:
        assert module_name in text, f"landing missing {module_name}"


def test_render_python_landing_uses_native_cards(
    all_modules: list[api_discovery.Module],
) -> None:
    """Module entry points render as native cards linking to Fern routes,
    not a hand-styled grid emitting ``.mdx`` hrefs."""
    text = api_rendering.render_landing_page(all_modules)
    assert "ApiPythonIndex" not in text
    assert "<llms-only>" not in text
    assert "<CardGroup" in text
    for module in all_modules:
        assert f'href="python/{module.slug}"' in text
    assert '.mdx"' not in text


# ---------------------------------------------------------------------------
# Generator I/O + --check
# ---------------------------------------------------------------------------


def test_generator_writes_landing_data_and_every_module_page(
    workspace: Path,
    cached_discovery: list[api_discovery.Module],
) -> None:
    fern = workspace / "docs" / "fern"
    rc = gen_python_api.main(["--fern-root", str(fern)])
    assert rc == 0
    assert (fern / "pages" / "reference" / "api" / "python" / "README.mdx").is_file()
    for spec in api_discovery.MODULES:
        page = fern / "pages" / "reference" / "api" / "python" / f"{spec[1]}.mdx"
        assert page.is_file(), f"module page not written: {page}"


def test_check_mode_returns_zero_on_fresh_outputs(
    workspace: Path,
    cached_discovery: list[api_discovery.Module],
) -> None:
    fern = workspace / "docs" / "fern"
    assert gen_python_api.main(["--fern-root", str(fern)]) == 0
    assert gen_python_api.main(["--fern-root", str(fern), "--check"]) == 0


def test_check_mode_flags_module_page_drift(
    workspace: Path,
    cached_discovery: list[api_discovery.Module],
) -> None:
    fern = workspace / "docs" / "fern"
    assert gen_python_api.main(["--fern-root", str(fern)]) == 0
    page = fern / "pages" / "reference" / "api" / "python" / "runtime.mdx"
    page.write_text(
        page.read_text(encoding="utf-8") + "\n<!-- drift -->\n", encoding="utf-8"
    )
    assert gen_python_api.main(["--fern-root", str(fern), "--check"]) == 1


def test_check_mode_flags_landing_page_drift(
    workspace: Path,
    cached_discovery: list[api_discovery.Module],
) -> None:
    fern = workspace / "docs" / "fern"
    assert gen_python_api.main(["--fern-root", str(fern)]) == 0
    landing = fern / "pages" / "reference" / "api" / "python" / "README.mdx"
    landing.write_text(
        landing.read_text(encoding="utf-8") + "\n<!-- drift -->\n", encoding="utf-8"
    )
    assert gen_python_api.main(["--fern-root", str(fern), "--check"]) == 1


# ---------------------------------------------------------------------------
# Cross-checks against the shipped tree
# ---------------------------------------------------------------------------


def test_no_hardcoded_dev_paths_in_any_generated_output(
    cached_discovery: list[api_discovery.Module],
) -> None:
    """No output the generator produces may bake in a ``/dynamo/dev`` URL:
    Fern serves each doc version under its own prefix, and a hardcoded
    ``/dev`` path in a versioned snapshot links back to the dev site.

    The pages are publish-time artifacts (not committed), so the check runs
    against a fresh render rather than files on disk."""
    rendered = {"README.mdx": api_rendering.render_landing_page(cached_discovery)}
    for module in cached_discovery:
        rendered[f"{module.slug}.mdx"] = api_rendering.render_module_page(module)
    for name, text in rendered.items():
        assert "/dynamo/dev" not in text, f"{name}: hardcoded '/dynamo/dev' path found"


def test_python_landing_links_to_fern_routes_not_mdx_files(
    cached_discovery: list[api_discovery.Module],
) -> None:
    """Card hrefs must be site routes; Fern only rewrites relative links in
    Markdown, not ``.mdx`` paths handed to a component."""
    source = api_rendering.render_landing_page(cached_discovery)
    hrefs = re.findall(r'href="([^"]+)"', source)
    assert hrefs
    for href in hrefs:
        assert href.startswith("python/"), f"unexpected landing href {href!r}"
        assert not href.endswith(".mdx")


def test_api_landing_links_resolve_through_the_file_graph() -> None:
    """Card hrefs use source paths, not site routes.

    Fern resolves a relative ``.mdx`` path through the file graph and its
    broken-link checker verifies it, so the links survive a slug rename. Site
    routes look right and silently rot -- the React hero these cards replaced
    pointed at ``api/python``, which never resolved.
    """
    source = (FERN_ROOT / "pages" / "reference" / "api" / "README.mdx").read_text(
        encoding="utf-8"
    )
    assert 'href="python/README.mdx"' in source
    assert 'href="rust/README.mdx"' in source
    assert 'href="../kubernetes-api/full-api-reference.mdx"' in source


def test_generated_pages_do_not_leak_maintainer_instructions(
    cached_discovery: list[api_discovery.Module],
) -> None:
    """Generated pages are reader-facing; the regeneration workflow belongs
    in the API overview, not in the middle of a reference page."""
    for module in cached_discovery:
        text = api_rendering.render_module_page(module)
        body = text.split("---\n", 2)[-1].replace(
            api_rendering.MDX_GENERATED_MARKER, ""
        )
        assert "re-run the generator" not in body


def test_index_yml_registers_every_generated_module_page() -> None:
    """Every generated per-module MDX page under pages/reference/api/python/
    must be referenced from docs/fern/index.yml; an unregistered page is
    unreachable and fails Fern's broken-link check."""
    index_text = (FERN_ROOT / "index.yml").read_text(encoding="utf-8")
    for spec in api_discovery.MODULES:
        expected = f"pages/reference/api/python/{spec[1]}.mdx"
        assert (
            expected in index_text
        ), f"index.yml does not reference generated page {expected}"
    assert "pages/reference/api/README.mdx" in index_text
    assert "pages/reference/api/python/README.mdx" in index_text


def test_index_yml_python_registrations_do_not_shadow_each_other() -> None:
    """Only real generated paths may appear under the Python API
    navigation; a stale registration would 404 during Fern build."""
    doc = yaml.safe_load((FERN_ROOT / "index.yml").read_text(encoding="utf-8"))
    registered = _collect_python_pages(doc)
    expected = {
        f"pages/reference/api/python/{spec[1]}.mdx" for spec in api_discovery.MODULES
    }
    expected.add("pages/reference/api/python/README.mdx")
    unexpected = registered - expected
    assert not unexpected, f"index.yml references stale python pages: {unexpected}"


def test_python_landing_owns_the_python_section_slug() -> None:
    """The landing page must be the Python section path.

    The section carries the ``api/python`` slug and the README landing; every
    generated module page sits underneath as a visible sibling — the earlier
    hidden-children pattern hid the whole surface from the sidebar.

    The slug is nested even though the section is not: Python API sits at the
    top level of the tab as a peer of Rust and Kubernetes, and pinning the slug
    is what keeps every published ``/reference/api/python/...`` URL from moving
    with it. See ``test_api_slugs_stay_nested_under_api``.
    """
    doc = yaml.safe_load((FERN_ROOT / "index.yml").read_text(encoding="utf-8"))
    landing = _find_node_by_path(doc, "pages/reference/api/python/README.mdx")
    assert landing.get("section") == "Python API"
    assert landing.get("slug") == "api/python"
    module_pages = landing.get("contents")
    assert isinstance(module_pages, list)
    assert not any(
        page.get("hidden") is True for page in module_pages
    ), "Python module pages must remain visible sidebar entries."


def _find_node_by_path(node: object, target: str) -> dict[str, object]:
    """Return the unique navigation node registered for ``target``."""
    matches: list[dict[str, object]] = []
    if isinstance(node, list):
        for item in node:
            found = _find_node_by_path(item, target)
            if found:
                matches.append(found)
    elif isinstance(node, dict):
        if node.get("path") == target:
            matches.append(node)
        for value in node.values():
            found = _find_node_by_path(value, target)
            if found:
                matches.append(found)
    assert len(matches) <= 1, f"duplicate navigation nodes for {target}"
    return matches[0] if matches else {}


def _collect_python_pages(node: object) -> set[str]:
    """Walk the parsed index.yml and return every ``path:`` value below
    the Python API navigation subtree."""
    found: set[str] = set()
    if isinstance(node, list):
        for item in node:
            found |= _collect_python_pages(item)
    elif isinstance(node, dict):
        path = node.get("path")
        if isinstance(path, str) and path.startswith("pages/reference/api/python/"):
            found.add(path)
        for value in node.values():
            found |= _collect_python_pages(value)
    return found


def test_method_anchors_are_unique_within_a_page(
    all_modules: list[api_discovery.Module],
) -> None:
    """Every anchor a module page emits must be unique on that page.

    Method names repeat across classes -- ``initialize`` sits on both
    ``Connection`` and ``Connector`` in ``dynamo.nixl_connect`` -- so an
    anchor derived from the method name alone would collide and Fern would
    resolve a deep link to whichever symbol rendered first.
    """
    for module in all_modules:
        anchors = [api_rendering.symbol_anchor(sym) for sym in module.symbols]
        anchors += [
            api_rendering.method_anchor(sym, method)
            for sym in module.symbols
            for method in sym.methods
        ]
        duplicates = {a for a in anchors if anchors.count(a) > 1}
        assert not duplicates, f"{module.name}: duplicate anchors {sorted(duplicates)}"


def test_every_public_method_renders_through_one_shape(
    modules_by_name: dict[str, api_discovery.Module],
) -> None:
    """Rendering must not depend on whether a docstring documents params.

    Promoting only the documented methods would let docstring coverage decide
    which methods look important, so a class whose trivial method happens to
    carry ``Args:`` would outrank the load-bearing one beside it.
    """
    module = modules_by_name["dynamo.nixl_connect"]
    page = api_rendering.render_module_page(module)
    for symbol in module.symbols:
        for method in symbol.methods:
            anchor = api_rendering.method_anchor(symbol, method)
            assert (
                f'<h4 id="{anchor}">' in page
            ), f"{symbol.name}.{method.name} rendered without its anchored heading"
    assert "- `" not in page.split("**Public methods**", 1)[-1].split("</Accordion>")[0]


def test_method_docstring_sections_reach_the_page(
    modules_by_name: dict[str, api_discovery.Module],
) -> None:
    """A method's parameters, returns, and raises must survive to the MDX.

    ``Method`` previously carried only a summary, so everything below a
    docstring's first line was dropped. ``Connector.begin_read`` documents
    all three section kinds, which makes it the regression canary.
    """
    module = modules_by_name["dynamo.nixl_connect"]
    connector = next(s for s in module.symbols if s.name == "Connector")
    begin_read = next(m for m in connector.methods if m.name == "begin_read")
    kinds = {section.kind for section in begin_read.docs}
    assert {"parameters", "returns", "raises"} <= kinds, f"got {sorted(kinds)}"

    page = api_rendering.render_module_page(module)
    body = page.split(
        f'<h4 id="{api_rendering.method_anchor(connector, begin_read)}">'
    )[1].split("<h4 ")[0]
    assert '<ParamField path="remote_metadata"' in body
    assert "**Returns**" in body
    assert "**Raises**" in body
