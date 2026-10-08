#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generate the Recipes tab's provider grouping from the recipe catalog.

The recipe catalog (``pages/recipes/_catalog/``) is the single source of truth
for which model recipes exist and who makes each model. This script derives
every surface that groups recipes by provider, so a recipe author edits only
the catalog and the provider order cannot drift between surfaces:

  * ``index.yml``: the Model Recipes sidebar, one section per provider
    (``recipe-nav`` span).
  * ``model-recipes/overview.mdx``: the provider filter's radio inputs and
    chips (``provider-inputs`` and ``provider-chips`` spans) and the catalog
    counts (``family-count`` and ``config-count`` spans).
  * ``components/RecipeStyles.tsx``: the per-provider chip-highlight and
    card-filter CSS rules (``provider-chip-rules`` and
    ``provider-filter-rules`` spans).
  * ``_catalog/index.yaml``: the ``recipes:`` list, re-sorted so it reads in
    sidebar order.

Provider order comes from ``_catalog/providers.yaml``: ranked providers first,
in their listed order, then every other provider alphabetically by name. A
provider with no active recipe is dropped. Within a provider, recipes keep
their ``index.yaml`` order (newest first).

The overview's model cards stay hand-written, so this script also checks them:
every card's ``data-provider`` must name a provider that has an active recipe,
and every such provider must have at least one card, so no filter chip is a
dead end.

Usage (from any cwd; paths resolve relative to this file):

    python3 gen_recipe_nav.py          # rewrite the generated spans
    python3 gen_recipe_nav.py --check  # exit 1 if any span is stale, no writes

The ``gen-recipe-nav`` pre-commit hook runs the write mode, so a commit that
changes the catalog regenerates the spans and asks for them to be staged.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
FERN_DIR = SCRIPT_DIR.parent
CATALOG_DIR = FERN_DIR / "pages" / "recipes" / "_catalog"
PROVIDERS_YAML = CATALOG_DIR / "providers.yaml"
INDEX_YAML = CATALOG_DIR / "index.yaml"
NAV_YML = FERN_DIR / "index.yml"
OVERVIEW_MDX = FERN_DIR / "pages" / "recipes" / "model-recipes" / "overview.mdx"
STYLES_TSX = FERN_DIR / "components" / "RecipeStyles.tsx"

PROVIDER_KEY = re.compile(r"^[a-z0-9]+$")
CARD_PROVIDER = re.compile(r'data-recipe-card\b[^>]*?\bdata-provider="([^"]*)"')


class CatalogError(Exception):
    """The catalog or a target file is inconsistent; nothing is written."""


@dataclass(frozen=True)
class Provider:
    key: str
    name: str
    chip: str


@dataclass(frozen=True)
class Recipe:
    id: str
    title: str
    provider: str
    page: str
    slug: str
    targets: int


@dataclass(frozen=True)
class Group:
    provider: Provider
    recipes: tuple[Recipe, ...]


# ---------------------------------------------------------------- loading


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise CatalogError(f"{path}: expected a mapping at the top level")
    return data


def parse_providers(data: dict) -> tuple[dict[str, Provider], list[str]]:
    """Return the provider registry and the ranked key order."""
    raw = data.get("providers") or {}
    providers: dict[str, Provider] = {}
    for key, fields in raw.items():
        if not PROVIDER_KEY.match(str(key)):
            raise CatalogError(
                f"providers.yaml: key {key!r} must be lowercase letters and digits"
            )
        if not isinstance(fields, dict) or not fields.get("name"):
            raise CatalogError(f"providers.yaml: provider {key!r} needs a name")
        name = str(fields["name"])
        providers[key] = Provider(key, name, str(fields.get("chip") or name))

    ranked = [str(k) for k in data.get("ranked") or []]
    seen: set[str] = set()
    for key in ranked:
        if key not in providers:
            raise CatalogError(
                f"providers.yaml: ranked provider {key!r} is not under providers:"
            )
        if key in seen:
            raise CatalogError(f"providers.yaml: {key!r} is ranked twice")
        seen.add(key)
    return providers, ranked


def parse_recipe(entry: dict, providers: dict[str, Provider], source: str) -> Recipe:
    rid = entry.get("id")
    provider = entry.get("provider")
    if provider not in providers:
        raise CatalogError(
            f"{source}: provider {provider!r} is not in providers.yaml; "
            "add it under providers: (rank it only if it belongs above the "
            "alphabetical tail)"
        )
    page = entry.get("page")
    if not page:
        raise CatalogError(f"{source}: an active recipe needs a page:")
    return Recipe(
        id=rid,
        title=entry["title"],
        provider=provider,
        page=page,
        slug=entry.get("slug") or Path(page).stem,
        targets=len(entry.get("targets") or []),
    )


def load_catalog() -> tuple[dict[str, Provider], list[str], list[Recipe]]:
    providers, ranked = parse_providers(load_yaml(PROVIDERS_YAML))
    index = load_yaml(INDEX_YAML)
    recipes = []
    for rid in index.get("recipes") or []:
        path = CATALOG_DIR / "recipes" / f"{rid}.yaml"
        if not path.exists():
            raise CatalogError(f"index.yaml lists {rid!r} but {path} does not exist")
        recipes.append(parse_recipe(load_yaml(path), providers, f"recipes/{rid}.yaml"))
    for rid in index.get("deferred_recipes") or []:
        path = CATALOG_DIR / "recipes" / f"{rid}.yaml"
        if path.exists():
            provider = load_yaml(path).get("provider")
            if provider not in providers:
                raise CatalogError(
                    f"recipes/{rid}.yaml: provider {provider!r} is not in providers.yaml"
                )
    return providers, ranked, recipes


# --------------------------------------------------------------- ordering


def group_recipes(
    providers: dict[str, Provider], ranked: list[str], recipes: list[Recipe]
) -> list[Group]:
    """Group active recipes by provider in display order.

    Ranked providers come first in ranked order; the rest follow alphabetically
    by display name. Providers without an active recipe are omitted. Recipes
    keep their catalog order within a provider.
    """
    by_provider: dict[str, list[Recipe]] = {}
    for recipe in recipes:
        by_provider.setdefault(recipe.provider, []).append(recipe)
    head = [k for k in ranked if k in by_provider]
    tail = sorted(
        (k for k in by_provider if k not in ranked),
        key=lambda k: (providers[k].name.casefold(), k),
    )
    return [Group(providers[k], tuple(by_provider[k])) for k in head + tail]


# -------------------------------------------------------------- rendering


def render_nav(groups: list[Group], indent: str) -> list[str]:
    lines = []
    for g in groups:
        lines += [
            f"{indent}- section: {g.provider.name}",
            f"{indent}  skip-slug: true",
            f"{indent}  collapsed: true",
            f"{indent}  contents:",
        ]
        for r in g.recipes:
            lines += [
                f"{indent}    - page: {r.title}",
                f"{indent}      path: {r.page}",
                f"{indent}      slug: {r.slug}",
            ]
    return lines


def render_provider_inputs(groups: list[Group]) -> list[str]:
    return [
        f'<input className="dynamo-filter-input" type="radio" '
        f'id="provider-{g.provider.key}" name="provider" />'
        for g in groups
    ]


def render_provider_chips(groups: list[Group]) -> list[str]:
    return [
        f'<label className="dynamo-recipe-chip" '
        f'htmlFor="provider-{g.provider.key}">{g.provider.chip}</label>'
        for g in groups
    ]


def render_family_count(recipes: list[Recipe], indent: str) -> list[str]:
    return [f"{indent}<h3>{len(recipes)} model families</h3>"]


def render_config_count(recipes: list[Recipe], indent: str) -> list[str]:
    return [f"{indent}<strong>{sum(r.targets for r in recipes)}</strong>"]


def render_chip_rules(groups: list[Group]) -> list[str]:
    return [
        f"#provider-{k}:checked ~ .dynamo-recipe-browser " f'label[for="provider-{k}"],'
        for k in (g.provider.key for g in groups)
    ]


def render_filter_rules(groups: list[Group]) -> list[str]:
    return [
        f"#provider-{k}:checked ~ .dynamo-model-grid "
        f'[data-recipe-card]:not([data-provider~="{k}"]),'
        for k in (g.provider.key for g in groups)
    ]


def render_index(text: str, groups: list[Group]) -> str:
    """Rewrite index.yaml's ``recipes:`` list in sidebar order."""
    lines = text.split("\n")
    try:
        start = lines.index("recipes:") + 1
    except ValueError:
        raise CatalogError("index.yaml: no top-level 'recipes:' list") from None
    end = start
    while end < len(lines) and lines[end].startswith("- "):
        end += 1
    ordered = [f"- {r.id}" for g in groups for r in g.recipes]
    return "\n".join(lines[:start] + ordered + lines[end:])


# --------------------------------------------------------------- splicing


def splice(text: str, name: str, render, source: Path) -> str:
    """Replace the lines between the ``<name>:begin`` and ``<name>:end`` markers.

    ``render`` receives the begin marker's indentation and returns the body
    lines. The marker lines themselves are kept, so their comment syntax can
    suit each file type.
    """
    lines = text.split("\n")
    begins = [i for i, line in enumerate(lines) if f"{name}:begin" in line]
    ends = [i for i, line in enumerate(lines) if f"{name}:end" in line]
    if len(begins) != 1 or len(ends) != 1 or begins[0] > ends[0]:
        raise CatalogError(
            f"{source.relative_to(FERN_DIR)}: expected one {name}:begin marker "
            f"followed by one {name}:end marker"
        )
    b, e = begins[0], ends[0]
    indent = lines[b][: len(lines[b]) - len(lines[b].lstrip())]
    return "\n".join(lines[: b + 1] + render(indent) + lines[e:])


def check_cards(overview: str, groups: list[Group]) -> None:
    """Every card must name an active provider, and every one must have a card."""
    active = {g.provider.key for g in groups}
    carded: set[str] = set()
    for value in CARD_PROVIDER.findall(overview):
        for key in value.split():
            if key not in active:
                raise CatalogError(
                    f"overview.mdx: a card has data-provider={key!r}, which has "
                    "no active recipe in the catalog, so it has no filter chip"
                )
            carded.add(key)
    missing = [g.provider.name for g in groups if g.provider.key not in carded]
    if missing:
        raise CatalogError(
            "overview.mdx: no model card for "
            + ", ".join(missing)
            + "; add one so the provider's filter chip is not empty"
        )


def build() -> dict[Path, str]:
    """Return the generated contents of every output file."""
    providers, ranked, recipes = load_catalog()
    groups = group_recipes(providers, ranked, recipes)

    nav = splice(
        NAV_YML.read_text(encoding="utf-8"),
        "recipe-nav",
        lambda ind: render_nav(groups, ind),
        NAV_YML,
    )

    overview = OVERVIEW_MDX.read_text(encoding="utf-8")
    check_cards(overview, groups)
    overview = splice(
        overview,
        "provider-inputs",
        lambda _: render_provider_inputs(groups),
        OVERVIEW_MDX,
    )
    overview = splice(
        overview,
        "provider-chips",
        lambda _: render_provider_chips(groups),
        OVERVIEW_MDX,
    )
    overview = splice(
        overview,
        "family-count",
        lambda ind: render_family_count(recipes, ind),
        OVERVIEW_MDX,
    )
    overview = splice(
        overview,
        "config-count",
        lambda ind: render_config_count(recipes, ind),
        OVERVIEW_MDX,
    )

    styles = STYLES_TSX.read_text(encoding="utf-8")
    styles = splice(
        styles, "provider-chip-rules", lambda _: render_chip_rules(groups), STYLES_TSX
    )
    styles = splice(
        styles,
        "provider-filter-rules",
        lambda _: render_filter_rules(groups),
        STYLES_TSX,
    )

    index = render_index(INDEX_YAML.read_text(encoding="utf-8"), groups)

    return {NAV_YML: nav, OVERVIEW_MDX: overview, STYLES_TSX: styles, INDEX_YAML: index}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit 1 if any output is stale; write nothing",
    )
    args = parser.parse_args(argv)

    try:
        outputs = build()
    except CatalogError as err:
        print(f"gen_recipe_nav: {err}", file=sys.stderr)
        return 1

    stale = [p for p, text in outputs.items() if p.read_text(encoding="utf-8") != text]
    for path in stale:
        rel = path.relative_to(FERN_DIR.parent.parent)
        if args.check:
            print(f"stale: {rel}", file=sys.stderr)
        else:
            path.write_text(outputs[path], encoding="utf-8")
            print(f"updated: {rel}")
    if args.check and stale:
        print(
            "Run python3 docs/fern/scripts/gen_recipe_nav.py and commit the result.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
