# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for gen_recipe_nav.py.

Run: pytest -c docs/fern/scripts/pytest.ini docs/fern/scripts/test_gen_recipe_nav.py

CI runs this through the `gen-recipe-nav-tests` pre-commit hook. The repo-root
pytest config ignores docs/, so the standalone pytest.ini beside this file
supplies the marker registrations.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import gen_recipe_nav as gen  # noqa: E402

# Pure-Python catalog rendering: no GPU, no engine, no network.
pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0, pytest.mark.unit]


def providers(ranked: list[str], names: dict[str, str]):
    return gen.parse_providers(
        {"ranked": ranked, "providers": {k: {"name": v} for k, v in names.items()}}
    )


def recipe(rid: str, provider: str, targets: int = 1) -> gen.Recipe:
    return gen.Recipe(rid, rid.upper(), provider, f"pages/{rid}.mdx", rid, targets)


def order(groups: list[gen.Group]) -> list[str]:
    return [g.provider.key for g in groups]


def test_ranked_providers_lead_and_the_rest_sort_by_name():
    registry, ranked = providers(
        ["zai", "qwen"],
        {
            "zai": "Z.ai",
            "qwen": "Qwen",
            "motif": "Motif",
            "acme": "Acme",
            "beta": "beta",
        },
    )
    recipes = [
        recipe(r, p)
        for r, p in [
            ("m", "motif"),
            ("q", "qwen"),
            ("b", "beta"),
            ("z", "zai"),
            ("a", "acme"),
        ]
    ]
    # Unranked providers sort case-insensitively by display name, after every ranked one.
    assert order(gen.group_recipes(registry, ranked, recipes)) == [
        "zai",
        "qwen",
        "acme",
        "beta",
        "motif",
    ]


def test_provider_without_active_recipe_is_dropped():
    registry, ranked = providers(["zai", "meta"], {"zai": "Z.ai", "meta": "Meta"})
    assert order(gen.group_recipes(registry, ranked, [recipe("z", "zai")])) == ["zai"]


def test_recipes_keep_catalog_order_within_a_provider():
    registry, ranked = providers(["qwen", "zai"], {"zai": "Z.ai", "qwen": "Qwen"})
    recipes = [recipe("q2", "qwen"), recipe("z1", "zai"), recipe("q1", "qwen")]
    groups = gen.group_recipes(registry, ranked, recipes)
    assert [r.id for g in groups for r in g.recipes] == ["q2", "q1", "z1"]


def test_unknown_provider_fails_with_a_fix():
    registry, _ = providers([], {"zai": "Z.ai"})
    entry = {"id": "x", "title": "X", "provider": "acme", "page": "pages/x.mdx"}
    with pytest.raises(gen.CatalogError, match="not in providers.yaml"):
        gen.parse_recipe(entry, registry, "recipes/x.yaml")


def test_slug_defaults_to_page_stem_and_can_be_overridden():
    registry, _ = providers([], {"zai": "Z.ai"})
    entry = {"id": "x", "title": "X", "provider": "zai", "page": "pages/x-nvfp4.mdx"}
    assert gen.parse_recipe(entry, registry, "x").slug == "x-nvfp4"
    assert gen.parse_recipe({**entry, "slug": "x"}, registry, "x").slug == "x"


@pytest.mark.parametrize(
    "data, message",
    [
        ({"ranked": ["zai"], "providers": {}}, "not under providers"),
        (
            {"ranked": ["zai", "zai"], "providers": {"zai": {"name": "Z.ai"}}},
            "ranked twice",
        ),
        ({"providers": {"Z-ai": {"name": "Z.ai"}}}, "lowercase"),
        ({"providers": {"zai": {}}}, "needs a name"),
    ],
)
def test_invalid_providers_yaml_is_rejected(data, message):
    with pytest.raises(gen.CatalogError, match=message):
        gen.parse_providers(data)


def test_chip_label_defaults_to_name():
    registry, _ = gen.parse_providers(
        {
            "providers": {
                "zai": {"name": "Z.ai", "chip": "Z.AI"},
                "qwen": {"name": "Qwen"},
            }
        }
    )
    assert registry["zai"].chip == "Z.AI"
    assert registry["qwen"].chip == "Qwen"


def groups_for(*keys: str) -> list[gen.Group]:
    registry, ranked = providers(list(keys), {k: k.title() for k in keys})
    return gen.group_recipes(registry, ranked, [recipe(k, k) for k in keys])


def test_cards_must_cover_every_provider_and_no_more():
    groups = groups_for("zai", "qwen")
    card = '<div className="dynamo-model-card" data-recipe-card data-provider="{}">'
    gen.check_cards(card.format("zai") + card.format("qwen"), groups)
    with pytest.raises(gen.CatalogError, match="no model card for Qwen"):
        gen.check_cards(card.format("zai"), groups)
    with pytest.raises(gen.CatalogError, match="'meta'"):
        gen.check_cards(card.format("zai qwen meta"), groups)


def test_render_index_reorders_only_the_active_list():
    text = "schema_version: 1\nrecipes:\n- q\n- z\ndeferred_recipes:\n- d\n"
    out = gen.render_index(text, groups_for("z", "q"))
    assert out == "schema_version: 1\nrecipes:\n- z\n- q\ndeferred_recipes:\n- d\n"


def test_splice_replaces_between_markers_and_keeps_indent():
    text = "a\n    # nav:begin\n    stale\n    # nav:end\nb"
    out = gen.splice(text, "nav", lambda ind: [f"{ind}fresh"], gen.NAV_YML)
    assert out == "a\n    # nav:begin\n    fresh\n    # nav:end\nb"


@pytest.mark.parametrize(
    "text",
    ["no markers", "# nav:end\n# nav:begin", "# nav:begin\n# nav:begin\n# nav:end"],
)
def test_splice_requires_one_ordered_marker_pair(text):
    with pytest.raises(gen.CatalogError, match="expected one nav:begin"):
        gen.splice(text, "nav", lambda _: [], gen.NAV_YML)


def test_render_nav_matches_the_sidebar_shape():
    registry, ranked = providers(["zai"], {"zai": "Z.ai"})
    lines = gen.render_nav(
        gen.group_recipes(registry, ranked, [recipe("glm", "zai")]), "  "
    )
    assert lines == [
        "  - section: Z.ai",
        "    skip-slug: true",
        "    collapsed: true",
        "    contents:",
        "      - page: GLM",
        "        path: pages/glm.mdx",
        "        slug: glm",
    ]


def test_committed_outputs_are_current():
    assert gen.main(["--check"]) == 0
