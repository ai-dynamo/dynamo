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


def recipe(
    rid: str, provider: str, targets: int = 1, generation: tuple[int, ...] = (1,)
) -> gen.Recipe:
    return gen.Recipe(
        rid, rid.upper(), provider, f"pages/{rid}.mdx", rid, targets, generation
    )


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


def test_recipes_keep_catalog_order_within_a_generation():
    registry, ranked = providers(["qwen", "zai"], {"zai": "Z.ai", "qwen": "Qwen"})
    recipes = [recipe("q2", "qwen"), recipe("z1", "zai"), recipe("q1", "qwen")]
    groups = gen.group_recipes(registry, ranked, recipes)
    assert [r.id for g in groups for r in g.recipes] == ["q2", "q1", "z1"]


def test_newest_generation_leads_within_a_provider():
    registry, ranked = providers(["qwen"], {"qwen": "Qwen"})
    recipes = [
        recipe("q3-a", "qwen", generation=(3,)),
        recipe("q3.10", "qwen", generation=(3, 10)),
        recipe("q3.8", "qwen", generation=(3, 8)),
        recipe("q3-b", "qwen", generation=(3,)),
    ]
    groups = gen.group_recipes(registry, ranked, recipes)
    assert [r.id for r in groups[0].recipes] == ["q3.10", "q3.8", "q3-a", "q3-b"]


def test_pro_then_flash_then_untagged_within_a_generation():
    registry, ranked = providers(["deepseek"], {"deepseek": "DeepSeek"})
    titles = ["V4", "V4-Flash", "V4-Pro-0813", "V4-Pro", "V4.1-Flash", "V4-Profile"]
    recipes = [
        gen.Recipe(
            t, t, "deepseek", f"pages/{t}.mdx", t, 1, (4, 1) if "4.1" in t else (4,)
        )
        for t in titles
    ]
    groups = gen.group_recipes(registry, ranked, recipes)
    assert [r.title for r in groups[0].recipes] == [
        "V4.1-Flash",
        "V4-Pro-0813",
        "V4-Pro",
        "V4-Flash",
        "V4",
        "V4-Profile",
    ]


@pytest.mark.parametrize(
    "title, expected",
    [
        ("DeepSeek-V4.1-Flash", (4, 1)),
        ("DeepSeek-V4-Pro-0813", (4,)),
        ("GLM-5.3/5.2", (5, 3)),
        ("GLM-5 NVFP4", (5,)),
        ("Kimi-K2.6", (2, 6)),
        ("Qwen3.8-2.4T-A95B", (3, 8)),
        ("Qwen3-235B-A22B FP8", (3,)),
        ("Nemotron 3.5 Lightning", (3, 5)),
        ("K-EXAONE 2.0", (2,)),
        ("Solar Open2 250B", (2,)),
        ("Qwen3.10-7B", (3, 10)),
        ("GPT-OSS-120B", (0,)),
        ("Inkling NVFP4", (0,)),
        ("Llama BF16 INT4 MXFP4", (0,)),
    ],
)
def test_generation_comes_from_the_title(title, expected):
    assert gen.parse_generation({"title": title}, "x") == expected


@pytest.mark.parametrize(
    "value, expected",
    [("4.1", (4, 1)), ("5.10", (5, 10)), ("2.0", (2,)), ("3", (3,))],
)
def test_generation_override_beats_the_title(value, expected):
    entry = {"title": "Model-9", "model": {"generation": value}}
    assert gen.parse_generation(entry, "x") == expected


@pytest.mark.parametrize("value", [5.1, 3, "v4", "4.1-flash"])
def test_generation_override_must_be_a_quoted_version(value):
    # An unquoted 5.10 loads as the float 5.1 and would sort below 5.9.
    with pytest.raises(gen.CatalogError, match="quoted version number"):
        gen.parse_generation({"model": {"generation": value}}, "x")


def test_unknown_provider_fails_with_a_fix():
    registry, _ = providers([], {"zai": "Z.ai"})
    entry = {"id": "x", "title": "X", "provider": "acme", "page": "pages/x.mdx"}
    with pytest.raises(gen.CatalogError, match="not in providers.yaml"):
        gen.parse_recipe(entry, registry, "recipes/x.yaml")


def test_slug_defaults_to_page_stem_and_can_be_overridden():
    registry, _ = providers([], {"zai": "Z.ai"})
    entry = {
        "id": "x",
        "title": "X",
        "provider": "zai",
        "page": "pages/x-nvfp4.mdx",
    }
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


def card(page: str, provider: str, indent: str = "  ") -> list[str]:
    return [
        f'{indent}<div className="dynamo-model-card" data-recipe-card '
        f'data-provider="{provider}">',
        f"{indent}  <div><h3>{page}</h3></div>",
        f'{indent}  <a className="dynamo-card-link" href="{page}.mdx">Open</a>',
        f"{indent}</div>",
    ]


def test_cards_follow_sidebar_order_and_drop_blank_lines():
    groups = groups_for("zai", "qwen")
    body = card("qwen", "qwen") + [""] + card("zai", "zai")
    assert gen.sort_cards(body, groups) == card("zai", "zai") + card("qwen", "qwen")


@pytest.mark.parametrize(
    "body, message",
    [
        (card("zai", "zai"), "no model card for QWEN"),
        (card("zai", "zai") + card("qwen", "zai"), 'data-provider="qwen"'),
        (card("zai", "zai") * 2 + card("qwen", "qwen"), "two model cards"),
        (card("zai", "zai") + card("qwen", "qwen") + card("meta", "meta"), "meta.mdx"),
        (card("zai", "zai") + ["<p>stray</p>"], "only model cards"),
        (card("zai", "zai")[:-1], "no '</div>'"),
    ],
)
def test_cards_must_match_active_recipes_one_to_one(body, message):
    with pytest.raises(gen.CatalogError, match=message):
        gen.sort_cards(body, groups_for("zai", "qwen"))


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


def test_provider_row_closes_on_the_last_chip_line():
    # MDX parses the row as one paragraph: a line that starts with `{` or with
    # the closing tags ends the paragraph before the <div>s close, and Fern
    # fails to parse the page.
    lines = gen.render_provider_chips(groups_for("zai", "qwen"))
    assert lines[0].startswith('<div className="dynamo-filter-row">')
    assert lines[0].endswith('htmlFor="provider-all">All</label>')
    assert lines[-1].endswith('htmlFor="provider-qwen">Qwen</label></div></div>')
    assert all(line.startswith("<") and not line.startswith("</") for line in lines)


def test_committed_outputs_are_current():
    assert gen.main(["--check"]) == 0
