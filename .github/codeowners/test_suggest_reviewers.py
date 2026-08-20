# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for suggest_reviewers -- the reviewer answer posted on a pull request.

Resolution is not retested here. ``resolve_owners`` and ``match`` are covered by
``test_codeowners.py``, and this module calls the same functions, so duplicating
those cases would only assert that imports work. What is tested is the part that
is new: grouping many files into a readable answer, keeping the "any one
co-owner satisfies the gate" distinction that a flat union destroys, surfacing
unowned paths instead of quietly reporting no reviewer, and bounding the comment
so a large pull request cannot produce one GitHub refuses to accept.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from codeowners_match import parse_codeowners  # noqa: E402
from suggest_reviewers import (  # noqa: E402
    COMMENT_MARKER,
    MAX_COMMENT_BYTES,
    render_comment,
    suggest,
)

CODEOWNERS = """\
* @ai-dynamo/dynamo-codeowners
/lib/llm/ @ai-dynamo/dynamo-llm-codeowners
/deploy/ @ai-dynamo/dynamo-deploy-codeowners @ai-dynamo/dynamo-ops-codeowners
/docs/ @ai-dynamo/dynamo-docs-codeowners @janedoe
"""


@pytest.fixture
def rules() -> list[tuple[str, list[str]]]:
    return parse_codeowners(CODEOWNERS)


# ------------------------------------------------------------------- resolution


def test_the_union_names_every_team_github_will_request(rules) -> None:
    result = suggest(["lib/llm/a.rs", "deploy/b.yaml"], rules)
    assert result.requested_teams == (
        "@ai-dynamo/dynamo-deploy-codeowners",
        "@ai-dynamo/dynamo-llm-codeowners",
        "@ai-dynamo/dynamo-ops-codeowners",
    )


def test_co_owners_stay_grouped_because_any_one_satisfies_the_gate(rules) -> None:
    """The union alone is misleading, so the grouping has to survive.

    ``/deploy/`` lists two teams on one line, which means either team's approval
    satisfies the gate. Flattened into a list of three requested teams, that
    reads as three required approvals. The group preserves the distinction.
    """
    result = suggest(["deploy/b.yaml"], rules)
    assert len(result.groups) == 1
    assert result.groups[0].owners == (
        "@ai-dynamo/dynamo-deploy-codeowners",
        "@ai-dynamo/dynamo-ops-codeowners",
    )


def test_files_sharing_an_owner_set_collapse_into_one_group(rules) -> None:
    """A forty-file pull request must not render forty lines."""
    result = suggest(["lib/llm/a.rs", "lib/llm/b.rs", "lib/llm/c.rs"], rules)
    assert len(result.groups) == 1
    assert result.groups[0].files == ("lib/llm/a.rs", "lib/llm/b.rs", "lib/llm/c.rs")


def test_last_match_wins_exactly_as_github_resolves(rules) -> None:
    """``/lib/llm/`` is more specific than ``*``, so the catch-all loses."""
    result = suggest(["lib/llm/a.rs"], rules)
    assert result.requested_teams == ("@ai-dynamo/dynamo-llm-codeowners",)


def test_an_individual_login_from_codeowners_may_be_named(rules) -> None:
    """Public-tier disclosure, and it looks like a contradiction without this.

    The agent's capability contract forbids naming individuals to someone who
    could not look them up. A login written into CODEOWNERS is not that case:
    the file ships in a public repository, so ``@janedoe`` owning ``/docs/`` is
    already public. What stays behind the org boundary is *team membership* and
    per-person review load, neither of which this answer reads.
    """
    result = suggest(["docs/x.md"], rules)
    assert "@janedoe" in result.requested_teams


# --------------------------------------------------------------- unowned paths


def test_unowned_files_are_surfaced_not_silently_dropped() -> None:
    """Reporting no reviewer and reporting no rule are different answers."""
    result = suggest(["stray.txt"], parse_codeowners("/lib/ @ai-dynamo/x\n"))
    assert result.unowned == ("stray.txt",)
    assert result.requested_teams == ()


def test_unowned_files_do_not_suppress_the_owned_ones() -> None:
    rules = parse_codeowners("/lib/ @ai-dynamo/x\n")
    result = suggest(["lib/a.rs", "stray.txt"], rules)
    assert result.requested_teams == ("@ai-dynamo/x",)
    assert result.unowned == ("stray.txt",)


# ------------------------------------------------------------------ rendering


def test_the_comment_carries_a_marker_so_it_can_be_updated_in_place(rules) -> None:
    """Without a marker the bot appends a new comment on every push."""
    body = render_comment(suggest(["lib/llm/a.rs"], rules))
    assert COMMENT_MARKER in body


def test_the_comment_names_the_requested_teams(rules) -> None:
    body = render_comment(suggest(["lib/llm/a.rs"], rules))
    assert "@ai-dynamo/dynamo-llm-codeowners" in body


def test_the_comment_says_any_one_co_owner_suffices(rules) -> None:
    body = render_comment(suggest(["deploy/b.yaml"], rules))
    assert "any one" in body.lower()


def test_the_comment_points_at_the_self_service_tool(rules) -> None:
    """The answer is reproducible without the bot, and saying so is the point."""
    body = render_comment(suggest(["lib/llm/a.rs"], rules))
    assert "who_owns.py" in body


def test_a_huge_pull_request_still_fits_in_one_comment(rules) -> None:
    """GitHub rejects a comment body over 65536 characters."""
    files = [f"lib/llm/f{i:05d}.rs" for i in range(5000)]
    body = render_comment(suggest(files, rules))
    assert len(body.encode()) <= MAX_COMMENT_BYTES


def test_truncation_says_so_rather_than_appearing_complete(rules) -> None:
    files = [f"lib/llm/f{i:05d}.rs" for i in range(5000)]
    body = render_comment(suggest(files, rules))
    assert "more" in body.lower()


def test_truncation_never_drops_a_team_only_the_file_list(rules) -> None:
    """The team list is the answer; the file list is evidence for it."""
    files = [f"lib/llm/f{i:05d}.rs" for i in range(5000)] + ["deploy/b.yaml"]
    body = render_comment(suggest(files, rules))
    assert "@ai-dynamo/dynamo-llm-codeowners" in body
    assert "@ai-dynamo/dynamo-deploy-codeowners" in body


def test_many_distinct_owner_groups_are_counted_when_they_do_not_fit() -> None:
    """The overflow note must name a plausible remainder, not a stale total."""
    rules = parse_codeowners(
        "".join(f"/area{i:04d}/ @ai-dynamo/team-{i:04d}\n" for i in range(4000))
    )
    files = [f"area{i:04d}/f.rs" for i in range(4000)]
    body = render_comment(suggest(files, rules))
    assert len(body.encode()) <= MAX_COMMENT_BYTES
    assert "more owner group" in body


def test_a_pull_request_touching_nothing_owned_says_what_to_do() -> None:
    result = suggest(["stray.txt"], parse_codeowners("/lib/ @ai-dynamo/x\n"))
    body = render_comment(result)
    assert "stray.txt" in body


def test_rendering_is_stable_for_the_same_input(rules) -> None:
    """An unstable render would edit the comment on every run, mailing everyone."""
    files = ["deploy/b.yaml", "lib/llm/a.rs", "docs/x.md"]
    assert render_comment(suggest(files, rules)) == render_comment(
        suggest(files, rules)
    )


def test_file_order_does_not_change_the_answer(rules) -> None:
    forward = render_comment(suggest(["lib/llm/a.rs", "deploy/b.yaml"], rules))
    reverse = render_comment(suggest(["deploy/b.yaml", "lib/llm/a.rs"], rules))
    assert forward == reverse


def test_no_third_party_import_so_the_workflow_needs_no_pip_install() -> None:
    """A dependency here means a pip step in the workflow that can fail offline."""
    import ast

    source = Path(__file__).parent / "suggest_reviewers.py"
    tree = ast.parse(source.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert "yaml" not in imported
