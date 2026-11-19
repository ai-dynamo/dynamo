"""Regression cases for delegated ownership decisions and review state."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import check_ownership_approvals as gate
import pytest
import yaml
from codeowners_match import (
    DELEGATED_POLICY_PATHS,
    compute_resolution,
    parse_codeowners,
    resolve_owners,
)
from ownership_approvals import (
    OPS,
    PROCESS,
    approved_reviewers,
    enforcement_changed,
    governance,
    missing_approvals,
    policy_fingerprint,
    rendered_policy,
    required_approvals,
)


@pytest.mark.parametrize(
    "old,new,expected",
    [
        ("@ops", "@ops @runtime", {"@ops", "@runtime"}),
        ("@docs", "@operator", {"@docs", "@operator"}),
        ("@ops @runtime", "@runtime", {"@ops", "@runtime"}),
        ("@a @b", "@a @c", {"@a", "@b", "@c"}),
    ],
)
def test_every_previous_and_proposed_owner_has_a_say(old, new, expected):
    required, changes = required_approvals(
        f"/src/ {old}", f"/src/ {new}", ["src/a.py", "src/b.py"]
    )
    assert required == expected
    assert [c.path for c in changes] == ["src/a.py", "src/b.py"]


def test_only_effective_last_matches_matter():
    old = "/src/ @a\n/src/a.py @old\n/src/a.py @stable"
    new = "/src/ @a\n/src/a.py @other\n/src/a.py @stable"
    assert required_approvals(old, new, ["src/a.py"]) == (set(), [])


def test_new_exact_override_delegates_to_current_owner_and_recipient():
    old = "/docs/ @docs"
    new = old + "\n/docs/generated.md @operator"
    required, changes = required_approvals(
        old, new, ["docs/generated.md", "docs/manual.md"]
    )
    assert required == {"@docs", "@operator"}
    assert [c.path for c in changes] == ["docs/generated.md"]


def test_order_of_coowners_does_not_change_ownership():
    assert required_approvals("/a @one @two", "/a @two @one", ["a"]) == (set(), [])


def test_future_only_scope_loss_requires_governance():
    required, changes = required_approvals("/src/ @a", "/src/*.py @a", ["src/a.py"])
    assert required == {PROCESS}
    assert changes == []


def test_future_only_rule_overlap_cannot_change_without_review():
    old = "/src/*/left.py @a\n/src/new/*.py @b"
    new = "/src/new/*.py @b\n/src/*/left.py @a"
    required, changes = required_approvals(
        old, new, ["src/old/left.py", "src/new/right.py"]
    )
    assert required == {PROCESS}
    assert changes == []


@pytest.mark.parametrize(
    "old,new,paths",
    [
        (
            "/src/*/left.py @a\n/src/new/*.py @b",
            "/src/new/*.py @b\n/src/*/left.py @a",
            ["src/old/left.py", "src/new/right.py", "concrete"],
        ),
        (
            "/src/ @a\n/src/current.py @stable",
            "/src/ @c\n/src/current.py @stable",
            ["src/current.py", "concrete"],
        ),
    ],
)
def test_unrelated_transfer_does_not_hide_future_scope_changes(old, new, paths):
    required, _ = required_approvals(
        old + "\n/concrete @x",
        new + "\n/concrete @y",
        paths,
    )
    assert PROCESS in required
    assert {"@x", "@y"} <= required


def test_delegated_files_do_not_reintroduce_their_blanket_owners():
    old = (
        "* @runtime\n/CODEOWNERS @process\n/.github/codeowners/areas.yaml @ops @process"
    )
    new = "* @runtime\n/CODEOWNERS\n/.github/codeowners/areas.yaml"
    assert required_approvals(old, new, list(DELEGATED_POLICY_PATHS)) == (set(), [])
    assert resolve_owners(parse_codeowners(new), "CODEOWNERS") == []
    assert resolve_owners(parse_codeowners(new), "src/a.py") == ["@runtime"]


@pytest.fixture
def spec():
    return {
        "meta": {"catch_all": "@root", "delegated_policy_reviews": True},
        "areas": [
            {"label": "a", "github_team": "@org/a", "path_globs": ["src/"]},
            {"label": "b", "github_team": "@org/b", "path_globs": ["other/"]},
        ],
    }


def test_only_two_fixed_paths_can_be_delegated(spec):
    model = compute_resolution(spec)
    assert model.delegated_paths() == DELEGATED_POLICY_PATHS
    assert model.unmatched_paths([*DELEGATED_POLICY_PATHS, "unclaimed.txt"]) == [
        "unclaimed.txt"
    ]
    rules = parse_codeowners(rendered_policy(spec, []))
    assert resolve_owners(rules, "CODEOWNERS") == []
    assert resolve_owners(rules, ".github/codeowners/areas.yaml") == []
    assert resolve_owners(rules, ".github/codeowners/malicious.yaml") == ["@root"]


@pytest.mark.parametrize("owner", ["", " ", "@", "@team # hidden", "not-a-handle"])
def test_an_area_cannot_clear_native_owners_with_an_empty_or_invalid_team(spec, owner):
    spec["areas"][0]["github_team"] = owner
    with pytest.raises(SystemExit, match="nonempty GitHub owner handle"):
        compute_resolution(spec)


def test_bulk_rollout_stays_below_the_matrix_limit():
    prs = [{"pr": i, "head": str(i), "policy": False} for i in range(1075)]
    grouped = gate.batches(prs)
    assert len(grouped) == 8
    assert sorted(p["pr"] for group in grouped for p in group["prs"]) == list(
        range(1075)
    )


def test_delegation_is_opt_in_and_rejects_string_flags(spec):
    spec["meta"]["delegated_policy_reviews"] = False
    assert compute_resolution(spec).delegated_paths() == frozenset()
    spec["meta"]["delegated_policy_reviews"] = "false"
    with pytest.raises(SystemExit, match="must be a boolean"):
        compute_resolution(spec)


def test_fingerprint_ignores_yaml_mapping_order_and_comments(spec):
    reverse = dict(reversed(list(spec.items())))
    assert policy_fingerprint(spec, []) == policy_fingerprint(reverse, [])


def test_another_transfer_invalidates_prior_policy_approval(spec):
    changed = deepcopy(spec)
    changed["shared"] = [{"glob": "src/new.py", "owners": ["a", "b"]}]
    assert policy_fingerprint(spec, []) != policy_fingerprint(changed, [])


def test_contract_changes_require_process_even_without_new_routing(spec):
    changed = deepcopy(spec)
    changed["required_owners"] = [{"glob": "src/", "owners": ["a"]}]
    assert rendered_policy(spec, []) == rendered_policy(changed, [])
    assert governance(spec) != governance(changed)
    assert policy_fingerprint(spec, []) != policy_fingerprint(changed, [])


@pytest.mark.parametrize(
    "path",
    [
        ".github/codeowners/ownership_approvals.py",
        ".github/workflows/codeowners-approval.yml",
    ],
)
def test_enforcement_cannot_delegate_its_own_changes(path):
    assert enforcement_changed([path])
    required, _ = required_approvals("/a @one", "/a @one", ["a"], protected=True)
    assert required == {OPS, PROCESS}


def review(review_id, login, state, *, user_type="User"):
    return {
        "id": review_id,
        "user": {"login": login, "type": user_type},
        "state": state,
        "submitted_at": f"2026-11-{review_id:02d}T00:00:00Z",
        "commit_id": "same-policy",
    }


def test_comments_preserve_approval_but_changes_requested_revoke_it():
    reviews = [review(1, "alice", "APPROVED"), review(2, "alice", "COMMENTED")]
    assert set(approved_reviewers(reviews, "author")) == {"alice"}
    reviews.append(review(3, "alice", "CHANGES_REQUESTED"))
    assert approved_reviewers(reviews, "author") == {}


def test_dismissed_author_and_bot_approvals_do_not_count():
    reviews = [
        review(1, "alice", "DISMISSED"),
        review(2, "author", "APPROVED"),
        review(3, "bot", "APPROVED", user_type="Bot"),
    ]
    assert approved_reviewers(reviews, "author") == {}


def test_one_person_can_approve_for_multiple_groups():
    members = {"@org/a": {"alice"}, "@org/b": {"alice", "bob"}}
    assert missing_approvals(set(members), {"alice"}, members) == set()
    assert missing_approvals(set(members), {"bob"}, members) == {"@org/a"}
    with pytest.raises(KeyError):
        missing_approvals(set(members), {"alice"}, {})


def test_named_owner_must_approve():
    assert missing_approvals({"@alice"}, {"bob"}, {}) == {"@alice"}
    assert missing_approvals({"@alice"}, {"alice"}, {}) == set()


def test_team_membership_and_reviews_are_not_truncated_at_100(monkeypatch):
    calls = []

    def fake_api(endpoint, **kwargs):
        calls.append(endpoint)
        return list(range(100)) if endpoint.endswith("&page=1") else ["last-reviewer"]

    monkeypatch.setattr(gate, "api", fake_api)
    assert gate.pages("orgs/org/teams/team/members")[-1] == "last-reviewer"
    assert len(calls) == 2


def test_failed_api_reads_do_not_turn_into_empty_membership(monkeypatch):
    def unavailable(*args, **kwargs):
        raise ValueError("API unavailable")

    monkeypatch.setattr(gate, "api", unavailable)
    with pytest.raises(ValueError, match="API unavailable"):
        gate.pages("orgs/org/teams/team/members")


def test_noop_discovery_cannot_hide_an_ownership_edit(monkeypatch):
    pr = {
        "state": "open",
        "head": {"sha": "head"},
        "base": {"ref": "main", "sha": "base"},
    }
    published = []

    def fake_api(endpoint, **kwargs):
        if "/reviews?" in endpoint:
            return []
        if "/check-runs?" in endpoint:
            return {"check_runs": [{"id": 42, "app": {"id": 15368}}]}
        if "/pulls/1" in endpoint:
            return pr
        if "/compare/" in endpoint:
            return {"files": [{"filename": "CODEOWNERS"}]}
        raise AssertionError(endpoint)

    def needs_review(*args, **kwargs):
        raise ValueError("Missing affected owners")

    monkeypatch.setattr(gate, "api", fake_api)
    monkeypatch.setattr(gate, "evaluate", needs_review)

    def publish(*a, **kw):
        published.append(kw["conclusion"])
        return 42

    monkeypatch.setattr(gate, "check_run", publish)
    gate.evaluate_one("org/repo", {"pr": 1, "head": "head", "policy": False}, {})
    assert published == [None, "failure"]


def test_review_relay_and_bootstrap_cannot_replace_native_structure():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load(
        (root / ".github/workflows/codeowners.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert "edited" in workflow["on"]["pull_request"]["types"]
    assert (
        workflow["jobs"]["gate-mode"]["steps"][0]["with"]["ref"]
        == "${{ github.event.repository.default_branch }}"
    )
    assert "ownership-review-event" in workflow["jobs"]["codeowners"]["name"]
    assert "base.ref == 'main'" in workflow["jobs"]["approval-bootstrap"]["name"]
    assert "delegated == 'false'" in workflow["jobs"]["approval-bootstrap"]["name"]


@pytest.mark.parametrize("newer_check", [False, True])
def test_bulk_cannot_publish_stale_green_after_a_dismissal(monkeypatch, newer_check):
    pr = {
        "state": "open",
        "head": {"sha": "head"},
        "base": {"ref": "main", "sha": "base"},
    }
    published = []
    reviewed = review(1, "alice", "APPROVED")
    reads = 0

    def fake_api(endpoint, **kwargs):
        nonlocal reads
        if "/reviews?" in endpoint:
            reads += 1
            return [reviewed if reads == 1 else dict(reviewed, state="DISMISSED")]
        if "/check-runs?" in endpoint:
            return {
                "check_runs": [{"id": 43 if newer_check else 42, "app": {"id": 15368}}]
            }
        if "/pulls/1" in endpoint:
            return pr
        raise AssertionError(endpoint)

    def publish(*a, **kw):
        published.append(kw["conclusion"])
        return 42

    monkeypatch.setattr(gate, "api", fake_api)
    monkeypatch.setattr(gate, "evaluate", lambda *a, **kw: "approved before dismissal")
    monkeypatch.setattr(gate, "check_run", publish)
    gate.evaluate_one("org/repo", {"pr": 1, "head": "head", "policy": True}, {})
    assert published == ([None] if newer_check else [None, "failure"])


def test_approval_of_enforcement_code_cannot_be_reused_after_code_changes(spec):
    approved = {".github/codeowners/ownership_approvals.py": "reviewed-blob"}
    changed = {".github/codeowners/ownership_approvals.py": "new-enforcement-blob"}
    assert policy_fingerprint(spec, [], approved) != policy_fingerprint(
        spec, [], changed
    )


def test_native_exemptions_require_a_separately_required_check(
    spec, monkeypatch, tmp_path
):
    monkeypatch.chdir(tmp_path)
    areas = tmp_path / gate.AREAS
    areas.parent.mkdir(parents=True)
    areas.write_text("meta: {delegated_policy_reviews: true}")
    monkeypatch.setattr(gate, "api", lambda *a, **kw: [])
    with pytest.raises(ValueError, match="Before enabling delegation"):
        gate.check_activation("org/repo")
    rules = [
        {
            "type": "required_status_checks",
            "parameters": {
                "required_status_checks": [
                    {"context": gate.CHECK_NAME, "integration_id": 15368},
                    {"context": "codeowners", "integration_id": 15368},
                ]
            },
        }
    ]
    monkeypatch.setattr(gate, "api", lambda *a, **kw: rules)
    gate.check_activation("org/repo")
    rules[0]["parameters"]["required_status_checks"][0]["integration_id"] = None
    with pytest.raises(ValueError, match="Before enabling delegation"):
        gate.check_activation("org/repo")
