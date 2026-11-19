"""Regression cases for delegated ownership decisions and review state."""

from __future__ import annotations

import subprocess
import sys
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


def test_delegation_is_opt_in_and_rejects_string_flags(spec):
    spec["meta"]["delegated_policy_reviews"] = False
    assert compute_resolution(spec).delegated_paths() == frozenset()
    spec["meta"]["delegated_policy_reviews"] = "false"
    with pytest.raises(SystemExit, match="must be a boolean"):
        compute_resolution(spec)


def test_contract_changes_require_process_even_without_new_routing(spec):
    changed = deepcopy(spec)
    changed["required_owners"] = [{"glob": "src/", "owners": ["a"]}]
    assert rendered_policy(spec, []) == rendered_policy(changed, [])
    assert governance(spec) != governance(changed)


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
        "commit_id": "head",
    }


def test_comments_preserve_approval_but_changes_requested_revoke_it():
    reviews = [review(1, "alice", "APPROVED"), review(2, "alice", "COMMENTED")]
    assert set(approved_reviewers(reviews, "author", "head")) == {"alice"}
    reviews.append(review(3, "alice", "CHANGES_REQUESTED"))
    assert approved_reviewers(reviews, "author", "head") == set()


def test_dismissed_author_and_bot_approvals_do_not_count():
    reviews = [
        review(1, "alice", "DISMISSED"),
        review(2, "author", "APPROVED"),
        review(3, "bot", "APPROVED", user_type="Bot"),
    ]
    assert approved_reviewers(reviews, "author", "head") == set()


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


def test_only_current_head_approvals_count():
    old = dict(review(1, "alice", "APPROVED"), commit_id="old")
    current = review(2, "bob", "APPROVED")
    assert approved_reviewers([old, current], "author", "head") == {"bob"}
    withdrawn = dict(review(3, "bob", "CHANGES_REQUESTED"), commit_id="old")
    assert approved_reviewers([current, withdrawn], "author", "head") == set()


def test_rollout_preserves_native_owners_until_activation():
    root = Path(__file__).resolve().parents[2]
    workflow = yaml.load(
        (root / ".github/workflows/codeowners.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert "edited" in workflow["on"]["pull_request"]["types"]
    assert "ownership-review-event" in workflow["jobs"]["codeowners"]["name"]
    assert "approval-bootstrap" not in workflow["jobs"]
    assert gate.CHECK_NAME != "ownership-approvals"
    spec = yaml.safe_load((root / gate.AREAS).read_text())
    active = spec["meta"]["delegated_policy_reviews"]
    rules = parse_codeowners((root / "CODEOWNERS").read_text())
    assert resolve_owners(rules, gate.AREAS) == ([] if active else [OPS, PROCESS])
    assert resolve_owners(rules, "CODEOWNERS") == ([] if active else [PROCESS])


@pytest.mark.parametrize("newer_check", [False, True])
def test_concurrent_evaluation_cannot_restore_a_withdrawn_approval(
    monkeypatch, newer_check
):
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
        if "/compare/" in endpoint:
            return {"files": [{"filename": "CODEOWNERS"}]}
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
    gate.evaluate_one("org/repo", 1)
    assert published == ([None] if newer_check else [None, "failure"])


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
    checks = rules[0]["parameters"]["required_status_checks"]
    checks[0]["context"] = "ownership-approvals"
    with pytest.raises(ValueError, match="Before enabling delegation"):
        gate.check_activation("org/repo")
    checks[0]["context"] = gate.CHECK_NAME
    rules[0]["parameters"]["required_status_checks"][0]["integration_id"] = None
    with pytest.raises(ValueError, match="Before enabling delegation"):
        gate.check_activation("org/repo")


def test_dispatch_targets_one_pr_and_review_events_use_live_source_branch(monkeypatch):
    assert gate.discover("org/repo", {"inputs": {"pr": "42"}}) == [42]
    with pytest.raises(ValueError, match="positive PR"):
        gate.discover("org/repo", {"inputs": {"pr": "0"}})
    run = {
        "workflow_id": 17,
        "event": "pull_request_review",
        "pull_requests": [],
        "head_sha": "old-head",
        "head_branch": "feature/x?look=1",
        "head_repository": {"id": 2, "owner": {"login": "fork"}},
    }
    calls = []

    def fake_api(endpoint, **kwargs):
        calls.append(endpoint)
        if endpoint.endswith("/actions/workflows/codeowners.yml"):
            return {"id": 17}
        assert "head=fork%3Afeature%2Fx%3Flook%3D1" in endpoint
        head = {"ref": run["head_branch"], "sha": "new-head"}
        return [
            {"number": i, "head": dict(head, repo={"id": repo_id})}
            for i, repo_id in [(42, 2), (99, 3)]
        ]

    monkeypatch.setattr(gate, "api", fake_api)
    assert gate.discover("org/repo", {"workflow_run": run}) == [42]
    assert len(calls) == 2
    run["workflow_id"] = 18
    assert gate.discover("org/repo", {"workflow_run": run}) == []


@pytest.mark.parametrize("policy,tree", [(False, False), (True, False), (False, True)])
def test_main_refreshes_only_policy_prs_when_policy_or_tree_changes(
    monkeypatch, policy, tree
):
    def git(*args):
        if "--diff-filter=ADRT" in args:
            return "new.py" if tree else ""
        return "CODEOWNERS\0" if policy and "diff" in args else ""

    monkeypatch.setattr(gate, "command", git)
    monkeypatch.setattr(gate, "open_policies", lambda repo: [42])
    assert gate.discover("org/repo", {"before": "old-base"}) == (
        [42] if policy or tree else []
    )


def test_policy_discovery_checks_large_diffs_and_renames(monkeypatch):
    def node(number, path, kind="MODIFIED", more=False):
        return {
            "number": number,
            "files": {
                "nodes": [{"path": path, "changeType": kind}],
                "pageInfo": {"hasNextPage": more},
            },
        }

    def api(endpoint, **kwargs):
        if endpoint == "graphql":
            nodes = [
                node(1, "source.py"),
                node(2, "CODEOWNERS"),
                node(3, "source.py", more=True),
                node(4, "renamed.yml", "RENAMED"),
            ]
            return {
                "data": {
                    "repository": {
                        "pullRequests": {
                            "nodes": nodes,
                            "pageInfo": {"hasNextPage": False},
                        }
                    }
                }
            }
        raise AssertionError(endpoint)

    monkeypatch.setattr(gate, "api", api)
    assert gate.open_policies("org/repo") == [2, 3, 4]


@pytest.mark.parametrize("large", [False, True])
def test_noop_check_uses_captured_commits_and_full_evaluation_at_api_cap(
    monkeypatch, large
):
    pr = {
        "state": "open",
        "head": {"sha": "head"},
        "base": {"ref": "main", "sha": "base"},
    }
    published = []

    def api(endpoint, **kwargs):
        if endpoint.endswith("/pulls/1"):
            return pr
        if endpoint.endswith("/compare/base...head"):
            return {
                "files": [{"filename": "source.py"}] * 300
                if large
                else [{"filename": "CODEOWNERS"}]
            }
        if "/reviews?" in endpoint:
            return []
        if "/check-runs?" in endpoint:
            return {"check_runs": [{"id": 42, "app": {"id": 15368}}]}
        raise AssertionError(endpoint)

    def evaluate(*args):
        raise ValueError("Missing affected-owner approvals")

    def publish(*args, **kwargs):
        published.append(kwargs["conclusion"])
        return 42

    monkeypatch.setattr(gate, "api", api)
    monkeypatch.setattr(gate, "evaluate", evaluate)
    monkeypatch.setattr(gate, "check_run", publish)
    gate.evaluate_one("org/repo", 1)
    assert published == [None, "failure"]


@pytest.mark.parametrize(
    "glob,path", [("Cargo.toml", "Cargo.toml"), ("src/", "src/a.py")]
)
def test_rendered_shared_owner_uses_affected_groups_without_process(spec, glob, path):
    spec["areas"][0]["path_globs"] = [glob]
    spec["areas"][1]["path_globs"] = []
    changed = deepcopy(spec)
    changed["shared"] = [{"glob": glob, "owners": ["a", "b"]}]
    required, _ = required_approvals(
        rendered_policy(spec, []), rendered_policy(changed, []), [path]
    )
    assert required == {"@org/a", "@org/b"}


@pytest.mark.parametrize("external", ["", "contributors:", "{}", "contributors: []"])
def test_empty_external_policy_matches_generator_semantics(monkeypatch, external):
    monkeypatch.setattr(
        gate, "blob", lambda ref, path: "{}" if path == gate.AREAS else external
    )
    assert gate.inputs("head") == ({}, [])


@pytest.mark.parametrize(
    "external", ["[]", "false", "42", "contributors: text", "contributors: [null]"]
)
def test_malformed_external_policy_fails_explicitly(monkeypatch, external):
    monkeypatch.setattr(
        gate, "blob", lambda ref, path: "{}" if path == gate.AREAS else external
    )
    with pytest.raises(ValueError):
        gate.inputs("head")


def test_unwitnessed_scope_reordering_requires_process_even_for_same_owners():
    old = "/src/*/left.py @a\n/src/new/*.py @b\n/exact @a"
    new = "/src/new/*.py @b\n/src/*/left.py @a\n/exact @b"
    required, _ = required_approvals(
        old, new, ["src/old/left.py", "src/new/right.py", "exact"]
    )
    assert required == {"@a", "@b", PROCESS}


@pytest.mark.parametrize(
    "mode", ["once", "always", "head", "read_error", "review_error", "malformed"]
)
def test_snapshot_retry_is_bounded_and_postevaluation_errors_terminal_fail(
    monkeypatch, mode
):
    initial = {
        "state": "open",
        "head": {"sha": "head"},
        "base": {"ref": "main", "sha": "base0"},
    }
    reads, review_reads = 0, 0
    evaluated, published = [], []

    def api(endpoint, **kwargs):
        nonlocal reads, review_reads
        if endpoint.endswith("/pulls/1"):
            reads += 1
            if reads > 1 and mode == "read_error":
                raise OSError("temporary read failure")
            base = (
                f"base{reads-1}"
                if mode == "always"
                else ("base1" if reads > 1 and mode == "once" else "base0")
            )
            return dict(
                initial,
                head={"sha": "new" if reads > 1 and mode == "head" else "head"},
                base={"ref": "main", "sha": base},
            )
        if "/compare/" in endpoint:
            return {"files": [{"filename": "CODEOWNERS"}]}
        if "/reviews?" in endpoint:
            review_reads += 1
            if review_reads > 1 and mode == "review_error":
                raise OSError("review read failure")
            return []
        if "/check-runs?" in endpoint:
            return {"check_runs": [{"id": 42, "app": {"id": 15368}}]}
        raise AssertionError(endpoint)

    def evaluate(repo, pr, reviews):
        evaluated.append(pr["base"]["sha"])
        if mode == "malformed":
            return rendered_policy({"meta": None}, [])
        return "approved"

    def publish(*args, **kwargs):
        published.append(kwargs)
        return 42

    monkeypatch.setattr(gate, "api", api)
    monkeypatch.setattr(gate, "evaluate", evaluate)
    monkeypatch.setattr(gate, "check_run", publish)
    gate.evaluate_one("org/repo", 1)
    if mode == "head":
        assert [p["conclusion"] for p in published] == [None]
    else:
        assert published[-1]["conclusion"] == (
            "success" if mode == "once" else "failure"
        )
        assert published[-1]["check_id"] == 42
    assert evaluated == (
        ["base0", "base1", "base2"]
        if mode == "always"
        else ["base0", "base1"]
        if mode == "once"
        else ["base0"]
    )


@pytest.mark.parametrize("failure", ["lookup", "publication", "malformed"])
def test_one_pr_api_failure_does_not_skip_remaining_prs(monkeypatch, tmp_path, failure):
    event = tmp_path / "event.json"
    event.write_text("{}")
    monkeypatch.setenv("GITHUB_EVENT_PATH", str(event))
    monkeypatch.setenv("GITHUB_REPOSITORY", "org/repo")
    monkeypatch.setattr(sys, "argv", ["check"])
    monkeypatch.setattr(gate, "discover", lambda *a: [1, 2])
    completed = []

    def api(endpoint, **kwargs):
        if "/pulls/" in endpoint:
            number = int(endpoint.rsplit("/", 1)[1])
            if number == 1 and failure == "lookup":
                raise subprocess.CalledProcessError(1, "gh api", stderr="502")
            return {
                "state": "open",
                "head": {"sha": str(number)},
                "base": {"sha": "base", "ref": "main"},
            }
        if "/compare/" in endpoint:
            return {
                "files": [{"filename": "CODEOWNERS"}]
                if failure == "malformed" and endpoint.endswith("...1")
                else []
            }
        if "/reviews?" in endpoint:
            return []
        if "/check-runs?" in endpoint:
            return {"check_runs": [{"id": 42, "app": {"id": 15368}}]}
        raise AssertionError(endpoint)

    def publish(repo, head, **kwargs):
        if kwargs["conclusion"] is not None:
            if head == "1" and failure == "publication":
                raise OSError("publication failed")
            completed.append((head, kwargs["conclusion"]))
        return 42

    monkeypatch.setattr(gate, "api", api)
    monkeypatch.setattr(gate, "check_run", publish)
    monkeypatch.setattr(
        gate, "evaluate", lambda *args: rendered_policy({"meta": None}, [])
    )
    if failure == "malformed":
        gate.main()
        assert completed == [("1", "failure"), ("2", "success")]
    else:
        with pytest.raises(SystemExit, match=r"Retry PRs individually: \[1\]"):
            gate.main()
        assert completed == [("2", "success")]
