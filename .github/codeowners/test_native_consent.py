"""Native consent generation, routing and validation."""

import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from codeowners_match import compute_resolution, parse_codeowners, resolve_owners
from emit_codeowners import _render_codeowners
from native_consent import DIRECTORY, POLICY, PROCESS, plan, receipt_path, validate

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0]


@pytest.fixture
def policy():
    teams = {
        "ops": "@org/ops",
        "runtime": "@org/runtime",
        "docs": "@org/docs",
        "operator": "@org/operator",
        "process": PROCESS,
        "ci": "@org/ci",
    }
    gap = "# independent rule section\n" * 12
    rules = (
        "* @org/runtime\n/.github/codeowners/ @org/ci\n/Cargo.toml @org/ops\n"
        + gap
        + "/other.toml @org/ops\n"
        + gap
        + "/docs/ @org/docs\n/src/ @org/ops\n*Dockerfile* @org/ci\n"
    )
    rules += "".join(f"/{DIRECTORY}{name} {owner}\n" for name, owner in teams.items())
    rules += "/CODEOWNERS\n/.github/codeowners/areas.yaml\n"
    return {
        "CODEOWNERS": rules,
        POLICY[1]: "areas: []\n",
        POLICY[2]: "contributors: []\n",
        "Cargo.toml": "",
        "other.toml": "",
        "docs/page.md": "",
        "src/a.py": "",
        **{DIRECTORY + name: "initial\n" for name in teams},
    }


def proposal(before, old, new):
    after = deepcopy(before)
    after["CODEOWNERS"] = after["CODEOWNERS"].replace(old, new, 1)
    expected = plan(before, after)
    after.update(expected)
    return after, expected


@pytest.mark.unit
@pytest.mark.parametrize(
    "old,new,groups",
    [
        ("@org/ops", "@org/ops @org/runtime", {"ops", "runtime"}),
        ("@org/ops", "@org/runtime", {"ops", "runtime"}),
        ("@org/ops @org/runtime", "@org/runtime", {"ops", "runtime"}),
        ("@org/ops @org/runtime", "@org/ops @org/docs", {"ops", "runtime", "docs"}),
    ],
)
def test_old_and_new_groups_are_native_file_owners(policy, old, new, groups):
    policy["CODEOWNERS"] = policy["CODEOWNERS"].replace(
        "/Cargo.toml @org/ops", "/Cargo.toml " + old
    )
    after, expected = proposal(policy, "/Cargo.toml " + old, "/Cargo.toml " + new)
    assert set(expected) == {DIRECTORY + group for group in groups}
    assert {
        owner
        for path in expected
        for owner in resolve_owners(parse_codeowners(policy["CODEOWNERS"]), path)
    } == {"@org/" + group for group in groups}
    assert validate(policy, after) == expected
    after[DIRECTORY + "ops"] = "initial\n"
    with pytest.raises(ValueError, match="Regenerate"):
        validate(policy, after)


@pytest.mark.unit
@pytest.mark.parametrize(
    "attack", ["delete", "reroute", "ownerless", "unknown", "old-marker"]
)
def test_receipts_cannot_evade_native_review(policy, attack):
    after, _ = proposal(
        policy, "/Cargo.toml @org/ops", "/Cargo.toml @org/ops @org/runtime"
    )
    if attack == "delete":
        del after[DIRECTORY + "ops"]
    elif attack in {"reroute", "ownerless"}:
        after["CODEOWNERS"] = after["CODEOWNERS"].replace(
            f"/{DIRECTORY}ops @org/ops",
            f"/{DIRECTORY}ops" + (" @org/runtime" if attack == "reroute" else ""),
        )
    elif attack == "unknown":
        after["CODEOWNERS"] = after["CODEOWNERS"].replace(
            "/Cargo.toml @org/ops @org/runtime", "/Cargo.toml @org/unregistered"
        )
    else:
        after["CODEOWNERS"] = after["CODEOWNERS"].replace(
            "/Cargo.toml @org/ops @org/runtime", "/Cargo.toml @org/runtime"
        )
    with pytest.raises(ValueError):
        validate(policy, after)


@pytest.mark.unit
def test_ordinary_code_and_two_policy_exceptions(policy):
    after = dict(policy, **{"src/new.py": "pass"})
    assert validate(policy, after) == {}
    rules = parse_codeowners(after["CODEOWNERS"])
    assert resolve_owners(rules, "CODEOWNERS") == []
    assert resolve_owners(rules, POLICY[1]) == []
    assert resolve_owners(rules, "src/new.py") == ["@org/ops"]


@pytest.mark.unit
def test_other_policy_inputs_still_require_their_old_and_new_owners(policy):
    after = dict(policy)
    after["CODEOWNERS"] += f"/{POLICY[2]} @org/docs\n"
    expected = plan(policy, after)
    assert set(expected) == {DIRECTORY + "ci", DIRECTORY + "docs"}


@pytest.mark.unit
def test_shadowed_broad_owner_change_still_requires_governance(policy):
    before = dict(policy)
    before["CODEOWNERS"] += "/src/a.py @org/runtime\n"
    after = dict(before)
    after["CODEOWNERS"] = after["CODEOWNERS"].replace(
        "/src/ @org/ops", "/src/ @org/docs"
    )
    assert set(plan(before, after)) == {DIRECTORY + "process"}


@pytest.mark.unit
def test_higher_priority_codeowners_cannot_bypass_the_unchanged_policy(policy):
    after = dict(policy, **{".github/CODEOWNERS": "* @org/runtime\n"})
    with pytest.raises(ValueError, match="higher-priority"):
        validate(policy, after)


@pytest.mark.unit
def test_exact_docs_override_uses_docs_and_operator_without_process(policy):
    after = dict(policy)
    after["CODEOWNERS"] += "/docs/page.md @org/operator\n"
    assert set(plan(policy, after)) == {
        DIRECTORY + "docs",
        DIRECTORY + "operator",
    }


@pytest.mark.unit
def test_installer_registers_owners_before_delegating_policy():
    spec = {
        "meta": {"native_policy_consents": False},
        "areas": [
            {"label": "ops", "github_team": "@org/ops", "path_globs": [".github/"]},
            {"label": "process", "github_team": PROCESS, "path_globs": ["CODEOWNERS"]},
        ],
    }
    lines, _ = _render_codeowners(compute_resolution(spec), group=True)
    installer = "\n".join(lines)
    owners = parse_codeowners(installer)
    assert resolve_owners(owners, "CODEOWNERS") == [PROCESS]
    assert resolve_owners(owners, POLICY[1]) == ["@org/ops"]
    assert resolve_owners(owners, receipt_path("@org/ops")) == ["@org/ops"]
    before = {
        "CODEOWNERS": "/CODEOWNERS " + PROCESS + "\n/.github/ @org/ops\n",
        POLICY[1]: "{}",
    }
    installed = {
        "CODEOWNERS": installer,
        POLICY[1]: "meta: {native_policy_consents: false}",
        receipt_path("@org/ops"): "registered\n",
        receipt_path(PROCESS): "registered\n",
    }
    assert validate(before, installed) == {}
    missing = dict(installed)
    del missing[receipt_path("@org/ops")]
    with pytest.raises(ValueError, match="Commit every registered"):
        validate(before, missing)
    bad = dict(installed, **{"app.py": ""})
    bad["CODEOWNERS"] += "\n/app.py @org/new\n"
    with pytest.raises(ValueError, match="separately"):
        validate(before, bad)
    spec["meta"]["native_policy_consents"] = True
    active, _ = _render_codeowners(compute_resolution(spec), group=True)
    active_owners = parse_codeowners("\n".join(active))
    assert resolve_owners(active_owners, "CODEOWNERS") == []
    assert resolve_owners(active_owners, POLICY[1]) == []
    assert resolve_owners(active_owners, ".github/workflows/codeowners.yml") == [
        "@org/ops"
    ]


@pytest.mark.integration
@pytest.mark.timeout(30)
@pytest.mark.parametrize("symlink", [False, True])
def test_cli_generates_and_checks_real_receipts(policy, tmp_path, symlink):
    script = Path(__file__).with_name("native_consent.py").resolve()

    def git(*args):
        return subprocess.check_output(
            ["git", *args], cwd=tmp_path, text=True, timeout=10
        ).strip()

    for path, value in policy.items():
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(value)
    git("init", "-q", "-b", "consent-test")
    git("config", "user.name", "Consent Test")
    git("config", "user.email", "test@example.com")
    git("add", *policy)
    git("commit", "-q", "-s", "-m", "base")
    base = git("rev-parse", "HEAD")
    (tmp_path / "CODEOWNERS").write_text(
        policy["CODEOWNERS"].replace(
            "/Cargo.toml @org/ops", "/Cargo.toml @org/ops @org/runtime"
        )
    )
    outside = tmp_path / "outside"
    outside.write_text("unchanged")
    if symlink:
        receipt = tmp_path / (DIRECTORY + "ops")
        receipt.unlink()
        receipt.symlink_to(outside)

    def run(*args):
        return subprocess.run(
            [sys.executable, str(script), "--base", base, *args],
            cwd=tmp_path,
            text=True,
            capture_output=True,
            timeout=10,
        )

    assert run().returncode != 0
    written = run("--write")
    if symlink:
        assert written.returncode != 0
        assert outside.read_text() == "unchanged"
    else:
        assert written.returncode == 0, written.stderr
        assert run().returncode == 0
        assert (tmp_path / (DIRECTORY + "ops")).read_text() != "initial\n"
