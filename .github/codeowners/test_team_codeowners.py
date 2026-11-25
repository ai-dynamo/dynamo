"""Native team-file generation and approval coverage."""

import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from codeowners_match import compute_resolution, parse_codeowners, resolve_owners
from emit_codeowners import _render_codeowners
from team_codeowners import (
    POLICY,
    PREFIX,
    PROCESS,
    render_team_files,
    team_path,
    validate,
)

pytestmark = [pytest.mark.pre_merge, pytest.mark.gpu_0]
TEAMS = {name: "@org/" + name for name in ("ops", "runtime", "docs", "operator", "ci")}
TEAMS["process"] = PROCESS


def generate(snapshot):
    config = yaml.safe_load(snapshot.get(POLICY[1], "{}")) or {}
    snapshot.update(render_team_files(snapshot["CODEOWNERS"], config))
    return snapshot


def changed_teams(before, after):
    return {
        p
        for p in before.keys() | after.keys()
        if p.startswith(PREFIX) and before.get(p) != after.get(p)
    }


@pytest.fixture
def policy():
    rules = (
        "* @org/runtime\n/.github/codeowners/ @org/ci\n/Cargo.toml @org/ops\n"
        "/other.toml @org/ops\n/docs/ @org/docs\n/src/ @org/ops\n*Dockerfile* @org/ci\n"
    )
    rules += "".join(f"/{team_path(owner)} {owner}\n" for owner in TEAMS.values())
    rules += "/CODEOWNERS\n/.github/codeowners/areas.yaml\n"
    return generate(
        {
            "CODEOWNERS": rules,
            POLICY[1]: "areas: []\n",
            ".github/codeowners/external_contributors.yaml": "contributors: []\n",
            "Cargo.toml": "",
            "other.toml": "",
            "docs/page.md": "",
            "src/a.py": "",
        }
    )


def proposal(before, old, new):
    after = deepcopy(before)
    after["CODEOWNERS"] = after["CODEOWNERS"].replace(old, new, 1)
    return generate(after)


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
def test_old_and_new_groups_have_readable_changed_entries(policy, old, new, groups):
    policy["CODEOWNERS"] = policy["CODEOWNERS"].replace(
        "/Cargo.toml @org/ops", "/Cargo.toml " + old
    )
    generate(policy)
    after = proposal(policy, "/Cargo.toml " + old, "/Cargo.toml " + new)
    assert changed_teams(policy, after) == {team_path(TEAMS[group]) for group in groups}
    for group in groups:
        path = team_path(TEAMS[group])
        assert "/Cargo.toml" in policy[path] or "/Cargo.toml" in after[path]
    validate(policy, after)
    after[team_path("@org/ops")] = policy[team_path("@org/ops")]
    with pytest.raises(ValueError, match="Regenerate"):
        validate(policy, after)


@pytest.mark.unit
@pytest.mark.parametrize(
    "attack", ["delete", "reroute", "ownerless", "unknown", "stale"]
)
def test_team_files_cannot_evade_native_review(policy, attack):
    after = proposal(
        policy, "/Cargo.toml @org/ops", "/Cargo.toml @org/ops @org/runtime"
    )
    path = team_path("@org/ops")
    if attack == "delete":
        del after[path]
    elif attack in {"reroute", "ownerless"}:
        after["CODEOWNERS"] = after["CODEOWNERS"].replace(
            f"/{path} @org/ops",
            f"/{path}" + (" @org/runtime" if attack == "reroute" else ""),
        )
    elif attack == "unknown":
        after["CODEOWNERS"] = after["CODEOWNERS"].replace(
            "@org/ops @org/runtime", "@org/unregistered"
        )
    else:
        after[path] = policy[path]
    with pytest.raises(ValueError):
        validate(policy, after)


@pytest.mark.unit
def test_normal_source_additions_do_not_change_team_files(policy):
    after = dict(policy, **{"src/new.py": "pass"})
    generate(after)
    assert changed_teams(policy, after) == set()
    validate(policy, after)
    assert resolve_owners(parse_codeowners(after["CODEOWNERS"]), "src/new.py") == [
        "@org/ops"
    ]


@pytest.mark.unit
def test_new_override_records_loss_in_old_owner_file(policy):
    after = deepcopy(policy)
    after["CODEOWNERS"] += "/docs/page.md @org/operator\n"
    generate(after)
    assert changed_teams(policy, after) == {
        team_path("@org/docs"),
        team_path("@org/operator"),
    }
    assert "/docs/page.md\n" in after[team_path("@org/docs")]
    assert "/docs/page.md @org/operator\n" in after[team_path("@org/operator")]
    validate(policy, after)


@pytest.mark.unit
def test_already_excluded_scope_does_not_request_original_owner_again(policy):
    policy["CODEOWNERS"] += "/src/private/ @org/runtime\n"
    policy["src/private/a.py"] = ""
    generate(policy)
    after = deepcopy(policy)
    after["CODEOWNERS"] += "/src/private/a.py @org/docs\n"
    generate(after)
    assert changed_teams(policy, after) == {
        team_path("@org/runtime"),
        team_path("@org/docs"),
    }
    validate(policy, after)


@pytest.mark.unit
def test_other_policy_file_keeps_its_native_owners(policy):
    after = deepcopy(policy)
    after["CODEOWNERS"] += "/.github/codeowners/external_contributors.yaml @org/docs\n"
    generate(after)
    assert changed_teams(policy, after) == {
        team_path("@org/ci"),
        team_path("@org/docs"),
    }
    validate(policy, after)


@pytest.mark.unit
def test_process_sees_readable_contract_changes(policy):
    policy[POLICY[1]] = yaml.safe_dump(
        {"required_owners": [{"glob": "src/", "owners": ["ops"]}]}
    )
    generate(policy)
    after = dict(policy, **{POLICY[1]: "{}"})
    generate(after)
    assert changed_teams(policy, after) == {team_path(PROCESS)}
    assert "# required_owners:" in policy[team_path(PROCESS)]
    validate(policy, after)


@pytest.mark.unit
def test_higher_priority_codeowners_is_rejected(policy):
    after = dict(policy, **{".github/CODEOWNERS": "* @org/runtime\n"})
    with pytest.raises(ValueError, match="higher-priority"):
        validate(policy, after)


@pytest.mark.unit
def test_generator_delegates_only_policy_and_registers_empty_teams():
    spec = {
        "areas": [
            {"label": "ops", "github_team": "@org/ops", "path_globs": [".github/"]},
            {"label": "process", "github_team": PROCESS, "path_globs": ["CODEOWNERS"]},
            {"label": "new", "github_team": "@org/new", "path_globs": []},
        ]
    }
    before = {
        "CODEOWNERS": "/.github/ @org/ops\n/CODEOWNERS " + PROCESS + "\n",
        POLICY[1]: yaml.safe_dump(spec),
    }
    lines, _ = _render_codeowners(compute_resolution(spec), group=True)
    after = generate({"CODEOWNERS": "\n".join(lines), POLICY[1]: yaml.safe_dump(spec)})
    validate(before, after)
    assert after[team_path("@org/new")] == ""
    owners = parse_codeowners(after["CODEOWNERS"])
    assert resolve_owners(owners, "CODEOWNERS") == []
    assert resolve_owners(owners, POLICY[1]) == []
    assert resolve_owners(owners, ".github/workflows/codeowners.yml") == ["@org/ops"]

    # GitHub uses the base rules when the next PR introduces a team file.
    new_path = team_path("@org/later")
    assert resolve_owners(owners, new_path) == [PROCESS]
    spec["areas"].append(
        {"label": "later", "github_team": "@org/later", "path_globs": []}
    )
    lines, _ = _render_codeowners(compute_resolution(spec), group=True)
    registered = generate(
        {"CODEOWNERS": "\n".join(lines), POLICY[1]: yaml.safe_dump(spec)}
    )
    validate(after, registered)
    assert registered[new_path] == ""
    assert changed_teams(after, registered) == {new_path}
    rules = parse_codeowners(registered["CODEOWNERS"])
    assert resolve_owners(rules, new_path) == ["@org/later"]
    assert resolve_owners(rules, team_path("@org/ops")) == ["@org/ops"]
    assert PREFIX + "*" not in registered

    bad = dict(after, **{"app.py": ""})
    bad["CODEOWNERS"] += "\n/app.py @org/new\n"
    generate(bad)
    with pytest.raises(ValueError, match="separately"):
        validate(before, bad)


@pytest.mark.unit
@pytest.mark.parametrize("owners", [[], ["@org/ops"], [PROCESS, "@org/ops"]])
def test_new_team_fallback_requires_process(owners):
    rules = f"/{PREFIX}* {' '.join(owners)}\n"
    with pytest.raises(ValueError, match="exact native owner"):
        render_team_files(rules, {})


@pytest.mark.unit
def test_new_team_must_be_registered_before_receiving_code(policy):
    after = deepcopy(policy)
    after["CODEOWNERS"] += f'/{team_path("@org/new")} @org/new\n/src/new/ @org/new\n'
    generate(after)
    with pytest.raises(ValueError, match="separately"):
        validate(policy, after)


@pytest.mark.unit
def test_owner_without_team_file_cannot_gain_paths(policy):
    after = proposal(
        policy, "/Cargo.toml @org/ops", "/Cargo.toml @org/ops @org/unregistered"
    )
    with pytest.raises(ValueError, match="must update"):
        validate(policy, after)


@pytest.mark.unit
def test_regenerated_policy_cannot_drop_a_team_file(policy):
    path = team_path("@org/docs")
    after = deepcopy(policy)
    after["CODEOWNERS"] = after["CODEOWNERS"].replace(f"/{path} @org/docs\n", "")
    del after[path]
    generate(after)
    with pytest.raises(ValueError, match="cannot be deleted"):
        validate(policy, after)


@pytest.mark.unit
@pytest.mark.parametrize("path", POLICY)
def test_regenerated_policy_cannot_own_delegated_paths(policy, path):
    after = deepcopy(policy)
    after["CODEOWNERS"] += f"/{path} @org/ops\n"
    generate(after)
    with pytest.raises(ValueError, match="ownerless"):
        validate(policy, after)


@pytest.mark.integration
@pytest.mark.parametrize("symlink", [False, True])
def test_generator_and_validator_use_real_git_trees(tmp_path, symlink):
    scripts = Path(__file__).parent.resolve()

    def run(*args):
        return subprocess.run(
            args, cwd=tmp_path, text=True, capture_output=True, timeout=10
        )

    def git(*args):
        result = run("git", *args)
        assert result.returncode == 0, result.stderr
        return result.stdout.strip()

    spec = {
        "areas": [
            {
                "label": "ops",
                "github_team": "@org/ops",
                "path_globs": ["Cargo.toml", ".github/"],
            },
            {"label": "runtime", "github_team": "@org/runtime", "path_globs": ["src/"]},
            {"label": "process", "github_team": PROCESS, "path_globs": ["CODEOWNERS"]},
        ],
    }
    areas = tmp_path / POLICY[1]
    areas.parent.mkdir(parents=True)
    areas.write_text(yaml.safe_dump(spec))
    (tmp_path / "Cargo.toml").write_text("")
    emit = [
        sys.executable,
        str(scripts / "emit_codeowners.py"),
        "--areas",
        POLICY[1],
        "--out",
        "CODEOWNERS",
    ]
    assert run(*emit).returncode == 0
    git("init", "-q", "-b", "team-codeowners-test")
    git("config", "user.name", "Ownership Test")
    git("config", "user.email", "test@example.com")
    git(
        "add",
        *[
            str(p.relative_to(tmp_path))
            for p in tmp_path.rglob("*")
            if p.is_file() and ".git" not in p.parts
        ],
    )
    git("commit", "-q", "-s", "-m", "base")
    base = git("rev-parse", "HEAD")
    spec["shared"] = [{"glob": "Cargo.toml", "owners": ["ops", "runtime"]}]
    areas.write_text(yaml.safe_dump(spec))
    outside = tmp_path / "outside"
    outside.write_text("unchanged")
    if symlink:
        target = tmp_path / team_path("@org/ops")
        target.unlink()
        target.symlink_to(outside)
    generated = run(*emit)
    check = [sys.executable, str(scripts / "team_codeowners.py"), "--base", base]
    if symlink:
        assert generated.returncode != 0
        assert run(*check).returncode != 0
        assert outside.read_text() == "unchanged"
    else:
        assert generated.returncode == 0, generated.stderr
        assert run(*check).returncode == 0
        assert (
            "/Cargo.toml @org/ops @org/runtime\n"
            in (tmp_path / team_path("@org/ops")).read_text()
        )
        (tmp_path / team_path("@org/ops")).write_text("")
        assert run(*check).returncode != 0
