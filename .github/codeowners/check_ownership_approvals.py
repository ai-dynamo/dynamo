"""Trusted workflow entry point. Read PR blobs as data; never execute PR code."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from urllib.parse import urlencode

import yaml
from ownership_approvals import (
    PROCESS,
    approved_reviewers,
    enforcement_changed,
    governance,
    missing_approvals,
    rendered_policy,
    required_approvals,
)

AREAS = ".github/codeowners/areas.yaml"
EXTERNAL = ".github/codeowners/external_contributors.yaml"
CHECK_NAME = "ownership-approvals"


def command(*args: str, **kwargs) -> str:
    return subprocess.run(
        args, check=True, capture_output=True, text=True, timeout=120, **kwargs
    ).stdout


def api(
    endpoint: str,
    *,
    method: str = "GET",
    body: dict | None = None,
    token: str | None = None,
):
    args = ["gh", "api", endpoint, "--method", method]
    if body is not None:
        args += ["--input", "-"]
    env = dict(os.environ)
    if token:
        env["GH_TOKEN"] = token
    return json.loads(
        command(*args, input=json.dumps(body) if body is not None else None, env=env)
    )


def pages(endpoint: str, *, key: str | None = None, token: str | None = None) -> list:
    result = []
    for page in range(1, 1001):
        sep = "&" if "?" in endpoint else "?"
        data = api(f"{endpoint}{sep}per_page=100&page={page}", token=token)
        batch = data[key] if key else data
        if not isinstance(batch, list):
            raise ValueError(f"Expected a paginated list from {endpoint}")
        result.extend(batch)
        if len(batch) < 100:
            return result
    raise ValueError(f"Pagination limit reached for {endpoint}")


def policy_path(path: str) -> bool:
    return path in {
        AREAS,
        EXTERNAL,
        "CODEOWNERS",
        ".github/CODEOWNERS",
    } or enforcement_changed([path])


def open_policies(repo: str) -> list[int]:
    owner, name = repo.split("/")
    query = """query($owner:String!,$name:String!,$cursor:String) {
      repository(owner:$owner,name:$name) {
        pullRequests(states:OPEN,baseRefName:"main",first:100,after:$cursor) {
          nodes { number files(first:100) { nodes { path changeType } pageInfo { hasNextPage } } }
          pageInfo { hasNextPage endCursor }
        }
      }
    }"""
    cursor = None
    result = []
    while True:
        response = api(
            "graphql",
            method="POST",
            body={
                "query": query,
                "variables": {"owner": owner, "name": name, "cursor": cursor},
            },
        )
        if response.get("errors"):
            raise ValueError(str(response["errors"]))
        connection = response["data"]["repository"]["pullRequests"]
        for pr in connection["nodes"]:
            paths = [f["path"] for f in pr["files"]["nodes"]]
            if (
                pr["files"]["pageInfo"]["hasNextPage"]
                or any(f["changeType"] == "RENAMED" for f in pr["files"]["nodes"])
                or any(map(policy_path, paths))
            ):
                result.append(pr["number"])
        if not connection["pageInfo"]["hasNextPage"]:
            return result
        cursor = connection["pageInfo"]["endCursor"]


def discover(repo: str, event: dict) -> list[int]:
    if "inputs" in event:
        number = int(event["inputs"]["pr"])
        if number < 1:
            raise ValueError("A positive PR number is required")
        return [number]
    run = event.get("workflow_run")
    if run:
        if run["event"] not in {"pull_request", "pull_request_review"}:
            return []
        workflow = api(f"repos/{repo}/actions/workflows/codeowners.yml")
        if run["workflow_id"] != workflow["id"]:
            return []
        if run["pull_requests"]:
            return sorted({p["number"] for p in run["pull_requests"]})

        # Fork review events may omit pull_requests; query only the source branch.
        source = run["head_repository"]
        query = urlencode(
            {
                "state": "open",
                "base": "main",
                "head": f"{source['owner']['login']}:{run['head_branch']}",
            }
        )
        return [
            p["number"]
            for p in pages(f"repos/{repo}/pulls?{query}")
            if p["head"]["repo"]["id"] == source["id"]
            and p["head"]["ref"] == run["head_branch"]
        ]
    before = event["before"]
    command("git", "fetch", "--no-tags", "origin", before)
    changed = command("git", "diff", "--name-only", "-z", before, "HEAD").split("\0")
    tree_changed = command(
        "git", "diff", "--name-only", "--diff-filter=ADRT", before, "HEAD"
    ).strip()
    return open_policies(repo) if tree_changed or any(map(policy_path, changed)) else []


def blob(ref: str, path: str) -> str:
    return command("git", "show", f"{ref}:{path}")


def inputs(ref: str) -> tuple[dict, list[dict]]:
    spec = yaml.safe_load(blob(ref, AREAS))
    external = yaml.safe_load(blob(ref, EXTERNAL)).get("contributors") or []
    if not isinstance(spec, dict) or not isinstance(external, list):
        raise ValueError("Malformed ownership policy")
    return spec, external


def check_run(
    repo: str,
    head: str,
    *,
    conclusion: str | None,
    summary: str,
    check_id: int | None = None,
) -> int:
    body = {
        "name": CHECK_NAME,
        "head_sha": head,
        "external_id": "delegated-codeowners",
        "status": "completed" if conclusion else "in_progress",
        "output": {"title": "Ownership policy approvals", "summary": summary[:65000]},
    }
    if conclusion:
        body["conclusion"] = conclusion
    endpoint = f"repos/{repo}/check-runs"
    if check_id is not None:
        del body["head_sha"]
        return api(f"{endpoint}/{check_id}", method="PATCH", body=body)["id"]
    else:
        return api(endpoint, method="POST", body=body)["id"]


def evaluate(repo: str, pr: dict, reviews: list[dict]) -> str:
    number, base, head = pr["number"], pr["base"]["sha"], pr["head"]["sha"]
    duplicates = [
        p
        for p in pages(f"repos/{repo}/commits/{head}/pulls")
        if p["state"] == "open"
        and p["base"]["ref"] == "main"
        and p["head"]["sha"] == head
    ]
    if [p["number"] for p in duplicates] != [number]:
        raise ValueError(
            "Policy PRs must have a unique head SHA for commit-scoped checks"
        )
    command("git", "fetch", "--no-tags", "origin", base, f"refs/pull/{number}/head")
    proposed = command("git", "merge-tree", "--write-tree", base, head).splitlines()[0]
    # Read both snapshots without checking out code from the PR.
    before_paths = set(
        command("git", "ls-tree", "-rz", "--name-only", base).split("\0")
    ) - {""}
    proposed_paths = set(
        command("git", "ls-tree", "-rz", "--name-only", proposed).split("\0")
    ) - {""}
    paths = sorted(before_paths | proposed_paths)
    changed = (
        command("git", "diff", "--no-renames", "--name-only", "-z", base, proposed)
        .rstrip("\0")
        .split("\0")
    )
    if ".github/CODEOWNERS" in paths:
        raise ValueError(
            "A higher-priority .github/CODEOWNERS would bypass the root ownership policy"
        )
    if not ({AREAS, EXTERNAL, "CODEOWNERS"} & set(changed)) and not enforcement_changed(
        changed
    ):
        return "Ownership structure/tests passed; no ownership policy change."
    old_spec, old_external = inputs(base)
    new_spec, new_external = inputs(proposed)
    if new_spec.get("meta", {}).get("delegated_policy_reviews", False):
        essential = {
            ".github/workflows/codeowners.yml",
            ".github/workflows/codeowners-approval.yml",
            ".github/codeowners/check_ownership_approvals.py",
            ".github/codeowners/ownership_approvals.py",
        }
        if not essential <= proposed_paths:
            raise ValueError(
                "Disable delegation before removing or renaming its enforcement entry points"
            )
    protected = enforcement_changed(changed)
    before, after = blob(base, "CODEOWNERS"), blob(proposed, "CODEOWNERS")
    if not protected and after != rendered_policy(new_spec, new_external):
        raise ValueError(
            "CODEOWNERS differs from trusted generation; regenerate it from the policy inputs"
        )
    required, changes = required_approvals(before, after, paths, protected=protected)
    if governance(old_spec) != governance(new_spec):
        required.add(PROCESS)
    approved = approved_reviewers(reviews, pr["user"]["login"], head)
    members = {}
    for principal in sorted(required):
        if "/" not in principal:
            continue
        org, team = principal[1:].split("/", 1)
        roster = pages(
            f"orgs/{org}/teams/{team}/members", token=os.environ.get("ORG_READ_TOKEN")
        )
        if not roster:
            raise ValueError(
                f"No readable members for {principal}; configure ORG_READ_TOKEN with org members:read"
            )
        members[principal] = {u["login"].lower() for u in roster}
    missing = missing_approvals(required, approved, members)
    summary = [
        "## Required ownership approvals",
        "",
        "One reviewer may represent several groups.",
        "",
    ]
    summary += [f"- {'❌' if p in missing else '✅'} {p}" for p in sorted(required)]
    if not required:
        summary.append("No effective ownership change requires an additional approval.")
    summary += ["", "| Path | Before | After |", "| --- | --- | --- |"]
    summary += [
        f"| `{c.path}` | {', '.join(c.before) or 'none'} | {', '.join(c.after) or 'none'} |"
        for c in changes[:200]
    ]
    if len(changes) > 200:
        summary.append(f"\n{len(changes)-200} additional changed paths omitted.")
    result = "\n".join(summary)
    if missing:
        raise ValueError(
            result
            + "\n\nApprovals must reference the current head commit; every push requires fresh approvals."
        )
    return result


def check_activation(repo: str) -> None:
    spec = yaml.safe_load(Path(AREAS).read_text())
    if not spec.get("meta", {}).get("delegated_policy_reviews", False):
        return
    rules = api(f"repos/{repo}/rules/branches/main")
    required = [
        check
        for rule in rules
        if rule["type"] == "required_status_checks"
        for check in rule["parameters"]["required_status_checks"]
    ]
    for name in ("codeowners", CHECK_NAME):
        if not any(
            c["context"] == name and c.get("integration_id") == 15368 for c in required
        ):
            raise ValueError(
                f"Before enabling delegation, require {name!r} from GitHub Actions (15368) in the main ruleset"
            )


def review_version(reviews: list[dict]) -> list[tuple]:
    return sorted(
        (r["id"], r["state"], r["commit_id"], r["user"]["login"], r["user"]["type"])
        for r in reviews
        if r["state"] in {"APPROVED", "CHANGES_REQUESTED", "DISMISSED"}
    )


def evaluate_one(repo: str, number: int) -> None:
    pr = api(f"repos/{repo}/pulls/{number}")
    if pr["state"] != "open" or pr["base"]["ref"] != "main":
        return
    head = pr["head"]["sha"]
    check_id = check_run(
        repo, head, conclusion=None, summary="Checking ownership approvals."
    )
    reviews = None
    try:
        comparison = api(f"repos/{repo}/compare/{pr['base']['sha']}...{head}")
        files = comparison["files"]
        paths = [
            p for f in files for p in (f["filename"], f.get("previous_filename", ""))
        ]
        # The compare API caps files at 300; a full evaluation handles larger diffs.
        if len(files) >= 300 or any(map(policy_path, paths)):
            reviews = pages(f"repos/{repo}/pulls/{number}/reviews")
            summary = evaluate(repo, pr, reviews)
        else:
            summary = (
                "No ownership change; native GitHub CODEOWNERS reviews still apply."
            )
        conclusion = "success"
    except (
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
        yaml.YAMLError,
        SystemExit,
    ) as error:
        summary = str(error)
        conclusion = "failure"
    current = api(f"repos/{repo}/pulls/{number}")
    if (
        current["head"]["sha"],
        current["base"]["sha"],
        current["base"]["ref"],
        current["state"],
    ) != (head, pr["base"]["sha"], "main", "open"):
        return
    if reviews is not None and review_version(reviews) != review_version(
        pages(f"repos/{repo}/pulls/{number}/reviews")
    ):
        conclusion, summary = (
            "failure",
            "Reviews changed during evaluation; awaiting a fresh evaluation.",
        )
    checks = pages(
        f"repos/{repo}/commits/{head}/check-runs?check_name={CHECK_NAME}&filter=all",
        key="check_runs",
    )
    own = [c for c in checks if c["app"]["id"] == 15368]
    if not own or max(c["id"] for c in own) != check_id:
        return
    check_run(repo, head, conclusion=conclusion, summary=summary, check_id=check_id)
    print(f"PR #{number}: {summary}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-activation", action="store_true")
    args = parser.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    if args.check_activation:
        check_activation(repo)
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    for number in discover(repo, event):
        evaluate_one(repo, number)


if __name__ == "__main__":
    main()
