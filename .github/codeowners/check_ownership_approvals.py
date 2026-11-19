"""Trusted workflow entry point. Read PR blobs as data; never execute PR code."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import yaml
from ownership_approvals import (
    PROCESS,
    approved_reviewers,
    enforcement_changed,
    governance,
    missing_approvals,
    policy_fingerprint,
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


def changed_paths(repo: str, number: int) -> list[str]:
    return [
        path
        for f in pages(f"repos/{repo}/pulls/{number}/files")
        for path in (f["filename"], f.get("previous_filename", ""))
        if path
    ]


def open_policies(repo: str) -> list[dict]:
    owner, name = repo.split("/")
    query = """query($owner:String!,$name:String!,$cursor:String) {
      repository(owner:$owner,name:$name) {
        pullRequests(states:OPEN,baseRefName:"main",first:100,after:$cursor) {
          nodes { number headRefOid files(first:100) { nodes { path changeType } pageInfo { hasNextPage } } }
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
            if pr["files"]["pageInfo"]["hasNextPage"] or any(
                f["changeType"] == "RENAMED" for f in pr["files"]["nodes"]
            ):
                paths = changed_paths(repo, pr["number"])
            result.append(
                {
                    "pr": pr["number"],
                    "head": pr["headRefOid"],
                    "policy": any(map(policy_path, paths)),
                }
            )
        if not connection["pageInfo"]["hasNextPage"]:
            return result
        cursor = connection["pageInfo"]["endCursor"]


def batches(prs: list[dict]) -> list[dict]:
    count = min(8, len(prs))
    return [{"key": f"refresh-{i}", "prs": prs[i::count]} for i in range(count)]


def discover(repo: str, event: dict) -> list[dict]:
    run = event.get("workflow_run")
    if run:
        if run["name"] != "codeowners" or run["event"] not in {
            "pull_request",
            "pull_request_review",
        }:
            return []
        workflow = api(f"repos/{repo}/actions/workflows/codeowners.yml")
        if run["workflow_id"] != workflow["id"]:
            return []
        numbers = {p["number"] for p in run.get("pull_requests", [])}
        opened = (
            [api(f"repos/{repo}/pulls/{n}") for n in numbers]
            if numbers
            else pages(f"repos/{repo}/pulls?state=open&base=main")
        )
        opened = [
            p
            for p in opened
            if p["number"] in numbers
            or p["head"]["sha"] == run["head_sha"]
            or (
                p["head"]["ref"] == run["head_branch"]
                and p["head"]["repo"]["id"] == run["head_repository"]["id"]
            )
        ]
        tasks = []
        for pr in opened:
            if pr["state"] == "open" and pr["base"]["ref"] == "main":
                paths = changed_paths(repo, pr["number"])
                tasks.append(
                    {
                        "pr": pr["number"],
                        "head": pr["head"]["sha"],
                        "policy": any(map(policy_path, paths)),
                    }
                )
        return [{"key": p["head"], "prs": [p]} for p in tasks]
    seed = "before" not in event
    if "before" in event:
        before = event["before"]
        command("git", "fetch", "--no-tags", "--depth=1", "origin", before)
        changed = command("git", "diff", "--name-only", "-z", before, "HEAD").split(
            "\0"
        )
        tree_changed = command(
            "git", "diff", "--name-only", "--diff-filter=ADRT", before, "HEAD"
        ).strip()
        if not any(map(policy_path, changed)) and not tree_changed:
            return []
        seed = (
            ".github/workflows/codeowners-approval.yml"
            not in command("git", "ls-tree", "-r", "--name-only", before).splitlines()
        )
    tasks = open_policies(repo)
    return batches([p for p in tasks if seed or p["policy"]])


def blob(ref: str, path: str) -> str:
    return command("git", "show", f"{ref}:{path}")


def inputs(ref: str) -> tuple[dict, list[dict]]:
    spec = yaml.safe_load(blob(ref, AREAS))
    external = yaml.safe_load(blob(ref, EXTERNAL)).get("contributors") or []
    if not isinstance(spec, dict) or not isinstance(external, list):
        raise ValueError("Malformed ownership policy")
    return spec, external


def merge_tree(base: str, head: str, *, tolerate_conflicts: bool = False) -> str:
    try:
        return command("git", "merge-tree", "--write-tree", base, head).splitlines()[0]
    except subprocess.CalledProcessError as error:
        tree = (error.stdout or "").splitlines()[0] if error.stdout else ""
        if (
            not tolerate_conflicts
            or error.returncode != 1
            or not re.fullmatch(r"[a-f0-9]{40,64}", tree)
        ):
            raise
        # Conflicts outside policy/enforcement inputs do not invalidate a policy
        # approval. Conflicted inputs themselves fail parsing or the fingerprint.
        return tree


def fingerprint(ref: str, protected: bool) -> str:
    spec, external = inputs(ref)
    files = None
    if protected:
        files = {}
        for entry in command("git", "ls-tree", "-rz", ref).split("\0"):
            if entry:
                metadata, path = entry.split("\t", 1)
                if enforcement_changed([path]) or path.startswith(".github/workflows/"):
                    files[path] = metadata.split()[2]
    return policy_fingerprint(spec, external, files)


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


def evaluate(
    repo: str, pr: dict, event: dict, reviews: list[dict] | None = None
) -> str:
    number, base, head = pr["number"], pr["base"]["sha"], pr["head"]["sha"]
    duplicates = [
        p
        for p in pages(f"repos/{repo}/commits/{head}/pulls")
        if p["state"] == "open"
        and p["base"]["ref"] == "main"
        and p["head"]["sha"] == head
    ]
    if not duplicates:
        duplicates = [
            p
            for p in pages(f"repos/{repo}/pulls?state=open&base=main")
            if p["head"]["sha"] == head
        ]
    if len(duplicates) != 1:
        raise ValueError(
            "This head SHA belongs to multiple open PRs; use distinct commits for independent approvals"
        )
    command("git", "fetch", "--no-tags", "origin", base, f"refs/pull/{number}/head")
    proposed = merge_tree(base, head)
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
    expected_fingerprint = fingerprint(proposed, protected)
    approvals = approved_reviewers(
        reviews
        if reviews is not None
        else pages(f"repos/{repo}/pulls/{number}/reviews"),
        pr["user"]["login"],
    )
    approved = set()
    fingerprints = {}
    for login, review in approvals.items():
        if not required:
            break
        commit = review["commit_id"]
        if commit not in fingerprints:
            try:
                command("git", "fetch", "--no-tags", "origin", commit)
                fingerprints[commit] = fingerprint(
                    merge_tree(base, commit, tolerate_conflicts=True), protected
                )
            except (
                subprocess.SubprocessError,
                ValueError,
                KeyError,
                TypeError,
                yaml.YAMLError,
                SystemExit,
            ):
                fingerprints[commit] = None
        if fingerprints[commit] == expected_fingerprint:
            approved.add(login)
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
            + "\n\nApprovals must cover the current ownership policy; unrelated commits do not invalidate them."
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


def evaluate_one(repo: str, task: dict, event: dict) -> None:
    number, head = task["pr"], task["head"]
    pr = api(f"repos/{repo}/pulls/{number}")
    if (
        pr["state"] != "open"
        or pr["base"]["ref"] != "main"
        or pr["head"]["sha"] != head
    ):
        return
    if not task["policy"]:
        # Discovery is an optimization, not authority to skip review. Compare
        # immutable commits again before issuing a lightweight result.
        comparison = api(f"repos/{repo}/compare/{pr['base']['sha']}...{head}")
        files = comparison["files"]
        if len(files) >= 300 or any(
            policy_path(path)
            for f in files
            for path in (f["filename"], f.get("previous_filename", ""))
        ):
            task = dict(task, policy=True)
    if not task["policy"]:
        api(
            f"repos/{repo}/check-runs",
            method="POST",
            body={
                "name": CHECK_NAME,
                "head_sha": head,
                "external_id": "delegated-codeowners-noop",
                "status": "completed",
                "conclusion": "success",
                "output": {
                    "title": "No ownership change",
                    "summary": "This PR changes no ownership policy or enforcement files. The separate required codeowners check validates structure.",
                },
            },
        )
        return
    check_id = check_run(
        repo, head, conclusion=None, summary="Checking ownership approvals."
    )
    base = pr["base"]["sha"]
    reviews = None
    try:
        reviews = pages(f"repos/{repo}/pulls/{number}/reviews")
        summary = evaluate(repo, pr, event, reviews=reviews)
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
    if current["head"]["sha"] != head or current["base"]["sha"] != base:
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
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--pr", type=int)
    parser.add_argument("--head")
    parser.add_argument("--batch")
    parser.add_argument("--check-activation", action="store_true")
    args = parser.parse_args()
    repo = os.environ["GITHUB_REPOSITORY"]
    if args.check_activation:
        check_activation(repo)
        return
    event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text())
    if args.discover:
        print(json.dumps(discover(repo, event)))
        return
    if args.batch:
        for task in json.loads(args.batch):
            evaluate_one(repo, task, event)
            if "workflow_run" not in event:
                time.sleep(1)
        return
    if not args.pr or not args.head:
        parser.error("--pr and --head are required")
    evaluate_one(repo, {"pr": args.pr, "head": args.head, "policy": True}, event)


if __name__ == "__main__":
    main()
