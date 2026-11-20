"""Generate team CODEOWNERS files and verify their approval coverage."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

import yaml
from codeowners_match import DELEGATED, match, parse_codeowners, resolve_owners

PREFIX = ".github/codeowners/CODEOWNERS."
POLICY = ("CODEOWNERS", ".github/codeowners/areas.yaml")
PROCESS = "@ai-dynamo/dynamo-process-codeowners"


def team_path(owner: str) -> str:
    if not re.fullmatch(
        r"@[A-Za-z0-9][A-Za-z0-9_-]*(?:/[A-Za-z0-9][A-Za-z0-9_-]*)?", owner
    ):
        raise ValueError(f"Invalid native owner: {owner!r}")
    name = owner[1:].replace("/", ".") if "/" in owner else owner
    if owner.startswith("@ai-dynamo/dynamo-") and owner.endswith("-codeowners"):
        name = owner.removeprefix("@ai-dynamo/dynamo-").removesuffix("-codeowners")
    return PREFIX + name


def write_team_file(path: Path, value: str) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Team CODEOWNERS input must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _overlap(left: str, right: str) -> bool:
    for exact, other in ((left, right), (right, left)):
        if (
            exact.startswith("/")
            and not exact.endswith("/")
            and not any(c in exact for c in "*?[")
        ):
            return match(other, exact[1:])

    # Disjoint literal directory prefixes cannot overlap; uncertain globs can.
    prefixes = [
        re.split(r"[*?\[]", p)[0].rsplit("/", 1)[0] + "/" if p.startswith("/") else "/"
        for p in (left, right)
    ]
    return prefixes[0].startswith(prefixes[1]) or prefixes[1].startswith(prefixes[0])


def team_routes(rules: list[tuple[str, list[str]]]) -> dict[str, str]:
    routes = {}
    for glob, owners in rules:
        if glob.lstrip("/").startswith(PREFIX):
            if len(owners) != 1 or glob != "/" + team_path(owners[0]):
                raise ValueError(
                    "Each team CODEOWNERS file must have its own exact native owner"
                )
            routes[glob[1:]] = owners[0]
    return routes


def governance(spec: dict) -> dict:
    return {k: v for k, v in spec.items() if k not in {"areas", "shared"}}


def render_team_files(codeowners: str, spec: dict) -> dict[str, str]:
    rules = parse_codeowners(codeowners)
    routes = team_routes(rules)
    rules = [(p, owners) for p, owners in rules if not p.lstrip("/").startswith(PREFIX)]
    files = {}
    for path, owner in routes.items():
        projected = []
        for glob, owners in rules:
            if owner in owners:
                projected.append((glob, owners))
                continue
            for previous, own in reversed(projected):
                if own and _overlap(previous, glob):
                    projected.append((glob, []))
                    break
                if previous == glob or (
                    previous.endswith("/")
                    and not any(c in previous for c in "*?[")
                    and glob.startswith(previous)
                ):
                    break
        lines = [" ".join([glob, *owners]) for glob, owners in projected]
        if owner == PROCESS and governance(spec):
            lines += ["", "# Generator settings and ownership contracts:"]
            lines += [
                "# " + line
                for line in yaml.safe_dump(
                    governance(spec), sort_keys=True
                ).splitlines()
            ]
        files[path] = "\n".join(lines) + ("\n" if lines else "")
    return files


def validate(before: dict[str, str], after: dict[str, str]) -> None:
    old, new = (parse_codeowners(s["CODEOWNERS"]) for s in (before, after))
    previous, proposed = team_routes(old), team_routes(new)
    if any(
        proposed.get(p) != owner or p not in after or resolve_owners(new, p) != [owner]
        for p, owner in previous.items()
    ):
        raise ValueError(
            "Existing team CODEOWNERS files cannot be deleted or reassigned"
        )
    if ".github/CODEOWNERS" in after:
        raise ValueError(
            "A higher-priority CODEOWNERS would replace native team routing"
        )
    config = yaml.safe_load(after.get(POLICY[1], "{}")) or {}
    if any(resolve_owners(new, p) for p in DELEGATED):
        raise ValueError("Keep exactly the two delegated policy paths ownerless")
    for path, value in render_team_files(after["CODEOWNERS"], config).items():
        if after.get(path) != value:
            raise ValueError(f"Regenerate team CODEOWNERS files: {path}")

    # A new principal needs a natively owned file before it can receive code.
    application = [
        [
            r
            for r in rules
            if not r[0].lstrip("/").startswith(PREFIX)
            and r[0].lstrip("/") not in DELEGATED
        ]
        for rules in (old, new)
    ]
    if proposed.keys() - previous.keys() and application[0] != application[1]:
        raise ValueError("Register new owners separately from ownership transfers")
    old_config = yaml.safe_load(before.get(POLICY[1], "{}")) or {}
    if not previous or (old == new and governance(old_config) == governance(config)):
        return
    required = set()
    for path in (before.keys() | after.keys()) - DELEGATED:
        if path.startswith(PREFIX):
            continue
        old_owners, new_owners = set(resolve_owners(old, path)), set(
            resolve_owners(new, path)
        )
        if old_owners != new_owners:
            required.update(old_owners | new_owners)
    if governance(old_config) != governance(config):
        required.add(PROCESS)
    for owner in required:
        path = team_path(owner)
        if previous.get(path) != owner or before.get(path) == after.get(path):
            raise ValueError(
                f"Ownership change must update {path}; regenerate or split the owning glob"
            )


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, timeout=10)


def snapshot(ref: str | None) -> dict[str, str]:
    paths = (
        git("ls-tree", "-rz", "--name-only", ref)
        if ref
        else git("ls-files", "--cached", "--others", "--exclude-standard", "-z")
    )
    files = {p: "" for p in paths.split("\0") if p}
    for path in files.keys() & set(POLICY) | {p for p in files if p.startswith(PREFIX)}:
        if ref:
            if git("ls-tree", ref, "--", path).split()[0] != "100644":
                raise ValueError(f"Policy/team input must be a regular file: {path}")
            files[path] = git("show", f"{ref}:{path}")
        else:
            if any(p.is_symlink() for p in (Path(path), *Path(path).parents)):
                raise ValueError(f"Policy/team input must not be a symlink: {path}")
            if Path(path).exists():
                files[path] = Path(path).read_text()
            else:
                del files[path]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    args = parser.parse_args()
    validate(snapshot(git("rev-parse", args.base).strip()), snapshot(None))
    print("Team CODEOWNERS routing is valid")


if __name__ == "__main__":
    main()
