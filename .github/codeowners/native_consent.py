"""Encode affected-owner consent as native CODEOWNERS file reviews."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

import yaml
from codeowners_match import DELEGATED, parse_codeowners, resolve_owners

DIRECTORY = ".github/codeowners/consents/"
POLICY = (
    "CODEOWNERS",
    ".github/codeowners/areas.yaml",
    ".github/codeowners/external_contributors.yaml",
)
PROCESS = "@ai-dynamo/dynamo-process-codeowners"


def receipt_path(owner: str) -> str:
    if not re.fullmatch(
        r"@[A-Za-z0-9][A-Za-z0-9_-]*(?:/[A-Za-z0-9][A-Za-z0-9_-]*)?", owner
    ):
        raise ValueError(f"Invalid native owner: {owner!r}")
    return DIRECTORY + owner[1:] + ".txt"


def write_receipt(path: Path, value: str) -> None:
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError(f"Consent input must not be a symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def plan(before: dict[str, str], after: dict[str, str]) -> dict[str, str]:
    """Return the exact receipt edits needed for this captured policy transition."""
    old, new = (
        parse_codeowners(snapshot["CODEOWNERS"]) for snapshot in (before, after)
    )
    receipts = {p: resolve_owners(old, p) for p in before if p.startswith(DIRECTORY)}
    registered = [
        (g.lstrip("/"), owners) for g, owners in new if g.startswith("/" + DIRECTORY)
    ]
    if any(path not in after or len(owners) != 1 for path, owners in registered):
        raise ValueError(
            "Commit every registered native consent file with its owning team"
        )
    if any(
        len(owners) != 1 or resolve_owners(new, p) != owners or p not in after
        for p, owners in receipts.items()
    ):
        raise ValueError(
            "Consent-file ownership/deletion requires separate governance setup"
        )
    if ".github/CODEOWNERS" in after:
        raise ValueError(
            "A higher-priority CODEOWNERS would replace native consent routing"
        )
    if all(before.get(p) == after.get(p) for p in POLICY):
        return {}
    config = yaml.safe_load(after.get(POLICY[1], "{}")) or {}
    if config.get("meta", {}).get("native_policy_consents") and any(
        resolve_owners(new, p) for p in DELEGATED
    ):
        raise ValueError(
            "Keep the two delegated paths and root CODEOWNERS routing fixed"
        )

    required = set()
    for path in (before.keys() | after.keys()) - DELEGATED:
        if path.startswith(DIRECTORY):
            continue
        previous, proposed = set(resolve_owners(old, path)), set(
            resolve_owners(new, path)
        )
        if previous != proposed:
            required.update(previous | proposed)

    # The initial registry install keeps native governance review and cannot transfer code.
    if not receipts:
        if required or not all(resolve_owners(old, p) for p in DELEGATED):
            raise ValueError(
                "Install native consent routing separately from ownership transfers"
            )
        return {}

    # Changes beyond witnessed routing stay with governance.
    broad_rules = [
        [
            (g, set(owners))
            for g, owners in rules
            if not g.lstrip("/").startswith(DIRECTORY)
            and (not g.startswith("/") or g.endswith("/") or any(c in g for c in "*?["))
        ]
        for rules in (old, new)
    ]
    scopes = [
        list(dict.fromkeys(g for g, _ in reversed(rules))) for rules in broad_rules
    ]
    maps = [dict(rules) for rules in broad_rules]
    unwitnessed = any(
        maps[0].get(g) != maps[1].get(g)
        and not maps[0].get(g, set()) | maps[1].get(g, set()) <= required
        for g in maps[0].keys() | maps[1].keys()
    )
    configs = [
        yaml.safe_load(snapshot.get(POLICY[1], "{}")) or {}
        for snapshot in (before, after)
    ]
    governance = [
        {k: v for k, v in spec.items() if k not in {"areas", "shared"}}
        for spec in configs
    ]
    if scopes[0] != scopes[1] or governance[0] != governance[1] or unwitnessed:
        required.add(PROCESS)
    routes = {owners[0]: path for path, owners in receipts.items()}
    if required - routes.keys():
        raise ValueError(
            f"Register native consent files first: {sorted(required - routes.keys())}"
        )
    digest = hashlib.sha256(
        json.dumps(
            [{p: s.get(p) for p in POLICY} for s in (before, after)], sort_keys=True
        ).encode()
    ).hexdigest()
    return {routes[owner]: f"{digest}\n" for owner in sorted(required)}


def validate(before: dict[str, str], after: dict[str, str]) -> dict[str, str]:
    expected = plan(before, after)
    missing = [
        p
        for p, value in expected.items()
        if after.get(p) != value or before.get(p) == value
    ]
    if missing:
        raise ValueError(f"Regenerate the affected teams' consent files: {missing}")
    return expected


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], text=True, timeout=10)


def snapshot(ref: str | None) -> dict[str, str]:
    paths = (
        git("ls-tree", "-rz", "--name-only", ref).split("\0")
        if ref
        else git("ls-files", "--cached", "--others", "--exclude-standard", "-z").split(
            "\0"
        )
    )
    files = {p: "" for p in paths if p}
    for path in files.keys() & set(POLICY) | {
        p for p in files if p.startswith(DIRECTORY)
    }:
        if ref:
            mode = git("ls-tree", ref, "--", path).split()[0]
            if mode != "100644":
                raise ValueError(f"Policy/consent input must be a regular file: {path}")
            files[path] = git("show", f"{ref}:{path}")
        else:
            if any(p.is_symlink() for p in (Path(path), *Path(path).parents)):
                raise ValueError(f"Policy/consent input must not be a symlink: {path}")
            if Path(path).exists():
                files[path] = Path(path).read_text()
            else:
                del files[path]
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    base = git("rev-parse", args.base).strip()
    before, after = snapshot(base), snapshot(None)
    expected = plan(before, after)
    if args.write:
        for path, value in expected.items():
            write_receipt(Path(path), value)
            after[path] = value
    validate(before, after)
    print("Native review files: " + (", ".join(expected) or "none"))


if __name__ == "__main__":
    main()
