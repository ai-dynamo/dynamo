"""Resolve a CODEOWNERS area taxonomy and validate coverage (repo-agnostic).

Reads an `areas.yaml` (each area declares its path globs directly), assigns
every tracked file in the target repo to an area via three layers, in order:

    declared path_globs  ->  substring carve-out overrides  ->  keyword auto-classify

then reports how much of the tree is EXPLICITLY owned vs. falls to the catch-all,
and writes a resolved table that `emit_codeowners.py` turns into a CODEOWNERS file.

No external services, no ownership-inference engine -- just the `areas.yaml`
plus `git -C <repo> ls-files`. The companion that infers WHO should own code
(from git history / trackers) is deliberately out of scope; this produces
ownership STRUCTURE (path -> team), and teams are populated by the adopting org.

Usage:
  uv run python scripts/build_codeowners.py --areas areas.yaml --repo /path/to/repo
"""

from __future__ import annotations

import argparse
import fnmatch
import subprocess
from collections import Counter
from pathlib import Path

import yaml


def load_tree(repo: Path) -> list[str]:
    out = subprocess.check_output(["git", "-C", str(repo), "ls-files"], text=True)
    return [p for p in out.splitlines() if p.strip()]


def override_globs(tree: list[str], substr: str) -> set[str]:
    """Concrete directory globs for every tree path that contains `substr` in a
    directory segment. A carve-out like `multimodal_handlers` becomes
    `components/.../multimodal_handlers/` wherever it appears in the tree."""
    out: set[str] = set()
    s = substr.lower().strip("/")
    for p in tree:
        segs = p.split("/")
        for i, seg in enumerate(segs[:-1]):  # directory segments only
            if s in seg.lower():
                out.add("/".join(segs[: i + 1]) + "/")
                break
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--areas", required=True, help="path to areas.yaml (source of truth)"
    )
    ap.add_argument("--repo", required=True, help="path to the target git repo")
    ap.add_argument(
        "--out",
        default=None,
        help="resolved yaml output (default: <areas>.resolved.yaml)",
    )
    ap.add_argument(
        "--strict",
        action="store_true",
        help="exit non-zero if any file falls to the catch-all (CI gate)",
    )
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.areas).read_text())
    areas = spec["areas"]
    overrides = spec.get("overrides", {})
    classify = spec.get("classify", {})
    out_path = (
        Path(args.out) if args.out else Path(args.areas).with_suffix(".resolved.yaml")
    )

    def name(a: dict) -> str:
        return a.get("name", a["label"])

    # accept override / classify targets as either label or name
    label_by = {a["label"]: a["label"] for a in areas}
    label_by.update({name(a): a["label"] for a in areas})

    tree = load_tree(Path(args.repo))

    # area label -> set of path globs (start with what each area declares)
    globs: dict[str, set[str]] = {
        a["label"]: set(a.get("path_globs", [])) for a in areas
    }

    # overrides: substring carve-outs resolved to concrete dir globs (more specific,
    # so they win via last-match in the emitter).
    for area_key, subs in overrides.items():
        lbl = label_by.get(area_key, area_key)
        globs.setdefault(lbl, set())
        for sub in subs:
            globs[lbl] |= override_globs(tree, sub)

    def index():
        dirs = [(g, lbl) for lbl, s in globs.items() for g in s if g.endswith("/")]
        files = {g for s in globs.values() for g in s if not g.endswith("/")}
        return dirs, files

    def covered(p: str, dirs, files) -> bool:
        # file globs match like CODEOWNERS: `*.md` (no slash) matches a basename
        # anywhere, an exact path matches itself. Without this, `*.md` and
        # `LICENSE`-style file globs were ignored and their files looked unowned.
        base = p.rsplit("/", 1)[-1]
        if any(fnmatch.fnmatch(p, g) or fnmatch.fnmatch(base, g) for g in files):
            return True
        return any(p == d.rstrip("/") or p.startswith(d) for d, _ in dirs)

    dirs, files = index()
    unmatched = [p for p in tree if not covered(p, dirs, files)]

    # auto-classify: any unmatched (new) directory whose path contains a keyword
    # gets owned without a human editing areas.yaml. First matching rule wins.
    kw = classify.get("keyword_rules", [])

    def classify_dir(path: str) -> str | None:
        pl = ("/" + path + "/").lower()
        for r in kw:
            if r["match"].lower() in pl and r.get("area"):
                return label_by.get(r["area"], r["area"])
        return None

    def dir_prefix(p: str) -> str:
        # Group an unmatched FILE by a directory prefix (<=3 segments), never
        # including the filename itself -- otherwise a file at depth 3 like
        # recipes/foo/Dockerfile would become a bogus `recipes/foo/Dockerfile/` glob.
        segs = p.split("/")
        depth = min(3, len(segs) - 1)
        return "/".join(segs[:depth])  # "" for a repo-root file -> skipped

    auto: list[tuple[str, str]] = []
    if unmatched and kw:
        bydir = sorted({d for p in unmatched if (d := dir_prefix(p))})
        for d in bydir:
            lbl = classify_dir(d)
            if lbl:
                globs.setdefault(lbl, set()).add(d.rstrip("/") + "/")
                auto.append((d, lbl))
        dirs, files = index()
        unmatched = [p for p in tree if not covered(p, dirs, files)]

    # file-type co-ownership (blocking): each matching file is co-owned by its
    # ENCLOSING area plus the rule's coowner -- so e.g. ops owns every Dockerfile
    # while the area stays in the loop. Files already in the coowner's area stay sole.
    enc_pairs = sorted(
        ((g, lbl) for lbl, s in globs.items() for g in s), key=lambda kv: -len(kv[0])
    )

    def enclosing(path: str) -> str | None:
        for g, lbl in enc_pairs:
            gg = g.rstrip("/")
            if path == gg or path.startswith(gg + "/"):
                return lbl
        return None

    ft_rules = classify.get("filetype_rules", [])
    filetype_shared = []
    for r in (x for x in ft_rules if not x.get("advisory")):
        pat, co = r.get("pattern"), r["coowner"]
        for p in tree:
            base = p.rsplit("/", 1)[-1]
            if not (
                fnmatch.fnmatch(base, pat) if pat else r["match"].lower() in p.lower()
            ):
                continue
            enc = enclosing(p)
            owners = [enc] if enc and enc != co else []
            owners.append(co)
            filetype_shared.append({"glob": p, "owners": owners})

    # Honest coverage: a file is "explicitly owned" if ANY emitted CODEOWNERS line
    # matches it -- a base/auto area glob, a shared multi-owner glob, OR a file-type
    # co-ownership line. (The earlier `unmatched` only knew about area globs, so it
    # over-counted the catch-all by ignoring `*.md` and Dockerfile-style lines.)
    dirs, files = index()
    shared_specs = spec.get("shared", [])
    shared_dirs = [s["glob"] for s in shared_specs if s["glob"].endswith("/")]
    owned_files = {s["glob"] for s in shared_specs if not s["glob"].endswith("/")}
    owned_files |= {s["glob"] for s in filetype_shared}

    def owned(p: str) -> bool:
        if covered(p, dirs, files) or p in owned_files:
            return True
        return any(p == d.rstrip("/") or p.startswith(d) for d in shared_dirs)

    unmatched = [p for p in tree if not owned(p)]

    resolved = {
        "meta": dict(
            spec.get("meta", {}),
            tree_files=len(tree),
            explicit_pct=round(100 * (len(tree) - len(unmatched)) / len(tree), 2),
        ),
        "areas": [
            {
                "label": a["label"],
                "name": name(a),
                "github_team": a["github_team"],
                "new_team": a.get("new_team", False),
                "path_globs": sorted(globs[a["label"]]),
            }
            for a in areas
        ],
        "shared": spec.get("shared", []),
        "advisory": spec.get("advisory", []),
        "filetype_shared": filetype_shared,
        "filetype_advisory": [r for r in ft_rules if r.get("advisory")],
    }
    out_path.write_text(yaml.safe_dump(resolved, sort_keys=False, width=120))

    covn = len(tree) - len(unmatched)
    print(f"areas: {len(areas)} | tree files: {len(tree)}")
    print(
        f"explicitly owned: {covn}/{len(tree)} ({100 * covn / len(tree):.2f}%) | "
        f"catch-all only: {len(unmatched)}"
    )
    print(f"auto-classified new dirs: {len(auto)}")
    for d, lbl in auto[:20]:
        print(f"    {d} -> {lbl}")
    if unmatched:
        print("catch-all-only sample (add an area or classify rule to cover these):")
        print("   ", unmatched[:15])
    print("\nper-area glob counts:")
    counts = Counter({a["label"]: len(globs[a["label"]]) for a in areas})
    for lbl, c in counts.most_common():
        print(f"  {lbl:<22} {c}")
    print("\nwrote", out_path)
    if args.strict and unmatched:
        print(
            f"!! strict: {len(unmatched)} file(s) fall to the catch-all -- cover them in areas.yaml"
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
