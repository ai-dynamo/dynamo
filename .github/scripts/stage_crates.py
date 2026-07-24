#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publish Dynamo's publishable workspace crates to an Artifactory cargo registry.

Mirrors GITLAB/runbooks/rcs/dynamo/crates_*.sh, but derives the crate set and
dependency order from `cargo metadata` instead of a hardcoded list, so it stays
correct as the workspace graph changes. Crates are published leaves-first so each
crate's intra-workspace deps are indexed before its dependents.

The Artifactory reference token (ARTIFACTORY_TOKEN, the same one the wheel upload
uses) is sent as a Bearer header for both the cargo registry and the idempotency
HEAD check. The registry location is read from ARTIFACTORY_CARGO_INDEX (a secret,
e.g. sparse+https://<host>/artifactory/api/cargo/<repo>/index/) so it is not
hardcoded in the workflow. Idempotent: crates already present are skipped.

Fail-soft: a crate that fails to publish is recorded (FAILED_CRATE=...), its
workspace dependents are skipped with a reason, and the rest of the graph keeps
staging; the exit code is nonzero when anything failed so the workflow can
report loudly without blocking the release.

Non-publishable members (the python/c bindings, example binaries) are excluded by
manifest-path; anything with `publish = false` is excluded too.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Manifest-path substrings whose crates are never published: python/C bindings,
# example binaries, and deploy/ components — deployable binaries/services (e.g. the
# inference-gateway ext-proc) that are not library crates consumers pull from the
# registry, and whose build scripts (envoy/protoc codegen) aren't wired up here.
EXCLUDE_PATH_SUBSTRINGS = ("/bindings/", "/examples/", "/deploy/")
# Workspace members never published from the on-demand flow: internal crates that
# carry an independent version (not the workspace version) and that no published
# crate depends on. kvbm-consolidator pins its own 1.2.0, so the version gate would
# (correctly) abort the release; it is a leaf, so excluding it is closure-safe.
EXCLUDE_NAMES = frozenset({"kvbm-consolidator"})
RETRYABLE = re.compile(r"HTTP/2|stream error|INTERNAL_ERROR|connection error|timed out|\b(429|50[0-9])\b")
ALREADY = re.compile(r"already exists|409 Conflict|already uploaded")
# Sparse-index propagation lag after publishing a dep — retried before failing.
DEP_NOT_INDEXED = re.compile(r"no matching package named")


def cargo_metadata(root: Path) -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=root, check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)


def publishable(meta: dict) -> dict[str, dict]:
    members = set(meta["workspace_members"])
    pkgs: dict[str, dict] = {}
    for p in meta["packages"]:
        if p["id"] not in members:
            continue
        if any(s in p["manifest_path"] for s in EXCLUDE_PATH_SUBSTRINGS):
            continue
        if p["name"] in EXCLUDE_NAMES:
            continue
        if p.get("publish") == []:  # `publish = false` serializes to []
            continue
        pkgs[p["name"]] = p
    return pkgs


def topo_order(pkgs: dict[str, dict]) -> list[str]:
    names = set(pkgs)
    incoming = {
        n: {d["name"] for d in pkgs[n]["dependencies"]
            if d["name"] in names and d["name"] != n and d.get("kind") != "dev"}
        for n in names
    }
    order: list[str] = []
    ready = sorted(n for n in names if not incoming[n])
    while ready:
        n = ready.pop(0)
        order.append(n)
        for m in names:
            if n in incoming[m]:
                incoming[m].discard(n)
                if not incoming[m] and m not in order and m not in ready:
                    ready.append(m)
        ready.sort()
    if len(order) != len(names):
        raise RuntimeError(f"dependency cycle among crates: {sorted(names - set(order))}")
    return order


def write_cargo_config(root: Path, alias: str, index: str) -> None:
    cfg = root / ".cargo" / "config.toml"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(
        f'[registries]\n{alias} = {{ index = "{index}" }}\n\n'
        f'[registry]\ndefault = "{alias}"\n'
        'global-credential-providers = ["cargo:token"]\n'
    )


def inject_registry(root: Path, names: list[str], alias: str, versions: dict[str, str]) -> None:
    # Internal deps must carry a registry + a version when published, else cargo
    # publish defaults them to crates.io / rejects them. `workspace = true` deps
    # inherit both from the root table (which we also rewrite), so skip them.
    # A direct path dep (e.g. `dynamo-llm = { path = "../llm", default-features = false }`
    # in backend-common) has neither — add the registry AND the depended-on crate's
    # publish version so `cargo publish` accepts it (cargo drops the path on publish).
    pat = re.compile(
        r"^(" + "|".join(re.escape(n) for n in names) + r")(\s*=\s*)\{([^}]*)\}",
        re.MULTILINE,
    )

    def repl(m: re.Match) -> str:
        name, inner = m.group(1), m.group(3).strip()
        if "registry" in inner or "workspace" in inner:
            return m.group(0)
        parts = inner.rstrip(",")
        if "version" not in inner and name in versions:
            parts = f'{parts}, version = "{versions[name]}"'
        return f'{m.group(1)}{m.group(2)}{{ {parts}, registry = "{alias}" }}'

    for manifest in [root / "Cargo.toml", *sorted((root / "lib").glob("*/Cargo.toml"))]:
        text = manifest.read_text()
        new = pat.sub(repl, text)
        if new != text:
            manifest.write_text(new)


def rewrite_versions(root: Path, old: str, new: str) -> int:
    """Replace `version = "<old>"` with `version = "<new>"` across the workspace
    manifests (root + lib/*) — both [workspace.package].version and the path-dep
    pins that share it. Used to stamp the internal staging rc version (e.g.
    0.1.0-rc0) without re-bumping the commit; mirrors the bump's version replace.
    Returns the number of manifests changed."""
    pat = re.compile(rf'(\bversion\s*=\s*"){re.escape(old)}(")')
    changed = 0
    for manifest in [root / "Cargo.toml", *sorted((root / "lib").glob("*/Cargo.toml"))]:
        text = manifest.read_text()
        new_text = pat.sub(lambda m: f"{m.group(1)}{new}{m.group(2)}", text)
        if new_text != text:
            manifest.write_text(new_text)
            changed += 1
    return changed


def strip_git_deps(root: Path, manifests: list[Path]) -> None:
    """Remove git-sourced deps (and feature refs to them) before publish: cargo
    publish rejects deps without a registry version, and a git dep — including a
    versioned dep redirected by [patch.crates-io] — can't resolve faithfully for
    a registry consumer. Only optional deps are dropped (non-optional is a hard
    error); the feature KEY is kept (emptied) so `<crate>/<feature>` refs resolve."""
    git_names = set(re.findall(
        r'(?m)^\s*([A-Za-z0-9_-]+)\s*=\s*\{[^}\n]*\bgit\s*=',
        (root / "Cargo.toml").read_text()))
    if not git_names:
        return
    for mp in manifests:
        orig = mp.read_text()
        kept, removed = [], set()
        for line in orig.splitlines(keepends=True):
            m = re.match(r'^[ \t]*([A-Za-z0-9_-]+)[ \t]*=[ \t]*\{(.*)\}[ \t]*$', line)
            if m and (re.search(r'\bgit\s*=', m.group(2))
                      or (m.group(1) in git_names and re.search(r'\bworkspace\s*=\s*true', m.group(2)))):
                removed.add(m.group(1))
                continue
            if m and m.group(1) in git_names and re.search(r'\bversion\s*=', m.group(2)):
                # Versioned dep git-patched via [patch.crates-io]: droppable only if optional.
                if not re.search(r'\boptional\s*=\s*true', m.group(2)):
                    raise RuntimeError(
                        f"{mp}: non-optional dependency '{m.group(1)}' is git-patched in the "
                        "workspace; a registry consumer would resolve different code. "
                        "Vendor/publish the patched crate or make the dependency optional.")
                removed.add(m.group(1))
                continue
            kept.append(line)
        if not removed:
            continue
        text = "".join(kept)
        for nm in removed:
            # scrub feature-array tokens: "dep:nm", "nm", "nm/feat", "nm?/feat"
            text = re.sub(rf'"(?:dep:)?{re.escape(nm)}(?:\?)?(?:/[^"]*)?"', "", text)
        text = re.sub(r'\[\s*,', "[", text)          # [ , ...   -> [ ...
        text = re.sub(r',\s*,', ",", text)            # , ,       -> ,
        text = re.sub(r',(\s*[\]\n])', r'\1', text)   # , ]  / ,\n -> ] / \n
        mp.write_text(text)
        print(f"strip-git-deps: removed {sorted(removed)} from {mp}")


def crate_exists(raw_base: str, name: str, version: str, token: str) -> bool:
    # 200 -> present, 404 -> absent; anything else (auth/outage) is a hard error.
    url = f"{raw_base}/{name}/{name}-{version}.crate"
    req = urllib.request.Request(url, method="HEAD", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise RuntimeError(f"registry HEAD {url} failed: HTTP {e.code} (bad ARTIFACTORY_TOKEN or registry outage?)") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"registry HEAD {url} unreachable: {e.reason}") from e


def publish(manifest: str, alias: str, env: dict) -> str:
    # Returns 'published' / 'already' (both staged); raises on definitive
    # failure — the caller records it and continues (fail-soft).
    for attempt in range(1, 4):
        r = subprocess.run(
            ["cargo", "publish", "--allow-dirty", "--no-verify",
             "--manifest-path", manifest, "--registry", alias],
            env=env, capture_output=True, text=True,
        )
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        out = r.stdout + r.stderr
        if r.returncode == 0:
            return "published"
        if ALREADY.search(out):
            print(f"  -> {manifest}: skip (already present)")
            return "already"
        if attempt < 3 and DEP_NOT_INDEXED.search(out):
            print(f"  -> dep not yet indexed (sparse-index lag?), retry in 15s ({attempt}/3)")
            time.sleep(15)
            continue
        if attempt < 3 and RETRYABLE.search(out):
            print(f"  -> transient error, retry in 10s ({attempt}/3)")
            time.sleep(10)
            continue
        raise RuntimeError(f"cargo publish failed for {manifest}")
    raise RuntimeError(f"exhausted retries for {manifest}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--registry", default="artifactory", help="cargo registry alias")
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--expect-version", default=os.environ.get("EXPECT_VERSION", ""),
                    help="fail fast if any publishable crate's version differs from this")
    ap.add_argument("--only", default=os.environ.get("CRATE_SUBSET", ""),
                    help="csv of crate names to publish (strict: must be dependency-closed); "
                         "empty/'all' = every publishable crate")
    ap.add_argument("--stage-version", default=os.environ.get("STAGE_VERSION", ""),
                    help="rewrite every publishable crate's version to this exact value before "
                         "publishing (internal staging rc disambiguator, e.g. 0.1.0-rc0), so a "
                         "re-stage of the same release doesn't collide on the immutable registry. "
                         "Empty = leave the committed versions as-is")
    args = ap.parse_args()

    # ARTIFACTORY ONLY — never crates.io. Refuse a crates.io alias outright; every
    # `cargo publish` below passes --registry <this alias> and write_cargo_config
    # sets it as the default registry, so crates.io is never a publish target.
    if args.registry in ("crates-io", "crates.io", "crates_io"):
        print("::error::refusing to publish to crates.io; this stages to Artifactory only", file=sys.stderr)
        return 1

    # Reuse the wheel-upload token; keep the registry location out of the workflow
    # via the ARTIFACTORY_CARGO_INDEX secret.
    token = os.environ.get("ARTIFACTORY_TOKEN", "")
    index = os.environ.get("ARTIFACTORY_CARGO_INDEX", "")
    if not token or not index:
        print("::error::ARTIFACTORY_TOKEN and ARTIFACTORY_CARGO_INDEX must be set", file=sys.stderr)
        return 1
    m = re.match(r"sparse\+(https://.+?)/api/cargo/([^/]+)/index/?$", index)
    if not m:
        print("::error::ARTIFACTORY_CARGO_INDEX must be 'sparse+https://<host>/.../api/cargo/<repo>/index/'", file=sys.stderr)
        return 1
    raw_base = f"{m.group(1)}/{m.group(2)}/crates"

    root = Path(args.root).resolve()
    pkgs = publishable(cargo_metadata(root))
    order = topo_order(pkgs)
    print("Publish order:", " ".join(order))

    # Narrow to an explicit subset (strict: no silent auto-expansion). Every
    # intra-workspace dependency of a selected crate must also be selected, or a
    # dependent would fail to resolve its dep from the registry.
    only = args.only.strip()
    if only and only != "all":
        selected = {c.strip() for c in only.split(",") if c.strip()}
        # The --only list is a curated wishlist that may name crates absent on the
        # branch being staged (the workspace graph differs across release branches).
        # Drop those with a warning rather than failing the whole run; closure is then
        # checked on what actually exists here.
        unknown = selected - set(pkgs)
        if unknown:
            print(f"::warning::skipping requested crate(s) not publishable on this branch: "
                  f"{sorted(unknown)} (branch publishable set: {sorted(pkgs)})", file=sys.stderr)
            selected -= unknown
        if not selected:
            print("::error::none of the requested crates are publishable on this branch", file=sys.stderr)
            return 1
        gaps = {n: sorted({d["name"] for d in pkgs[n]["dependencies"]
                           if d["name"] in pkgs and d["name"] != n and d.get("kind") != "dev"} - selected)
                for n in selected}
        gaps = {n: g for n, g in gaps.items() if g}
        if gaps:
            for n, g in gaps.items():
                print(f"::error::crate {n} depends on un-selected workspace crate(s) {g}", file=sys.stderr)
            print("::error::crate subset is not dependency-closed; add the missing crate(s) or use 'all'", file=sys.stderr)
            return 1
        order = [n for n in order if n in selected]
        print("Selected crates (dependency-closed):", " ".join(order))

    # Internal staging: rewrite the bumped workspace version to the rc-suffixed staging
    # version (e.g. 0.1.0 -> 0.1.0-rc0) so each re-stage publishes a distinct,
    # non-colliding version on the immutable registry. The expect-version gate below
    # then validates the new value, and cargo publish uploads it (--allow-dirty).
    stage_version = args.stage_version.strip()
    if stage_version:
        wm = re.search(r'\[workspace\.package\][^\[]*?\n\s*version\s*=\s*"([^"]+)"',
                       (root / "Cargo.toml").read_text())
        if not wm:
            print("::error::--stage-version: cannot read [workspace.package].version", file=sys.stderr)
            return 1
        cur = wm.group(1)
        if cur != stage_version:
            n = rewrite_versions(root, cur, stage_version)
            print(f"stage-version: rewrote {cur} -> {stage_version} across {n} manifest(s)")
            for nm in pkgs:
                if pkgs[nm]["version"] == cur:
                    pkgs[nm]["version"] = stage_version

    # Fail fast BEFORE any build/publish if a crate carries an unexpected version
    # (e.g. a hardcoded version the bump missed) — never silently publish wrong tags.
    if args.expect_version:
        mismatched = [(n, pkgs[n]["version"]) for n in order if pkgs[n]["version"] != args.expect_version]
        if mismatched:
            for n, v in mismatched:
                print(f"::error::crate {n} is at version {v}, expected {args.expect_version} (hardcoded/un-bumped version)", file=sys.stderr)
            print(f"::error::aborting before publish: {len(mismatched)} crate(s) carry an unexpected version; "
                  "set 'version.workspace = true' (or fix Cargo.toml) and re-run", file=sys.stderr)
            return 1

    write_cargo_config(root, args.registry, index)
    inject_registry(root, order, args.registry, {n: pkgs[n]["version"] for n in order})
    # cargo publish rejects deps without a registry version; drop the optional,
    # private git deps (e.g. aiconfigurator-core) from the crates being published.
    strip_git_deps(root, [Path(pkgs[n]["manifest_path"]) for n in order])

    env = dict(os.environ)
    env[f"CARGO_REGISTRIES_{args.registry.upper()}_TOKEN"] = f"Bearer {token}"
    env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} --cfg tokio_unstable".strip()
    # Resync Cargo.lock after the stage-version rewrite; fail loudly here rather
    # than as a confusing publish error later.
    if stage_version:
        r = subprocess.run(["cargo", "update", "--workspace"], cwd=root, env=env)
        if r.returncode != 0:
            print("::error::cargo update --workspace failed after the stage-version rewrite", file=sys.stderr)
            return 1

    # Fail-soft loop: a failed crate skips its dependents but not the rest.
    # Machine-readable lines consumed by the workflow:
    #   STAGED_CRATE=<name>            on the registry (this run or earlier)
    #   FAILED_CRATE=<name> <reason>   failed, or skipped due to a failed dep
    published = already = 0
    staged: list[str] = []
    failed: dict[str, str] = {}
    deps_of = {
        n: {d["name"] for d in pkgs[n]["dependencies"]
            if d["name"] in pkgs and d["name"] != n and d.get("kind") != "dev"}
        for n in order
    }
    for name in order:
        p = pkgs[name]
        manifest, version = p["manifest_path"], p["version"]
        print(f"=== {name} {version} ===")
        bad_deps = sorted(deps_of[name] & set(failed))
        if bad_deps:
            failed[name] = f"dependency {','.join(bad_deps)} failed"
            print(f"::error::{name} {version} skipped: {failed[name]}", file=sys.stderr)
            print(f"FAILED_CRATE={name} {failed[name]}", flush=True)
            continue
        try:
            if crate_exists(raw_base, name, version, token):
                print(f"  -> {name} {version} already on the registry (skip)")
                already += 1
                staged.append(name)
                print(f"STAGED_CRATE={name}", flush=True)  # captured by the staging job
                continue
            if subprocess.run(["cargo", "check", "--manifest-path", manifest], cwd=root, env=env).returncode != 0:
                raise RuntimeError(f"cargo check failed for {manifest}")
            status = publish(manifest, args.registry, env)
        except RuntimeError as e:
            failed[name] = str(e)
            print(f"::error::{name} {version} not staged: {e}", file=sys.stderr)
            print(f"FAILED_CRATE={name} {e}", flush=True)
            continue
        if status == "published":
            published += 1
        else:
            already += 1
        staged.append(name)
        print(f"STAGED_CRATE={name}", flush=True)  # captured by the staging job

    print(f"Done: {len(order)} crates ({published} published, {already} already present, "
          f"{len(staged)} staged, {len(failed)} failed).")
    print(f"STAGED_CRATES={','.join(staged)}")
    if failed:
        print(f"FAILED_CRATES={','.join(failed)}")
        print(f"::error::{len(failed)} crate(s) not staged: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
