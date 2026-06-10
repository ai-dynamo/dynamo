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

EXCLUDE_PATH_SUBSTRINGS = ("/bindings/", "/examples/")
RETRYABLE = re.compile(r"HTTP/2|stream error|INTERNAL_ERROR|connection error|timed out|50[23] ")
ALREADY = re.compile(r"already exists|409 Conflict|already uploaded")


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


def inject_registry(root: Path, names: list[str], alias: str) -> None:
    # Internal deps carry path + version but no registry, so published metadata
    # would default to crates.io. Attribute them to the artifactory registry.
    # `workspace = true` deps inherit from the root table, which we also rewrite.
    pat = re.compile(
        r"^(" + "|".join(re.escape(n) for n in names) + r")(\s*=\s*)\{([^}]*)\}",
        re.MULTILINE,
    )

    def repl(m: re.Match) -> str:
        inner = m.group(3).strip()
        if "registry" in inner or "workspace" in inner:
            return m.group(0)
        return f'{m.group(1)}{m.group(2)}{{ {inner.rstrip(",")}, registry = "{alias}" }}'

    for manifest in [root / "Cargo.toml", *sorted((root / "lib").glob("*/Cargo.toml"))]:
        text = manifest.read_text()
        new = pat.sub(repl, text)
        if new != text:
            manifest.write_text(new)


def crate_exists(raw_base: str, name: str, version: str, token: str) -> bool:
    url = f"{raw_base}/{name}/{name}-{version}.crate"
    req = urllib.request.Request(url, method="HEAD", headers={"Authorization": f"Bearer {token}"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        return e.code == 200
    except Exception:
        return False


def publish(manifest: str, alias: str, env: dict) -> None:
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
            return
        if ALREADY.search(out) or "no matching package named" in out:
            print(f"  -> {manifest}: skip ({'already present' if ALREADY.search(out) else 'dep not yet indexed'})")
            return
        if RETRYABLE.search(out) and attempt < 3:
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
    args = ap.parse_args()

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
    inject_registry(root, order, args.registry)

    env = dict(os.environ)
    env[f"CARGO_REGISTRIES_{args.registry.upper()}_TOKEN"] = f"Bearer {token}"
    env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} --cfg tokio_unstable".strip()
    subprocess.run(["cargo", "update", "tokio", "--precise", "1.43.0"], cwd=root, env=env)

    published = skipped = 0
    for name in order:
        p = pkgs[name]
        manifest, version = p["manifest_path"], p["version"]
        print(f"=== {name} {version} ===")
        if crate_exists(raw_base, name, version, token):
            print(f"  -> {name} {version} already on the registry (skip)")
            skipped += 1
            continue
        if subprocess.run(["cargo", "check", "--manifest-path", manifest], cwd=root, env=env).returncode != 0:
            print(f"::error::cargo check failed for {manifest}", file=sys.stderr)
            return 1
        publish(manifest, args.registry, env)
        published += 1

    print(f"Done: {len(order)} crates ({published} published, {skipped} already present).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
