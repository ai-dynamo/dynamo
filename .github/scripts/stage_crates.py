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
HEAD check. The registry location is read from ARTIFACTORY_CARGO_INDEX (configuration,
e.g. sparse+https://<host>/artifactory/api/cargo/<repo>/index/) so it is not
hardcoded in the workflow. Idempotent: crates already present are skipped.

Fail-soft: a crate that fails to publish is recorded (FAILED_CRATE=...), its
workspace dependents are skipped with a reason, and the rest of the graph keeps
staging; the exit code is nonzero when anything failed so the workflow can
report loudly without blocking the release.

External Git-patched dependencies can declare publication counterparts in the
source Cargo.toml, independently of the normal source-build requirement:

    [workspace.metadata.artifactory.dependencies.example-core]
    version = "=1.2.0-rc.0"
    registry = "example"
    git-rev = "<the full revision in patch.crates-io.example-core>"

ARTIFACTORY_CARGO_DEPENDENCY_REGISTRIES is non-secret JSON configuration mapping
these aliases to sparse Cargo index URLs on the same Artifactory server as the
publication registry. ARTIFACTORY_TOKEN must have read access to those indexes.

For AISim on release/1.7.0, create a repository variable in ai-dynamo/dynamo:
Settings > Secrets and variables > Actions > Variables > New repository variable.
Name: ARTIFACTORY_CARGO_DEPENDENCY_REGISTRIES
Value (a JSON object, not a quoted JSON string):

    {"aisim":"sparse+https://artifactory.nvidia.com/artifactory/api/cargo/sw-dynamo-aisimulate-cargo-local/index/"}

The key "aisim" matches registry = "aisim" in the source Cargo.toml's
workspace.metadata.artifactory.dependencies. That metadata selects the exact
aisim-core version (for example, =0.14.0-rc.1); versions and crate download paths do not belong
in this variable. Keep the sparse+ prefix and trailing /index/.

Store credentials only in the existing ARTIFACTORY_TOKEN secret, which needs
read access to the dependency repository as well as publication access to the
Dynamo repository. ARTIFACTORY_CARGO_INDEX still selects the Dynamo publication
destination; this mapping only adds dependency registries. The automated-release
environment must not override the repository variable with a different value.
An unset variable is allowed only when the source has no publication references.

Only this disposable staging checkout gets rewritten; ordinary builds keep their
Git pins. The matching patch is removed so cargo check/package use the published
counterpart too. Start retries from the committed source checkout.

Non-publishable members (the python/c bindings, example binaries) are excluded by
manifest-path; anything with `publish = false` is excluded too.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import subprocess
import sys
import time
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Manifest-path substrings whose crates are never published: python/C bindings,
# example binaries, and deploy/ components — deployable binaries/services (e.g. the
# inference-gateway ext-proc) that are not library crates consumers pull from the
# registry, and whose build scripts (envoy/protoc codegen) aren't wired up here.
EXCLUDE_PATH_SUBSTRINGS = ("/bindings/", "/examples/", "/deploy/")
# Artifactory < 7.148 rejects the `Content-Type: application/octet-stream` header
# cargo >= 1.96 sends on publish with a 415 (rust-lang/cargo#17086),
# so the upload runs on the last pre-header toolchain. Remove once the
# registry host runs Artifactory 7.148+.
PUBLISH_TOOLCHAIN = "1.95.0"
# No bare "HTTP/2" alternation here: cargo dumps the raw status line ("HTTP/2 415")
# on every HTTP error, which made deterministic 4xx failures retry as transient.
RETRYABLE = re.compile(
    r"stream error|INTERNAL_ERROR|connection error|timed out|\b(429|50[0-9])\b"
)
ALREADY = re.compile(r"already exists|409 Conflict|already uploaded")
# Sparse-index propagation lag after publishing a dep — retried before failing.
DEP_NOT_INDEXED = re.compile(r"no matching package named")


def cargo_metadata(root: Path) -> dict:
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--no-deps"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
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
        n: {
            d["name"]
            for d in pkgs[n]["dependencies"]
            if d["name"] in names and d["name"] != n and d.get("kind") != "dev"
        }
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
        raise RuntimeError(
            f"dependency cycle among crates: {sorted(names - set(order))}"
        )
    return order


def dependency_tables(document: dict) -> dict[str, dict]:
    """Enumerate actual Cargo dependency tables, excluding package metadata."""
    kinds = ("dependencies", "build-dependencies", "dev-dependencies")
    tables = {kind: document.get(kind, {}) for kind in kinds}
    tables["workspace.dependencies"] = document.get("workspace", {}).get(
        "dependencies", {}
    )
    for target, fields in document.get("target", {}).items():
        for spelling in (target, json.dumps(target), f"'{target}'"):
            for kind in kinds:
                tables[f"target.{spelling}.{kind}"] = fields.get(kind, {})
    return tables


def configure_dependency_registries(
    root: Path, manifests: list[Path], metadata: dict, alias: str, index: str, env: dict
) -> dict[str, str]:
    """Apply reviewed publication references only to the staging checkout."""
    configuration = metadata.get("artifactory", {})
    if not isinstance(configuration, dict):
        raise RuntimeError("workspace.metadata.artifactory must be a table")
    references = configuration.get("dependencies", {})
    if not isinstance(references, dict):
        raise RuntimeError(
            "workspace.metadata.artifactory.dependencies must be a table"
        )
    if not references:
        return {}
    try:
        indexes = json.loads(env.get("ARTIFACTORY_CARGO_DEPENDENCY_REGISTRIES", "{}"))
    except json.JSONDecodeError as error:
        raise RuntimeError(
            "ARTIFACTORY_CARGO_DEPENDENCY_REGISTRIES must be a JSON object"
        ) from error
    if not isinstance(indexes, dict):
        raise RuntimeError(
            "ARTIFACTORY_CARGO_DEPENDENCY_REGISTRIES must be a JSON object"
        )

    source = tomllib.loads((root / "Cargo.toml").read_text())
    patches = source.get("patch", {}).get("crates-io", {})
    approved_base = index.removeprefix("sparse+").split("/api/cargo/", 1)[0]
    registries: dict[str, str] = {}
    for package, reference in references.items():
        if not isinstance(reference, dict) or set(reference) != {
            "version",
            "registry",
            "git-rev",
        }:
            raise RuntimeError(
                f"{package}: expected version, registry, and git-rev publication fields"
            )
        version, registry, revision = (
            reference[key] for key in ("version", "registry", "git-rev")
        )
        if not isinstance(version, str) or not re.fullmatch(
            r"=\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?", version
        ):
            raise RuntimeError(
                f"{package}: publication version must be an exact Cargo requirement"
            )
        if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise RuntimeError(f"{package}: git-rev must be a full commit SHA")
        if not isinstance(registry, str) or not re.fullmatch(
            r"[a-z][a-z0-9_]*", registry
        ):
            raise RuntimeError(f"{package}: invalid dependency registry alias")
        if registry in {alias, "crates_io"}:
            raise RuntimeError(
                f"{package}: dependency registry alias collides with a publish/default registry"
            )
        url = indexes.get(registry)
        if not isinstance(url, str):
            raise RuntimeError(
                f"{package}: registry {registry} is missing from ARTIFACTORY_CARGO_DEPENDENCY_REGISTRIES"
            )
        parsed = urllib.parse.urlsplit(url.removeprefix("sparse+"))
        if (
            not url.startswith(f"sparse+{approved_base}/api/cargo/")
            or not url.endswith("/index/")
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise RuntimeError(
                f"{package}: dependency index must use the publication Artifactory origin and /api/cargo/<repo>/index/ path"
            )
        patch = patches.get(package)
        if (
            not isinstance(patch, dict)
            or not patch.get("git")
            or patch.get("rev") != revision
            or patch.get("package", package) != package
        ):
            raise RuntimeError(
                f"{package}: publication reference does not match the current Git source pin"
            )
        registries[registry] = url

    # Existing staging rewrites use inline dependency declarations. Fail closed
    # on other layouts rather than silently leaving a crates.io dependency.
    assignment = re.compile(r"^(\s*)([A-Za-z0-9_-]+)\s*=\s*(\{.*\})(\s*(?:#.*)?)$")
    edits: dict[Path, str] = {}
    expected_documents: dict[Path, dict] = {}
    rewritten, removed = set(), set()
    for manifest in dict.fromkeys(manifests):
        original = manifest.read_text()
        document = tomllib.loads(original)
        sections = dependency_tables(document)
        expected = copy.deepcopy(document)
        tables = {id(table): table for table in dependency_tables(expected).values()}
        for table in tables.values():
            for name, attributes in list(table.items()):
                if isinstance(attributes, str):
                    attributes = {"version": attributes}
                package = attributes.get("package", name)
                if package in references and not attributes.get("workspace"):
                    if any(
                        key in attributes
                        for key in ("git", "rev", "branch", "tag", "path", "registry")
                    ):
                        raise RuntimeError(
                            f"{manifest}: {name} must use the reviewed crates.io Git patch"
                        )
                    attributes.update(
                        {
                            key: references[package][key]
                            for key in ("version", "registry")
                        }
                    )
                    table[name] = attributes
        if manifest == root / "Cargo.toml":
            for package in references:
                del expected["patch"]["crates-io"][package]
        expected_documents[manifest] = expected
        section = ""
        lines = []
        for line in original.splitlines(keepends=True):
            header = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
            if header:
                section = header.group(1)
            match = assignment.match(line.rstrip("\n"))
            if (
                match
                and manifest == root / "Cargo.toml"
                and section == "patch.crates-io"
                and match[2] in references
            ):
                removed.add(match[2])
                continue
            if match and section in sections:
                attributes = tomllib.loads(f"dependency = {match[3]}")["dependency"]
                package = attributes.get("package", match[2])
                if package in references and "workspace" not in attributes:
                    if any(
                        key in attributes
                        for key in ("git", "rev", "branch", "tag", "path", "registry")
                    ):
                        raise RuntimeError(
                            f"{manifest}: {match[2]} must use the reviewed crates.io Git patch"
                        )
                    attributes.update(
                        {
                            key: references[package][key]
                            for key in ("version", "registry")
                        }
                    )
                    fields = ", ".join(
                        f"{key} = {json.dumps(value)}"
                        for key, value in attributes.items()
                    )
                    line = f"{match[1]}{match[2]} = {{ {fields} }}{match[4]}\n"
                    rewritten.add(package)
            lines.append(line)
        edits[manifest] = "".join(lines)
    if rewritten != set(references) or removed != set(references):
        raise RuntimeError(
            "external publication references require matching inline dependencies and Git patches"
        )
    for manifest, text in edits.items():
        # The line rewrite must change exactly the approved dependency fields,
        # never metadata or strings that resemble Cargo declarations.
        if tomllib.loads(text) != expected_documents[manifest]:
            raise RuntimeError(
                f"{manifest}: unsupported declaration layout for publication references"
            )
    for manifest, text in edits.items():
        manifest.write_text(text)
    for registry in registries:
        env[
            f"CARGO_REGISTRIES_{registry.upper()}_TOKEN"
        ] = f"Bearer {env['ARTIFACTORY_TOKEN']}"
    for package, reference in references.items():
        print(
            f"publication dependency: {package} {reference['version']} via {reference['registry']}"
        )
    return registries


def write_cargo_config(
    root: Path,
    alias: str,
    index: str,
    dependency_registries: dict[str, str] | None = None,
) -> None:
    # Append to the workspace .cargo/config.toml -- it carries required build
    # settings (rustflags, PCRE2_SYS_STATIC) that an overwrite would drop.
    # The marker makes re-runs idempotent (no duplicated tables).
    marker = "# --- stage_crates registry (generated) ---"
    cfg = root / ".cargo" / "config.toml"
    cfg.parent.mkdir(exist_ok=True)
    text = cfg.read_text() if cfg.exists() else ""
    if marker in text:
        # Drop our previously generated tail (anything after the marker is ours).
        text = text.split(marker)[0].rstrip() + "\n"
    # Header-anchored so [registries.x]/[registry] count but comments don't;
    # checked AFTER stripping our block so a re-run can't mask a foreign table.
    if re.search(r"(?m)^\s*\[registr(?:y|ies)[\].]", text):
        raise RuntimeError(
            ".cargo/config.toml already defines registry tables; refusing to append conflicting ones"
        )
    cfg.write_text(
        f"{text}\n{marker}\n"
        f'[registries]\n{alias} = {{ index = "{index}" }}\n'
        + "".join(
            f"{name} = {{ index = {json.dumps(url)} }}\n"
            for name, url in sorted((dependency_registries or {}).items())
        )
        + "\n"
        f'[registry]\ndefault = "{alias}"\n'
        'global-credential-providers = ["cargo:token"]\n'
    )


def inject_registry(
    manifests: list[Path], names: list[str], alias: str, versions: dict[str, str]
) -> None:
    # Internal deps must carry a registry + a version when published, else cargo
    # publish defaults them to crates.io / rejects them. `workspace = true` deps
    # inherit both from the root table (which we also rewrite), so skip them.
    # A direct path dep (e.g. `dynamo-vllm-sidecar = { path = "../../../sidecar/vllm" }`
    # in the vllm mocker server) has neither — add the registry AND the depended-on
    # crate's publish version so `cargo publish` accepts it (cargo drops the path on
    # publish). Manifests come from cargo metadata, not a glob: a shallow lib/* glob
    # missed nested members like lib/mocker/servers/vllm, whose un-rewritten path dep
    # then failed `cargo publish` with "does not specify a version".
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

    for manifest in manifests:
        text = manifest.read_text()
        new = pat.sub(repl, text)
        if new != text:
            manifest.write_text(new)


def rewrite_versions(manifests: list[Path], old: str, new: str) -> int:
    """Replace `version = "<old>"` with `version = "<new>"` across the given
    manifests (root + every workspace member, from cargo metadata) — both
    [workspace.package].version and the path-dep pins that share it. Used to stamp
    the internal staging rc version (e.g. 0.1.0-rc0) without re-bumping the commit;
    mirrors the bump's version replace. Returns the number of manifests changed."""
    pat = re.compile(rf'(\bversion\s*=\s*"){re.escape(old)}(")')
    changed = 0
    for manifest in manifests:
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
    error); the feature KEY is kept (emptied) so `<crate>/<feature>` refs resolve.
    A non-optional `workspace = true` dep that is only [patch]-overridden (not
    declared as a git dep) keeps its registry requirement: dropping it would
    leave the crate uncompilable."""
    git_names: set[str] = set()
    patched_only: set[str] = set()
    section = ""
    for line in (root / "Cargo.toml").read_text().splitlines():
        h = re.match(r"^\s*\[([^\]]+)\]\s*$", line)
        if h:
            section = h.group(1).strip()
            continue
        g = re.match(r"^\s*([A-Za-z0-9_-]+)\s*=\s*\{[^}\n]*\bgit\s*=", line)
        if g:
            (patched_only if section.startswith("patch.") else git_names).add(
                g.group(1)
            )
    patched_only -= git_names
    git_names |= patched_only
    if not git_names:
        return
    for mp in manifests:
        orig = mp.read_text()
        kept, removed = [], set()
        for line in orig.splitlines(keepends=True):
            m = re.match(r"^[ \t]*([A-Za-z0-9_-]+)[ \t]*=[ \t]*\{(.*)\}[ \t]*$", line)
            if m and (
                re.search(r"\bgit\s*=", m.group(2))
                or (
                    m.group(1) in git_names
                    and re.search(r"\bworkspace\s*=\s*true", m.group(2))
                    and not (
                        m.group(1) in patched_only
                        and not re.search(r"\boptional\s*=\s*true", m.group(2))
                    )
                )
            ):
                removed.add(m.group(1))
                continue
            if (
                m
                and m.group(1) in git_names
                and re.search(r"\bversion\s*=", m.group(2))
            ):
                # Versioned dep git-patched via [patch.crates-io]: droppable only if optional.
                if not re.search(r"\boptional\s*=\s*true", m.group(2)):
                    raise RuntimeError(
                        f"{mp}: non-optional dependency '{m.group(1)}' is git-patched in the "
                        "workspace; a registry consumer would resolve different code. "
                        "Vendor/publish the patched crate or make the dependency optional."
                    )
                removed.add(m.group(1))
                continue
            kept.append(line)
        if not removed:
            continue
        text = "".join(kept)
        for nm in removed:
            # scrub feature-array tokens: "dep:nm", "nm", "nm/feat", "nm?/feat"
            text = re.sub(rf'"(?:dep:)?{re.escape(nm)}(?:\?)?(?:/[^"]*)?"', "", text)
        text = re.sub(r"\[\s*,", "[", text)  # [ , ...   -> [ ...
        text = re.sub(r",\s*,", ",", text)  # , ,       -> ,
        text = re.sub(r",(\s*[\]\n])", r"\1", text)  # , ]  / ,\n -> ] / \n
        mp.write_text(text)
        print(f"strip-git-deps: removed {sorted(removed)} from {mp}")


def crate_exists(raw_base: str, name: str, version: str, token: str) -> bool:
    # 200 -> present, 404 -> absent; anything else (auth/outage) is a hard error.
    url = f"{raw_base}/{name}/{name}-{version}.crate"
    req = urllib.request.Request(
        url, method="HEAD", headers={"Authorization": f"Bearer {token}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status == 200
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        # Path only — the registry host is workflow-secret-derived and error
        # strings end up in public Actions logs.
        raise RuntimeError(
            f"registry HEAD {urllib.parse.urlparse(url).path} failed: HTTP {e.code} (bad ARTIFACTORY_TOKEN or registry outage?)"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"registry HEAD {urllib.parse.urlparse(url).path} unreachable: {e.reason}"
        ) from e


def publish(manifest: str, alias: str, env: dict) -> str:
    # Returns 'published' / 'already' (both staged); raises on definitive
    # failure — the caller records it and continues (fail-soft).
    for attempt in range(1, 4):
        r = subprocess.run(
            [
                "cargo",
                f"+{PUBLISH_TOOLCHAIN}",
                "publish",
                "--allow-dirty",
                "--no-verify",
                "--manifest-path",
                manifest,
                "--registry",
                alias,
            ],
            env=env,
            capture_output=True,
            text=True,
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
            print(
                f"  -> dep not yet indexed (sparse-index lag?), retry in 15s ({attempt}/3)"
            )
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
    ap.add_argument(
        "--expect-version",
        default=os.environ.get("EXPECT_VERSION", ""),
        help="fail fast if any publishable crate's version differs from this",
    )
    ap.add_argument(
        "--only",
        default=os.environ.get("CRATE_SUBSET", ""),
        help="csv of crate names to publish (strict: must be dependency-closed); "
        "empty/'all' = every publishable crate",
    )
    ap.add_argument(
        "--stage-version",
        default=os.environ.get("STAGE_VERSION", ""),
        help="rewrite every publishable crate's version to this exact value before "
        "publishing (internal staging rc disambiguator, e.g. 0.1.0-rc0), so a "
        "re-stage of the same release doesn't collide on the immutable registry. "
        "Empty = leave the committed versions as-is",
    )
    args = ap.parse_args()

    # ARTIFACTORY ONLY — never crates.io. Refuse a crates.io alias outright; every
    # `cargo publish` below passes --registry <this alias> and write_cargo_config
    # sets it as the default registry, so crates.io is never a publish target.
    if args.registry in ("crates-io", "crates.io", "crates_io"):
        print(
            "::error::refusing to publish to crates.io; this stages to Artifactory only",
            file=sys.stderr,
        )
        return 1

    # Reuse the wheel-upload token; registry locations are supplied by the workflow.
    token = os.environ.get("ARTIFACTORY_TOKEN", "")
    index = os.environ.get("ARTIFACTORY_CARGO_INDEX", "")
    if not token or not index:
        print(
            "::error::ARTIFACTORY_TOKEN and ARTIFACTORY_CARGO_INDEX must be set",
            file=sys.stderr,
        )
        return 1
    m = re.match(r"sparse\+(https://.+?)/api/cargo/([^/]+)/index/?$", index)
    if not m:
        print(
            "::error::ARTIFACTORY_CARGO_INDEX must be 'sparse+https://<host>/.../api/cargo/<repo>/index/'",
            file=sys.stderr,
        )
        return 1
    raw_base = f"{m.group(1)}/{m.group(2)}/crates"

    root = Path(args.root).resolve()
    meta = cargo_metadata(root)
    pkgs = publishable(meta)
    # Every workspace member's manifest (nested ones included — lib/sidecar/*,
    # lib/mocker/servers/*), for the version/registry rewrites below.
    members = set(meta["workspace_members"])
    member_manifests = sorted(
        Path(p["manifest_path"]) for p in meta["packages"] if p["id"] in members
    )
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
            print(
                f"::warning::skipping requested crate(s) not publishable on this branch: "
                f"{sorted(unknown)} (branch publishable set: {sorted(pkgs)})",
                file=sys.stderr,
            )
            selected -= unknown
        if not selected:
            print(
                "::error::none of the requested crates are publishable on this branch",
                file=sys.stderr,
            )
            return 1
        gaps = {
            n: sorted(
                {
                    d["name"]
                    for d in pkgs[n]["dependencies"]
                    if d["name"] in pkgs and d["name"] != n and d.get("kind") != "dev"
                }
                - selected
            )
            for n in selected
        }
        gaps = {n: g for n, g in gaps.items() if g}
        if gaps:
            for n, g in gaps.items():
                print(
                    f"::error::crate {n} depends on un-selected workspace crate(s) {g}",
                    file=sys.stderr,
                )
            print(
                "::error::crate subset is not dependency-closed; add the missing crate(s) or use 'all'",
                file=sys.stderr,
            )
            return 1
        order = [n for n in order if n in selected]
        print("Selected crates (dependency-closed):", " ".join(order))

    # Internal staging: rewrite the bumped workspace version to the rc-suffixed staging
    # version (e.g. 0.1.0 -> 0.1.0-rc0) so each re-stage publishes a distinct,
    # non-colliding version on the immutable registry. The expect-version gate below
    # then validates the new value, and cargo publish uploads it (--allow-dirty).
    stage_version = args.stage_version.strip()
    if stage_version:
        wm = re.search(
            r'\[workspace\.package\][^\[]*?\n\s*version\s*=\s*"([^"]+)"',
            (root / "Cargo.toml").read_text(),
        )
        if not wm:
            print(
                "::error::--stage-version: cannot read [workspace.package].version",
                file=sys.stderr,
            )
            return 1
        cur = wm.group(1)
        # STAGE_VERSION's base comes from pyproject.toml (release.yml); nothing
        # else gates pyproject against Cargo.toml, so a bump that missed one
        # file would otherwise publish crates under the wrong base silently.
        if not (stage_version == cur or stage_version.startswith(f"{cur}-")):
            print(
                f"::warning::stage-version '{stage_version}' does not extend the "
                f"workspace version '{cur}' — pyproject.toml and Cargo.toml may "
                f"have drifted",
                file=sys.stderr,
            )
        if cur != stage_version:
            n = rewrite_versions(
                [root / "Cargo.toml", *member_manifests], cur, stage_version
            )
            print(
                f"stage-version: rewrote {cur} -> {stage_version} across {n} manifest(s)"
            )
            for nm in pkgs:
                if pkgs[nm]["version"] == cur:
                    pkgs[nm]["version"] = stage_version

    # Fail fast BEFORE any build/publish if a crate carries an unexpected version
    # (e.g. a hardcoded version the bump missed) — never silently publish wrong tags.
    if args.expect_version:
        mismatched = [
            (n, pkgs[n]["version"])
            for n in order
            if pkgs[n]["version"] != args.expect_version
        ]
        if mismatched:
            for n, v in mismatched:
                print(
                    f"::error::crate {n} is at version {v}, expected {args.expect_version} (hardcoded/un-bumped version)",
                    file=sys.stderr,
                )
            print(
                f"::error::aborting before publish: {len(mismatched)} crate(s) carry an unexpected version; "
                "set 'version.workspace = true' (or fix Cargo.toml) and re-run",
                file=sys.stderr,
            )
            return 1

    env = dict(os.environ)
    try:
        dependency_registries = configure_dependency_registries(
            root,
            [root / "Cargo.toml", *member_manifests],
            meta.get("metadata") or {},
            args.registry,
            index,
            env,
        )
    except RuntimeError as error:
        print(f"::error::{error}", file=sys.stderr)
        return 1
    write_cargo_config(root, args.registry, index, dependency_registries)
    inject_registry(
        [root / "Cargo.toml", *(Path(pkgs[n]["manifest_path"]) for n in order)],
        order,
        args.registry,
        {n: pkgs[n]["version"] for n in order},
    )
    # cargo publish rejects deps without a registry version; drop optional,
    # private git deps from the crates being published (none on current main;
    # older release branches still carry e.g. aiconfigurator-core).
    strip_git_deps(root, [Path(pkgs[n]["manifest_path"]) for n in order])

    env[f"CARGO_REGISTRIES_{args.registry.upper()}_TOKEN"] = f"Bearer {token}"
    env["RUSTFLAGS"] = f"{env.get('RUSTFLAGS', '')} --cfg tokio_unstable".strip()
    # Only the upload PUT runs on PUBLISH_TOOLCHAIN (--no-verify skips compilation);
    # cargo check/update stay on the workspace's rust-toolchain.toml toolchain.
    r = subprocess.run(
        [
            "rustup",
            "toolchain",
            "install",
            "--profile",
            "minimal",
            "--no-self-update",
            PUBLISH_TOOLCHAIN,
        ],
        env=env,
    )
    if r.returncode != 0:
        print(
            f"::error::rustup install of publish toolchain {PUBLISH_TOOLCHAIN} failed",
            file=sys.stderr,
        )
        return 1
    # Resync Cargo.lock after the stage-version rewrite; fail loudly here rather
    # than as a confusing publish error later.
    if stage_version:
        r = subprocess.run(["cargo", "update", "--workspace"], cwd=root, env=env)
        if r.returncode != 0:
            print(
                "::error::cargo update --workspace failed after the stage-version rewrite",
                file=sys.stderr,
            )
            return 1

    # Fail-soft loop: a failed crate skips its dependents but not the rest.
    # Machine-readable lines consumed by the workflow:
    #   STAGED_CRATE=<name>            on the registry (this run or earlier)
    #   FAILED_CRATE=<name> <reason>   failed, or skipped due to a failed dep
    published = already = 0
    staged: list[str] = []
    failed: dict[str, str] = {}
    deps_of = {
        n: {
            d["name"]
            for d in pkgs[n]["dependencies"]
            if d["name"] in pkgs and d["name"] != n and d.get("kind") != "dev"
        }
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
            if (
                subprocess.run(
                    ["cargo", "check", "--manifest-path", manifest], cwd=root, env=env
                ).returncode
                != 0
            ):
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

    print(
        f"Done: {len(order)} crates ({published} published, {already} already present, "
        f"{len(staged)} staged, {len(failed)} failed)."
    )
    print(f"STAGED_CRATES={','.join(staged)}")
    if failed:
        print(f"FAILED_CRATES={','.join(failed)}")
        print(
            f"::error::{len(failed)} crate(s) not staged: {', '.join(failed)}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
