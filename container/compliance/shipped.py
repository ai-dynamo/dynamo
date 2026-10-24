# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Supplement wheel inventories with packages found in shipped files.

Syft provides discovery, not approval: every additional component still goes
through the existing license-policy gate. Keep source declarations, fixtures,
and foreign-platform resources: distributing a file is sufficient to inventory it.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import tarfile
import tomllib
import urllib.parse
import urllib.request
from dataclasses import asdict
from pathlib import Path
from posixpath import normpath

from . import overrides
from .generators import common
from .generators import dpkg as dpkg_generator
from .generators import python as python_generator
from .identity import ARTIFACT_ECOSYSTEMS, identity
from .policy.validate import load_policy, validate_row

logger = logging.getLogger(__name__)

_TYPES = ARTIFACT_ECOSYSTEMS
# No dynamic-linking assumption for archive retention. An allowed license can
# still require source distribution (MPL), so policy.allow is NOT this list.
_PERMISSIVE = frozenset(
    {
        "MIT",
        "Apache-2.0",
        "BSD-2-Clause",
        "BSD-3-Clause",
        "ISC",
        "0BSD",
        "Unlicense",
        "CC0-1.0",
        "Zlib",
        "BSL-1.0",
        "Unicode-3.0",
    }
)


def _path(path: str) -> str:
    return normpath("/" + path.lstrip("/"))


def _read_url(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "dynamo-compliance/1 (https://github.com/ai-dynamo/dynamo)"
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read()


def _license_files(archive: tarfile.TarFile) -> str:
    texts = []
    for member in sorted(archive.getmembers(), key=lambda m: m.name):
        # Root-level upstream notices, including LICENSE-MIT/LICENSE-APACHE.
        path = Path(member.name)
        if not member.isfile() or len(path.parts) != 2:
            continue
        if not path.name.upper().startswith(
            ("LICENSE", "COPYING", "NOTICE", "COPYRIGHT")
        ):
            continue
        stream = archive.extractfile(member)
        if stream is not None:
            with stream:
                texts.append(
                    f"--- {path.name} ---\n\n"
                    + stream.read().decode("utf-8", errors="replace")
                )
    return "\n\n".join(texts)


def _crate(name: str, version: str, cache: Path) -> tuple[str, str, str, str]:
    """Read exact-version metadata and checksum-verified upstream license texts."""
    # Components are untrusted scanner input, never path/URL fragments verbatim.
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name) or not re.fullmatch(
        r"[A-Za-z0-9.+_-]+", version
    ):
        raise ValueError(f"invalid registry crate identity: {name}@{version}")
    cache.mkdir(parents=True, exist_ok=True)
    metadata = cache / f"{name}@{version}.json"
    if not metadata.exists():
        normalized = name.lower()
        if len(normalized) < 3:
            prefix = str(len(normalized))
        elif len(normalized) == 3:
            prefix = "3/" + normalized[0]
        else:
            prefix = normalized[:2] + "/" + normalized[2:4]
        # The static registry index supplies the published archive checksum.
        # Read licensing from that archive's manifest, not a mutable API or a
        # second version's metadata. This needs no Rust toolchain in the image.
        index = _read_url(f"https://index.crates.io/{prefix}/{normalized}")
        versions = [json.loads(line) for line in index.splitlines() if line.strip()]
        info = next((entry for entry in versions if entry["vers"] == version), None)
        if info is None:
            raise ValueError(f"crate version missing from registry: {name}@{version}")
        metadata.write_text(json.dumps(info))
    info = json.loads(metadata.read_text())
    if info["vers"] != version:
        raise ValueError(f"cached crate version mismatch: {name}@{version}")
    archive_path = cache / f"{name}@{version}.crate"
    url = f"https://static.crates.io/crates/{name}/{name}-{version}.crate"
    if not archive_path.exists():
        archive_path.write_bytes(_read_url(url))
    body = archive_path.read_bytes()
    checksum = hashlib.sha256(body).hexdigest()
    if checksum != info["cksum"]:
        raise ValueError(f"crate checksum mismatch: {name}@{version}")
    with tarfile.open(fileobj=io.BytesIO(body)) as archive:
        manifest = archive.extractfile(f"{name}-{version}/Cargo.toml")
        if manifest is None:
            raise ValueError(f"crate manifest missing: {name}@{version}")
        with manifest:
            package = tomllib.loads(manifest.read().decode("utf-8"))["package"]
        text = _license_files(archive)
        if package.get("license-file"):
            license_path = f"{name}-{version}/" + package["license-file"]
            notice = archive.extractfile(license_path)
            if notice is not None:
                with notice:
                    text += "\n\n" + notice.read().decode("utf-8", errors="replace")
    spdx = (
        overrides.lookup("rust", name, version)
        or package.get("license")
        or common.UNKNOWN
    )
    # Cargo historically accepted slash-separated alternatives. Normalize only
    # that legacy form at the Cargo boundary, never in the SPDX policy parser.
    if re.fullmatch(r"[A-Za-z0-9.+-]+(?:/[A-Za-z0-9.+-]+)+", spdx):
        spdx = spdx.replace("/", " OR ")
    return spdx, text, url, checksum


def _licenses(artifact: dict) -> tuple[str, str]:
    licenses = artifact.get("licenses") or []
    if artifact.get("type") == "go-module":
        # Remote Go discovery also finds license *templates* in source files
        # (e.g. cobra/cmd/license_gpl.go). Those strings do not relicense the
        # module. Retain actual license/notice files and declared metadata;
        # the unfiltered evidence is preserved in shipped-provenance.json.
        licenses = [
            entry
            for entry in licenses
            if not entry.get("urls")
            or any(
                re.fullmatch(
                    r"(LICENSE|LICENCE|COPYING|NOTICE)([.-](txt|md|rst|apache|mit|bsd))?",
                    url.rsplit("/", 1)[-1],
                    re.IGNORECASE,
                )
                for url in entry["urls"]
            )
        ]
    expressions = sorted(
        {entry["spdxExpression"] for entry in licenses if entry.get("spdxExpression")}
    )
    # Multiple independent license detections are obligations, not an invented OR.
    expression = " AND ".join(f"({expr})" for expr in expressions) or common.UNKNOWN
    text = "\n\n".join(
        sorted({entry["contents"] for entry in licenses if entry.get("contents")})
    )
    return expression, text


def _local_crate(artifact: dict, root: Path) -> tuple[str, str]:
    """Resolve local/path crates only from a manifest naming that exact crate."""
    for location in artifact.get("locations", []):
        lock = root / _path(location["path"]).lstrip("/")
        if lock.name != "Cargo.lock":
            continue
        for manifest in sorted(lock.parent.rglob("Cargo.toml")):
            data = tomllib.loads(manifest.read_text())
            package = data.get("package", {})
            if package.get("name") != artifact["name"]:
                continue
            declared_version = package.get("version")
            if (
                isinstance(declared_version, str)
                and declared_version != artifact["version"]
            ):
                continue
            expr = package.get("license")
            if isinstance(expr, dict) and expr.get("workspace"):
                for ancestor in manifest.parents:
                    workspace_manifest = ancestor / "Cargo.toml"
                    if workspace_manifest.is_file():
                        workspace = tomllib.loads(workspace_manifest.read_text()).get(
                            "workspace", {}
                        )
                        expr = workspace.get("package", {}).get("license")
                        if isinstance(expr, str):
                            break
                    if ancestor == root:
                        break
            texts = []
            for path in sorted(manifest.parent.iterdir()):
                if path.is_file() and path.name.upper().startswith(
                    ("LICENSE", "COPYING", "NOTICE")
                ):
                    texts.append(path.read_text(errors="replace"))
            return expr if isinstance(expr, str) else common.UNKNOWN, "\n\n".join(texts)
    return common.UNKNOWN, ""


def _owners(artifacts: list[dict]) -> dict[str, list[dict]]:
    owners: dict[str, list[dict]] = {}
    for artifact in artifacts:
        if artifact["type"] != "python":
            continue
        metadata = artifact.get("metadata", {})
        site = metadata.get("sitePackagesRootPath")
        if not site:
            continue
        for entry in metadata.get("files", []):
            owners.setdefault(_path(site + "/" + entry["path"]), []).append(artifact)
    return owners


def _owning_project(
    artifact: dict, owners: dict[str, list[dict]], root: Path
) -> tuple[str, str, str]:
    """License a synthetic/local project from its exact recorded wheel owner.

    This applies only to the project's own identity, never its registry
    dependencies or arbitrary bundled binaries. Conflicting/absent owners fail.
    """
    paths = [_path(loc["path"]) for loc in artifact.get("locations", [])]
    owning = [owners.get(path, []) for path in paths]
    if not owning or not all(len(items) == 1 for items in owning):
        return common.UNKNOWN, "", ""
    if len({items[0]["id"] for items in owning}) != 1:
        return common.UNKNOWN, "", ""
    owner = owning[0][0]
    expr, text, source = _python_metadata(owner, root)
    return expr, text, source


def _python_metadata(artifact: dict, root: Path) -> tuple[str, str, str]:
    """Use installed metadata and the same resolver as the wheel inventory."""
    for location in artifact.get("locations", []):
        path = root / _path(location["path"]).lstrip("/")
        if path.is_dir() and path.name.endswith(".egg-info"):
            path = path / "PKG-INFO"
        if path.name not in {"METADATA", "PKG-INFO"} and not path.name.endswith(
            ".egg-info"
        ):
            continue
        if not path.is_file():
            continue
        (
            name,
            version,
            expr,
            free,
            classifiers,
        ) = python_generator._extract_metadata_field(path.read_text(errors="replace"))
        if (
            identity("python", name or "") != identity("python", artifact["name"])
            or version != artifact["version"]
        ):
            raise ValueError(f"installed Python metadata mismatch: {path}")
        spdx = python_generator._resolve_spdx(name, version, expr, free, classifiers)
        text = python_generator._read_license_text(path.parent) or ""
        return spdx, text, artifact.get("purl", "")
    spdx, text = _licenses(artifact)
    return spdx, text, artifact.get("purl", "")


def _system_python(
    artifact: dict, artifacts: list[dict], root: Path
) -> tuple[str, str, str]:
    """Resolve unknown system distributions from their exact dpkg file owner."""
    paths = {
        _path(loc["path"])
        for loc in artifact.get("locations", [])
        if loc.get("annotations", {}).get("evidence") == "primary"
    }
    if not paths:
        return common.UNKNOWN, "", ""
    owners = []
    for package in artifacts:
        if package["type"] != "deb":
            continue
        files = {
            _path(entry["path"])
            for entry in package.get("metadata", {}).get("files", [])
        }
        if paths <= files:
            owners.append(package)
    if len(owners) != 1:
        return common.UNKNOWN, "", ""
    package = owners[0]
    spdx = dpkg_generator._resolve_license(package["name"], package["version"], root)
    copyright_path = root / "usr/share/doc" / package["name"] / "copyright"
    text = (
        copyright_path.read_text(errors="replace") if copyright_path.is_file() else ""
    )
    return spdx, text, package.get("purl", "")


def _resource(
    artifact: dict, owners: dict[str, list[dict]], root: Path
) -> tuple[str, str, str]:
    """Explicit resource-specific licensing; never inherit a wheel's license blindly.

    Rules below describe file families with reviewed ownership, not exceptions
    to policy. Unrecognized resources remain UNKNOWN and fail the policy gate.
    """
    resolutions = []
    for location in artifact.get("locations", []):
        before = len(resolutions)
        path = _path(location["path"])
        candidates = owners.get(path, [])
        for owner in candidates:
            name = owner["name"]
            notice = "\n\n".join(
                python_generator._read_license_text(
                    (root / _path(loc["path"]).lstrip("/")).parent
                )
                or ""
                for loc in owner.get("locations", [])
                if loc["path"].endswith("/METADATA")
            )
            if name == "pulp" and "/pulp/solverdir/cbc/" in path:
                license_path = root / path.lstrip("/")
                license_path = license_path.parent / "coin-license.txt"
                if not license_path.is_file():
                    continue
                text = license_path.read_text()
                # The shipped solver's own notice controls, not PuLP's MIT license.
                if text.startswith("Eclipse Public License - v 1.0"):
                    resolutions.append(
                        ("EPL-1.0", text, "https://github.com/coin-or/Cbc")
                    )
            elif name == "setuptools" and re.search(
                r"/setuptools/(cli|gui)(-(32|64|arm64))?\.exe$", path
            ):
                resolutions.append(
                    (
                        "MIT",
                        notice,
                        "https://github.com/pypa/setuptools/blob/main/launcher.c",
                    )
                )
            elif name in {"distlib", "pip"} and re.search(
                r"/distlib/[tw](32|64|64-arm)\.exe$", path
            ):
                resolutions.append(
                    (
                        "BSD-2-Clause",
                        (
                            Path(__file__).parent
                            / "resource_notices/distlib-launcher.txt"
                        ).read_text(),
                        "https://github.com/pypa/distlib/blob/0.4.3/PC/launcher.c",
                    )
                )
            elif name == "pre-commit" and path.endswith(
                "/pre_commit/resources/empty_template_pre-commit-package-dev-1.rockspec"
            ):
                resolutions.append(
                    ("MIT", notice, "https://github.com/pre-commit/pre-commit")
                )
            elif (
                name == "setuptools"
                and "/pkg_resources/tests/data/my-test-package" in path
            ):
                resolutions.append(
                    ("MIT", notice, "https://github.com/pypa/setuptools")
                )
        # Each detected location must have a reviewed resolution.
        if len(resolutions) == before:
            return common.UNKNOWN, "", ""
    if not resolutions:
        return common.UNKNOWN, "", ""
    exprs = {entry[0] for entry in resolutions}
    return (
        " AND ".join(sorted(exprs)),
        "\n\n".join(sorted({e[1] for e in resolutions if e[1]})),
        resolutions[0][2],
    )


def supplement(
    scan: dict,
    existing: list[common.Component],
    root: Path,
    cache: Path,
    subtract: set[tuple[str, str]],
    output: Path,
    sources_output: Path | None = None,
) -> list[common.Component]:
    """Add discovered components, recording paths, upstream evidence and checksums."""
    sources_output = sources_output or output / "source-archives"
    sources_output.mkdir(parents=True, exist_ok=True)
    artifacts = scan["artifacts"]
    owners = _owners(artifacts)
    keys = {(identity(c.ecosystem, c.name), c.version) for c in existing}
    records = []
    added = []
    for artifact in sorted(
        artifacts, key=lambda a: (a["name"], a.get("version") or "", a["id"])
    ):
        name, version = artifact["name"], artifact.get("version") or "UNKNOWN"
        if (name, version) in subtract:
            continue
        ecosystem = _TYPES.get(artifact["type"], "native")
        if (identity(ecosystem, name), version) in keys:
            continue
        spdx, text = _licenses(artifact)
        source = artifact.get("purl") or ""
        checksum = ""
        if artifact["type"] == "rust-crate":
            origin = artifact.get("metadata", {}).get("source", "")
            if artifact.get("purl", "").startswith("pkg:cargo/") and origin in {
                "",
                "crates.io",
                "registry+https://github.com/rust-lang/crates.io-index",
            }:
                spdx, text, source, checksum = _crate(name, version, cache)
                expected = artifact.get("metadata", {}).get("checksum")
                if expected and checksum != expected:
                    raise ValueError(
                        f"shipped lockfile checksum mismatch: {name}@{version}"
                    )
                if any(token not in _PERMISSIVE for token in common._spdx_ids(spdx)):
                    dest = sources_output
                    dest.mkdir(parents=True, exist_ok=True)
                    (dest / f"{name}-{version}.crate").write_bytes(
                        (cache / f"{name}@{version}.crate").read_bytes()
                    )
            else:
                spdx, text = _local_crate(artifact, root)
                spdx = overrides.lookup("rust", name, version) or spdx
                if (
                    spdx == common.UNKNOWN
                    and not text
                    and not artifact.get("metadata", {}).get("source")
                ):
                    spdx, text, source = _owning_project(artifact, owners, root)
        elif (
            artifact["type"] == "python"
            and name != "my-test-package"
            and spdx == common.UNKNOWN
        ):
            spdx, text, source = _python_metadata(artifact, root)
            if spdx == common.UNKNOWN:
                spdx, text, source = _system_python(artifact, artifacts, root)
        elif artifact["type"] not in _TYPES or name == "my-test-package":
            spdx, text, source = _resource(artifact, owners, root)
        elif (
            artifact["type"] == "go-module"
            and spdx == common.UNKNOWN
            and name.startswith("go.etcd.io/etcd/")
        ):
            # etcd's release archive contains local workspace modules whose Go
            # buildinfo has no published module version. All go.etcd.io/etcd/*
            # modules use the release's Apache license; third-party modules do
            # not enter this mapping. The native release row comes from the
            # same Dockerfile COPY of the downloaded release directory.
            release = next(
                (c for c in existing if c.ecosystem == "native" and c.name == "etcd"),
                None,
            )
            locations = artifact.get("locations", [])
            if (
                release
                and locations
                and all(
                    _path(loc["path"]).startswith("/usr/local/bin/etcd/")
                    for loc in locations
                )
            ):
                spdx, text = release.spdx, release.license_text or ""
                source = f"https://github.com/etcd-io/etcd/tree/{release.version}"
        elif artifact["type"] == "go-module" and name == "command-line-arguments":
            # Go's synthetic main package belongs to the wheel that compiled it;
            # all independently named module dependencies remain separate rows.
            spdx, text, source = _owning_project(artifact, owners, root)
        if (
            artifact["type"] == "go-module"
            and spdx != common.UNKNOWN
            and version != "UNKNOWN"
            and any(token not in _PERMISSIVE for token in common._spdx_ids(spdx))
        ):
            # Escaped module paths are the Go proxy's canonical source URLs.
            escaped = re.sub(r"[A-Z]", lambda match: "!" + match.group().lower(), name)
            url = (
                "https://proxy.golang.org/"
                + urllib.parse.quote(escaped, safe="/!")
                + "/@v/"
                + urllib.parse.quote(version, safe="")
                + ".zip"
            )
            body = _read_url(url)
            checksum = hashlib.sha256(body).hexdigest()
            (sources_output / (checksum + ".zip")).write_bytes(body)
            source = url
        canonical = not bool(text)
        if canonical:
            text = common.spdx_license_text(spdx) or ""
        component = common.Component(
            ecosystem, name, version, spdx, source or None, text or None, canonical
        )
        added.append(component)
        component_record = asdict(component)
        component_record.pop("license_text")
        evidence = []
        for entry in artifact.get("licenses", []):
            record = {key: value for key, value in entry.items() if key != "contents"}
            if entry.get("contents"):
                record["contents_sha256"] = hashlib.sha256(
                    entry["contents"].encode()
                ).hexdigest()
            evidence.append(record)
        records.append(
            {
                "component": component_record,
                "artifact_id": artifact["id"],
                "cataloger": artifact["foundBy"],
                "locations": artifact.get("locations", []),
                "owning_packages": [
                    {key: owner.get(key) for key in ("id", "name", "version", "purl")}
                    for owner in {
                        owner["id"]: owner
                        for location in artifact.get("locations", [])
                        for owner in owners.get(_path(location["path"]), [])
                    }.values()
                ],
                "source_sha256": checksum,
                "license_evidence": evidence,
            }
        )
    output.mkdir(parents=True, exist_ok=True)
    (output / "shipped-provenance.json").write_text(
        json.dumps(records, indent=2) + "\n"
    )
    return added


def write_license_review(
    components: list[common.Component], policy_path: Path, output: Path
) -> None:
    """Report reciprocal/restricted/unknown licenses, even if policy allows them."""
    policy = load_policy(policy_path)
    records = []
    for component in sorted(components, key=lambda c: (c.ecosystem, c.name, c.version)):
        terms = [
            term
            for term in common._spdx_ids(component.spdx)
            if re.match(r"(?:A?GPL|LGPL|MPL|EPL|CPL|CDDL|EUPL|OSL|BUSL|SSPL)-", term)
            or term.startswith("LicenseRef-")
            or term == common.UNKNOWN
        ]
        violation = validate_row(
            policy,
            component.ecosystem,
            component.name,
            component.version,
            component.spdx,
        )
        if terms or violation:
            records.append(
                {
                    "ecosystem": component.ecosystem,
                    "name": component.name,
                    "version": component.version,
                    "spdx": component.spdx,
                    "review_terms": terms,
                    "source_url": component.source_url,
                    "policy_violation": asdict(violation) if violation else None,
                }
            )
    (output / "license-review.json").write_text(json.dumps(records, indent=2) + "\n")
