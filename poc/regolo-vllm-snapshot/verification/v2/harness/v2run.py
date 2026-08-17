#!/usr/bin/env python3
"""Production-only wiring for the sealed V2-A runner."""

import argparse
import hashlib
import importlib.util
import json
import os
import pathlib
import re
import stat
import sys


def _load_module(name):
    try:
        return __import__(name)
    except ModuleNotFoundError:
        spec = importlib.util.spec_from_file_location(name + "_local", pathlib.Path(__file__).with_name(name + ".py"))
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module


_live = _load_module("v2_live")
_production = _load_module("v2_production")

LiveRunner = _live.LiveRunner
SubprocessTransport = _production.SubprocessTransport
ProductionCheckpointValidator = _production.ProductionCheckpointValidator
ProductionClusterPreflight = _production.ProductionClusterPreflight
ProductionCollector = _production.ProductionCollector
CacheAdvisor = _production.CacheAdvisor
SEAL_ROOT = pathlib.Path(__file__).resolve().parents[3]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _safe_bytes(path):
    """Read one regular, non-symlinked local file without following it."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(os.fspath(path), flags)
    except OSError as exc:
        raise ValueError("cannot safely read local input") from exc
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise ValueError("local input is not a regular file")
        chunks = []
        while True:
            chunk = os.read(fd, 65536)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        os.close(fd)


def _json_file(path, digest=None, *, object_only=True):
    if digest is not None and (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
        raise ValueError("invalid local digest")
    body = _safe_bytes(path)
    if digest is not None and hashlib.sha256(body).hexdigest() != digest:
        raise ValueError("local input digest mismatch")
    try:
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError("duplicate JSON key")
                result[key] = value
            return result
        value = json.loads(body.decode("utf-8"), object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError("invalid local JSON") from exc
    if object_only and not isinstance(value, dict):
        raise ValueError("local JSON must be an object")
    return value


def _exact(value, fields, name):
    if not isinstance(value, dict) or set(value) != set(fields):
        raise ValueError("invalid " + name)


def _attestation(path, digest):
    value = _json_file(path, digest)
    _exact(value, {"checkpoint", "agent", "pvc", "pv"}, "checkpoint attestation")
    checkpoint_fields = {
        "id", "compatibility_hash", "location", "total_size_bytes", "pages_12_size_bytes",
        "rootfs_size_bytes", "metadata_size_bytes", "manifest_sha256",
    }
    if set(value["checkpoint"]) != checkpoint_fields | {"inventory"}:
        raise ValueError("invalid checkpoint attestation")
    inventory = value["checkpoint"]["inventory"]
    _exact(inventory, {"regular_file_count", "regular_file_size_bytes", "inventory_sha256"}, "checkpoint inventory")
    if (
        not isinstance(inventory["regular_file_count"], int) or isinstance(inventory["regular_file_count"], bool)
        or inventory["regular_file_count"] <= 0
        or not isinstance(inventory["regular_file_size_bytes"], int) or isinstance(inventory["regular_file_size_bytes"], bool)
        or inventory["regular_file_size_bytes"] <= 0
        or not isinstance(inventory["inventory_sha256"], str)
        or not _SHA256.fullmatch(inventory["inventory_sha256"])
    ):
        raise ValueError("invalid checkpoint inventory")
    _exact(value["agent"], {"namespace", "name", "uid", "image", "node"}, "checkpoint attestation")
    _exact(value["pvc"], {"name", "uid", "pv"}, "checkpoint attestation")
    _exact(value["pv"], {"uid", "local_path", "claim_uid", "node", "reclaim_policy"}, "checkpoint attestation")
    return value


def _sealed_path(root, relative):
    """Resolve a SHA256SUMS entry without accepting escapes or symlinks."""
    pure = pathlib.PurePosixPath(relative)
    if not relative or pure.is_absolute() or ".." in pure.parts or any(not part for part in pure.parts):
        raise ValueError("unsafe sealed path")
    path = pathlib.Path(root).joinpath(*pure.parts)
    try:
        current = pathlib.Path(root)
        for part in pure.parts:
            current = current / part
            if stat.S_ISLNK(os.lstat(current).st_mode):
                raise ValueError("sealed path is a symlink")
    except OSError as exc:
        raise ValueError("cannot safely read sealed input") from exc
    return path


def _verify_seal():
    manifest = pathlib.Path(SEAL_ROOT) / "verification" / "v2" / "SHA256SUMS"
    body = _safe_bytes(manifest)
    entries = {}
    try:
        lines = body.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError("invalid SHA256SUMS entry") from exc
    for line in lines:
        parts = line.split("  ")
        if len(parts) != 2 or len(parts[0]) != 64 or any(c not in "0123456789abcdef" for c in parts[0]):
            raise ValueError("invalid SHA256SUMS entry")
        digest, relative = parts
        if relative in entries:
            raise ValueError("duplicate SHA256SUMS entry")
        entries[relative] = digest
    if not entries:
        raise ValueError("empty SHA256SUMS")
    for relative, digest in entries.items():
        if hashlib.sha256(_safe_bytes(_sealed_path(SEAL_ROOT, relative))).hexdigest() != digest:
            raise ValueError("sealed file digest mismatch")
    return hashlib.sha256(body).hexdigest()


def _execution_digest(**identities):
    return hashlib.sha256(
        json.dumps(identities, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _content_identity(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _campaign(path):
    value = _json_file(path)
    _exact(value, {"namespace", "node", "snapshotctl", "snapshotctl_sha256", "checkpoint"}, "campaign")
    checkpoint = value["checkpoint"]
    _exact(checkpoint, {"checkpoint_id", "compatibility_hash", "attestation_path", "attestation_sha256"}, "campaign checkpoint")
    if (not isinstance(value["namespace"], str) or not value["namespace"]
            or not isinstance(value["node"], str) or not value["node"]
            or not isinstance(value["snapshotctl"], str) or not pathlib.PurePath(value["snapshotctl"]).is_absolute()
            or not isinstance(value["snapshotctl_sha256"], str)
            or not isinstance(checkpoint["attestation_path"], str) or not pathlib.PurePath(checkpoint["attestation_path"]).is_absolute()
            or not isinstance(checkpoint["attestation_sha256"], str)):
        raise ValueError("invalid campaign values")
    return value


def _cluster(path, digest, agent_name):
    value = _json_file(path, digest)
    _exact(value, {"namespace", "node", "agent", "reserves"}, "cluster attestation")
    _exact(value["agent"], {"name", "uid", "image", "node"}, "cluster attestation")
    required = {"name", "uid", "container", "image", "node", "gpu_uuid"}
    reserves = value["reserves"]
    if (not isinstance(value["namespace"], str) or not value["namespace"]
            or not isinstance(value["node"], str) or not value["node"]
            or value["agent"].get("name") != agent_name
            or not isinstance(reserves, list) or len(reserves) != 3
            or any(not isinstance(row, dict) or set(row) != required for row in reserves)
            or any(row.get("node") != value["node"] for row in reserves)):
        raise ValueError("invalid cluster attestation")
    return value


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", type=pathlib.Path, required=True)
    parser.add_argument("--auth", type=pathlib.Path, required=True)
    parser.add_argument("--schedule", type=pathlib.Path, required=True)
    parser.add_argument("--key", type=pathlib.Path, required=True)
    parser.add_argument("--ledger", type=pathlib.Path, required=True)
    parser.add_argument("--campaign", type=pathlib.Path, required=True)
    parser.add_argument("--artifact-dir", type=pathlib.Path, required=True)
    parser.add_argument("--cluster-attestation", type=pathlib.Path, required=True)
    parser.add_argument("--cluster-attestation-sha", required=True)
    parser.add_argument("--agent", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv=None):
    try:
        args = _parser().parse_args(argv)
        if args.limit is not None and args.limit < 0:
            raise ValueError("invalid limit")
        # Read every local control input safely before wiring a transport.
        seal_digest = _verify_seal()
        lane = _json_file(args.lane)
        authorization = _json_file(args.auth)
        if authorization != {"execution_authorized": True}:
            raise ValueError("execution authorization is required")
        schedule = _json_file(args.schedule, object_only=False)
        key = _json_file(args.key)
        campaign = _campaign(args.campaign)
        cluster = _cluster(args.cluster_attestation, args.cluster_attestation_sha, args.agent)
        if campaign["namespace"] != cluster["namespace"] or campaign["node"] != cluster["node"]:
            raise ValueError("campaign and cluster attestation disagree")
        checkpoint = _attestation(
            campaign["checkpoint"]["attestation_path"], campaign["checkpoint"]["attestation_sha256"]
        )
        checkpoint_metadata = checkpoint["checkpoint"]
        collector_attestation = {
            "checkpoint_id": checkpoint_metadata["id"],
            "compatibility_hash": checkpoint_metadata["compatibility_hash"],
            "checkpoint_size_bytes": checkpoint_metadata["total_size_bytes"],
            "pages_12_size_bytes": checkpoint_metadata["pages_12_size_bytes"],
            "rootfs_size_bytes": checkpoint_metadata["rootfs_size_bytes"],
            "metadata_size_bytes": checkpoint_metadata["metadata_size_bytes"],
            "checkpoint_inventory": checkpoint_metadata.get("inventory"),
        }
        execution_digest = _execution_digest(
            seal=seal_digest,
            lane=_content_identity(lane),
            authorization=_content_identity(authorization),
            schedule=_content_identity(schedule),
            key=_content_identity(key),
            campaign=_content_identity({
                "namespace": campaign["namespace"],
                "node": campaign["node"],
                "snapshotctl_sha256": campaign["snapshotctl_sha256"],
                "checkpoint": {
                    "checkpoint_id": campaign["checkpoint"]["checkpoint_id"],
                    "compatibility_hash": campaign["checkpoint"]["compatibility_hash"],
                    "attestation_sha256": campaign["checkpoint"]["attestation_sha256"],
                },
            }),
            checkpoint_attestation=_content_identity(checkpoint),
            cluster_attestation=_content_identity(cluster),
        )
        transport = SubprocessTransport(timeout_s=1800)
        validator = ProductionCheckpointValidator(campaign["namespace"], args.agent, transport, timeout_s=1800)
        preflight = ProductionClusterPreflight(
            cluster["namespace"], cluster["node"], cluster["reserves"], cluster["agent"], transport, timeout_s=1800
        )
        collector = ProductionCollector(
            campaign["namespace"], args.agent, args.artifact_dir, collector_attestation, transport, timeout_s=1800
        )
        advisor = CacheAdvisor(campaign["namespace"], args.agent, checkpoint["checkpoint"]["location"], transport, timeout_s=1800)
        runner = LiveRunner(
            lane_path=args.lane, authorization_path=args.auth, schedule_path=args.schedule,
            key_path=args.key, ledger_path=args.ledger, campaign=campaign,
            command_runner=transport, collector=collector, artifact_dir=args.artifact_dir,
            checkpoint_inventory=checkpoint_metadata.get("inventory"),
            checkpoint_validator=validator, cluster_preflight=preflight, fadvise=advisor.advise_inventory,
            execution_digest=execution_digest,
            dry_run=args.dry_run,
        )
        result = runner.run(limit=args.limit)
        print(json.dumps(result, sort_keys=True, separators=(",", ":")))
        return 0
    except (OSError, ValueError, RuntimeError) as exc:
        print("v2run: " + str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
