# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Detect drift in tracked base-image SBOMs.

Reads container/compliance/base_sboms/manifest.json and, for each entry,
runs three structural checks against the live registry:

  1. from_image digest matches the recorded from_digest.
     Catches the vendor pushing the same tag with different bits.
  2. baseline_image digest matches the recorded baseline_digest.
     Catches NGC (or whoever) rebuilding the floor underneath us.
  3. The from_image's current layer list still starts with the
     baseline_image's current full layer list (layer-prefix invariant).
     Catches the subtle case: vendor silently switched their FROM line
     to a different upstream while keeping the from_image tag stable.

Any one failing surfaces as build-blocking output with the action to
take: rerun capture_baseline_sbom.py for the affected entry, and update
manifest.json with the new digests.

Resolution uses `docker buildx imagetools inspect --raw <ref>` and
hashes the canonical manifest bytes locally; that hash is the OCI
manifest digest by spec. Same approach for the per-platform layer
lists, with one extra fetch for multi-arch indices.

Exit codes:
  0  all manifest entries match the live registry, OR manifest is empty
  1  one or more entries drifted (output explains which and how to fix)
  2  registry/network failure on at least one entry (treated as drift -
     CI should not silently pass if we can't verify)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

# Reuse digest + layer-list helpers from the capture tool. Both files
# live in the same dir, so the bare module import works whether the
# script is invoked as `python3 .../check_drift.py` or via `-m`.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from capture_baseline_sbom import (  # noqa: E402
    resolve_index_digest,
    resolve_platform_layers,
)

logger = logging.getLogger(__name__)

_MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
_DEFAULT_PLATFORM = "linux/amd64"


def check_entry(entry: dict) -> list[str]:
    """Run all three checks for one manifest entry. Return list of problem strings."""
    problems: list[str] = []

    required = (
        "from_image",
        "from_tag",
        "from_digest",
        "baseline_image",
        "baseline_tag",
        "baseline_digest",
    )
    missing = [k for k in required if not entry.get(k)]
    if missing:
        return [f"manifest entry malformed (missing {missing}): {entry!r}"]

    from_ref = f"{entry['from_image']}:{entry['from_tag']}"
    baseline_ref = f"{entry['baseline_image']}:{entry['baseline_tag']}"
    platform = entry.get("platform", _DEFAULT_PLATFORM)

    # Check 1: from_image digest still matches.
    try:
        from_current = resolve_index_digest(from_ref)
    except subprocess.CalledProcessError as exc:
        return [
            f"{from_ref}: from_image registry lookup failed "
            f"({(exc.stderr or b'').decode().strip()})"
        ]
    if from_current != entry["from_digest"]:
        problems.append(
            f"{from_ref}: FROM-IMAGE DRIFT\n"
            f"      recorded: {entry['from_digest']}\n"
            f"      current : {from_current}\n"
            f"      action  : rerun capture_baseline_sbom.py "
            f"--from {from_ref} --baseline {baseline_ref}"
        )

    # Check 2: baseline_image digest still matches.
    try:
        baseline_current = resolve_index_digest(baseline_ref)
    except subprocess.CalledProcessError as exc:
        problems.append(
            f"{baseline_ref}: baseline_image registry lookup failed "
            f"({(exc.stderr or b'').decode().strip()})"
        )
        baseline_current = None
    if baseline_current and baseline_current != entry["baseline_digest"]:
        problems.append(
            f"{baseline_ref}: BASELINE DRIFT (underneath {from_ref})\n"
            f"      recorded: {entry['baseline_digest']}\n"
            f"      current : {baseline_current}\n"
            f"      action  : rerun capture_baseline_sbom.py "
            f"--from {from_ref} --baseline {baseline_ref}"
        )

    # Check 3: layer-prefix invariant. Skipped only if explicitly opted out
    # at capture time (recorded on the entry).
    if entry.get("layer_prefix_check_skipped"):
        logger.debug(
            "skipping layer-prefix check for %s (opted out at capture)", from_ref
        )
        return problems

    try:
        baseline_layers = resolve_platform_layers(baseline_ref, platform)
        from_layers = resolve_platform_layers(from_ref, platform)
    except (subprocess.CalledProcessError, ValueError) as exc:
        problems.append(
            f"{from_ref}/{baseline_ref}: layer-prefix lookup failed ({exc})"
        )
        return problems

    n = len(baseline_layers)
    if from_layers[:n] != baseline_layers:
        problems.append(
            f"{from_ref}: LAYER-PREFIX MISMATCH (built on a DIFFERENT base than recorded)\n"
            f"      recorded baseline: {baseline_ref}\n"
            f"      baseline layers ({n}): {baseline_layers}\n"
            f"      from_image's first {n} layers do not match.\n"
            f"      from_image layers[:{n}]: {from_layers[:n]}\n"
            f"      action: investigate; vendor may have switched their FROM line. "
            f"Either pick a different baseline or pin the from_image to a "
            f"version that does match."
        )

    return problems


_CAPTURE_SCRIPT = "container/compliance/base_sboms/capture_baseline_sbom.py"


def _remediation_command(entry: dict) -> str:
    """The exact capture_baseline_sbom.py invocation that refreshes this entry."""
    cmd = (
        f"python3 {_CAPTURE_SCRIPT} \\\n"
        f"    --from {entry['from_image']}:{entry['from_tag']} \\\n"
        f"    --baseline {entry['baseline_image']}:{entry['baseline_tag']}"
    )
    platform = entry.get("platform", _DEFAULT_PLATFORM)
    if platform != _DEFAULT_PLATFORM:
        cmd += f" \\\n    --platform {platform}"
    return cmd


def _render_markdown(failed: list[tuple[dict, list[str]]], total: int) -> str:
    """Build a GitHub-flavoured-markdown drift report for the job summary."""
    if not failed:
        return (
            "## ✅ Base-image drift check passed\n\n"
            f"All **{total}** tracked baseline SBOM(s) still match the registry.\n"
        )

    # Real drift (fix = re-capture) vs transient registry failures (fix = retry).
    recapture = [(e, p) for e, p in failed if not all("lookup failed" in s for s in p)]
    transient = [(e, p) for e, p in failed if all("lookup failed" in s for s in p)]

    out = [
        "## ⚠️ Base-image drift detected\n",
        f"**{len(failed)}** of **{total}** tracked baseline SBOM(s) no longer match "
        "the registry. The committed baselines are stale, so per-image NOTICES are "
        "subtracting an out-of-date floor.\n",
    ]

    if recapture:
        seen: set[str] = set()
        blocks: list[str] = []
        for entry, _ in recapture:
            ref = f"{entry['from_image']}:{entry['from_tag']}"
            if ref in seen:  # one re-capture fixes all of an entry's failed checks
                continue
            seen.add(ref)
            blocks.append(f"# {ref}\n{_remediation_command(entry)}")
        out.append("### Re-capture the drifted baselines\n")
        out.append(
            "Run each command, then commit the regenerated SBOM + the updated "
            "`container/compliance/base_sboms/manifest.json` and open a PR — its CI "
            "re-runs the license policy gate against the new SBOM:\n"
        )
        out.append("```bash\n" + "\n\n".join(blocks) + "\n```\n")

    if transient:
        refs = ", ".join(f"`{e['from_image']}:{e['from_tag']}`" for e, _ in transient)
        out.append(
            "### ⏳ Registry lookup failed (transient)\n\n"
            f"Could not reach the registry for: {refs}. Treated as drift so CI never "
            "silently passes — re-run the job; if it persists, check registry "
            "auth/availability.\n"
        )

    detail = "\n".join(f"- {p}" for _, problems in failed for p in problems)
    out.append(
        "<details><summary>Full drift report</summary>\n\n"
        f"```\n{detail}\n```\n\n</details>\n"
    )
    return "\n".join(out)


def _write_step_summary(markdown: str) -> None:
    """Append the markdown report to GITHUB_STEP_SUMMARY when running in CI."""
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(markdown + "\n")
    except OSError as exc:
        logger.warning("could not write GITHUB_STEP_SUMMARY: %s", exc)


def check(manifest_path: Path) -> int:
    if not manifest_path.is_file():
        logger.error("manifest.json not found at %s", manifest_path)
        return 2

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("entries", []) or []
    if not entries:
        print("base_sboms manifest is empty; nothing to verify.")
        _write_step_summary(
            "## ✅ Base-image drift check passed\n\n"
            "The baseline SBOM manifest is empty; nothing to verify.\n"
        )
        return 0

    failed: list[tuple[dict, list[str]]] = []
    all_problems: list[str] = []
    for entry in entries:
        problems = check_entry(entry)
        if problems:
            failed.append((entry, problems))
            all_problems.extend(problems)
        else:
            from_ref = f"{entry['from_image']}:{entry['from_tag']}"
            print(f"  {from_ref} OK ({entry['from_digest'][:19]}...)")

    # Always emit a job-summary report (✅ on success, fix-it commands on drift).
    _write_step_summary(_render_markdown(failed, len(entries)))

    if not all_problems:
        print(f"All {len(entries)} tracked base images current.")
        return 0

    print("", file=sys.stderr)
    print(
        f"Base-image drift detected ({len(all_problems)} problem"
        f"{'' if len(all_problems) == 1 else 's'}):",
        file=sys.stderr,
    )
    for p in all_problems:
        print(f"  - {p}", file=sys.stderr)
    # Distinguish drift from registry failure for the exit code — both fail
    # CI, but the cause matters for triage.
    if any("registry lookup failed" in p or "lookup failed" in p for p in all_problems):
        return 2
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="check_drift",
        description=(
            "Verify base_sboms/manifest.json entries against the live registry: "
            "from_digest, baseline_digest, and the layer-prefix invariant."
        ),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=_MANIFEST_PATH,
        help="Path to manifest.json (default: %(default)s)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    return check(args.manifest)


if __name__ == "__main__":
    sys.exit(main())
