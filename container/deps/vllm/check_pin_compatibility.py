# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Resolve container/context.yaml's vLLM pins against PyPI BEFORE building.

`context.yaml` pins several packages independently, but they constrain each
other, and the resolver only says so ~15 minutes into a multi-arch build:

    Because vllm-omni>=0.28.0rc1 depends on transformers>=5.10.1,<5.15 and
    transformers==5.15.1, we can conclude that vllm-omni>=0.28.0rc1 cannot be used.

That pairing is easy to create by accident: each pin is defensible on its own
(omni must track vLLM's major/minor; transformers wants to track the vLLM
release), and nothing checks them together. Checking a pin against a *running
image* is not a substitute either -- the thin dev image carries transformers
5.15.1 and does not install omni at all, so it "proves" a combination that
cannot actually resolve.

    python3 container/deps/vllm/check_pin_compatibility.py [context.yaml]

Exit 0 if every device stanza resolves, 1 otherwise. Requires network.
"""

from __future__ import annotations

import json
import re
import sys
import urllib.request
from pathlib import Path

import yaml

# (pinned package, the key holding its version) -- extend as pins are added.
PINNED = {"vllm-omni": "vllm_omni_ref", "transformers": "transformers_version"}


def _requires(package: str, version: str) -> list[str]:
    url = f"https://pypi.org/pypi/{package}/{version}/json"
    with urllib.request.urlopen(url, timeout=30) as resp:
        return json.load(resp)["info"].get("requires_dist") or []


def _parse(spec: str) -> list[tuple[str, tuple[int, ...]]]:
    """`transformers<5.15,>=5.10.1` -> [('<', (5,15)), ('>=', (5,10,1))]."""
    out = []
    for clause in spec.split(";")[0].split(","):
        clause = clause.strip()
        for op in (">=", "<=", "==", "!=", "<", ">", "~="):
            _, sep, rest = clause.partition(op)
            if sep:
                # Take the leading numeric release only. A plain rstrip of digits
                # would turn "5.15" into "5" and make every bound nonsense.
                m = re.match(r"(\d+(?:\.\d+)*)", rest.strip())
                if m:
                    out.append((op, tuple(int(n) for n in m.group(1).split("."))))
                break
    return out


def _satisfies(version: str, spec: str) -> bool:
    base = version.lstrip("v").split("rc")[0]
    try:
        have = tuple(int(n) for n in base.split("."))
    except ValueError:
        return True  # unparseable -- do not fail the build on a guess
    for op, want in _parse(spec):
        n = min(len(have), len(want))
        h, w = have[:n], want[:n]
        if op == ">=" and not h >= w:
            return False
        if op == "<" and not h < w:
            return False
        if op == "<=" and not h <= w:
            return False
        if op == ">" and not h > w:
            return False
    return True


def main(path: str) -> int:
    ctx = yaml.safe_load(Path(path).read_text())["vllm"]
    devices = [
        k for k, v in ctx.items() if isinstance(v, dict) and "runtime_image_tag" in v
    ]
    failures = []

    for dev in devices:
        stanza = {
            **{k: v for k, v in ctx.items() if not isinstance(v, dict)},
            **ctx[dev],
        }
        pinned = {
            pkg: str(stanza[key]).lstrip("v")
            for pkg, key in PINNED.items()
            if stanza.get(key)
        }
        for pkg, ver in pinned.items():
            try:
                reqs = _requires(pkg, ver)
            except Exception as exc:  # noqa: BLE001 - reported, not masked
                print(f"  {dev}: could not read {pkg}=={ver} from PyPI: {exc}")
                continue
            for req in reqs:
                name = (
                    req.split()[0]
                    .split("[")[0]
                    .split(">")[0]
                    .split("<")[0]
                    .split("=")[0]
                )
                if name in pinned and name != pkg:
                    if not _satisfies(pinned[name], req):
                        failures.append(
                            f"{dev}: {pkg}=={ver} requires `{req.split(';')[0].strip()}`, "
                            f"but {name} is pinned to {pinned[name]}"
                        )
        print(f"  {dev}: {pinned}")

    if failures:
        print("\nPIN CONFLICT -- this would fail mid-build:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll device stanzas resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        main(sys.argv[1] if len(sys.argv) > 1 else "container/context.yaml")
    )
