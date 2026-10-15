# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate the preflight contract from a checkout of the bundled Grove release (requires PyYAML)."""

import argparse
import importlib.util
import json
import subprocess
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grove_checkout", type=Path)
    args = parser.parse_args()
    chart = Path(__file__).resolve().parents[3]
    dependency = next(
        item
        for item in yaml.safe_load((chart / "Chart.yaml").read_text())["dependencies"]
        if item["name"] == "grove-charts"
    )
    files = chart / "components/operator/files/grove-preflight"
    spec = importlib.util.spec_from_file_location("grove_preflight", files / "check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Verify provenance so a different local Grove revision cannot silently become the baseline.
    release = dependency["version"]
    expected = subprocess.check_output(
        ["git", "rev-parse", f"refs/tags/{release}^{{commit}}"],
        cwd=args.grove_checkout,
        text=True,
    ).strip()
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=args.grove_checkout, text=True
    ).strip()
    if actual != expected:
        parser.error(f"Grove checkout is not at bundled release {release}")
    crd_dir = args.grove_checkout / "operator/api/core/v1alpha1/crds"
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--", str(crd_dir.resolve())],
        cwd=args.grove_checkout,
        text=True,
    ).strip():
        parser.error("Grove CRD source has local changes")
    names = json.loads((files / "compatibility.json").read_text())["crdSchemaDigests"]
    digests = {}
    for name in sorted(names):
        group, plural = name.split(".", 1)[1], name.split(".", 1)[0]
        crd = yaml.safe_load((crd_dir / f"{group}_{plural}.yaml").read_text())
        digests[name] = [module.schema_digest(crd)]
    contract = {"minimumVersion": release, "crdSchemaDigests": digests}
    (files / "compatibility.json").write_text(json.dumps(contract, indent=2) + "\n")


if __name__ == "__main__":
    main()
