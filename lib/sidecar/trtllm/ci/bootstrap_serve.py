#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Add isolated Dynamo/pytest tooling without changing the native engine stack."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Reuse repository pins only for packages absent from the immutable stock image.
REQUIRED = {
    "aiohttp",
    "boto3",
    "filelock",
    "nats-py",
    "openai",
    "pillow",
    "prometheus-client",
    "psutil",
    "pytest",
    "pytest-asyncio",
    "pytest-benchmark",
    "pytest-httpserver",
    "pytest-timeout",
    "pyyaml",
    "requests",
}


@dataclass
class RuntimeIdentity:
    source_commit: str
    native_commit: str | None
    native_version: str | None
    native_python: str
    python_version: str
    stock_packages: dict[str, str]
    bootstrap_state: str = "starting"
    image: str | None = None
    expected_native_commit: str | None = None
    test_python: str | None = None
    overlay_native_packages: dict[str, str] | None = None


def normalized(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def installed():
    return {
        normalized(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--venv", type=Path, required=True)
    parser.add_argument("--github-env", type=Path)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[4]
    args.results.mkdir(parents=True, exist_ok=True)
    stock = installed()
    identity = RuntimeIdentity(
        source_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        native_commit=os.environ.get("TRT_LLM_GIT_COMMIT"),
        native_version=stock.get("tensorrt-llm"),
        native_python=sys.executable,
        python_version=sys.version,
        stock_packages=stock,
    )
    manifest = args.results / "runtime.json"
    manifest.write_text(json.dumps(asdict(identity), indent=2) + "\n")
    config = json.loads((args.artifact / "nightly.json").read_text())
    build = json.loads((args.artifact / "build.json").read_text())
    identity.image = config["image"]
    identity.expected_native_commit = config["upstream_commit"]
    manifest.write_text(json.dumps(asdict(identity), indent=2) + "\n")
    if identity.native_commit != config["upstream_commit"]:
        raise RuntimeError("Installed TRT-LLM source differs from the resolved nightly")
    if (
        build["native_image"] != config["image"]
        or build["native_commit"] != config["upstream_commit"]
    ):
        raise RuntimeError("Test artifacts were built against a different native image")
    if build["source_commit"] != identity.source_commit:
        raise RuntimeError("Dynamo test artifact does not match the checked-out source")
    for relative, expected in build["files"].items():
        if (
            hashlib.sha256((args.artifact / relative).read_bytes()).hexdigest()
            != expected
        ):
            raise RuntimeError(f"Artifact checksum mismatch: {relative}")
    (args.artifact / "bin/dynamo-trtllm-sidecar").chmod(0o755)
    wheels = list((args.artifact / "wheels").glob("ai_dynamo_runtime-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError("Expected exactly one current-source Dynamo runtime wheel")
    constraints = args.results / "stock-constraints.txt"
    constraints.write_text(
        "".join(
            f"{name}=={version}\n"
            for name, version in sorted(stock.items())
            if name != "ai-dynamo-runtime"
        )
    )
    requirements = {}
    for filename in ("requirements.common.txt", "requirements.test.txt"):
        for line in (repo / "container/deps" / filename).read_text().splitlines():
            match = re.match(r"([\w.-]+)(?:[<>=!~\[]|$)", line)
            if match and normalized(match[1]) in REQUIRED:
                requirements[normalized(match[1])] = line.split("#", 1)[0].strip()
    missing = REQUIRED - stock.keys()
    if missing - requirements.keys():
        raise RuntimeError(
            f"Missing repository dependency declarations: {sorted(missing - requirements.keys())}"
        )
    subprocess.run(
        [sys.executable, "-m", "venv", "--system-site-packages", str(args.venv)],
        check=True,
    )
    python = args.venv.resolve() / "bin/python"
    subprocess.run(
        [
            str(python),
            "-m",
            "pip",
            "install",
            "--constraint",
            str(constraints.resolve()),
            str(wheels[0].resolve()),
            *[requirements[name] for name in sorted(missing)],
        ],
        check=True,
    )
    environment = {**os.environ, "PYTHONPATH": f"{repo}/components/src:{repo}"}
    native_packages = json.loads(
        subprocess.check_output(
            [
                str(python),
                "-c",
                "import importlib.metadata as m, json; "
                "print(json.dumps({n: m.version(n) for n in ('torch', 'tensorrt-llm')}))",
            ],
            env=environment,
            text=True,
        )
    )
    identity.bootstrap_state = "validating"
    identity.test_python = str(python)
    identity.overlay_native_packages = native_packages
    manifest.write_text(json.dumps(asdict(identity), indent=2) + "\n")
    if any(stock.get(name) != version for name, version in native_packages.items()):
        raise RuntimeError(
            "The test virtual environment shadows the stock native stack"
        )
    # Import and parse the actual frontend before spending GPU time on engine startup.
    with (args.results / "frontend-smoke.log").open("w") as output:
        subprocess.run(
            [str(python), "-m", "dynamo.frontend", "--help"],
            env=environment,
            stdout=output,
            stderr=subprocess.STDOUT,
            check=True,
        )
    if installed() != stock:
        raise RuntimeError(
            "Bootstrap changed the stock interpreter's installed packages"
        )
    identity.bootstrap_state = "ready"
    manifest.write_text(json.dumps(asdict(identity), indent=2) + "\n")
    if args.github_env:
        with args.github_env.open("a") as output:
            output.write(
                f"DYNAMO_TEST_PYTHON={python}\nTRTLLM_PYTHON={sys.executable}\n"
            )
            output.write(
                f"DYNAMO_SIDECAR_BIN={args.artifact.resolve()}/bin/dynamo-trtllm-sidecar\n"
            )
            output.write(f"PYTHONPATH={repo}/components/src:{repo}\n")
            output.write(f"PATH={python.parent}:{os.environ['PATH']}\n")


if __name__ == "__main__":
    main()
