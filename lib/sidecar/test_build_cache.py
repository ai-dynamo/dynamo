# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

HELPER = Path(__file__).with_name("build-with-cache.sh")


def check_cached_checkout_inputs(tmp_path: Path, generated: bool):
    cargo = shutil.which("cargo")
    assert cargo is not None, "Rust toolchain required for Cargo cache regression"
    root = tmp_path / "workspace with spaces"
    runtime = root / "runtime"
    consumer = root / "consumer"
    (runtime / "src").mkdir(parents=True)
    (consumer / "src").mkdir(parents=True)
    (root / "Cargo.toml").write_text(
        '[workspace]\nmembers = ["runtime", "consumer"]\nresolver = "2"\n'
    )
    (runtime / "Cargo.toml").write_text(
        '[package]\nname = "runtime"\nversion = "0.1.0"\nedition = "2021"\n'
    )
    (consumer / "Cargo.toml").write_text(
        '[package]\nname = "consumer"\nversion = "0.1.0"\nedition = "2021"\n'
        '[dependencies]\nruntime = { path = "../runtime" }\n'
    )
    runtime_input = runtime / "src/lib.rs"
    if generated:
        runtime_input.write_text('include!(concat!(env!("OUT_DIR"), "/api.rs"));\n')
        runtime_input = runtime / "api.in"
        (runtime / "build.rs").write_text(
            "fn main() {\n"
            '    let out = std::env::var("OUT_DIR").unwrap();\n'
            '    std::fs::copy("api.in", format!("{out}/api.rs")).unwrap();\n'
            '    println!("cargo:rerun-if-changed=api.in");\n'
            "}\n"
        )
    consumer_input = consumer / "src/main.rs"
    runtime_input.write_text("pub fn old_api() {}\n")
    consumer_input.write_text("fn main() { runtime::old_api(); }\n")
    env = {
        key: os.environ[key]
        for key in ("PATH", "HOME", "RUSTUP_HOME", "CARGO_HOME", "TMPDIR")
        if key in os.environ
    }
    env["CARGO_TARGET_DIR"] = str(root / "target")

    def run(*args: str, helper: bool = False) -> subprocess.CompletedProcess:
        command = [str(HELPER)] if helper else [cargo]
        return subprocess.run(
            command + list(args),
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    assert run("generate-lockfile").returncode == 0
    baseline = run("build", "--locked")
    assert baseline.returncode == 0, baseline.stderr
    old_mtime = runtime_input.stat().st_mtime_ns - 86_400_000_000_000
    runtime_input.write_text("pub fn new_api() {}\n")
    os.utime(runtime_input, ns=(old_mtime, old_mtime))
    consumer_input.write_text("fn main() { runtime::new_api(); }\n")
    stale = run("build", "--locked")
    assert stale.returncode != 0
    assert "cannot find function `new_api`" in stale.stderr

    repaired = run("build", "--locked", helper=True)
    assert repaired.returncode == 0, repaired.stderr
    stamp = root / "target/.workspace-inputs"
    stamp_mtime = stamp.stat().st_mtime_ns
    assert runtime_input.stat().st_mtime_ns == stamp_mtime

    # A sibling build gets another checkout's old mtimes, with identical bytes.
    os.utime(runtime_input, ns=(old_mtime, old_mtime))
    warm = run("build", "--locked", "-v", helper=True)
    assert warm.returncode == 0, warm.stderr
    assert "Fresh runtime" in warm.stderr
    assert stamp.stat().st_mtime_ns == stamp_mtime

    # Publish a new source stamp then interrupt before compiling any package.
    runtime_input.write_text("pub fn old_api() {}\n")
    consumer_input.write_text("fn main() { runtime::old_api(); }\n")
    interrupted = run("build", "--locked", "-p", "missing-package", helper=True)
    assert interrupted.returncode != 0
    interrupted_mtime = stamp.stat().st_mtime_ns
    assert interrupted_mtime > stamp_mtime
    os.utime(runtime_input, ns=(old_mtime, old_mtime))
    resumed = run("build", "--locked", helper=True)
    assert resumed.returncode == 0, resumed.stderr
    assert stamp.stat().st_mtime_ns == interrupted_mtime
    assert list((root / "target").glob(".workspace-inputs.*")) == []


class BuildCacheTest(unittest.TestCase):
    def test_helper_owns_all_target_cache_writers(self):
        dockerfile = HELPER.with_name("Dockerfile").read_text()
        target_caches = re.findall(
            r"id=(dynamo-sidecar-target-[^,]+),target=/src/target", dockerfile
        )
        self.assertEqual(
            target_caches, ["dynamo-sidecar-target-${TARGETARCH}-inputs-v3"] * 3
        )
        self.assertEqual(dockerfile.count("build-with-cache.sh build --release"), 3)

    def test_source_and_generated_inputs(self):
        for generated in (False, True):
            with self.subTest(
                generated=generated
            ), tempfile.TemporaryDirectory() as tmpdir:
                check_cached_checkout_inputs(Path(tmpdir), generated)
