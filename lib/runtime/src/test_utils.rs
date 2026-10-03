// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::Command;

/// Carry execution support without inheriting process-global test configuration.
pub(crate) fn isolated_command(exact_test_name: &str) -> Command {
    let mut command = Command::new(std::env::current_exe().expect("current test executable"));
    // The test owns parser/TLS/HTTP settings; carry only execution support, never credentials.
    command.env_clear();
    for key in [
        "PATH",
        "LD_LIBRARY_PATH",
        "HOME",
        "TMPDIR",
        "RUST_BACKTRACE",
    ] {
        if let Some(value) = std::env::var_os(key) {
            command.env(key, value);
        }
    }
    command.args(["--exact", exact_test_name, "--nocapture"]);
    command
}

/// Keep temporary environment settings out of sibling tests and their cached configuration.
pub(crate) fn run_isolated(test: &str, env_overrides: &[(&str, &str)]) -> bool {
    const CHILD: &str = "DYNAMO_RUNTIME_ISOLATED_TEST";
    if std::env::var(CHILD).as_deref() == Ok(test) {
        return false;
    }
    let (_, test_name) = test.split_once("::").expect("fully qualified test name");
    let mut command = isolated_command(test_name);
    command.env(CHILD, test).envs(env_overrides.iter().copied());
    let output = command.output().expect("run isolated test executable");
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stdout.contains("running 1 test"),
        "isolated test was not selected: {test_name}\n{stdout}\n{stderr}"
    );
    assert!(
        output.status.success(),
        "isolated test failed: {test_name}\n{stdout}\n{stderr}"
    );
    true
}
