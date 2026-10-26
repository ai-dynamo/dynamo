// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::{Command, Output};

/// Prevent an uncaptured thread from caching a shared callsite's interest for a sole capture.
pub(crate) fn tracing_interest_guard() -> tracing::Dispatch {
    tracing::Dispatch::new(tracing::subscriber::NoSubscriber::default())
}

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
    // Harmony's downloader publishes directly to its cache path; child processes must not share it.
    let tokenizer_cache = tempfile::Builder::new()
        .prefix("dynamo-tokenizer-")
        .tempdir()
        .expect("create isolated tokenizer cache")
        .keep();
    command.env("TIKTOKEN_RS_CACHE_DIR", tokenizer_cache);
    // The parent harness owns ignoring tests; explicitly rerun its selected test in the child.
    command.args([
        "--exact",
        exact_test_name,
        "--include-ignored",
        "--nocapture",
    ]);
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
    assert_isolated_success(&output);
    true
}

/// A successful exit can mean that a stale exact-test filter matched no tests.
pub(crate) fn assert_isolated_success(output: &Output) {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "isolated test failed: {stdout}\n{stderr}"
    );
    assert!(
        stdout
            .lines()
            .any(|line| line.starts_with("test result: ok. 1 passed; 0 failed; 0 ignored;")),
        "isolated subprocess must run exactly one passing test: {stdout}\n{stderr}"
    );
}

#[test]
fn isolated_commands_own_distinct_tokenizer_caches() {
    let _interest_guard = tracing_interest_guard();
    let commands = [
        isolated_command("cache-probe"),
        isolated_command("cache-probe"),
    ];
    let paths: Vec<_> = commands
        .iter()
        .map(|command| {
            let path = command
                .get_envs()
                .find_map(|(key, value)| {
                    (key == "TIKTOKEN_RS_CACHE_DIR").then_some(value).flatten()
                })
                .expect("isolated cache override");
            let path = std::path::PathBuf::from(path);
            assert!(path.is_dir());
            assert_eq!(path.parent(), Some(std::env::temp_dir().as_path()));
            let child_tmpdir = command
                .get_envs()
                .find_map(|(key, value)| (key == "TMPDIR").then_some(value).flatten());
            assert_eq!(child_tmpdir, std::env::var_os("TMPDIR").as_deref());
            path
        })
        .collect();
    assert_ne!(paths[0], paths[1]);
}

#[test]
fn isolated_success_rejects_unmatched_test_filter() {
    let output = isolated_command("__dynamo_nonexistent_isolated_test__")
        .output()
        .expect("run isolated test executable");
    assert!(
        output.status.success(),
        "libtest accepts an unmatched filter"
    );
    assert!(std::panic::catch_unwind(|| assert_isolated_success(&output)).is_err());
}
