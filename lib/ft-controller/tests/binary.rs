// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The `dynamo-ft-controller` binary end to end: arguments, the summary lines
//! on stdout, the escalation command and the exit codes.

mod common;

use std::path::Path;
use std::process::Stdio;
use std::time::Duration;

use common::{FakeWorld, Timing, ms, wait_until};
use dynamo_ft_controller::round::{FileRoundStore, RoundStore};
use tokio::io::{AsyncBufReadExt, BufReader, Lines};
use tokio::process::{Child, ChildStdout, Command};

const BINARY: &str = env!("CARGO_BIN_EXE_dynamo-ft-controller");

/// Arguments for `world` over `ranks` of `fake`, with timings scaled down
/// for tests. Every fake rank listens on 127.0.0.1, as in a single-node
/// test, so each gets an explicit loopback store host.
fn base_args(world: &str, fake: &FakeWorld, ranks: &[u32]) -> Vec<String> {
    let mut args = vec!["--world".to_string(), world.to_string()];
    for (flag, value) in [
        ("--poll-interval", "20ms"),
        ("--call-timeout", "250ms"),
        ("--dead-after", "400ms"),
        ("--wake-after", "100ms"),
        ("--wake-max-gap", "400ms"),
        ("--wake-timeout", "250ms"),
        ("--checkin-timeout", "300ms"),
        ("--recovery-deadline", "5s"),
        ("--store-port-base", "40000"),
    ] {
        args.extend([flag.to_string(), value.to_string()]);
    }
    for &rank in ranks {
        args.extend(["--rank".to_string(), format!("{rank}={}", fake.url(rank))]);
        args.extend(["--store-host".to_string(), format!("{rank}=127.0.0.1")]);
    }
    args
}

/// Starts the binary with its log in `log`.
fn spawn(args: &[String], log: &Path) -> (Child, Lines<BufReader<ChildStdout>>) {
    let mut child = Command::new(BINARY)
        .args(args)
        .env("RUST_LOG", "info")
        .stdout(Stdio::piped())
        .stderr(std::fs::File::create(log).unwrap())
        .kill_on_drop(true)
        .spawn()
        .unwrap();
    let lines = BufReader::new(child.stdout.take().unwrap()).lines();
    (child, lines)
}

async fn line_starting_with(
    lines: &mut Lines<BufReader<ChildStdout>>,
    prefix: &str,
    log: &Path,
) -> String {
    let found = tokio::time::timeout(Duration::from_secs(10), async {
        while let Some(line) = lines.next_line().await.unwrap() {
            if line.starts_with(prefix) {
                return Some(line);
            }
        }
        None
    })
    .await;
    match found {
        Ok(Some(line)) => line,
        other => panic!(
            "no {prefix} line ({other:?}); controller log:\n{}",
            std::fs::read_to_string(log).unwrap_or_default()
        ),
    }
}

/// Waits until the controller has polled every rank a few times, so it has
/// formed.
async fn polled(fake: &FakeWorld, ranks: &[u32]) {
    wait_until(|| ranks.iter().all(|&rank| fake.polls(rank) >= 3)).await;
}

async fn exit_code(child: &mut Child) -> Option<i32> {
    tokio::time::timeout(Duration::from_secs(10), child.wait())
        .await
        .expect("the controller did not exit")
        .unwrap()
        .code()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn recovers_then_escalates_with_exit_code_3() {
    let fake = FakeWorld::start(3, Timing::default()).await;
    let dir = tempfile::tempdir().unwrap();
    let state = dir.path().join("state");
    let log = dir.path().join("controller.log");
    let marker = dir.path().join("escalated");
    let mut args = base_args("bin", &fake, &[0, 1, 2]);
    args.extend([
        "--state-dir".to_string(),
        state.display().to_string(),
        "--round-timeout".to_string(),
        "3s".to_string(),
        "--escalate-cmd".to_string(),
        format!(
            "printf '%s|%s' {{world}} {{reason}} > '{}'",
            marker.display()
        ),
    ]);
    let (mut child, mut lines) = spawn(&args, &log);
    polled(&fake, &[0, 1, 2]).await;

    fake.engine_dies(2);
    let recovered = line_starting_with(&mut lines, "FT_CONTROLLER_RECOVERED ", &log).await;
    for field in [
        "world=bin",
        "instruction=scale_down",
        "removed=[2]",
        "survivors=[0,1]",
        "rounds=1",
        "request_id=bin-e1-r1-",
    ] {
        assert!(
            recovered.contains(field),
            "{field} missing from {recovered}"
        );
    }

    fake.engine_dies(0);
    fake.engine_dies(1);
    let escalated = line_starting_with(&mut lines, "FT_CONTROLLER_ESCALATED ", &log).await;
    assert!(escalated.contains("no survivors"), "{escalated}");
    assert_eq!(exit_code(&mut child).await, Some(3));
    assert_eq!(
        std::fs::read_to_string(&marker).unwrap(),
        "bin|no survivors: every rank is dead"
    );
    // The supervisor restarts the whole world, so a restarted controller may
    // list rank 2 again.
    let last = FileRoundStore::new(&state, "bin").last().unwrap().unwrap();
    assert_eq!(last.survivors, [0, 1]);
    assert!(last.escalated);
    assert_eq!(fake.violations(), Vec::<String>::new());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dry_run_prints_would_escalate_and_keeps_running() {
    let fake = FakeWorld::start(2, Timing::default()).await;
    let dir = tempfile::tempdir().unwrap();
    let log = dir.path().join("controller.log");
    let mut args = base_args("bin-dry", &fake, &[0, 1]);
    args.push("--dry-run".to_string());
    let (mut child, mut lines) = spawn(&args, &log);
    polled(&fake, &[0, 1]).await;

    fake.engine_dies(0);
    fake.engine_dies(1);
    let line = line_starting_with(&mut lines, "FT_CONTROLLER_WOULD_ESCALATE ", &log).await;
    assert!(line.contains("world=bin-dry"), "{line}");
    assert!(line.contains("no survivors"), "{line}");

    // A dry run never exits with code 3: it goes on watching...
    tokio::time::sleep(ms(200)).await;
    assert!(child.try_wait().unwrap().is_none(), "the dry run exited");
    // ...until SIGTERM, which stops it with code 0.
    let pid = child.id().unwrap();
    let signalled = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!("kill -TERM {pid}"))
        .status()
        .unwrap();
    assert!(signalled.success());
    assert_eq!(exit_code(&mut child).await, Some(0));
    assert!(fake.orders().is_empty());
    assert!(fake.wakes().is_empty());
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn swapped_rank_urls_exit_with_code_2() {
    let fake = FakeWorld::start(2, Timing::default()).await;
    let rank_0 = format!("0={}", fake.url(1));
    let rank_1 = format!("1={}", fake.url(0));
    let output = tokio::time::timeout(
        Duration::from_secs(10),
        Command::new(BINARY)
            .args([
                "--world",
                "w-swap",
                "--dry-run",
                "--poll-interval",
                "20ms",
                "--rank",
                rank_0.as_str(),
                "--rank",
                rank_1.as_str(),
                "--store-host",
                "0=127.0.0.1",
                "--store-host",
                "1=127.0.0.1",
            ])
            .env("RUST_LOG", "info")
            .output(),
    )
    .await
    .expect("the controller did not exit")
    .unwrap();
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert_eq!(output.status.code(), Some(2), "{stderr}");
    assert!(stderr.contains("--rank 0 reports engine 1"), "{stderr}");
    assert!(output.stdout.is_empty());
}

#[test]
fn bad_arguments_exit_with_code_2() {
    for (extra, expected) in [
        (&["--rank", "0=http://192.0.2.1:9"][..], "more than once"),
        (&["--rank", "1=http://192.0.2.1:9"][..], "same URL"),
        (&["--store-host", "7=10.0.0.7"][..], "not a member"),
        (&["--initial-store-rank", "5"][..], "--initial-store-rank"),
        (&["--rank", "1=http://127.0.0.1:9"][..], "loopback"),
        (
            &["--recovery-deadline", "1000000000y"][..],
            "--recovery-deadline",
        ),
    ] {
        let output = std::process::Command::new(BINARY)
            .args([
                "--world",
                "w",
                "--dry-run",
                "--rank",
                "0=http://192.0.2.1:9",
            ])
            .args(extra)
            .output()
            .unwrap();
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert_eq!(output.status.code(), Some(2), "{extra:?}: {stderr}");
        assert!(output.stdout.is_empty(), "{extra:?}");
        assert!(
            stderr.starts_with("error: ") && stderr.contains(expected),
            "{extra:?}: {stderr}"
        );
    }
}
