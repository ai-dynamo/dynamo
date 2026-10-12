// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runs one gate to a verdict. `Err` carries the reason shown on `/health`.

use std::process::Stdio;

use tokio::io::AsyncReadExt;

use super::{GateCheck, GateSpec};

/// How much of a failed check's stderr is kept as its reason.
const STDERR_TAIL_BYTES: usize = 2048;

pub(super) async fn run_gate(gate: &GateSpec) -> Result<(), String> {
    let run = async {
        match &gate.check {
            GateCheck::Exec(argv) => run_exec(&gate.name, argv).await,
        }
    };
    tokio::time::timeout(gate.timeout, run)
        .await
        .unwrap_or_else(|_| {
            Err(format!(
                "no verdict within timeout_secs={}",
                gate.timeout.as_secs()
            ))
        })
}

/// Run the command; exit status 0 is a pass. The child is killed if the
/// deadline drops this future.
async fn run_exec(gate: &str, argv: &[String]) -> Result<(), String> {
    let mut child = tokio::process::Command::new(&argv[0])
        .args(&argv[1..])
        .env("DYN_VALIDATION_GATE", gate)
        .stdin(Stdio::null())
        .stderr(Stdio::piped())
        .kill_on_drop(true)
        .spawn()
        .map_err(|error| format!("could not run {:?}: {error}", argv[0]))?;

    let mut stderr = Vec::new();
    let mut pipe = child.stderr.take().expect("stderr is piped");
    let (status, _) = tokio::join!(child.wait(), pipe.read_to_end(&mut stderr));
    let status = status.map_err(|error| format!("waiting for {:?}: {error}", argv[0]))?;
    if status.success() {
        return Ok(());
    }
    let tail = String::from_utf8_lossy(&stderr[stderr.len().saturating_sub(STDERR_TAIL_BYTES)..])
        .trim()
        .to_string();
    tracing::warn!(gate, %status, stderr = %tail, "Validation check failed");
    Err(format!("{status}: {tail}"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[tokio::test]
    async fn a_hung_check_fails_at_its_deadline() {
        let started = Instant::now();
        let gate = GateSpec {
            name: "g".into(),
            check: GateCheck::Exec(vec!["sleep".into(), "30".into()]),
            timeout: Duration::from_secs(1),
        };
        let reason = run_gate(&gate).await.unwrap_err();
        assert!(reason.contains("no verdict within"), "{reason}");
        assert!(started.elapsed() < Duration::from_secs(5));
    }
}
