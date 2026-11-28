// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Escalation: hand a world the FT API cannot recover to its supervisor.
//!
//! The controller always exits non-zero after escalating, so a supervisor
//! restarts the world (today's full restart). An [`Escalator`] adds whatever
//! the host wants on top, such as paging or deleting the world's pods.

use std::time::Duration;

use anyhow::{Context, Result, anyhow, ensure};
use async_trait::async_trait;

/// Hands a world over to its supervisor.
#[async_trait]
pub trait Escalator: Send + Sync {
    /// Called once, right before the controller exits non-zero.
    async fn escalate(&self, world: &str, reason: &str) -> Result<()>;
}

/// Runs `--escalate-cmd`, if one was given, through `sh -c`.
///
/// `{world}` and `{reason}` in the template become single-quoted shell words.
/// The command also gets `FT_CONTROLLER_WORLD` and `FT_CONTROLLER_REASON` in
/// its environment. It is killed after `timeout`.
#[derive(Clone, Debug)]
pub struct CommandEscalator {
    template: Option<String>,
    timeout: Duration,
}

impl CommandEscalator {
    pub fn new(template: Option<String>, timeout: Duration) -> Self {
        Self { template, timeout }
    }
}

#[async_trait]
impl Escalator for CommandEscalator {
    async fn escalate(&self, world: &str, reason: &str) -> Result<()> {
        let Some(template) = &self.template else {
            return Ok(());
        };
        let command = render(template, world, reason);
        tracing::warn!(%command, "running the escalation command");
        let mut child = tokio::process::Command::new("sh")
            .arg("-c")
            .arg(&command)
            .env("FT_CONTROLLER_WORLD", world)
            .env("FT_CONTROLLER_REASON", reason)
            .kill_on_drop(true)
            .spawn()
            .with_context(|| format!("cannot start the escalation command `{command}`"))?;
        let status = tokio::time::timeout(self.timeout, child.wait())
            .await
            .map_err(|_| {
                anyhow!(
                    "the escalation command did not finish within {:?}; killed it",
                    self.timeout
                )
            })?
            .context("cannot wait for the escalation command")?;
        ensure!(
            status.success(),
            "the escalation command exited with {status}"
        );
        Ok(())
    }
}

/// Replaces `{world}` and `{reason}` in one pass, quoting each value as one
/// shell word. Any other text, braces included, is kept as is.
pub fn render(template: &str, world: &str, reason: &str) -> String {
    let mut out = String::with_capacity(template.len() + world.len() + reason.len());
    let mut rest = template;
    while let Some(start) = rest.find('{') {
        out.push_str(&rest[..start]);
        let tail = &rest[start..];
        if let Some(after) = tail.strip_prefix("{world}") {
            out.push_str(&shell_quote(world));
            rest = after;
        } else if let Some(after) = tail.strip_prefix("{reason}") {
            out.push_str(&shell_quote(reason));
            rest = after;
        } else {
            out.push('{');
            rest = &tail[1..];
        }
    }
    out.push_str(rest);
    out
}

fn shell_quote(value: &str) -> String {
    format!("'{}'", value.replace('\'', r"'\''"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placeholders_become_quoted_words() {
        assert_eq!(
            render(
                "restart.sh --world {world} --why {reason} {x}",
                "w1",
                "rank 2's front end died"
            ),
            r#"restart.sh --world 'w1' --why 'rank 2'\''s front end died' {x}"#
        );
        // A value that looks like a placeholder is not expanded again.
        assert_eq!(render("{reason}{world}", "w", "{world}"), "'{world}''w'");
    }

    #[tokio::test]
    async fn runs_the_command_with_the_world_and_reason() {
        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("out");
        let template = format!(
            "printf '%s|%s|%s' {{world}} {{reason}} \"$FT_CONTROLLER_WORLD\" > {}",
            out.display()
        );
        CommandEscalator::new(Some(template), Duration::from_secs(5))
            .escalate("w1", "no survivors; it's over")
            .await
            .unwrap();
        assert_eq!(
            std::fs::read_to_string(&out).unwrap(),
            "w1|no survivors; it's over|w1"
        );
    }

    #[tokio::test]
    async fn reports_failure_and_timeouts() {
        let failing = CommandEscalator::new(Some("exit 7".to_string()), Duration::from_secs(5));
        let error = failing.escalate("w", "r").await.unwrap_err();
        assert!(error.to_string().contains("exited"), "{error:#}");

        let slow = CommandEscalator::new(Some("sleep 5".to_string()), Duration::from_millis(100));
        let error = slow.escalate("w", "r").await.unwrap_err();
        assert!(error.to_string().contains("did not finish"), "{error:#}");

        CommandEscalator::new(None, Duration::from_millis(1))
            .escalate("w", "r")
            .await
            .unwrap();
    }
}
