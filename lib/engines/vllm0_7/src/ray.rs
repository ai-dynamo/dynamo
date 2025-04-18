// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use regex::Regex;
use std::io::{BufRead, BufReader};
use std::net::SocketAddrV4;
use std::process::{Command, Stdio};
use std::time::Duration;
use thiserror::Error;
use tokio::io::AsyncBufReadExt;
use tokio::select;
use tokio::time;

use dynamo_runtime::CancellationToken;

/// Default is 16 seconds, we make it a bit shorter
const RAY_STOP_TIMEOUT_SECS: u32 = 10;

/// How long to wait for all the nodes to start.
/// This is either done manually or through some orchestration system, so either way it
/// can take some time.
const RAY_WAIT_SECS: u32 = 60 * 5;

#[derive(Debug, Error)]
pub enum RayError {
    #[error("Failed to execute Ray command: {0}")]
    CommandExecution(#[from] std::io::Error),

    #[error("Ray command failed with exit code: {0}")]
    CommandFailed(i32),

    #[error("Failed to parse Ray status output")]
    StatusParseError,

    #[error("Timeout waiting for nodes to become active")]
    WaitTimeout,

    #[error("Operation cancelled")]
    Cancelled,
}

#[derive(Debug, PartialEq)]
pub struct RayStatus {
    pub active_nodes: Vec<String>,
    pub pending_nodes_count: usize,
    pub recent_failures_count: usize,
}

pub struct Ray {
    #[allow(dead_code)]
    leader_address: SocketAddrV4,
}

pub fn start_leader(leader_address: SocketAddrV4) -> Result<Ray, RayError> {
    let ip = leader_address.ip().to_string();
    let port = leader_address.port().to_string();

    let mut cmd = Command::new("ray");
    cmd.args([
        "start",
        "--head",
        "--disable-usage-stats",
        "--log-style=record",
        &format!("--node-ip-address={}", ip),
        &format!("--port={}", port),
    ]);

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn()?;

    // Process stdout
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            tracing::info!("RAY: {line}");
        }
    }

    // Process stderr
    if let Some(stderr) = child.stderr.take() {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            tracing::info!("RAY: {line}");
        }
    }

    let status = child.wait()?;

    if !status.success() {
        return Err(RayError::CommandFailed(status.code().unwrap_or(-1)));
    }

    Ok(Ray { leader_address })
}

pub fn start_follower(leader_address: SocketAddrV4) -> Result<Ray, RayError> {
    let address = leader_address.to_string();

    let mut cmd = Command::new("ray");
    cmd.args(["start", &format!("--address={address}")]);

    cmd.stdout(Stdio::piped());
    cmd.stderr(Stdio::piped());

    let mut child = cmd.spawn()?;

    // Process stdout
    if let Some(stdout) = child.stdout.take() {
        let reader = BufReader::new(stdout);
        for line in reader.lines().map_while(Result::ok) {
            tracing::info!("RAY: {line}");
        }
    }

    // Process stderr
    if let Some(stderr) = child.stderr.take() {
        let reader = BufReader::new(stderr);
        for line in reader.lines().map_while(Result::ok) {
            tracing::info!("RAY: {line}");
        }
    }

    let status = child.wait()?;

    if !status.success() {
        return Err(RayError::CommandFailed(status.code().unwrap_or(-1)));
    }

    Ok(Ray { leader_address })
}

impl Ray {
    pub fn status(&self) -> Result<RayStatus, RayError> {
        let output = Command::new("ray").arg("status").output()?;

        if !output.status.success() {
            return Err(RayError::CommandFailed(output.status.code().unwrap_or(-1)));
        }

        let output_str = String::from_utf8_lossy(&output.stdout);
        parse_ray_status(&output_str).ok_or(RayError::StatusParseError)
    }

    pub async fn wait_for(
        &self,
        cancel_token: CancellationToken,
        num_nodes: u32,
    ) -> Result<(), RayError> {
        let timeout = time::sleep(Duration::from_secs(RAY_WAIT_SECS as u64));

        select! {
            _ = cancel_token.cancelled() => {
                Err(RayError::Cancelled)
            }
            _ = timeout => {
                Err(RayError::WaitTimeout)
            }
            result = self.wait_for_nodes(num_nodes) => {
                result
            }
        }
    }

    async fn wait_for_nodes(&self, num_nodes: u32) -> Result<(), RayError> {
        loop {
            let status = self.status()?;
            if status.active_nodes.len() as u32 == num_nodes {
                return Ok(());
            }
            time::sleep(Duration::from_millis(100)).await;
        }
    }

    pub async fn stop(&self) -> Result<(), RayError> {
        let mut cmd = tokio::process::Command::new("ray");
        cmd.args([
            "stop",
            &format!("--grace-period={RAY_STOP_TIMEOUT_SECS}"),
            "--log-style=record",
        ]);

        cmd.stdout(Stdio::piped());
        cmd.stderr(Stdio::piped());

        let mut child = cmd.spawn()?;

        // Process stdout
        if let Some(stdout) = child.stdout.take() {
            let reader = tokio::io::BufReader::new(stdout);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::info!("RAY: {line}");
            }
        }

        // Process stderr
        if let Some(stderr) = child.stderr.take() {
            let reader = tokio::io::BufReader::new(stderr);
            let mut lines = reader.lines();
            while let Ok(Some(line)) = lines.next_line().await {
                tracing::info!("RAY: {line}");
            }
        }

        let status = child.wait().await?;

        if !status.success() {
            return Err(RayError::CommandFailed(status.code().unwrap_or(-1)));
        }

        Ok(())
    }
}

/// Parse the output of "ray status" command into a RayStatus struct
fn parse_ray_status(output: &str) -> Option<RayStatus> {
    let mut active_nodes = Vec::new();
    let mut pending_nodes_count = 0;
    let mut recent_failures_count = 0;

    // Flags to track which section we're in
    let mut in_active_section = false;
    let mut in_pending_section = false;
    let mut in_failures_section = false;

    // Regex to match node IDs
    let node_regex = Regex::new(r"(\d+)\s+(node_[a-f0-9]+)").unwrap();
    let num_regex = Regex::new(r"(\d+)").unwrap();

    for line in output.lines() {
        let trimmed = line.trim();

        if trimmed == "Active:" {
            in_active_section = true;
            in_pending_section = false;
            in_failures_section = false;
            continue;
        } else if trimmed == "Pending:" {
            in_active_section = false;
            in_pending_section = true;
            in_failures_section = false;
            continue;
        } else if trimmed == "Recent failures:" {
            in_active_section = false;
            in_pending_section = false;
            in_failures_section = true;
            continue;
        } else if trimmed.starts_with("Resources") {
            // We've reached the end of the node status section
            break;
        }

        if in_active_section {
            if let Some(captures) = node_regex.captures(trimmed) {
                if let Some(node_id) = captures.get(2) {
                    active_nodes.push(node_id.as_str().to_string());
                }
            }
        } else if in_pending_section && trimmed != "(no pending nodes)" {
            // Count pending nodes
            if let Some(captures) = num_regex.captures(trimmed) {
                if let Some(count) = captures.get(1) {
                    if let Ok(count) = count.as_str().parse::<usize>() {
                        pending_nodes_count += count;
                    }
                }
            }
        } else if in_failures_section && trimmed != "(no failures)" {
            // Count failures
            if let Some(captures) = num_regex.captures(trimmed) {
                if let Some(count) = captures.get(1) {
                    if let Ok(count) = count.as_str().parse::<usize>() {
                        recent_failures_count += count;
                    }
                }
            }
        }
    }

    Some(RayStatus {
        active_nodes,
        pending_nodes_count,
        recent_failures_count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_ray_status() {
        let sample_output = r#"======== Autoscaler status: 2025-03-04 13:13:59.104771 ========
Node status
---------------------------------------------------------------
Active:
 1 node_b09a7440bd0987680f97c35206b2475251907d0c928fdd0f52b1b38f
 1 node_035ea3b640e13f3603d3debd97de8c569ed8c8b10e19ce00ea4fd070
Pending:
 (no pending nodes)
Recent failures:
 (no failures)

Resources
---------------------------------------------------------------
Usage:
 0.0/256.0 CPU
 0.0/16.0 GPU
 0B/1.58TiB memory
 0B/372.53GiB object_store_memory

Demands:
 (no resource demands)
"#;

        let expected = RayStatus {
            active_nodes: vec![
                "node_b09a7440bd0987680f97c35206b2475251907d0c928fdd0f52b1b38f".to_string(),
                "node_035ea3b640e13f3603d3debd97de8c569ed8c8b10e19ce00ea4fd070".to_string(),
            ],
            pending_nodes_count: 0,
            recent_failures_count: 0,
        };

        let result = parse_ray_status(sample_output);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), expected);
    }

    /// Test with pending nodes and failures
    #[test]
    fn test_parse_ray_status_with_failing() {
        let sample_output_with_pending = r#"======== Autoscaler status: 2025-03-04 13:13:59.104771 ========
Node status
---------------------------------------------------------------
Active:
 1 node_b09a7440bd0987680f97c35206b2475251907d0c928fdd0f52b1b38f
Pending:
 2 node_pending_1
 3 node_pending_2
Recent failures:
 1 node_failure_1
 4 node_failure_2

Resources
---------------------------------------------------------------
Usage:
 0.0/256.0 CPU
"#;

        let expected_with_pending = RayStatus {
            active_nodes: vec![
                "node_b09a7440bd0987680f97c35206b2475251907d0c928fdd0f52b1b38f".to_string(),
            ],
            pending_nodes_count: 5,   // 2 + 3
            recent_failures_count: 5, // 1 + 4
        };

        let result = parse_ray_status(sample_output_with_pending);
        assert!(result.is_some());
        assert_eq!(result.unwrap(), expected_with_pending);
    }

    /// Test with empty output
    #[test]
    fn test_parse_ray_status_empty() {
        let empty_output = "";
        let result = parse_ray_status(empty_output);
        assert!(result.is_some());
        assert_eq!(result.unwrap().active_nodes.len(), 0);
    }
}
