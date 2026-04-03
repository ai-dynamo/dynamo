// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![cfg(all(target_os = "linux", feature = "standalone-indexer"))]

use std::process::{Child, Command, Output, Stdio};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result, bail};
use dynamo_kv_router::protocols::WorkerId;
use dynamo_kv_router::standalone_indexer::registry::{IndexerKey, WorkerRegistry};
use tokio::time::{Instant, sleep};

const WORKER_ID: WorkerId = 1;
const MODEL_NAME: &str = "blackhole-model";
const TENANT_ID: &str = "default";
const BLOCK_SIZE: u32 = 1;
const PUBLISH_PORT: u16 = 5557;

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires CAP_NET_ADMIN and Linux network namespaces"]
async fn standalone_listener_recovers_after_blackhole_partition() -> Result<()> {
    if !command_available("ip") || !running_as_root()? {
        eprintln!("skipping blackhole repro: requires Linux 'ip' and root privileges");
        return Ok(());
    }

    let mut harness =
        NetworkHarness::new(env!("CARGO_BIN_EXE_standalone_indexer_blackhole_publisher"))?;
    let registry = Arc::new(WorkerRegistry::new(1));
    let key = IndexerKey {
        model_name: MODEL_NAME.to_string(),
        tenant_id: TENANT_ID.to_string(),
    };

    registry
        .register(
            WORKER_ID,
            harness.publisher_endpoint(),
            0,
            MODEL_NAME.to_string(),
            TENANT_ID.to_string(),
            BLOCK_SIZE,
            None,
        )
        .await?;
    registry.signal_ready();

    wait_for_event_count_at_least(&registry, &key, 1, Duration::from_secs(15)).await?;

    harness.add_blackhole()?;
    let stalled_count = wait_for_stable_count(
        &registry,
        &key,
        Duration::from_secs(2),
        Duration::from_secs(15),
    )
    .await?;
    harness.remove_blackhole()?;

    let recovered_count =
        wait_for_event_count_above(&registry, &key, stalled_count, Duration::from_secs(15)).await?;

    assert!(
        recovered_count > stalled_count,
        "event count should resume after connectivity is restored: stalled_count={stalled_count}, recovered_count={recovered_count}",
    );
    Ok(())
}

async fn wait_for_event_count_at_least(
    registry: &Arc<WorkerRegistry>,
    key: &IndexerKey,
    minimum: usize,
    timeout: Duration,
) -> Result<usize> {
    let deadline = Instant::now() + timeout;
    loop {
        let current = current_event_count(registry, key).await?;
        if current >= minimum {
            return Ok(current);
        }
        if Instant::now() >= deadline {
            bail!("timed out waiting for event count >= {minimum}, current={current}");
        }
        sleep(Duration::from_millis(200)).await;
    }
}

async fn wait_for_event_count_above(
    registry: &Arc<WorkerRegistry>,
    key: &IndexerKey,
    previous: usize,
    timeout: Duration,
) -> Result<usize> {
    let deadline = Instant::now() + timeout;
    loop {
        let current = current_event_count(registry, key).await?;
        if current > previous {
            return Ok(current);
        }
        if Instant::now() >= deadline {
            bail!("timed out waiting for event count to exceed {previous}, current={current}");
        }
        sleep(Duration::from_millis(200)).await;
    }
}

async fn wait_for_stable_count(
    registry: &Arc<WorkerRegistry>,
    key: &IndexerKey,
    stable_for: Duration,
    timeout: Duration,
) -> Result<usize> {
    let deadline = Instant::now() + timeout;
    let mut last_count = current_event_count(registry, key).await?;
    let mut stable_since = Instant::now();

    loop {
        sleep(Duration::from_millis(200)).await;
        let current = current_event_count(registry, key).await?;
        if current != last_count {
            last_count = current;
            stable_since = Instant::now();
        } else if Instant::now().duration_since(stable_since) >= stable_for {
            return Ok(current);
        }

        if Instant::now() >= deadline {
            bail!(
                "timed out waiting for stable event count, last_count={last_count}, current={current}"
            );
        }
    }
}

async fn current_event_count(registry: &Arc<WorkerRegistry>, key: &IndexerKey) -> Result<usize> {
    let Some(entry) = registry.get_indexer(key) else {
        bail!(
            "indexer missing for model={} tenant={}",
            key.model_name,
            key.tenant_id
        );
    };
    let indexer = entry.indexer.clone();
    drop(entry);

    Ok(indexer.dump_events().await?.len())
}

fn command_available(command: &str) -> bool {
    Command::new("sh")
        .args(["-c", &format!("command -v {command} >/dev/null 2>&1")])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn running_as_root() -> Result<bool> {
    let output = run_command("id", &["-u"])?;
    Ok(String::from_utf8_lossy(&output.stdout).trim() == "0")
}

struct NetworkHarness {
    namespace: String,
    host_if: String,
    namespace_ip: String,
    publisher: Child,
}

impl NetworkHarness {
    fn new(helper_binary: &str) -> Result<Self> {
        let suffix = unique_suffix();
        let namespace = format!("dzmqns{suffix}");
        let host_if = format!("dzmqh{suffix}");
        let namespace_if = format!("dzmqp{suffix}");
        let namespace_ip = "10.203.0.2".to_string();

        run_command("ip", &["netns", "add", &namespace])?;
        run_command(
            "ip",
            &[
                "link",
                "add",
                &host_if,
                "type",
                "veth",
                "peer",
                "name",
                &namespace_if,
            ],
        )?;
        run_command("ip", &["addr", "add", "10.203.0.1/24", "dev", &host_if])?;
        run_command("ip", &["link", "set", &host_if, "up"])?;
        run_command("ip", &["link", "set", &namespace_if, "netns", &namespace])?;
        run_command(
            "ip",
            &[
                "netns",
                "exec",
                &namespace,
                "ip",
                "addr",
                "add",
                "10.203.0.2/24",
                "dev",
                &namespace_if,
            ],
        )?;
        run_command(
            "ip",
            &[
                "netns",
                "exec",
                &namespace,
                "ip",
                "link",
                "set",
                &namespace_if,
                "up",
            ],
        )?;
        run_command(
            "ip",
            &["netns", "exec", &namespace, "ip", "link", "set", "lo", "up"],
        )?;

        let publisher = Command::new("ip")
            .args([
                "netns",
                "exec",
                &namespace,
                helper_binary,
                &format!("tcp://{}:{}", namespace_ip, PUBLISH_PORT),
                "100",
            ])
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to spawn publisher helper {helper_binary}"))?;

        Ok(Self {
            namespace,
            host_if,
            namespace_ip,
            publisher,
        })
    }

    fn publisher_endpoint(&self) -> String {
        format!("tcp://{}:{}", self.namespace_ip, PUBLISH_PORT)
    }

    fn add_blackhole(&self) -> Result<()> {
        run_command(
            "ip",
            &[
                "route",
                "add",
                "blackhole",
                &format!("{}/32", self.namespace_ip),
            ],
        )?;
        Ok(())
    }

    fn remove_blackhole(&self) -> Result<()> {
        run_command(
            "ip",
            &[
                "route",
                "del",
                "blackhole",
                &format!("{}/32", self.namespace_ip),
            ],
        )?;
        Ok(())
    }
}

impl Drop for NetworkHarness {
    fn drop(&mut self) {
        let _ = self.publisher.kill();
        let _ = self.publisher.wait();
        let _ = run_command(
            "ip",
            &[
                "route",
                "del",
                "blackhole",
                &format!("{}/32", self.namespace_ip),
            ],
        );
        let _ = run_command("ip", &["link", "del", &self.host_if]);
        let _ = run_command("ip", &["netns", "delete", &self.namespace]);
    }
}

fn run_command(command: &str, args: &[&str]) -> Result<Output> {
    let output = Command::new(command)
        .args(args)
        .output()
        .with_context(|| format!("failed to run `{command} {}`", args.join(" ")))?;
    if output.status.success() {
        return Ok(output);
    }

    bail!(
        "`{command} {}` failed: {}",
        args.join(" "),
        String::from_utf8_lossy(&output.stderr).trim()
    );
}

fn unique_suffix() -> String {
    let pid = std::process::id();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .subsec_nanos();
    format!("{:x}", pid ^ nanos).chars().take(6).collect()
}
