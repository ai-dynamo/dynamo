// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::{Command, Output};

use crate::component::Endpoint;
use crate::discovery::{
    Discovery, DiscoveryInstance, DiscoveryQuery, DiscoverySpec, DiscoveryStream,
    EndpointInstanceId, MockDiscovery,
};
use crate::pipeline::network::PushWorkHandler;
use anyhow::Result;
use async_trait::async_trait;
use tokio_util::sync::CancellationToken;

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

pub(crate) struct NoopHandler;

#[async_trait]
impl PushWorkHandler for NoopHandler {
    async fn handle_payload(
        &self,
        _payload: bytes::Bytes,
        _request_id: Option<String>,
    ) -> std::result::Result<(), crate::pipeline::PipelineError> {
        Ok(())
    }

    fn add_metrics(&self, _endpoint: &Endpoint, _labels: Option<&[(&str, &str)]>) -> Result<()> {
        Ok(())
    }
}

pub(crate) struct UnavailableUnregisterDiscovery {
    pub(crate) inner: MockDiscovery,
    pub(crate) hanging: bool,
    pub(crate) attempted: tokio::sync::Notify,
}

#[async_trait]
impl Discovery for UnavailableUnregisterDiscovery {
    fn instance_id(&self) -> u64 {
        self.inner.instance_id()
    }

    async fn register_internal(&self, spec: DiscoverySpec) -> Result<DiscoveryInstance> {
        self.inner.register_internal(spec).await
    }

    async fn unregister(&self, _instance: DiscoveryInstance) -> Result<()> {
        self.attempted.notify_one();
        if self.hanging {
            std::future::pending().await
        } else {
            anyhow::bail!("discovery unavailable")
        }
    }

    async fn get_endpoint(&self, id: &EndpointInstanceId) -> Result<Option<DiscoveryInstance>> {
        self.inner.get_endpoint(id).await
    }

    async fn list(&self, _query: DiscoveryQuery) -> Result<Vec<DiscoveryInstance>> {
        anyhow::bail!("endpoint lease acquisition must use a point lookup")
    }

    async fn list_and_watch(
        &self,
        query: DiscoveryQuery,
        cancel_token: Option<CancellationToken>,
    ) -> Result<DiscoveryStream> {
        self.inner.list_and_watch(query, cancel_token).await
    }
}
