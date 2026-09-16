// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

use dynamo_backend_common::{OutputOptions, PreprocessedRequest, SamplingOptions, StopConditions};
use opendal::raw::*;
use opendal::{
    Buffer, Builder, Capability, Error, ErrorKind, Metadata, MetadataBuilder, OperationContext,
    Operator, Result,
};
use serde_json::json;

use super::config::OperatorPolicy;
use super::{MetadataUploadService, OperatorCache, configure_operator};

fn policy(capacity: usize) -> OperatorPolicy {
    OperatorPolicy {
        capacity: NonZeroUsize::new(capacity).unwrap(),
        timeout: Duration::from_secs(10),
        io_timeout: Duration::from_secs(10),
        retry_max_times: 3,
        retry_min_delay: Duration::from_millis(1),
        retry_max_delay: Duration::from_millis(10),
        retry_factor: 2.0,
        retry_jitter: false,
    }
}

fn operator_cache(capacity: usize) -> OperatorCache {
    OperatorCache::new(policy(capacity))
}

fn request() -> PreprocessedRequest {
    PreprocessedRequest::builder()
        .model("model".to_string())
        .token_ids(vec![1])
        .sampling_options(SamplingOptions::default())
        .output_options(OutputOptions::default())
        .stop_conditions(StopConditions::default())
        .extra_args(Some(json!({
            "nvext": {"metadata_upload": {
                "url": "test://primary",
                "fallback_url": "test://fallback"
            }}
        })))
        .build()
        .unwrap()
}

#[tokio::test]
async fn concurrent_cache_misses_build_one_operator() {
    let directory = tempfile::tempdir().unwrap();
    let url = format!("fs://{}", directory.path().display());
    let cache = Arc::new(operator_cache(2));
    let barrier = Arc::new(tokio::sync::Barrier::new(33));
    let mut tasks = Vec::new();
    for _ in 0..32 {
        let cache = cache.clone();
        let url = url.clone();
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            cache.get(url, "url").await
        }));
    }
    barrier.wait().await;
    for task in tasks {
        task.await.unwrap().unwrap();
    }
    assert_eq!(cache.build_count.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn concurrent_cache_misses_share_construction_failure() {
    let cache = Arc::new(operator_cache(2));
    let barrier = Arc::new(tokio::sync::Barrier::new(33));
    let mut tasks = Vec::new();
    for _ in 0..32 {
        let cache = cache.clone();
        let barrier = barrier.clone();
        tasks.push(tokio::spawn(async move {
            barrier.wait().await;
            cache.get("unknown://destination".to_string(), "url").await
        }));
    }
    barrier.wait().await;
    for task in tasks {
        assert!(task.await.unwrap().is_err());
    }
    assert_eq!(cache.build_count.load(Ordering::Relaxed), 1);
}

#[tokio::test]
async fn retries_temporary_errors_until_success() {
    let (operator, attempts) = test_operator(WriteBehavior::FailUntil(2), policy(2));

    operator
        .write("metadata", Buffer::from("data"))
        .await
        .unwrap();

    assert_eq!(attempts.load(Ordering::Relaxed), 3);
}

#[tokio::test]
async fn stops_after_configured_retry_limit() {
    let mut retry_policy = policy(2);
    retry_policy.retry_max_times = 2;
    let (operator, attempts) = test_operator(WriteBehavior::FailUntil(usize::MAX), retry_policy);

    assert!(
        operator
            .write("metadata", Buffer::from("data"))
            .await
            .is_err()
    );
    assert_eq!(attempts.load(Ordering::Relaxed), 3);
}

#[tokio::test]
async fn timeout_exhausts_retries_before_fallback() {
    let cache = operator_cache(2);
    let mut timeout_policy = policy(2);
    timeout_policy.io_timeout = Duration::from_millis(5);
    timeout_policy.retry_max_times = 1;
    let (primary, primary_attempts) = test_operator(WriteBehavior::Pending, timeout_policy);
    let (fallback, fallback_attempts) = test_operator(WriteBehavior::Success, policy(2));
    cache.insert("test://primary", primary).await;
    cache.insert("test://fallback", fallback).await;
    let service = MetadataUploadService { operators: cache };
    let delivery = service.delivery_for(&request()).await.unwrap();

    delivery
        .finish(
            &std::collections::HashMap::from([("id".into(), "timeout".into())]),
            false,
        )
        .await
        .unwrap();

    assert_eq!(primary_attempts.load(Ordering::Relaxed), 2);
    assert_eq!(fallback_attempts.load(Ordering::Relaxed), 1);
}

#[derive(Clone, Copy, Debug)]
enum WriteBehavior {
    Success,
    FailUntil(usize),
    Pending,
}

#[derive(Clone, Debug)]
struct TestBuilder {
    behavior: WriteBehavior,
    attempts: Arc<AtomicUsize>,
}

impl Default for TestBuilder {
    fn default() -> Self {
        Self {
            behavior: WriteBehavior::Success,
            attempts: Arc::new(AtomicUsize::new(0)),
        }
    }
}

impl Builder for TestBuilder {
    type Config = ();

    fn build(self) -> Result<impl Service> {
        Ok(TestService {
            behavior: self.behavior,
            attempts: self.attempts,
        })
    }
}

#[derive(Clone, Debug)]
struct TestService {
    behavior: WriteBehavior,
    attempts: Arc<AtomicUsize>,
}

impl Service for TestService {
    type Reader = ();
    type Writer = TestWriter;
    type Lister = ();
    type Deleter = ();
    type Copier = ();
    type Composer = ();

    fn info(&self) -> ServiceInfo {
        ServiceInfo::with_scheme("metadata-test")
    }

    fn capability(&self) -> Capability {
        Capability {
            write: true,
            write_can_multi: true,
            ..Default::default()
        }
    }

    async fn create_dir(
        &self,
        _: &OperationContext,
        _: &str,
        _: OpCreateDir,
    ) -> Result<RpCreateDir> {
        unsupported()
    }

    async fn stat(&self, _: &OperationContext, _: &str, _: OpStat) -> Result<RpStat> {
        unsupported()
    }

    fn read(&self, _: &OperationContext, _: &str, _: OpRead) -> Result<Self::Reader> {
        unsupported()
    }

    fn write(&self, _: &OperationContext, _: &str, _: OpWrite) -> Result<Self::Writer> {
        Ok(TestWriter {
            behavior: self.behavior,
            attempts: self.attempts.clone(),
            size: 0,
        })
    }

    fn delete(&self, _: &OperationContext) -> Result<Self::Deleter> {
        unsupported()
    }

    fn list(&self, _: &OperationContext, _: &str, _: OpList) -> Result<Self::Lister> {
        unsupported()
    }

    fn copy(&self, _: &OperationContext, _: &str, _: &str, _: OpCopy) -> Result<Self::Copier> {
        unsupported()
    }

    async fn rename(
        &self,
        _: &OperationContext,
        _: &str,
        _: &str,
        _: OpRename,
    ) -> Result<RpRename> {
        unsupported()
    }

    async fn presign(&self, _: &OperationContext, _: &str, _: OpPresign) -> Result<RpPresign> {
        unsupported()
    }
}

#[derive(Debug)]
struct TestWriter {
    behavior: WriteBehavior,
    attempts: Arc<AtomicUsize>,
    size: usize,
}

impl oio::Write for TestWriter {
    async fn write(&mut self, data: Buffer) -> Result<()> {
        self.size += data.len();
        Ok(())
    }

    async fn close(&mut self) -> Result<Metadata> {
        let attempt = self.attempts.fetch_add(1, Ordering::Relaxed) + 1;
        match self.behavior {
            WriteBehavior::Success => Ok(file_metadata(self.size)),
            WriteBehavior::FailUntil(limit) if attempt <= limit => {
                Err(Error::new(ErrorKind::Unexpected, "injected temporary failure").set_temporary())
            }
            WriteBehavior::FailUntil(_) => Ok(file_metadata(self.size)),
            WriteBehavior::Pending => std::future::pending().await,
        }
    }

    async fn abort(&mut self) -> Result<()> {
        Ok(())
    }
}

fn test_operator(behavior: WriteBehavior, policy: OperatorPolicy) -> (Operator, Arc<AtomicUsize>) {
    let attempts = Arc::new(AtomicUsize::new(0));
    let operator = Operator::new(TestBuilder {
        behavior,
        attempts: attempts.clone(),
    })
    .unwrap();
    (configure_operator(operator, policy), attempts)
}

fn unsupported<T>() -> Result<T> {
    Err(Error::new(ErrorKind::Unsupported, "not used by this test"))
}

fn file_metadata(size: usize) -> Metadata {
    MetadataBuilder::file(size as u64).build()
}
