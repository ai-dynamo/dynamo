// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::io;
use std::num::NonZeroUsize;
use std::time::Duration;

use crate::GrpcTransportConfig;
use crate::transport::connect_pool_with;
use dynamo_backend_common::{BackendError, ErrorType};
use tokio::time::{Instant, sleep};

fn config() -> GrpcTransportConfig {
    GrpcTransportConfig {
        connections: NonZeroUsize::new(3).unwrap(),
        connect_attempt_timeout: Duration::from_millis(80),
        retry_interval: Duration::from_millis(10),
        startup_deadline: Duration::from_millis(100),
    }
}

sidecar_test! {
    lane: pre_merge;
    #[tokio::test(start_paused = true)]
    async fn failed_connection_retries_then_pool_contains_every_slot() {
        let attempts = Cell::new(0);
        let started = Instant::now();
        let channels = connect_pool_with("test", "in-memory", config(), false, |slot, timeout| {
            assert_eq!(timeout, Duration::from_millis(80));
            let attempt = attempts.get();
            attempts.set(attempt + 1);
            async move {
                if attempt == 0 {
                    Err(io::Error::other("initial failure"))
                } else {
                    Ok(slot)
                }
            }
        })
        .await
        .unwrap();
        assert_eq!(channels, vec![1, 2, 3]);
        assert_eq!(attempts.get(), 4);
        assert_eq!(started.elapsed(), Duration::from_millis(10));
    }
}

sidecar_test! {
    lane: pre_merge;
    #[tokio::test(start_paused = true)]
    async fn later_pool_slots_share_the_original_deadline() {
        let observed = std::cell::RefCell::new(Vec::new());
        let started = Instant::now();
        let error = connect_pool_with("test", "in-memory", config(), false, |slot, timeout| {
            observed.borrow_mut().push((slot, timeout));
            async move {
                if slot == 1 {
                    sleep(Duration::from_millis(70)).await;
                    Ok(slot)
                } else {
                    std::future::pending::<Result<usize, io::Error>>().await
                }
            }
        })
        .await
        .unwrap_err();
        assert_eq!(started.elapsed(), Duration::from_millis(100));
        assert_eq!(
            *observed.borrow(),
            vec![
                (1, Duration::from_millis(80)),
                (2, Duration::from_millis(30)),
                (3, Duration::from_millis(30))
            ]
        );
        assert_eq!(
            error.error_type(),
            ErrorType::Backend(BackendError::CannotConnect)
        );
        assert!(error.to_string().contains("pool slot 2"));
        assert!(error.to_string().contains("exceeded the startup deadline"));
    }
}

sidecar_test! {
    lane: pre_merge;
    #[tokio::test(start_paused = true)]
    async fn retry_wait_is_capped_and_last_failure_is_preserved() {
        let mut transport = config();
        transport.retry_interval = Duration::from_secs(1);
        let started = Instant::now();
        let error = connect_pool_with("test", "in-memory", transport, false, |_, _| async {
            Err::<(), _>(io::Error::other("peer rejected connection"))
        })
        .await
        .unwrap_err();
        assert_eq!(started.elapsed(), Duration::from_millis(100));
        assert!(error.to_string().contains("peer rejected connection"));
        assert!(error.to_string().contains("after 1 attempts"));
    }
}
