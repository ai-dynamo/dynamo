// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Round trips from this process to each pool, measured as the time to open
//! a TCP connection to the pool's first frontend: one round trip.

use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_global_router::{PoolSignals, PoolSnapshot};
use tokio::net::TcpStream;
use tokio::sync::watch;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

const SAMPLES: usize = 3;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

/// Measures every pool in `pools` each `interval` until `cancel` fires,
/// recording the median of a few connects.
pub async fn measure_round_trips(
    pools: watch::Receiver<Arc<PoolSnapshot>>,
    signals: Arc<PoolSignals>,
    interval: Duration,
    cancel: CancellationToken,
) {
    let mut ticks = tokio::time::interval(interval);
    ticks.set_missed_tick_behavior(MissedTickBehavior::Delay);
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            _ = ticks.tick() => {}
        }
        let snapshot = pools.borrow().clone();
        let measurements = snapshot.pools().iter().filter_map(|pool| {
            let frontend = pool.frontends.first()?;
            let host = frontend.host_str()?.to_owned();
            let port = frontend.port_or_known_default()?;
            let (pool, signals) = (pool.id.clone(), signals.clone());
            Some(async move {
                if let Some(round_trip) = connect_time(&host, port).await {
                    signals.record_round_trip(&pool, round_trip);
                }
            })
        });
        futures::future::join_all(measurements).await;
    }
}

/// The median time to connect, over the connects that succeed.
async fn connect_time(host: &str, port: u16) -> Option<Duration> {
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        let connected =
            tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect((host, port))).await;
        if matches!(connected, Ok(Ok(_))) {
            samples.push(start.elapsed());
        }
    }
    samples.sort();
    samples.get(samples.len() / 2).copied()
}
