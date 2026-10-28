// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Round trips from this process to each region, measured as the time to
//! open a TCP connection to a pool's first frontend: one round trip.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use dynamo_global_common::{Pool, PoolId};
use dynamo_global_router::PoolSignals;
use tokio::net::TcpStream;
use tokio::sync::watch;
use tokio::time::MissedTickBehavior;
use tokio_util::sync::CancellationToken;

use crate::PoolSnapshot;

const SAMPLES: usize = 3;
const CONNECT_TIMEOUT: Duration = Duration::from_secs(3);

/// Measures every pool in `pools` each `interval` until `cancel` fires, and
/// records each region's median in `signals`. A pool keeps its last round
/// trip when a later measurement fails.
pub async fn measure_round_trips(
    pools: watch::Receiver<Arc<PoolSnapshot>>,
    signals: Arc<PoolSignals>,
    interval: Duration,
    cancel: CancellationToken,
) {
    let mut last = HashMap::<PoolId, Duration>::new();
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
            let address = format!(
                "{}:{}",
                frontend.host_str()?,
                frontend.port_or_known_default()?
            );
            let pool = pool.id.clone();
            Some(async move { (pool, connect_time(&address).await) })
        });
        for (pool, round_trip) in futures::future::join_all(measurements).await {
            if let Some(round_trip) = round_trip {
                last.insert(pool, round_trip);
            }
        }
        last.retain(|pool, _| snapshot.pools().iter().any(|known| known.id == *pool));
        signals.set_region_round_trips(region_medians(snapshot.pools(), &last));
    }
}

/// The median round trip to each region's pools, over the pools measured.
fn region_medians(
    pools: &[Pool],
    round_trips: &HashMap<PoolId, Duration>,
) -> HashMap<String, Duration> {
    let mut by_region = HashMap::<&str, Vec<Duration>>::new();
    for pool in pools {
        if let Some(round_trip) = round_trips.get(&pool.id) {
            let region = pool.location.region.as_str();
            by_region.entry(region).or_default().push(*round_trip);
        }
    }
    by_region
        .into_iter()
        .map(|(region, mut round_trips)| {
            round_trips.sort();
            (region.to_owned(), round_trips[round_trips.len() / 2])
        })
        .collect()
}

/// The median time to connect, over the connects that succeed.
async fn connect_time(address: &str) -> Option<Duration> {
    let mut samples = Vec::with_capacity(SAMPLES);
    for _ in 0..SAMPLES {
        let start = Instant::now();
        let connected = tokio::time::timeout(CONNECT_TIMEOUT, TcpStream::connect(address)).await;
        if matches!(connected, Ok(Ok(_))) {
            samples.push(start.elapsed());
        }
    }
    samples.sort();
    samples.get(samples.len() / 2).copied()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::tests::pool;

    #[test]
    fn each_region_takes_the_median_of_its_measured_pools() {
        let mut pools = [
            pool("ohio-a", "qwen", &["http://ohio-a:8000"]),
            pool("ohio-b", "qwen", &["http://ohio-b:8000"]),
            pool("ohio-c", "qwen", &["http://ohio-c:8000"]),
            pool("tokyo-a", "qwen", &["http://tokyo-a:8000"]),
            pool("tokyo-b", "qwen", &["http://tokyo-b:8000"]),
        ];
        for pool in &mut pools[3..] {
            pool.location.region = "ap-northeast-1".to_owned();
        }
        let round_trips = HashMap::from([
            (pools[0].id.clone(), Duration::from_millis(3)),
            (pools[1].id.clone(), Duration::from_millis(1)),
            (pools[2].id.clone(), Duration::from_millis(2)),
            (pools[3].id.clone(), Duration::from_millis(140)),
        ]);
        let medians = region_medians(&pools, &round_trips);
        assert_eq!(medians["us-east-2"], Duration::from_millis(2));
        // Tokyo-b is not measured yet; Tokyo-a alone gives the region's.
        assert_eq!(medians["ap-northeast-1"], Duration::from_millis(140));
        assert_eq!(medians.len(), 2);
    }
}
