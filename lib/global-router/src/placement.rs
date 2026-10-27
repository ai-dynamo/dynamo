// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the session-aware selectors share: which pools are overloaded, how
//! far each pool's region is from the region a request entered, and the
//! order to try pools in.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Duration;

use crate::catalog::Pool;
use crate::signals::{PoolLoad, PoolSignals};

/// When a pool counts as overloaded: its requests in flight per worker are
/// more than `ratio` times the fleet's, and at least `floor` more per worker
/// than the fleet's. The floor keeps nearly idle pools from counting as
/// overloaded over a request or two.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Overload {
    pub ratio: f64,
    pub floor: f64,
}

impl Default for Overload {
    fn default() -> Self {
        Self {
            ratio: 1.25,
            floor: 1.0,
        }
    }
}

/// A reachable pool, with what the session-aware selectors weigh.
#[derive(Debug)]
pub(crate) struct Candidate<'a> {
    pub pool: &'a Pool,
    pub load: Option<PoolLoad>,
    /// Round trip to the pool's region: zero for the region the request
    /// entered, otherwise the median round trip measured to that region's
    /// pools. `None` until one is measured.
    pub distance: Option<Duration>,
    pub overloaded: bool,
}

/// The candidates that have a frontend, for a request entering `region`.
///
/// The fleet's load per worker is all requests in flight over all workers,
/// among the pools with a fresh load. A pool without a fresh load counts as
/// overloaded, so it is tried only after the pools known to have room. When
/// no pool has a fresh load, none is overloaded and distance alone orders
/// them.
pub(crate) fn place<'a>(
    candidates: &[&'a Pool],
    region: &str,
    signals: &PoolSignals,
    overload: Overload,
) -> Vec<Candidate<'a>> {
    let pools = candidates
        .iter()
        .copied()
        .filter(|pool| !pool.frontends.is_empty())
        .collect::<Vec<_>>();
    let loads = pools
        .iter()
        .map(|pool| signals.load(&pool.id))
        .collect::<Vec<_>>();
    let (in_flight, workers) = loads
        .iter()
        .flatten()
        .fold((0, 0), |(in_flight, workers), load| {
            (in_flight + load.in_flight, workers + load.workers)
        });
    let fleet = (workers > 0).then(|| in_flight as f64 / workers as f64);
    let distances = region_distances(&pools, region, signals);
    pools
        .into_iter()
        .zip(loads)
        .map(|(pool, load)| {
            let overloaded = match (load, fleet) {
                (Some(load), Some(fleet)) => {
                    let per_worker = load.per_worker();
                    per_worker > overload.ratio * fleet && per_worker - fleet >= overload.floor
                }
                (None, Some(_)) => true,
                (_, None) => false,
            };
            Candidate {
                pool,
                load,
                distance: distances[pool.location.region.as_str()],
                overloaded,
            }
        })
        .collect()
}

fn region_distances<'a>(
    pools: &[&'a Pool],
    region: &str,
    signals: &PoolSignals,
) -> HashMap<&'a str, Option<Duration>> {
    let mut measured = HashMap::<&str, Vec<Duration>>::new();
    for pool in pools {
        let round_trips = measured.entry(pool.location.region.as_str()).or_default();
        round_trips.extend(signals.round_trip(&pool.id));
    }
    measured
        .into_iter()
        .map(|(name, mut round_trips)| {
            let distance = if name == region {
                Some(Duration::ZERO)
            } else {
                round_trips.sort();
                round_trips.get(round_trips.len() / 2).copied()
            };
            (name, distance)
        })
        .collect()
}

/// Nearest region first; regions not yet measured last.
pub(crate) fn by_distance(left: &Candidate<'_>, right: &Candidate<'_>) -> Ordering {
    match (left.distance, right.distance) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// Lower load per worker first; unknown loads last.
pub(crate) fn by_load(left: &Candidate<'_>, right: &Candidate<'_>) -> Ordering {
    match (left.load, right.load) {
        (Some(left), Some(right)) => left.per_worker().total_cmp(&right.per_worker()),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// From least to most expensive: nearest region first, then lower load per
/// worker.
pub(crate) fn cost_order(candidates: &mut [Candidate<'_>]) {
    candidates.sort_by(|left, right| by_distance(left, right).then_with(|| by_load(left, right)));
}

/// For when every pool is overloaded: the lowest load per worker, nearest
/// on ties.
pub(crate) fn least_loaded<'a, 'b>(candidates: &'b [Candidate<'a>]) -> Option<&'b Candidate<'a>> {
    candidates
        .iter()
        .min_by(|left, right| by_load(left, right).then_with(|| by_distance(left, right)))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use crate::catalog::tests::regional_pool;
    use crate::load::InFlightTracker;

    use super::*;

    pub(crate) const OHIO: &str = "us-east-2";
    pub(crate) const TOKYO: &str = "ap-northeast-1";
    pub(crate) const OREGON: &str = "us-west-2";

    /// Two pools in Ohio, one in Oregon and one in Tokyo, with round trips
    /// measured from Ohio.
    pub(crate) fn fleet() -> (Vec<Pool>, Arc<PoolSignals>) {
        let pools = vec![
            regional_pool(OHIO, "ohio-a", "qwen", &["http://ohio-a:8000"]),
            regional_pool(OHIO, "ohio-b", "qwen", &["http://ohio-b:8000"]),
            regional_pool(OREGON, "oregon-a", "qwen", &["http://oregon-a:8000"]),
            regional_pool(TOKYO, "tokyo-a", "qwen", &["http://tokyo-a:8000"]),
        ];
        let signals = Arc::new(PoolSignals::new(
            Arc::new(InFlightTracker::new()),
            Duration::from_secs(60),
        ));
        for (pool, ms) in pools.iter().zip([0, 1, 50, 135]) {
            signals.record_round_trip(&pool.id, Duration::from_millis(ms));
        }
        (pools, signals)
    }

    /// Sets each pool's requests in flight, with 8 workers each.
    pub(crate) fn set_loads(pools: &[Pool], signals: &PoolSignals, in_flight: [u64; 4]) {
        for (pool, in_flight) in pools.iter().zip(in_flight) {
            let workers = 8;
            signals.record_load(&pool.id, PoolLoad { in_flight, workers });
        }
    }

    fn names(candidates: &[Candidate<'_>]) -> Vec<String> {
        candidates.iter().map(|c| c.pool.name.clone()).collect()
    }

    #[test]
    fn overload_is_relative_to_the_fleet_per_worker_with_a_floor() {
        let (pools, signals) = fleet();
        let refs = pools.iter().collect::<Vec<_>>();

        // Fleet: 400 / 32 = 12.5 per worker. Ohio-a has 25 per worker.
        set_loads(&pools, &signals, [200, 100, 80, 20]);
        let overloaded = place(&refs, OHIO, &signals, Overload::default())
            .iter()
            .map(|c| c.overloaded)
            .collect::<Vec<_>>();
        assert_eq!(overloaded, [true, false, false, false]);

        // Nearly idle: 0.6 per worker against 0.3 is under the floor.
        set_loads(&pools, &signals, [5, 3, 2, 0]);
        let placed = place(&refs, OHIO, &signals, Overload::default());
        assert!(placed.iter().all(|c| !c.overloaded));
    }

    #[test]
    fn orders_by_region_distance_then_load_per_worker() {
        let (pools, signals) = fleet();
        let refs = pools.iter().collect::<Vec<_>>();
        set_loads(&pools, &signals, [40, 20, 0, 0]);

        let mut placed = place(&refs, TOKYO, &signals, Overload::default());
        cost_order(&mut placed);
        // From Tokyo, Tokyo is home; Ohio's median measured round trip (1 ms)
        // is less than Oregon's (50 ms) here, and Ohio-b is less loaded.
        assert_eq!(
            names(&placed),
            [
                "ap-northeast-1/dynamo/tokyo-a",
                "us-east-2/dynamo/ohio-b",
                "us-east-2/dynamo/ohio-a",
                "us-west-2/dynamo/oregon-a",
            ]
        );
        assert_eq!(
            least_loaded(&placed).unwrap().pool.name,
            "ap-northeast-1/dynamo/tokyo-a"
        );
    }

    #[test]
    fn pools_without_fresh_load_are_tried_last() {
        let (pools, signals) = fleet();
        let refs = pools.iter().collect::<Vec<_>>();
        let placed = place(&refs, OHIO, &signals, Overload::default());
        assert!(placed.iter().all(|c| !c.overloaded && c.load.is_none()));

        signals.record_load(
            &pools[1].id,
            PoolLoad {
                in_flight: 8,
                workers: 8,
            },
        );
        let placed = place(&refs, OHIO, &signals, Overload::default());
        let overloaded = placed.iter().map(|c| c.overloaded).collect::<Vec<_>>();
        assert_eq!(overloaded, [true, false, true, true]);
    }
}
