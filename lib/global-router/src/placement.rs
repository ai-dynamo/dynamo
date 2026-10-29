// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the affinity selector weighs: how full each pool is, how far each
//! pool's region is from the region a request entered, and the order to try
//! pools in.

use std::cmp::Ordering;
use std::time::Duration;

use crate::Pool;
use crate::selector::reachable;
use crate::signals::PoolSignals;

/// A reachable pool, with what the affinity selector weighs.
#[derive(Debug)]
pub(crate) struct Candidate<'a> {
    pub pool: &'a Pool,
    /// How full the pool is against its own capacity, 1 being full (see
    /// [`PoolSignals::occupancy`]). `None` without a fresh report.
    pub load: Option<f64>,
    /// Round trip to the pool's region: zero for the region the request
    /// entered, otherwise as last measured. `None` until measured.
    pub distance: Option<Duration>,
}

/// The candidates that have a frontend, for a request entering `region`,
/// from least to most expensive: nearest region first, then less full, then
/// catalog order. Pools whose distance or load is unknown come after those
/// whose is known.
pub(crate) fn place<'a>(
    candidates: &[&'a Pool],
    region: &str,
    signals: &PoolSignals,
) -> Vec<Candidate<'a>> {
    let round_trips = signals.region_round_trips();
    let mut placed = candidates
        .iter()
        .copied()
        .filter(|pool| reachable(pool))
        .map(|pool| Candidate {
            pool,
            load: signals.occupancy(&pool.id).map(|occupancy| occupancy.load),
            distance: if pool.location.region == region {
                Some(Duration::ZERO)
            } else {
                round_trips.get(&pool.location.region).copied()
            },
        })
        .collect::<Vec<_>>();
    placed.sort_by(|left, right| by_distance(left, right).then_with(|| by_load(left, right)));
    placed
}

/// Nearest region first; regions not yet measured last.
fn by_distance(left: &Candidate<'_>, right: &Candidate<'_>) -> Ordering {
    match (left.distance, right.distance) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// Less full first; unknown loads last.
fn by_load(left: &Candidate<'_>, right: &Candidate<'_>) -> Ordering {
    match (left.load, right.load) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// `default`, unless another pool has more than `factor` times its spare
/// capacity (1 minus its load); then the first such pool in `candidates`'
/// order, nearest first. A full default leaves for the first pool with any
/// room, and stays if none has room. A default whose load is unknown counts
/// as full: sending more there blind could overfill it. Without a default,
/// the first candidate.
///
/// With `factor` 2, a pool at 80% gives requests to one at 10%, but one at
/// 10% keeps them from one at 2%. Coming back takes the same factor the
/// other way, so sessions do not flip between pools.
pub(crate) fn headroom<'a, 'b>(
    candidates: &'b [Candidate<'a>],
    default: Option<&'b Candidate<'a>>,
    factor: f64,
) -> Option<&'b Candidate<'a>> {
    let Some(default) = default else {
        return candidates.first();
    };
    let need = default
        .load
        .map_or(0.0, |load| (factor * (1.0 - load)).max(0.0));
    candidates
        .iter()
        .filter(|candidate| candidate.pool.id != default.pool.id)
        .find(|candidate| candidate.load.is_some_and(|load| 1.0 - load > need))
        .or(Some(default))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;
    use std::sync::Arc;

    use crate::load::InFlightTracker;
    use crate::selector::tests::regional_pool;
    use crate::signals::LoadReport;

    use super::*;

    pub(crate) const OHIO: &str = "us-east-2";
    pub(crate) const OREGON: &str = "us-west-2";
    pub(crate) const TOKYO: &str = "ap-northeast-1";

    /// Two pools in Ohio, one in Oregon and one in Tokyo, with round trips
    /// measured from Ohio.
    pub(crate) fn fleet() -> (Vec<Pool>, Arc<PoolSignals>) {
        let pools = vec![
            regional_pool(OHIO, "ohio-a", "qwen", &["http://ohio-a:8000"]),
            regional_pool(OHIO, "ohio-b", "qwen", &["http://ohio-b:8000"]),
            regional_pool(OREGON, "oregon-a", "qwen", &["http://oregon-a:8000"]),
            regional_pool(TOKYO, "tokyo-a", "qwen", &["http://tokyo-a:8000"]),
        ];
        let signals = Arc::new(PoolSignals::new(Arc::new(InFlightTracker::new())));
        signals.set_region_round_trips(HashMap::from([
            (OREGON.to_owned(), Duration::from_millis(50)),
            (TOKYO.to_owned(), Duration::from_millis(135)),
        ]));
        (pools, signals)
    }

    /// Sets each pool's occupancy, in reports fresh for a minute: requests in
    /// flight, all generating, over 128 slots, with too little KV in use to
    /// bind.
    pub(crate) fn set_occupancy(pools: &[Pool], signals: &PoolSignals, loads: [f64; 4]) {
        let window = Duration::from_secs(20);
        for (pool, load) in pools.iter().zip(loads) {
            let in_flight = (load * 128.0).round() as u64;
            let report = LoadReport {
                in_flight,
                generating: in_flight,
                max_concurrency: Some(128),
                window,
            };
            signals.record_load(&pool.id, report);
            signals.record_kv(&pool.id, 0, 16384, window);
        }
    }

    fn names(placed: &[Candidate<'_>]) -> Vec<String> {
        placed
            .iter()
            .map(|candidate| candidate.pool.frontends[0].host_str().unwrap().to_owned())
            .collect()
    }

    #[test]
    fn orders_by_region_distance_then_load() {
        let (pools, signals) = fleet();
        let candidates = pools.iter().collect::<Vec<_>>();
        set_occupancy(&pools, &signals, [0.6, 0.2, 0.1, 0.0]);
        assert_eq!(
            names(&place(&candidates, OHIO, &signals)),
            ["ohio-b", "ohio-a", "oregon-a", "tokyo-a"]
        );
        // From Tokyo, its own region is nearest.
        assert_eq!(names(&place(&candidates, TOKYO, &signals))[0], "tokyo-a");
    }

    #[test]
    fn pools_without_load_or_distance_are_tried_last() {
        let (pools, signals) = fleet();
        let candidates = pools.iter().collect::<Vec<_>>();
        set_occupancy(&pools, &signals, [0.6, 0.2, 0.1, 0.0]);
        signals.forget_load(&pools[1].id);
        signals.set_region_round_trips(HashMap::from([(
            TOKYO.to_owned(),
            Duration::from_millis(135),
        )]));
        let placed = place(&candidates, OHIO, &signals);
        assert_eq!(names(&placed), ["ohio-a", "ohio-b", "tokyo-a", "oregon-a"]);
        assert_eq!(placed[1].load, None);
        assert_eq!(placed[3].distance, None);
    }
}
