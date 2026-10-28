// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! What the session-aware selectors share: how busy each pool is, which
//! pools are overloaded, how far each pool's region is from the region a
//! request entered, and the order to try pools in.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Duration;

use crate::catalog::Pool;
use crate::signals::{PoolSignals, PoolTiming};

/// How the session-aware selectors measure how busy a pool is.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LoadMeasure {
    /// Requests in flight per worker. Pools differ in size, so loads are
    /// compared per worker; it assumes every worker serves as many.
    #[default]
    PerWorker,
    /// The expected time for a typical request: the wait for a first token
    /// plus the fleet's mean output length times the pool's time per output
    /// token (see [`PoolSignals::timing`]). It covers queueing, slower decode
    /// in fuller batches, and slower hardware, so it also compares pools
    /// whose workers differ in speed or size. A pool not yet measured while
    /// generating takes the mean time per token of those that have been.
    Latency,
    /// How full a pool is against its own capacity, 1 being full: the
    /// fuller of its request slots and its KV blocks (see
    /// [`PoolSignals::occupancy`]; the slots alone until the pool's KV is
    /// known). It compares pools of any speed or size, and a pool counts as
    /// overloaded only when it is full.
    Occupancy,
}

/// When a pool counts as overloaded: its load is more than `ratio` times
/// the fleet's, and more than the fleet's by at least a floor: `floor`
/// requests per worker, or `latency_floor` of expected time. The floor keeps
/// nearly idle pools from counting as overloaded over a request or two.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Overload {
    pub ratio: f64,
    pub floor: f64,
    pub latency_floor: Duration,
}

impl Default for Overload {
    fn default() -> Self {
        Self {
            ratio: 1.25,
            floor: 1.0,
            latency_floor: Duration::from_millis(500),
        }
    }
}

/// When a request leaves the pool a session-aware selector would otherwise
/// send it to: the session's pinned or home pool.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum SpillRule {
    /// Leave the pool only while it is overloaded (see [`Overload`]).
    #[default]
    FleetRatio,
    /// Leave the pool for the cheapest other pool when that one is at least
    /// `margin` cheaper. A pool's cost is the round trip to its region plus
    /// its expected time for a request, so it needs [`LoadMeasure::Latency`].
    /// Coming back
    /// takes the same margin the other way, so a session does not flip
    /// between pools that cost about the same.
    Cheapest { margin: Duration },
    /// Leave the pool for the nearest pool with more than `factor` times its
    /// spare capacity (1 minus [`LoadMeasure::Occupancy`]); a full pool
    /// leaves for the nearest pool with any room, and stays if none has
    /// room. With `factor` 2, a pool at 80% gives sessions to one at 10%,
    /// but one at 10% keeps them from one at 2%. Coming back takes the same
    /// factor the other way, so sessions do not flip between pools.
    Headroom { factor: f64 },
}

/// How the session-aware selectors decide when a request leaves its pool.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Spill {
    pub measure: LoadMeasure,
    pub overload: Overload,
    pub rule: SpillRule,
}

/// A reachable pool, with what the session-aware selectors weigh.
#[derive(Debug)]
pub(crate) struct Candidate<'a> {
    pub pool: &'a Pool,
    /// How busy the pool is by the [`LoadMeasure`] in use: requests per
    /// worker, or seconds of expected time for a request. `None` without a
    /// fresh report.
    pub busy: Option<f64>,
    /// Round trip to the pool's region: zero for the region the request
    /// entered, otherwise the median round trip measured to that region's
    /// pools. `None` until one is measured.
    pub distance: Option<Duration>,
    pub overloaded: bool,
}

/// The candidates that have a frontend, for a request entering `region`.
///
/// The fleet's load is taken over the pools with a fresh report: all
/// requests in flight over all workers, or, for [`LoadMeasure::Latency`],
/// the pools' expected times weighted by how many requests each serves. A
/// pool
/// without a fresh report counts as overloaded, so it is tried only after
/// the pools known to have room. When no pool has a fresh report, none is
/// overloaded and distance alone orders them.
pub(crate) fn place<'a>(
    candidates: &[&'a Pool],
    region: &str,
    signals: &PoolSignals,
    spill: Spill,
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
    let (busy, fleet, floor) =
        match spill.measure {
            LoadMeasure::PerWorker => {
                let (in_flight, workers) =
                    loads
                        .iter()
                        .flatten()
                        .fold((0, 0), |(in_flight, workers), load| {
                            (in_flight + load.in_flight, workers + load.workers)
                        });
                let busy = loads
                    .iter()
                    .map(|load| load.map(|load| load.per_worker()))
                    .collect::<Vec<_>>();
                let fleet = (workers > 0).then(|| in_flight as f64 / workers as f64);
                (busy, fleet, spill.overload.floor)
            }
            LoadMeasure::Latency => {
                let timings = pools
                    .iter()
                    .map(|pool| signals.timing(&pool.id))
                    .collect::<Vec<_>>();
                let busy = latencies(&timings);
                let (weighted, served) = timings.iter().zip(&busy).fold(
                    (0.0, 0.0),
                    |(weighted, served), (timing, busy)| match (timing, busy) {
                        (Some(timing), Some(busy)) => (
                            weighted + timing.served_per_s * busy,
                            served + timing.served_per_s,
                        ),
                        _ => (weighted, served),
                    },
                );
                let fleet = (served > 0.0).then(|| weighted / served);
                (busy, fleet, spill.overload.latency_floor.as_secs_f64())
            }
            LoadMeasure::Occupancy => {
                let busy = pools
                    .iter()
                    .map(|pool| signals.occupancy(&pool.id).map(|occupancy| occupancy.load))
                    .collect::<Vec<_>>();
                (busy, None, 0.0)
            }
        };
    let distances = region_distances(&pools, region, signals);
    pools
        .into_iter()
        .zip(busy)
        .map(|(pool, busy)| {
            let overloaded = match (spill.measure, busy, fleet) {
                // Relative to the pool's own capacity: full is 1.
                (LoadMeasure::Occupancy, busy, _) => busy.is_none_or(|busy| busy >= 1.0),
                (_, Some(busy), Some(fleet)) => {
                    busy > spill.overload.ratio * fleet && busy - fleet >= floor
                }
                (_, None, Some(_)) => true,
                (_, _, None) => false,
            };
            Candidate {
                pool,
                busy,
                distance: distances[pool.location.region.as_str()],
                overloaded,
            }
        })
        .collect()
}

/// Each pool's expected time for a typical request, in seconds: its wait for
/// a first token plus the fleet's mean output length (output tokens per
/// completed request over the window) times its time per output token.
fn latencies(timings: &[Option<PoolTiming>]) -> Vec<Option<f64>> {
    let (tokens, completed) = timings
        .iter()
        .flatten()
        .fold((0, 0), |(tokens, completed), timing| {
            (tokens + timing.output_tokens, completed + timing.completed)
        });
    let output_len = if completed > 0 {
        tokens as f64 / completed as f64
    } else {
        0.0
    };
    let measured = timings
        .iter()
        .flatten()
        .filter_map(|timing| timing.token_time)
        .map(|token_time| token_time.as_secs_f64())
        .collect::<Vec<_>>();
    let typical_token_time = if measured.is_empty() {
        0.0
    } else {
        measured.iter().sum::<f64>() / measured.len() as f64
    };
    timings
        .iter()
        .map(|timing| {
            timing.map(|timing| {
                let token_time = timing
                    .token_time
                    .map_or(typical_token_time, |token_time| token_time.as_secs_f64());
                timing.wait.as_secs_f64() + output_len * token_time
            })
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

/// Less busy first; unknown loads last.
pub(crate) fn by_load(left: &Candidate<'_>, right: &Candidate<'_>) -> Ordering {
    match (left.busy, right.busy) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

/// From least to most expensive: nearest region first, then less busy.
pub(crate) fn cost_order(candidates: &mut [Candidate<'_>]) {
    candidates.sort_by(|left, right| by_distance(left, right).then_with(|| by_load(left, right)));
}

/// For when every pool is overloaded: the least busy, nearest on ties.
pub(crate) fn least_loaded<'a, 'b>(candidates: &'b [Candidate<'a>]) -> Option<&'b Candidate<'a>> {
    candidates
        .iter()
        .min_by(|left, right| by_load(left, right).then_with(|| by_distance(left, right)))
}

/// What a request to the pool costs, in seconds: the round trip to its
/// region plus its expected wait. `None` unless both are known.
fn cost(candidate: &Candidate<'_>) -> Option<f64> {
    Some(candidate.distance?.as_secs_f64() + candidate.busy?)
}

/// For [`SpillRule::Headroom`]: `default`, unless another pool has more than
/// `factor` times its spare capacity (1 minus its occupancy); then the first
/// such pool in `candidates`' order, nearest first. A full default leaves
/// for the first pool with any room, and stays if none has room; a default
/// whose occupancy is unknown leaves for the first pool whose occupancy is
/// known. Without a default, the first candidate.
pub(crate) fn headroom<'a, 'b>(
    candidates: &'b [Candidate<'a>],
    default: Option<&'b Candidate<'a>>,
    factor: f64,
) -> Option<&'b Candidate<'a>> {
    let Some(default) = default else {
        return candidates.first();
    };
    // Less than none for a pool of unknown load, so any known pool beats it.
    let need = default
        .busy
        .map_or(f64::NEG_INFINITY, |busy| (factor * (1.0 - busy)).max(0.0));
    candidates
        .iter()
        .filter(|candidate| candidate.pool.id != default.pool.id)
        .find(|candidate| candidate.busy.is_some_and(|busy| 1.0 - busy > need))
        .or(Some(default))
}

/// For [`SpillRule::Cheapest`]: `default`, unless the cheapest other pool
/// is at least `margin` cheaper, or `default`'s cost is unknown; then that
/// pool. Without a default, the cheapest pool. When no other pool's cost
/// is known, `default`, or else the nearest pool.
pub(crate) fn cheapest<'a, 'b>(
    candidates: &'b [Candidate<'a>],
    default: Option<&'b Candidate<'a>>,
    margin: Duration,
) -> Option<&'b Candidate<'a>> {
    let best = candidates
        .iter()
        .filter(|candidate| default.is_none_or(|default| default.pool.id != candidate.pool.id))
        .filter_map(|candidate| Some((cost(candidate)?, candidate)))
        .min_by(|(left_cost, left), (right_cost, right)| {
            left_cost
                .total_cmp(right_cost)
                .then_with(|| by_distance(left, right))
                .then_with(|| left.pool.name.cmp(&right.pool.name))
        });
    match (default, best) {
        (Some(default), Some((best_cost, best))) => match cost(default) {
            Some(own) if own - best_cost < margin.as_secs_f64() => Some(default),
            _ => Some(best),
        },
        (Some(default), None) => Some(default),
        (None, Some((_, best))) => Some(best),
        (None, None) => candidates
            .iter()
            .min_by(|left, right| by_distance(left, right)),
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::sync::Arc;

    use crate::catalog::tests::regional_pool;
    use crate::load::InFlightTracker;
    use crate::signals::{LoadReport, PoolLoad};

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
            let load = PoolLoad {
                in_flight,
                workers: 8,
            };
            signals.record_load(
                &pool.id,
                LoadReport {
                    load,
                    awaiting: 0,
                    started_total: 0,
                    completed_total: 0,
                    output_tokens_total: 0,
                    max_concurrency: None,
                    observed_ms: 0,
                },
            );
        }
    }

    /// Gives every pool 40 requests in flight over 8 workers, and sets each
    /// pool's expected wait from its waiting requests and the requests it
    /// serves per second (with as many waiting at the start as at the end).
    pub(crate) fn set_waits(pools: &[Pool], signals: &PoolSignals, waits: [(u64, u64); 4]) {
        for (pool, (awaiting, per_second)) in pools.iter().zip(waits) {
            signals.forget_load(&pool.id);
            for (observed_ms, started_total) in [(0, 0), (1000, per_second)] {
                let load = PoolLoad {
                    in_flight: 40,
                    workers: 8,
                };
                let report = LoadReport {
                    load,
                    awaiting,
                    started_total,
                    completed_total: 0,
                    output_tokens_total: 0,
                    max_concurrency: None,
                    observed_ms,
                };
                signals.record_load(&pool.id, report);
            }
        }
    }

    pub(crate) fn by_latency() -> Spill {
        Spill {
            measure: LoadMeasure::Latency,
            ..Spill::default()
        }
    }

    /// By latency, leaving a pool only for one at least 1 s cheaper.
    pub(crate) fn by_cost() -> Spill {
        Spill {
            rule: SpillRule::Cheapest {
                margin: Duration::from_secs(1),
            },
            ..by_latency()
        }
    }

    /// Sets each pool's occupancy: requests in flight, all generating, over
    /// 128 slots, with too little KV in use to bind.
    pub(crate) fn set_occupancy(pools: &[Pool], signals: &PoolSignals, loads: [f64; 4]) {
        for (pool, load) in pools.iter().zip(loads) {
            let in_flight = (load * 128.0).round() as u64;
            let report = LoadReport {
                load: PoolLoad {
                    in_flight,
                    workers: 8,
                },
                awaiting: 0,
                started_total: in_flight,
                completed_total: 0,
                output_tokens_total: 0,
                max_concurrency: Some(128),
                observed_ms: 0,
            };
            signals.record_load(&pool.id, report);
            signals.record_kv(&pool.id, 0, 16384);
        }
    }

    pub(crate) fn by_headroom() -> Spill {
        Spill {
            measure: LoadMeasure::Occupancy,
            rule: SpillRule::Headroom { factor: 2.0 },
            ..Spill::default()
        }
    }

    fn names(candidates: &[Candidate<'_>]) -> Vec<String> {
        candidates.iter().map(|c| c.pool.name.clone()).collect()
    }

    fn overloaded(candidates: &[Candidate<'_>]) -> Vec<bool> {
        candidates.iter().map(|c| c.overloaded).collect()
    }

    #[test]
    fn overload_is_relative_to_the_fleet_per_worker_with_a_floor() {
        let (pools, signals) = fleet();
        let refs = pools.iter().collect::<Vec<_>>();

        // Fleet: 400 / 32 = 12.5 per worker. Ohio-a has 25 per worker.
        set_loads(&pools, &signals, [200, 100, 80, 20]);
        let placed = place(&refs, OHIO, &signals, Spill::default());
        assert_eq!(overloaded(&placed), [true, false, false, false]);

        // Nearly idle: 0.6 per worker against 0.3 is under the floor.
        set_loads(&pools, &signals, [5, 3, 2, 0]);
        let placed = place(&refs, OHIO, &signals, Spill::default());
        assert!(placed.iter().all(|c| !c.overloaded));
    }

    #[test]
    fn wait_tells_apart_pools_that_are_equally_full_but_not_equally_fast() {
        let (pools, signals) = fleet();
        let refs = pools.iter().collect::<Vec<_>>();
        // Every pool has 5 requests per worker. Ohio-a starts 4 requests a
        // second with 20 waiting (5 s); the others start 8 with 4 waiting
        // (0.5 s). The fleet waits 32 / 28 = 1.14 s.
        set_waits(&pools, &signals, [(20, 4), (4, 8), (4, 8), (4, 8)]);

        let mut placed = place(&refs, OHIO, &signals, Spill::default());
        assert!(placed.iter().all(|c| !c.overloaded));
        cost_order(&mut placed);
        assert_eq!(
            names(&placed)[..2],
            ["us-east-2/dynamo/ohio-a", "us-east-2/dynamo/ohio-b"]
        );

        let mut placed = place(&refs, OHIO, &signals, by_latency());
        assert_eq!(overloaded(&placed), [true, false, false, false]);
        cost_order(&mut placed);
        assert_eq!(
            names(&placed)[..2],
            ["us-east-2/dynamo/ohio-b", "us-east-2/dynamo/ohio-a"]
        );
    }

    #[test]
    fn latency_counts_decode_speed_so_an_idle_slow_pool_ranks_behind_a_fast_one() {
        let (pools, signals) = fleet();
        let refs = pools.iter().collect::<Vec<_>>();
        // No queues anywhere. Over 1 s each pool has 10 requests generating
        // and completes one: Ohio-a makes 200 tokens (50 ms per token), the
        // other two 500 (20 ms); Tokyo has never generated.
        for (pool, (generating, completed, tokens)) in
            pools
                .iter()
                .zip([(10, 1, 200), (10, 1, 500), (10, 1, 500), (0, 0, 0)])
        {
            for (observed_ms, completed_total, output_tokens_total) in
                [(0, 0, 0), (1000, completed, tokens)]
            {
                let load = PoolLoad {
                    in_flight: generating,
                    workers: 8,
                };
                let report = LoadReport {
                    load,
                    awaiting: 0,
                    started_total: completed_total + generating,
                    completed_total,
                    output_tokens_total,
                    max_concurrency: None,
                    observed_ms,
                };
                signals.record_load(&pool.id, report);
            }
        }

        // By requests per worker, the Ohio pools look the same.
        let mut placed = place(&refs, OHIO, &signals, Spill::default());
        cost_order(&mut placed);
        assert_eq!(
            names(&placed)[..2],
            ["us-east-2/dynamo/ohio-a", "us-east-2/dynamo/ohio-b"]
        );

        // A typical request makes 1,200 / 3 = 400 tokens: 20 s on Ohio-a,
        // 8 s on Ohio-b and Oregon. Tokyo takes the mean measured time per
        // token, 30 ms: 12 s.
        let mut placed = place(&refs, OHIO, &signals, by_latency());
        let busy = placed.iter().map(|c| c.busy.unwrap()).collect::<Vec<_>>();
        for (busy, want) in busy.iter().zip([20.0, 8.0, 8.0, 12.0]) {
            assert!((busy - want).abs() < 1e-6, "{busy} != {want}");
        }
        assert_eq!(overloaded(&placed), [true, false, false, false]);
        cost_order(&mut placed);
        assert_eq!(
            names(&placed)[..2],
            ["us-east-2/dynamo/ohio-b", "us-east-2/dynamo/ohio-a"]
        );
    }

    #[test]
    fn wait_overload_has_a_floor() {
        let (pools, signals) = fleet();
        let refs = pools.iter().collect::<Vec<_>>();
        // Ohio-a waits 0.2 s against a fleet of 0.05 s: four times longer,
        // but under the 0.5 s floor.
        set_waits(&pools, &signals, [(1, 5), (0, 5), (0, 5), (0, 5)]);
        let placed = place(&refs, OHIO, &signals, by_latency());
        assert!(placed.iter().all(|c| !c.overloaded));
    }

    #[test]
    fn orders_by_region_distance_then_load_per_worker() {
        let (pools, signals) = fleet();
        let refs = pools.iter().collect::<Vec<_>>();
        set_loads(&pools, &signals, [40, 20, 0, 0]);

        let mut placed = place(&refs, TOKYO, &signals, Spill::default());
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
        let placed = place(&refs, OHIO, &signals, Spill::default());
        assert!(placed.iter().all(|c| !c.overloaded && c.busy.is_none()));

        let load = PoolLoad {
            in_flight: 8,
            workers: 8,
        };
        let report = LoadReport {
            load,
            awaiting: 0,
            started_total: 0,
            completed_total: 0,
            output_tokens_total: 0,
            max_concurrency: None,
            observed_ms: 0,
        };
        signals.record_load(&pools[1].id, report);
        let placed = place(&refs, OHIO, &signals, Spill::default());
        assert_eq!(overloaded(&placed), [true, false, true, true]);

        // By wait, a pool needs two reports before it has a rate.
        let placed = place(&refs, OHIO, &signals, by_latency());
        assert!(placed.iter().all(|c| !c.overloaded && c.busy.is_none()));
    }
}
