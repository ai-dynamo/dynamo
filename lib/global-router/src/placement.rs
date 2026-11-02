// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The placement rule the load-aware policies share: nearest region first,
//! then the less loaded pool, and leave the default pool only for one with
//! much more room.
//!
//! Affinity weighs each pool's occupancy and defaults to the session's pin;
//! least-in-flight weighs its requests in flight against the pool's capacity
//! and defaults to the cheapest pool. Both selectors and both router core
//! pickers place pools with these functions, so the rule has one copy.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Duration;

/// One candidate pool, with what the rule weighs.
#[derive(Debug)]
pub(crate) struct Placed<T> {
    pub(crate) item: T,
    /// How full the pool is, 1 being full. `None` when unknown, which counts
    /// as full.
    pub(crate) load: Option<f64>,
    /// Round trip to the pool's region: zero for the region the request
    /// entered. `None` until measured.
    pub(crate) distance: Option<Duration>,
    /// Requests in flight, the last tie-break. Zero where a policy does not
    /// count them.
    pub(crate) in_flight: u64,
}

/// The round trip from `region` to a pool in `pool_region`.
pub(crate) fn distance(
    pool_region: &str,
    region: &str,
    round_trips: &HashMap<String, Duration>,
) -> Option<Duration> {
    if pool_region == region {
        Some(Duration::ZERO)
    } else {
        round_trips.get(pool_region).copied()
    }
}

/// Sorts `placed` from least to most expensive: nearest region first, then
/// less loaded, then fewer in flight. Unknown distances and loads come last.
/// The sort is stable, so ties keep the order given.
pub(crate) fn order<T>(placed: &mut [Placed<T>]) {
    placed.sort_by(|left, right| cost(left, right));
}

/// How many leading candidates of an ordered `placed` cost the same as the
/// first: the ones a policy without a pin rotates among.
pub(crate) fn leading_ties<T>(placed: &[Placed<T>]) -> usize {
    let Some(first) = placed.first() else {
        return 0;
    };
    placed
        .iter()
        .take_while(|candidate| cost(first, candidate) == Ordering::Equal)
        .count()
}

/// The index in an ordered `placed` of the pool to use: `default`, unless
/// another candidate has more than `factor` times its spare room (1 minus
/// its load); then the first such candidate. A default whose load is unknown
/// counts as full, so it leaves for the first candidate with any room, and
/// stays if none has.
pub(crate) fn headroom<T>(placed: &[Placed<T>], default: usize, factor: f64) -> usize {
    let need = placed[default]
        .load
        .map_or(0.0, |load| (factor * (1.0 - load)).max(0.0));
    placed
        .iter()
        .enumerate()
        .filter(|(index, _)| *index != default)
        .find(|(_, candidate)| candidate.load.is_some_and(|load| 1.0 - load > need))
        .map_or(default, |(index, _)| index)
}

fn cost<T>(left: &Placed<T>, right: &Placed<T>) -> Ordering {
    known_first(left.distance, right.distance)
        .then_with(|| known_first_f64(left.load, right.load))
        .then_with(|| left.in_flight.cmp(&right.in_flight))
}

/// Smaller first; unknown last.
fn known_first<T: Ord>(left: Option<T>, right: Option<T>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn known_first_f64(left: Option<f64>, right: Option<f64>) -> Ordering {
    match (left, right) {
        (Some(left), Some(right)) => left.total_cmp(&right),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn placed(
        name: &'static str,
        load: Option<f64>,
        rtt: Option<u64>,
        in_flight: u64,
    ) -> Placed<&'static str> {
        Placed {
            item: name,
            load,
            distance: rtt.map(Duration::from_millis),
            in_flight,
        }
    }

    fn names(placed: &[Placed<&'static str>]) -> Vec<&'static str> {
        placed.iter().map(|candidate| candidate.item).collect()
    }

    #[test]
    fn orders_nearest_then_least_loaded_then_fewest_in_flight_unknown_last() {
        let mut pools = [
            placed("far", Some(0.0), Some(80), 0),
            placed("unmeasured", Some(0.0), None, 0),
            placed("local-busy", Some(0.9), Some(0), 0),
            placed("local-unknown", None, Some(0), 0),
            placed("local-idle-2", Some(0.1), Some(0), 2),
            placed("local-idle-1", Some(0.1), Some(0), 1),
        ];
        order(&mut pools);
        assert_eq!(
            names(&pools),
            [
                "local-idle-1",
                "local-idle-2",
                "local-busy",
                "local-unknown",
                "far",
                "unmeasured"
            ]
        );
        assert_eq!(leading_ties(&pools), 1);
    }

    #[test]
    fn leaves_the_default_only_for_much_more_room() {
        // 0.4 spare at home; 2x needs more than 0.8 spare elsewhere.
        let mut pools = [
            placed("home", Some(0.6), Some(0), 0),
            placed("far", Some(0.25), Some(80), 0),
        ];
        order(&mut pools);
        assert_eq!(pools[headroom(&pools, 0, 2.0)].item, "home");
        pools[1].load = Some(0.1);
        assert_eq!(pools[headroom(&pools, 0, 2.0)].item, "far");
    }

    #[test]
    fn an_unknown_default_leaves_for_any_room_and_stays_if_none_is_known() {
        let mut pools = [
            placed("home", None, Some(0), 0),
            placed("far", None, Some(80), 0),
        ];
        order(&mut pools);
        assert_eq!(pools[headroom(&pools, 0, 2.0)].item, "home");
        pools[1].load = Some(0.99);
        order(&mut pools);
        // Known load sorts before unknown within a region, not across regions.
        assert_eq!(names(&pools), ["home", "far"]);
        assert_eq!(pools[headroom(&pools, 0, 2.0)].item, "far");
    }
}
