// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The placement rule shared by affinity and least-in-flight: nearest region
//! first, then less loaded; leave the default only for much more room.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::time::Duration;

#[derive(Debug)]
pub(crate) struct Placed<T> {
    pub(crate) item: T,
    /// 1 is full; `None` (unknown) counts as full.
    pub(crate) load: Option<f64>,
    /// Round trip to the pool's region; zero for the local region.
    pub(crate) distance: Option<Duration>,
    /// The last tie-break.
    pub(crate) in_flight: u64,
}

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

/// Cheapest first; unknowns last; stable.
pub(crate) fn order<T>(placed: &mut [Placed<T>]) {
    placed.sort_by(|left, right| cost(left, right));
}

/// How many leading candidates tie with the first.
pub(crate) fn leading_ties<T>(placed: &[Placed<T>]) -> usize {
    let Some(first) = placed.first() else {
        return 0;
    };
    placed
        .iter()
        .take_while(|candidate| cost(first, candidate) == Ordering::Equal)
        .count()
}

/// `default`, unless a candidate has more than `factor` times its spare room;
/// then the first such candidate.
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
