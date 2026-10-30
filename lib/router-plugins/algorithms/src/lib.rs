// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Worker-selection algorithms shared by Dynamo's router policies and `PushRouter`.
//!
//! Each function chooses an index into a candidate list of length `len` and reads any
//! per-candidate value through a closure, so callers keep their own candidate representation
//! and map the index back to a worker. Router policies belong in `dynamo-router-policies`;
//! this crate holds only the selection arithmetic they share with the runtime.

/// Uniform random indexes, injectable so tests can script each choice.
pub trait Samples {
    /// A uniform index in `0..upper`. Callers never pass zero.
    fn index(&mut self, upper: usize) -> usize;
}

/// The thread-local `fastrand` generator.
pub struct ThreadRandom;

impl Samples for ThreadRandom {
    #[inline(always)]
    fn index(&mut self, upper: usize) -> usize {
        fastrand::usize(..upper)
    }
}

/// One uniform draw.
#[inline(always)]
pub fn random(len: usize, samples: &mut impl Samples) -> Option<usize> {
    (len > 0).then(|| samples.index(len))
}

/// The position `cursor` reaches in a rotation over `len` candidates. The caller owns the
/// cursor, so it decides when a selection advances it and when a preview only reads it.
#[inline(always)]
pub fn round_robin(cursor: u64, len: usize) -> Option<usize> {
    (len > 0).then(|| cursor as usize % len)
}

/// The lowest load, with ties broken uniformly by reservoir sampling. Reads every load once.
#[inline(always)]
pub fn least_loaded(
    len: usize,
    load: impl Fn(usize) -> u64,
    samples: &mut impl Samples,
) -> Option<usize> {
    let mut best = None;
    let mut best_load = u64::MAX;
    let mut ties = 0usize;
    for index in 0..len {
        let candidate_load = load(index);
        if candidate_load < best_load {
            best = Some(index);
            best_load = candidate_load;
            ties = 1;
        } else if candidate_load == best_load {
            ties += 1;
            if samples.index(ties) == 0 {
                best = Some(index);
            }
        }
    }
    best
}

/// Two distinct uniform draws; the lower load wins, and the first on ties. Reads two loads.
#[inline(always)]
pub fn power_of_two(
    len: usize,
    load: impl Fn(usize) -> u64,
    samples: &mut impl Samples,
) -> Option<usize> {
    if len == 0 {
        return None;
    }
    let first = samples.index(len);
    if len == 1 {
        return Some(first);
    }
    let second = (first + 1 + samples.index(len - 1)) % len;
    Some(if load(first) <= load(second) {
        first
    } else {
        second
    })
}

/// The kind of device a worker runs on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Device {
    Cpu,
    #[default]
    Accelerator,
}

/// Request-level inputs to [`device_aware`].
#[derive(Clone, Copy, Debug)]
pub struct DeviceAwareContext {
    /// Distinct embedding-cache keys in the request. A worker holding all of them has a full hit.
    pub required_cache_hits: usize,
    /// How many accelerator workers' worth of load one CPU worker should carry, at least 1.
    pub non_cpu_to_cpu_ratio: usize,
}

/// The candidate [`device_aware`] chose, and whether it holds every embedding the request needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceAwareChoice {
    pub index: usize,
    pub full_cache_hit: bool,
}

/// Device-aware weighted selection for heterogeneous workers.
///
/// A worker holding every embedding the request needs wins outright, least loaded first.
/// Otherwise it keeps the least-loaded CPU worker and the least-loaded accelerator worker, and
/// sends work to the CPU worker while the CPU group's total load stays under its share,
/// weighted by `non_cpu_to_cpu_ratio`. Reads every load once.
#[inline(always)]
pub fn device_aware(
    len: usize,
    device: impl Fn(usize) -> Device,
    cache_hits: impl Fn(usize) -> usize,
    load: impl Fn(usize) -> u64,
    context: DeviceAwareContext,
    samples: &mut impl Samples,
) -> Option<DeviceAwareChoice> {
    let mut cpu = DeviceGroup::default();
    let mut accelerator = DeviceGroup::default();
    let mut full_cache = DeviceGroup::default();

    for index in 0..len {
        let candidate_load = load(index);
        match device(index) {
            Device::Cpu => cpu.consider(index, candidate_load, samples),
            Device::Accelerator => accelerator.consider(index, candidate_load, samples),
        }
        if context.required_cache_hits > 0 && cache_hits(index) >= context.required_cache_hits {
            full_cache.consider(index, candidate_load, samples);
        }
    }

    if let Some(index) = full_cache.best {
        return Some(DeviceAwareChoice {
            index,
            full_cache_hit: true,
        });
    }

    let index = match (cpu.best, accelerator.best) {
        (None, None) => return None,
        (Some(index), None) | (None, Some(index)) => index,
        (Some(cpu_index), Some(accelerator_index)) => {
            let ratio = context.non_cpu_to_cpu_ratio.max(1) as u64;
            let allowed_cpu = accelerator.total_load.saturating_mul(cpu.count)
                / ratio.saturating_mul(accelerator.count);
            if cpu.total_load < allowed_cpu {
                cpu_index
            } else {
                accelerator_index
            }
        }
    };
    Some(DeviceAwareChoice {
        index,
        full_cache_hit: false,
    })
}

#[derive(Default)]
struct DeviceGroup {
    count: u64,
    total_load: u64,
    best: Option<usize>,
    best_load: u64,
    best_ties: usize,
}

impl DeviceGroup {
    #[inline(always)]
    fn consider(&mut self, index: usize, load: u64, samples: &mut impl Samples) {
        self.count += 1;
        self.total_load = self.total_load.saturating_add(load);
        if self.best.is_none() || load < self.best_load {
            self.best = Some(index);
            self.best_load = load;
            self.best_ties = 1;
        } else if load == self.best_load {
            self.best_ties += 1;
            if samples.index(self.best_ties) == 0 {
                self.best = Some(index);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    struct ScriptedSamples {
        values: Vec<usize>,
        consumed: usize,
    }

    impl ScriptedSamples {
        fn new(values: impl Into<Vec<usize>>) -> Self {
            Self {
                values: values.into(),
                consumed: 0,
            }
        }
    }

    impl Samples for ScriptedSamples {
        fn index(&mut self, upper: usize) -> usize {
            let value = self.values[self.consumed];
            self.consumed += 1;
            value % upper
        }
    }

    #[test]
    fn empty_candidates_choose_nothing() {
        let mut samples = ScriptedSamples::new([]);
        assert_eq!(random(0, &mut samples), None);
        assert_eq!(round_robin(7, 0), None);
        assert_eq!(least_loaded(0, |_| 0, &mut samples), None);
        assert_eq!(power_of_two(0, |_| 0, &mut samples), None);
        let context = DeviceAwareContext {
            required_cache_hits: 0,
            non_cpu_to_cpu_ratio: 8,
        };
        assert_eq!(
            device_aware(0, |_| Device::Cpu, |_| 0, |_| 0, context, &mut samples),
            None
        );
        assert_eq!(samples.consumed, 0);
    }

    #[test]
    fn round_robin_rotates_with_the_cursor() {
        let picks: Vec<_> = (0..6)
            .map(|cursor| round_robin(cursor, 4).unwrap())
            .collect();
        assert_eq!(picks, [0, 1, 2, 3, 0, 1]);
    }

    #[test]
    fn random_uses_one_sample() {
        let mut samples = ScriptedSamples::new([2]);
        assert_eq!(random(4, &mut samples), Some(2));
        assert_eq!(samples.consumed, 1);
    }

    #[test]
    fn power_of_two_uses_exactly_two_samples_and_two_load_reads() {
        let mut samples = ScriptedSamples::new([1, 1]);
        let reads = Cell::new(0);
        let chosen = power_of_two(
            4,
            |index| {
                reads.set(reads.get() + 1);
                if index == 1 { 5 } else { 1 }
            },
            &mut samples,
        );
        assert_eq!(chosen, Some(3));
        assert_eq!(samples.consumed, 2);
        assert_eq!(reads.get(), 2);
    }

    #[test]
    fn least_loaded_ties_follow_scripted_reservoir_samples() {
        let mut samples = ScriptedSamples::new([1, 0]);
        assert_eq!(least_loaded(3, |_| 3, &mut samples), Some(2));
        assert_eq!(samples.consumed, 2);
    }

    #[test]
    fn least_loaded_picks_the_minimum() {
        let loads = [4, 1, 3];
        let mut samples = ScriptedSamples::new([]);
        assert_eq!(least_loaded(3, |index| loads[index], &mut samples), Some(1));
    }

    #[test]
    fn device_aware_prefers_a_full_cache_hit_and_reads_each_load_once() {
        let devices = [Device::Cpu, Device::Accelerator, Device::Accelerator];
        let hits = [0, 2, 1];
        let reads = Cell::new(0);
        let chosen = device_aware(
            3,
            |index| devices[index],
            |index| hits[index],
            |index| {
                reads.set(reads.get() + 1);
                index as u64
            },
            DeviceAwareContext {
                required_cache_hits: 2,
                non_cpu_to_cpu_ratio: 8,
            },
            &mut ScriptedSamples::new([]),
        );
        assert_eq!(
            chosen,
            Some(DeviceAwareChoice {
                index: 1,
                full_cache_hit: true
            })
        );
        assert_eq!(reads.get(), 3);
    }

    #[test]
    fn device_aware_sends_work_to_cpu_only_under_its_share() {
        let devices = [Device::Cpu, Device::Accelerator];
        let context = DeviceAwareContext {
            required_cache_hits: 0,
            non_cpu_to_cpu_ratio: 2,
        };
        // Accelerator load 10 allows the CPU group 10 * 1 / (2 * 1) = 5.
        let choose = |cpu_load: u64| {
            device_aware(
                2,
                |index| devices[index],
                |_| 0,
                |index| if index == 0 { cpu_load } else { 10 },
                context,
                &mut ScriptedSamples::new([]),
            )
            .unwrap()
            .index
        };
        assert_eq!(choose(4), 0);
        assert_eq!(choose(5), 1);
    }
}
