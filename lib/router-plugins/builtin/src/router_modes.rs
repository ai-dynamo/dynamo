// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The non-KV routing modes as worker-selection policies, one module per mode.
//!
//! `--router-mode random`, `round-robin` and `least-loaded` resolve to these types by name. Each picker declares at most [`WorkerInputs::ACTIVE_REQUESTS`].
//! `PushRouter` runs the same pickers through the instance selectors its caller passes, so each
//! algorithm lives only here.
//!
//! [`WorkerInputs::ACTIVE_REQUESTS`]: dynamo_kv_router::plugins::worker_selection::WorkerInputs::ACTIVE_REQUESTS

mod least_loaded;
mod random;
mod round_robin;

use std::sync::Arc;

use dynamo_kv_router::plugins::prelude::*;

pub use least_loaded::LEAST_LOADED;
pub use random::RANDOM;
pub use round_robin::ROUND_ROBIN;

/// Uniform random indexes, injectable so tests can script each choice.
trait Samples {
    /// A uniform index in `0..upper`. Callers never pass zero.
    fn index(&mut self, upper: usize) -> usize;
}

/// The thread-local `fastrand` generator.
struct ThreadRandom;

impl Samples for ThreadRandom {
    #[inline]
    fn index(&mut self, upper: usize) -> usize {
        fastrand::usize(..upper)
    }
}

fn no_candidates() -> WorkerSelectionPolicyError {
    WorkerSelectionPolicyError::failed("no eligible worker")
}

/// The row's active requests. The picker declared the input, so every in-range row reads `Some`.
fn active_requests(input: WorkerInputView<'_>) -> impl Fn(usize) -> u64 + '_ {
    move |row| input.active_requests(row).unwrap_or_default()
}

/// The routing modes take no parameters, so a policy file that passes one fails to resolve.
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct NoParameters {}

/// A provider whose factory builds a fresh picker per call, so every partition and selector
/// keeps its own state, such as its own rotation.
fn provider<P: WorkerPicker + 'static>(
    picker: fn() -> P,
) -> impl Fn(
    &WorkerSelectionPolicyParameters,
) -> Result<WorkerSelectionPolicyFactory, WorkerSelectionPolicyProviderError>
+ Send
+ Sync
+ 'static {
    move |parameters| {
        let _: NoParameters = parameters.deserialize()?;
        Ok(Arc::new(
            move |config: &SelectionConfig, worker_type, _partition| {
                WorkerSelectionPolicy::new(
                    config.clone(),
                    worker_type.as_str(),
                    Vec::new(),
                    Box::new(picker()),
                )
            },
        ))
    }
}

/// Add each routing mode `registry` doesn't already provide. Every host links them through
/// [`crate::with_defaults`].
pub(crate) fn register_missing(registry: &mut RouterPluginRegistry) {
    let modes: [(&str, WorkerSelectionPolicyProvider); 3] = [
        (RANDOM, Arc::new(provider(|| random::RandomPicker))),
        (
            ROUND_ROBIN,
            Arc::new(provider(round_robin::RoundRobinPicker::default)),
        ),
        (
            LEAST_LOADED,
            Arc::new(provider(|| least_loaded::LeastLoadedPicker)),
        ),
    ];
    for (name, provider) in modes {
        match registry.register_worker_selection(name, provider) {
            Ok(()) | Err(WorkerSelectionPolicyRegistryError::Duplicate { .. }) => {}
            Err(error) => unreachable!("{name} is a valid routing-mode type name: {error}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::Samples;
    use super::least_loaded::least_loaded;
    use super::random::random;
    use super::round_robin::round_robin;

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
}
