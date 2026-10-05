// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

use anyhow::{Context, Result, ensure};

use crate::loadgen::{AgenticProfileOptions, AgenticSnapshotOptions, AgenticTrace, WorkloadDriver};

/// Optional AgentX phases for the existing typed workload replay API.
#[derive(Clone, Debug)]
pub struct AgenticReplayOptions {
    pub snapshot: Option<AgenticSnapshotOptions>,
    pub warmup: bool,
    pub profile: Option<AgenticProfileOptions>,
    pub arrival_speedup_ratio: f64,
}

impl Default for AgenticReplayOptions {
    fn default() -> Self {
        Self {
            snapshot: None,
            warmup: false,
            profile: None,
            arrival_speedup_ratio: 1.0,
        }
    }
}

impl AgenticReplayOptions {
    pub fn has_phases(&self) -> bool {
        self.snapshot.is_some() || self.warmup || self.profile.is_some()
    }

    pub fn validate(&self, lanes: Option<usize>, max_sim_time_ms: Option<f64>) -> Result<()> {
        ensure!(
            self.arrival_speedup_ratio.is_finite() && self.arrival_speedup_ratio > 0.0,
            "arrival_speedup_ratio must be finite and greater than zero"
        );
        if let Some(profile) = &self.profile {
            profile.validate()?;
            ensure!(
                self.snapshot.is_some(),
                "agentic_profile requires agentic_snapshot"
            );
            ensure!(
                max_sim_time_ms.is_none(),
                "agentic_profile cannot be combined with max_sim_time_ms"
            );
        }
        ensure!(
            !self.warmup || self.snapshot.is_some(),
            "agentic_warmup requires agentic_snapshot"
        );
        ensure!(
            self.snapshot.is_none() || lanes.is_some_and(|lanes| lanes > 0),
            "agentic_snapshot requires positive agentic_lanes"
        );
        Ok(())
    }

    pub(crate) fn into_driver(
        self,
        trace: AgenticTrace,
        block_size: usize,
        include_replay_hashes: bool,
        lanes: Option<usize>,
        max_sim_time_ms: Option<f64>,
    ) -> Result<WorkloadDriver> {
        self.validate(lanes, max_sim_time_ms)?;
        let Some(snapshot) = self.snapshot else {
            return trace
                .speed_up_timing(self.arrival_speedup_ratio)?
                .into_trace_driver_with_options(block_size, include_replay_hashes, lanes);
        };
        // Sample recorded source time first; speedup applies only to the saved
        // frontier's remaining timers, just as it does in AISimulate's runner.
        let prepared = trace.prepare_snapshots(
            lanes.context("agentic_snapshot requires positive agentic_lanes")?,
            snapshot,
        )?;
        let mut driver = if self.warmup {
            WorkloadDriver::new_agentic_warmup(
                prepared,
                block_size,
                include_replay_hashes,
                self.arrival_speedup_ratio,
            )?
        } else {
            WorkloadDriver::new_agentic_snapshots(
                prepared,
                block_size,
                include_replay_hashes,
                self.arrival_speedup_ratio,
            )?
        };
        if let Some(profile) = self.profile {
            driver.enable_agentic_profile(profile)?;
        }
        Ok(driver)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loadgen::AgenticGraphBuilder;
    use serde_json::json;

    fn recorded_graph() -> AgenticTrace {
        let mut graph = AgenticGraphBuilder::new(
            serde_json::from_value(json!({
                "schema": "dynamo.agentic_mooncake",
                "version": 2,
                "block_size": 4,
                "hash_id_scope": "local",
                "source": {"format": "self-authored-test", "digest": "typed-agentic-options"}
            }))
            .unwrap(),
        )
        .unwrap();
        for (id, start) in [("first", 100.0), ("second", 220.0)] {
            graph
                .push(
                    serde_json::from_value(json!({
                        "request_id": id,
                        "play_id": "play",
                        "session_id": id,
                        "model": "fixture-model",
                        "input_length": 8,
                        "output_length": 1,
                        "hash_ids": [1, 2],
                        "not_before_ms": start,
                        "recorded_api_time_ms": 10.0,
                        "dependencies": []
                    }))
                    .unwrap(),
                )
                .unwrap();
        }
        graph.finish().unwrap()
    }

    #[test]
    fn snapshot_sampling_keeps_source_clock_and_requested_lane_count() {
        let build = |speedup| {
            AgenticReplayOptions {
                snapshot: Some(AgenticSnapshotOptions { seed: 7 }),
                arrival_speedup_ratio: speedup,
                ..Default::default()
            }
            .into_driver(recorded_graph(), 4, true, Some(3), None)
            .unwrap()
        };
        let regular = build(1.0);
        let fast = build(4.0);
        let snapshots = regular.agentic_snapshot_evidence().unwrap();
        assert_eq!(
            snapshots.len(),
            3,
            "snapshot lanes may exceed source play count"
        );
        assert_eq!(
            snapshots,
            fast.agentic_snapshot_evidence().unwrap(),
            "speedup must not change recorded-time sampling or cache identity"
        );
        assert!(snapshots.iter().all(|snapshot| snapshot.t_star_ms >= 100.0));
    }

    #[test]
    fn finite_workload_preserves_timestamps_and_scales_timers_once() {
        for (lanes, first_ms, second_ms) in [(None, 100.0, 220.0), (Some(1), 0.0, 120.0)] {
            let mut regular = AgenticReplayOptions::default()
                .into_driver(recorded_graph(), 4, true, lanes, None)
                .unwrap();
            let mut fast = AgenticReplayOptions {
                arrival_speedup_ratio: 4.0,
                ..Default::default()
            }
            .into_driver(recorded_graph(), 4, true, lanes, None)
            .unwrap();
            // Finite lanes start a play at zero; unbounded replay keeps its
            // timestamps. Both paths scale the next independent root once.
            assert_eq!(regular.next_ready_time_ms(), Some(first_ms));
            assert_eq!(fast.next_ready_time_ms(), Some(first_ms / 4.0));
            assert_eq!(regular.pop_ready(first_ms, 1).len(), 1);
            assert_eq!(fast.pop_ready(first_ms / 4.0, 1).len(), 1);
            assert_eq!(regular.next_ready_time_ms(), Some(second_ms));
            assert_eq!(fast.next_ready_time_ms(), Some(second_ms / 4.0));
            assert!(regular.agentic_snapshot_evidence().is_none());
            assert!(regular.agentic_profile_report().is_none());
        }
    }

    #[test]
    fn warmup_defers_profile_clock_until_runtime_barrier() {
        let options = AgenticReplayOptions {
            snapshot: Some(AgenticSnapshotOptions { seed: 7 }),
            warmup: true,
            profile: Some(AgenticProfileOptions {
                duration_seconds: 0.5,
                response_grace_seconds: 0.2,
                cancel_drain_seconds: 0.1,
                ..Default::default()
            }),
            ..Default::default()
        };
        let warm = options
            .clone()
            .into_driver(recorded_graph(), 4, true, Some(1), None)
            .unwrap();
        assert!(warm.is_agentic_preparing());
        assert_eq!(
            warm.agentic_profile_deadlines(),
            None,
            "creating a driver must not bypass the native preparation barrier"
        );
        let cold = AgenticReplayOptions {
            warmup: false,
            ..options
        }
        .into_driver(recorded_graph(), 4, true, Some(1), None)
        .unwrap();
        assert!(!cold.is_agentic_preparing());
        assert_eq!(
            cold.agentic_profile_deadlines(),
            Some((500.0, 700.0, 800.0))
        );
    }

    #[test]
    fn invalid_phase_combinations_fail_before_execution() {
        let cases = [
            (
                AgenticReplayOptions {
                    warmup: true,
                    ..Default::default()
                },
                Some(1),
                None,
                "agentic_warmup requires agentic_snapshot",
            ),
            (
                AgenticReplayOptions {
                    profile: Some(AgenticProfileOptions::default()),
                    ..Default::default()
                },
                Some(1),
                None,
                "agentic_profile requires agentic_snapshot",
            ),
            (
                AgenticReplayOptions {
                    snapshot: Some(AgenticSnapshotOptions { seed: 7 }),
                    ..Default::default()
                },
                None,
                None,
                "agentic_snapshot requires positive agentic_lanes",
            ),
            (
                AgenticReplayOptions {
                    snapshot: Some(AgenticSnapshotOptions { seed: 7 }),
                    profile: Some(AgenticProfileOptions::default()),
                    ..Default::default()
                },
                Some(1),
                Some(100.0),
                "agentic_profile cannot be combined with max_sim_time_ms",
            ),
            (
                AgenticReplayOptions {
                    arrival_speedup_ratio: f64::NAN,
                    ..Default::default()
                },
                None,
                None,
                "arrival_speedup_ratio must be finite and greater than zero",
            ),
        ];
        for (options, lanes, time_limit, expected) in cases {
            assert_eq!(
                options.validate(lanes, time_limit).unwrap_err().to_string(),
                expected
            );
        }
    }
}
