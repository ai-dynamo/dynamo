// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(crate) use crate::replay::normalize_trace_requests;

pub(crate) mod agg;
pub(crate) mod components;
pub(crate) mod core;
pub(crate) mod disagg;
mod entrypoints;
pub(crate) mod events;
pub(crate) mod extensions;
pub(crate) mod planner_hook;
mod progress;
pub(crate) mod runtime_utils;
pub(crate) mod single;
pub(crate) mod state;

pub use entrypoints::run_offline_handoff_conformance;
pub(crate) use entrypoints::{
    generate_trace_worker_artifacts, generate_trace_worker_artifacts_with_visibility,
    simulate_agentic_trace_workload, simulate_concurrency, simulate_concurrency_disagg,
    simulate_concurrency_workload, simulate_concurrency_workload_accumulating_deltas,
    simulate_concurrency_workload_disagg, simulate_trace, simulate_trace_disagg,
    simulate_trace_workload, simulate_trace_workload_accumulating_deltas,
    simulate_trace_workload_disagg,
};

#[cfg(test)]
mod firewall_tests {
    use std::fs;
    use std::mem::size_of;
    use std::path::Path;

    use dynamo_kv_router::protocols::RouterEvent;

    use super::core::{EngineObservation, NoEngineEvents};

    fn rust_sources(root: &Path, sources: &mut Vec<std::path::PathBuf>) {
        for entry in fs::read_dir(root).unwrap() {
            let path = entry.unwrap().path();
            if path.is_dir() {
                rust_sources(&path, sources);
            } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
                sources.push(path);
            }
        }
    }

    #[test]
    fn offline_core_has_no_dynamo_adapter_dependencies() {
        const FORBIDDEN: &[&str] = &[
            "dynamo_kv_router",
            "crate::loadgen",
            "crate::scheduler",
            "OfflineWorkerState",
            "WorkloadDriver",
            "ReplayRouterMode",
            "extensions::",
        ];

        let core = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/replay/offline/core");
        let mut sources = Vec::new();
        rust_sources(&core, &mut sources);
        for path in sources {
            let source = fs::read_to_string(&path).unwrap();
            for forbidden in FORBIDDEN {
                assert!(
                    !source.contains(forbidden),
                    "{} crosses the offline core firewall with `{forbidden}`",
                    path.display()
                );
            }
        }
    }

    #[test]
    fn default_round_robin_observation_is_zero_sized() {
        type Batch = <NoEngineEvents as EngineObservation<Vec<RouterEvent>>>::Batch;

        assert_eq!(size_of::<NoEngineEvents>(), 0);
        assert_eq!(size_of::<Batch>(), 0);
        assert_eq!(size_of::<()>(), 0);
    }
}
