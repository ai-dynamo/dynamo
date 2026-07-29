// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Prometheus metrics for block pool observability.
//!
//! Architecture:
//! - [`BlockPoolMetrics`]: Raw `AtomicU64`/`AtomicI64` counters and gauges. Zero overhead on hot path.
//! - [`MetricsAggregator`]: Custom `prometheus::core::Collector` that reads atomics at scrape time.
//! - [`StatsCollector`]: Optional periodic sampler computing rates, gradients, and hit ratios.

mod collector;
mod pool_metrics;
mod stats;

pub use collector::MetricsAggregator;
pub use pool_metrics::{BlockPoolMetrics, MetricsSnapshot};
pub use stats::{StatsCollector, StatsConfig, StatsSnapshot};

/// Returns the short (unqualified) type name for `T`.
///
/// Strips generic parameters and the module path, returning only the base
/// type name. May still be imperfect for deeply nested or anonymous types.
pub fn short_type_name<T: 'static>() -> String {
    let full = std::any::type_name::<T>();
    let base = full.split_once('<').map(|(b, _)| b).unwrap_or(full);
    base.rsplit("::").next().unwrap_or(base).to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MyType;

    #[test]
    fn test_short_type_name() {
        let name = short_type_name::<MyType>();
        assert_eq!(name, "MyType");
    }

    #[test]
    fn test_short_type_name_primitive() {
        let name = short_type_name::<u32>();
        assert_eq!(name, "u32");
    }

    #[test]
    fn test_short_type_name_generic() {
        let name = short_type_name::<Vec<String>>();
        assert_eq!(name, "Vec");
    }
}
