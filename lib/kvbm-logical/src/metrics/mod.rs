// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Re-export from kvbm-observability for backward compatibility
pub use kvbm_observability::{
    BlockPoolMetrics, MetricsAggregator, MetricsSnapshot, StatsCollector, StatsConfig,
    StatsSnapshot, short_type_name,
};
