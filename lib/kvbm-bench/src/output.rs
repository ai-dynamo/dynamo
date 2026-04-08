// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Output format selection.

/// Output format for benchmark results.
#[derive(Clone, Debug, clap::ValueEnum)]
pub enum OutputFormat {
    /// Pretty-printed table using comfy-table.
    Table,
    /// Comma-separated values.
    Csv,
    /// JSON array.
    Json,
}
