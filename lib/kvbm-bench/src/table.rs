// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! comfy-table output helper.

use comfy_table::{Table, presets};

/// A formatted table for benchmark output.
pub struct BenchTable {
    table: Table,
    headers: Vec<String>,
    rows: Vec<Vec<String>>,
}

impl BenchTable {
    /// Create a new table with the given column headers.
    pub fn new(headers: &[&str]) -> Self {
        let mut table = Table::new();
        table.load_preset(presets::UTF8_FULL);
        table.set_header(headers.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        Self {
            table,
            headers: headers.iter().map(|s| s.to_string()).collect(),
            rows: Vec::new(),
        }
    }

    /// Add a row of cells to the table.
    pub fn add_row(&mut self, cells: &[String]) {
        self.table.add_row(cells.iter().map(|s| s.to_string()).collect::<Vec<_>>());
        self.rows.push(cells.to_vec());
    }

    /// Print the table to stdout.
    pub fn print(&self) {
        println!("{}", self.table);
    }

    /// Render the table as CSV.
    pub fn to_csv(&self) -> String {
        let mut out = String::new();
        out.push_str(&self.headers.join(","));
        out.push('\n');
        for row in &self.rows {
            out.push_str(&row.join(","));
            out.push('\n');
        }
        out
    }
}
