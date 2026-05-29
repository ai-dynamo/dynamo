// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod bus;
pub mod jsonl;
pub mod jsonl_gz;
pub mod stream;

/// Parse a comma-separated list of sink names from an env-var value.
/// Empty/whitespace items are dropped; if the result is empty the list defaults
/// to `["stderr"]`. Used by audit and agent trace sink configuration.
pub fn parse_sink_names(value: &str) -> Vec<String> {
    let sinks: Vec<String> = value
        .split(',')
        .map(|v| v.trim().to_lowercase())
        .filter(|v| !v.is_empty())
        .collect();
    if sinks.is_empty() {
        vec!["stderr".to_string()]
    } else {
        sinks
    }
}

#[cfg(test)]
mod tests {
    use super::parse_sink_names;

    #[test]
    fn trims_and_normalizes() {
        assert_eq!(
            parse_sink_names(" jsonl, JSONL_GZ, STDERR "),
            vec![
                "jsonl".to_string(),
                "jsonl_gz".to_string(),
                "stderr".to_string(),
            ]
        );
    }

    #[test]
    fn defaults_empty_value_to_stderr() {
        assert_eq!(parse_sink_names(" , "), vec!["stderr".to_string()]);
    }
}
