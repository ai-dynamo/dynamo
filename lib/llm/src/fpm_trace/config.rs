// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::OnceLock;

use dynamo_runtime::config::environment_names::llm::fpm_trace as env_fpm_trace;

pub const DEFAULT_OUTPUT_PATH: &str = "/tmp/dynamo-fpm";
pub const DEFAULT_SAMPLE_INTERVAL_MS: u64 = 5_000;
pub const DEFAULT_JSONL_GZ_ROLL_BYTES: u64 = 256 * 1024 * 1024;
pub const DEFAULT_MAX_SEGMENTS: usize = 4;
pub(crate) const DEFAULT_CAPACITY: usize = 8_192;
pub(crate) const DEFAULT_JSONL_BUFFER_BYTES: usize = 1024 * 1024;
pub(crate) const DEFAULT_JSONL_FLUSH_INTERVAL_MS: u64 = 1_000;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq, serde::Serialize)]
#[serde(rename_all = "lowercase")]
pub enum FpmTraceMode {
    #[default]
    Sampled,
    Full,
}

#[derive(Clone, Debug)]
pub struct FpmTracePolicy {
    pub enabled: bool,
    pub output_path: String,
    pub mode: FpmTraceMode,
    pub sample_interval_ms: u64,
    pub jsonl_gz_roll_bytes: u64,
    pub max_segments: usize,
}

static POLICY: OnceLock<FpmTracePolicy> = OnceLock::new();

impl Default for FpmTracePolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            output_path: DEFAULT_OUTPUT_PATH.to_string(),
            mode: FpmTraceMode::Sampled,
            sample_interval_ms: DEFAULT_SAMPLE_INTERVAL_MS,
            jsonl_gz_roll_bytes: DEFAULT_JSONL_GZ_ROLL_BYTES,
            max_segments: DEFAULT_MAX_SEGMENTS,
        }
    }
}

fn parse_bool(value: &str) -> anyhow::Result<bool> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Ok(true),
        "0" | "false" | "off" | "no" => Ok(false),
        _ => anyhow::bail!(
            "{} must be one of true/false, 1/0, on/off, or yes/no",
            env_fpm_trace::DYN_FPM_TRACE
        ),
    }
}

fn positive_u64_from_env(name: &str, default: u64) -> anyhow::Result<u64> {
    let Some(value) = std::env::var(name).ok() else {
        return Ok(default);
    };
    let parsed = value
        .trim()
        .parse::<u64>()
        .map_err(|_| anyhow::anyhow!("{name} must be a positive integer"))?;
    if parsed == 0 {
        anyhow::bail!("{name} must be greater than zero");
    }
    Ok(parsed)
}

fn positive_usize_from_env(name: &str, default: usize) -> anyhow::Result<usize> {
    let Some(value) = std::env::var(name).ok() else {
        return Ok(default);
    };
    let parsed = value
        .trim()
        .parse::<usize>()
        .map_err(|_| anyhow::anyhow!("{name} must be a positive integer"))?;
    if parsed == 0 {
        anyhow::bail!("{name} must be greater than zero");
    }
    Ok(parsed)
}

fn load_enabled_policy() -> anyhow::Result<FpmTracePolicy> {
    let output_path = match std::env::var(env_fpm_trace::DYN_FPM_OUTPUT_PATH) {
        Ok(value) if value.trim().is_empty() => {
            anyhow::bail!("{} must not be empty", env_fpm_trace::DYN_FPM_OUTPUT_PATH)
        }
        Ok(value) => value.trim().to_string(),
        Err(_) => DEFAULT_OUTPUT_PATH.to_string(),
    };

    let mode = match std::env::var(env_fpm_trace::DYN_FPM_MODE) {
        Ok(value) if value.trim().eq_ignore_ascii_case("full") => FpmTraceMode::Full,
        Ok(value) if value.trim().eq_ignore_ascii_case("sampled") => FpmTraceMode::Sampled,
        Ok(_) => anyhow::bail!("{} must be sampled or full", env_fpm_trace::DYN_FPM_MODE),
        Err(_) => FpmTraceMode::Sampled,
    };

    Ok(FpmTracePolicy {
        enabled: true,
        output_path,
        mode,
        sample_interval_ms: positive_u64_from_env(
            env_fpm_trace::DYN_FPM_SAMPLE_INTERVAL_MS,
            DEFAULT_SAMPLE_INTERVAL_MS,
        )?,
        jsonl_gz_roll_bytes: positive_u64_from_env(
            env_fpm_trace::DYN_FPM_JSONL_GZ_ROLL_BYTES,
            DEFAULT_JSONL_GZ_ROLL_BYTES,
        )?,
        max_segments: positive_usize_from_env(
            env_fpm_trace::DYN_FPM_MAX_SEGMENTS,
            DEFAULT_MAX_SEGMENTS,
        )?,
    })
}

fn load_from_env() -> FpmTracePolicy {
    let enabled = match std::env::var(env_fpm_trace::DYN_FPM_TRACE) {
        Err(_) => return FpmTracePolicy::default(),
        Ok(value) => match parse_bool(&value) {
            Ok(enabled) => enabled,
            Err(error) => {
                tracing::warn!(%error, "invalid FPM trace configuration; tracing disabled");
                return FpmTracePolicy::default();
            }
        },
    };
    if !enabled {
        return FpmTracePolicy::default();
    }

    load_enabled_policy().unwrap_or_else(|error| {
        tracing::warn!(%error, "invalid FPM trace configuration; tracing disabled");
        FpmTracePolicy::default()
    })
}

pub fn policy() -> &'static FpmTracePolicy {
    POLICY.get_or_init(load_from_env)
}

pub fn is_enabled() -> bool {
    policy().enabled
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[serial_test::serial]
    fn disabled_defaults_are_bounded_and_sampled() {
        temp_env::with_vars(
            [
                (env_fpm_trace::DYN_FPM_TRACE, None::<&str>),
                (env_fpm_trace::DYN_FPM_OUTPUT_PATH, None),
                (env_fpm_trace::DYN_FPM_MODE, None),
                (env_fpm_trace::DYN_FPM_SAMPLE_INTERVAL_MS, None),
                (env_fpm_trace::DYN_FPM_JSONL_GZ_ROLL_BYTES, None),
                (env_fpm_trace::DYN_FPM_MAX_SEGMENTS, None),
            ],
            || {
                let policy = load_from_env();
                assert!(!policy.enabled);
                assert_eq!(policy.output_path, DEFAULT_OUTPUT_PATH);
                assert_eq!(policy.mode, FpmTraceMode::Sampled);
                assert_eq!(policy.sample_interval_ms, DEFAULT_SAMPLE_INTERVAL_MS);
                assert_eq!(policy.jsonl_gz_roll_bytes, DEFAULT_JSONL_GZ_ROLL_BYTES);
                assert_eq!(policy.max_segments, DEFAULT_MAX_SEGMENTS);
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn accepts_full_mode_and_numeric_overrides() {
        temp_env::with_vars(
            [
                (env_fpm_trace::DYN_FPM_TRACE, Some("yes")),
                (env_fpm_trace::DYN_FPM_OUTPUT_PATH, Some(" /var/log/fpm ")),
                (env_fpm_trace::DYN_FPM_MODE, Some(" FULL ")),
                (env_fpm_trace::DYN_FPM_SAMPLE_INTERVAL_MS, Some("250")),
                (env_fpm_trace::DYN_FPM_JSONL_GZ_ROLL_BYTES, Some("4096")),
                (env_fpm_trace::DYN_FPM_MAX_SEGMENTS, Some("7")),
            ],
            || {
                let policy = load_from_env();
                assert!(policy.enabled);
                assert_eq!(policy.output_path, "/var/log/fpm");
                assert_eq!(policy.mode, FpmTraceMode::Full);
                assert_eq!(policy.sample_interval_ms, 250);
                assert_eq!(policy.jsonl_gz_roll_bytes, 4096);
                assert_eq!(policy.max_segments, 7);
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn invalid_enabled_configuration_disables_trace_as_a_unit() {
        temp_env::with_vars(
            [
                (env_fpm_trace::DYN_FPM_TRACE, Some("true")),
                (env_fpm_trace::DYN_FPM_MODE, Some("sometimes")),
                (env_fpm_trace::DYN_FPM_SAMPLE_INTERVAL_MS, Some("0")),
                (env_fpm_trace::DYN_FPM_JSONL_GZ_ROLL_BYTES, Some("bad")),
                (env_fpm_trace::DYN_FPM_MAX_SEGMENTS, Some("0")),
            ],
            || {
                let policy = load_from_env();
                assert!(!policy.enabled);
                assert_eq!(policy.mode, FpmTraceMode::Sampled);
                assert_eq!(policy.sample_interval_ms, DEFAULT_SAMPLE_INTERVAL_MS);
                assert_eq!(policy.jsonl_gz_roll_bytes, DEFAULT_JSONL_GZ_ROLL_BYTES);
                assert_eq!(policy.max_segments, DEFAULT_MAX_SEGMENTS);
            },
        );
    }

    #[test]
    #[serial_test::serial]
    fn each_invalid_numeric_or_path_override_disables_trace() {
        for (invalid_name, invalid_value) in [
            (env_fpm_trace::DYN_FPM_SAMPLE_INTERVAL_MS, "0"),
            (env_fpm_trace::DYN_FPM_SAMPLE_INTERVAL_MS, "not-a-number"),
            (env_fpm_trace::DYN_FPM_JSONL_GZ_ROLL_BYTES, "0"),
            (env_fpm_trace::DYN_FPM_JSONL_GZ_ROLL_BYTES, "not-a-number"),
            (env_fpm_trace::DYN_FPM_MAX_SEGMENTS, "0"),
            (env_fpm_trace::DYN_FPM_MAX_SEGMENTS, "not-a-number"),
            (env_fpm_trace::DYN_FPM_OUTPUT_PATH, "   "),
        ] {
            let mut vars = vec![
                (env_fpm_trace::DYN_FPM_TRACE, Some("true")),
                (env_fpm_trace::DYN_FPM_OUTPUT_PATH, None),
                (env_fpm_trace::DYN_FPM_MODE, None),
                (env_fpm_trace::DYN_FPM_SAMPLE_INTERVAL_MS, None),
                (env_fpm_trace::DYN_FPM_JSONL_GZ_ROLL_BYTES, None),
                (env_fpm_trace::DYN_FPM_MAX_SEGMENTS, None),
            ];
            let (_, value) = vars
                .iter_mut()
                .find(|(name, _)| *name == invalid_name)
                .unwrap();
            *value = Some(invalid_value);

            temp_env::with_vars(vars, || {
                assert!(
                    !load_from_env().enabled,
                    "invalid setting should disable trace: {invalid_name}={invalid_value}"
                );
            });
        }
    }

    #[test]
    #[serial_test::serial]
    fn invalid_master_switch_disables_trace() {
        temp_env::with_var(env_fpm_trace::DYN_FPM_TRACE, Some("maybe"), || {
            assert!(!load_from_env().enabled);
        });
    }

    #[test]
    #[serial_test::serial]
    fn explicit_false_ignores_other_trace_settings() {
        temp_env::with_vars(
            [
                (env_fpm_trace::DYN_FPM_TRACE, Some("off")),
                (env_fpm_trace::DYN_FPM_MODE, Some("invalid")),
                (env_fpm_trace::DYN_FPM_MAX_SEGMENTS, Some("0")),
            ],
            || assert!(!load_from_env().enabled),
        );
    }

    #[test]
    fn master_switch_parser_accepts_only_documented_boolean_forms() {
        for value in ["true", "TRUE", "1", "on", "ON", "yes"] {
            assert!(parse_bool(value).unwrap(), "value={value}");
        }
        for value in ["false", "FALSE", "0", "off", "OFF", "no"] {
            assert!(!parse_bool(value).unwrap(), "value={value}");
        }
        for value in ["", "enabled", "2"] {
            assert!(parse_bool(value).is_err(), "value={value}");
        }
    }
}
