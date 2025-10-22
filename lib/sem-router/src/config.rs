// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Configuration for semantic router

use super::{Mode, OverridePolicy, RouteAction};

#[derive(Debug, Clone, serde::Deserialize)]
pub struct ClassRoute {
    /// Label id (classifier output key).
    pub label: String,
    /// Threshold to accept this class's route.
    pub threshold: f32,
    /// What to do when label score >= threshold.
    pub action: RouteAction,
}

/// Which classifier is used and any init params.
#[derive(Debug, Clone, serde::Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ClassifierConfig {
    /// fastText classifier (requires clf-fasttext feature).
    Fasttext {
        /// Path to .bin model file
        model_path: String,
    },
}

#[derive(Debug, Clone, serde::Deserialize)]
pub struct SemRouterConfig {
    #[serde(default = "default_enabled")]
    pub enabled: bool,
    #[serde(default)]
    pub mode: Mode,
    /// Policy when request already has a model set.
    #[serde(default)]
    pub default_policy: OverridePolicy,
    /// If present, treat this model value as an explicit "routing alias".
    /// E.g. user sends `model: "router"` to ask for routing.
    pub model_alias: Option<String>,
    /// Per-class routing actions (N classes supported).
    #[serde(default)]
    pub classes: Vec<ClassRoute>,
    /// Fallback action when no class crosses threshold (default: passthrough).
    #[serde(default)]
    pub fallback: Option<RouteAction>,
    /// Classifier selection.
    pub classifier: ClassifierConfig,
}

fn default_enabled() -> bool { true }

impl Default for SemRouterConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: Mode::Auto,
            default_policy: OverridePolicy::NeverWhenExplicit,
            model_alias: Some("router".to_string()),
            classes: vec![],
            fallback: None,
            classifier: ClassifierConfig::Fasttext {
                model_path: String::new(),
            },
        }
    }
}

impl SemRouterConfig {
    /// Load config from YAML file specified by env var, or use defaults.
    /// Looks for DYN_SEMROUTER_CONFIG env var pointing to a YAML file.
    pub fn load_from_env_and_defaults() -> anyhow::Result<Self> {
        if let Ok(config_path) = std::env::var("DYN_SEMROUTER_CONFIG") {
            tracing::info!("Loading semantic router config from: {}", config_path);
            let yaml_str = std::fs::read_to_string(&config_path)
                .map_err(|e| anyhow::anyhow!("Failed to read config file {}: {}", config_path, e))?;

            // Parse YAML with a top-level "semrouter" key (like example_sem_config_v2.yaml)
            #[derive(serde::Deserialize)]
            struct ConfigWrapper {
                semrouter: SemRouterConfig,
            }

            let wrapper: ConfigWrapper = serde_yaml::from_str(&yaml_str)
                .map_err(|e| anyhow::anyhow!("Failed to parse YAML config: {}", e))?;

            Ok(wrapper.semrouter)
        } else {
            tracing::debug!("DYN_SEMROUTER_CONFIG not set, using default config (disabled)");
            Ok(Self::default())
        }
    }
}

