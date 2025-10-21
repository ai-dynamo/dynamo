use serde::Deserialize;
use std::{collections::HashMap, fs, path::Path};

#[derive(Debug, Deserialize, Clone)]
pub struct Rule {
    pub when_any: Vec<RuleCond>,
    pub route_onprem_model: String,
    pub rationale: String,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RuleCond {
    pub label: String,
    #[serde(default = "default_min_conf")]
    pub min_conf: f32,
}

fn default_min_conf() -> f32 {
    0.5
}

#[derive(Debug, Deserialize, Clone)]
pub struct PolicyConfig {
    #[serde(default = "default_abstain_model")]
    pub abstain_onprem_model: String,
    #[serde(default = "default_min_conf")]
    pub threshold_min_conf: f32,
    #[serde(default)]
    pub weights: HashMap<String, i32>,
    #[serde(default)]
    pub rules: Vec<Rule>,
}

fn default_abstain_model() -> String {
    "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string()
}

impl PolicyConfig {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let data = fs::read_to_string(path)?;
        let cfg = serde_yaml::from_str::<PolicyConfig>(&data)?;
        Ok(cfg)
    }
}

