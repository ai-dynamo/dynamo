use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RequestTemplate {
    pub model: String,
    pub temperature: f32,
    pub max_completion_tokens: u32,
}

impl RequestTemplate {
    pub fn load(path: &Path) -> Result<Self> {
        let template = std::fs::read_to_string(path)?;
        let template: Self = serde_json::from_str(&template)?;
        Ok(template)
    }

}
