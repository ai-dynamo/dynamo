use std::collections::HashMap;

pub mod modernbert;

pub trait MultiClassifier: Send + Sync {
    fn probs(&self, text: &str) -> anyhow::Result<HashMap<String, f32>>;
}

pub struct HeuristicClassifier;

impl MultiClassifier for HeuristicClassifier {
    fn probs(&self, text: &str) -> anyhow::Result<HashMap<String, f32>> {
        let t = text.to_ascii_lowercase();
        let mut m = HashMap::new();
        let token_len = t.split_whitespace().count();
        if t.contains("think step by step")
            || t.contains("explain your reasoning")
            || token_len > 120
        {
            m.insert("reasoning".into(), 0.8);
        } else {
            m.insert("qa".into(), 0.6);
        }
        Ok(m)
    }
}

