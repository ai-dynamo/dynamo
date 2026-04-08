// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generic YAML cartesian-product sweep runner.

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;
use serde_yaml::Value;

/// A sweep runner that holds a list of parameter combinations.
///
/// Supports two YAML formats:
///
/// **Explicit mode** — list of configs under a `configs:` key:
/// ```yaml
/// configs:
///   - threads: 4
///     clients: 8
/// ```
///
/// **Cartesian product mode** — arrays at the top level generate all combinations:
/// ```yaml
/// threads: [1, 2, 4, 8]
/// clients: [1, 4, 16]
/// ```
pub struct SweepRunner<P> {
    pub points: Vec<P>,
}

impl<P: DeserializeOwned + Clone> SweepRunner<P> {
    /// Load sweep points from a YAML file.
    pub fn from_yaml_file(path: &str) -> Result<Self> {
        let contents = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read sweep file: {path}"))?;
        Self::from_yaml_str(&contents)
    }

    /// Load sweep points from a YAML string.
    pub fn from_yaml_str(s: &str) -> Result<Self> {
        let value: Value = serde_yaml::from_str(s).context("Failed to parse sweep YAML")?;

        let points = match &value {
            Value::Mapping(map) => {
                // Check for explicit `configs:` key
                if let Some(configs_val) = map.get("configs") {
                    // Explicit mode
                    let configs: Vec<P> = serde_yaml::from_value(configs_val.clone())
                        .context("Failed to deserialize explicit configs")?;
                    configs
                } else {
                    // Cartesian product mode
                    cartesian_product_from_mapping(map)?
                }
            }
            _ => anyhow::bail!("Sweep YAML must be a mapping at the top level"),
        };

        Ok(Self { points })
    }

    /// Number of sweep points.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns true if there are no sweep points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Generate cartesian product from a YAML mapping where each value may be
/// a scalar (single point) or a sequence (multiple values).
fn cartesian_product_from_mapping<P: DeserializeOwned>(
    map: &serde_yaml::Mapping,
) -> Result<Vec<P>> {
    // Collect (key, values) pairs
    let mut keys: Vec<Value> = Vec::new();
    let mut value_lists: Vec<Vec<Value>> = Vec::new();

    for (k, v) in map.iter() {
        keys.push(k.clone());
        match v {
            Value::Sequence(seq) => value_lists.push(seq.clone()),
            scalar => value_lists.push(vec![scalar.clone()]),
        }
    }

    if keys.is_empty() {
        return Ok(Vec::new());
    }

    // Generate cartesian product of all value lists
    let combinations = cartesian_product(&value_lists);

    let mut points = Vec::with_capacity(combinations.len());
    for combo in combinations {
        let mut mapping = serde_yaml::Mapping::new();
        for (k, v) in keys.iter().zip(combo.iter()) {
            mapping.insert(k.clone(), v.clone());
        }
        let p: P = serde_yaml::from_value(Value::Mapping(mapping))
            .context("Failed to deserialize sweep point")?;
        points.push(p);
    }

    Ok(points)
}

/// Compute cartesian product of a slice of value lists.
fn cartesian_product(lists: &[Vec<Value>]) -> Vec<Vec<Value>> {
    if lists.is_empty() {
        return vec![vec![]];
    }

    let mut result = vec![vec![]];
    for list in lists {
        let mut new_result = Vec::new();
        for existing in &result {
            for item in list {
                let mut combo = existing.clone();
                combo.push(item.clone());
                new_result.push(combo);
            }
        }
        result = new_result;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Clone, Deserialize, PartialEq)]
    struct TestParams {
        threads: usize,
        clients: usize,
    }

    #[test]
    fn test_explicit_mode() {
        let yaml = r#"
configs:
  - threads: 4
    clients: 8
  - threads: 2
    clients: 16
"#;
        let runner = SweepRunner::<TestParams>::from_yaml_str(yaml).unwrap();
        assert_eq!(runner.len(), 2);
        assert_eq!(
            runner.points[0],
            TestParams {
                threads: 4,
                clients: 8
            }
        );
        assert_eq!(
            runner.points[1],
            TestParams {
                threads: 2,
                clients: 16
            }
        );
    }

    #[test]
    fn test_cartesian_mode() {
        let yaml = r#"
threads: [1, 2]
clients: [4, 8]
"#;
        let runner = SweepRunner::<TestParams>::from_yaml_str(yaml).unwrap();
        // 2 * 2 = 4 combinations
        assert_eq!(runner.len(), 4);
    }

    #[test]
    fn test_single_values() {
        let yaml = r#"
threads: 4
clients: 8
"#;
        let runner = SweepRunner::<TestParams>::from_yaml_str(yaml).unwrap();
        assert_eq!(runner.len(), 1);
        assert_eq!(
            runner.points[0],
            TestParams {
                threads: 4,
                clients: 8
            }
        );
    }
}
