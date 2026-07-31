// SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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
        let template: Self = serde_json::from_str(&template)
            .inspect_err(|err| crate::log_json_err(&path.display().to_string(), &template, err))?;
        Ok(template)
    }
}

/// Resolve a request's `model` field against an optional [`RequestTemplate`].
///
/// Returns `raw` when it is non-empty; otherwise falls back to
/// `template.model`. If `raw` is empty and no template is configured, returns
/// the (empty) `raw` so the caller's downstream logic still observes the
/// missing-model condition.
///
/// HTTP handler wrappers use this to compute the effective model for
/// cancellation metric labels, keeping the wrapper and the inner handler's
/// own template merge consistent.
pub fn resolve_request_model<'a>(raw: &'a str, template: Option<&'a RequestTemplate>) -> &'a str {
    if raw.is_empty() {
        template.map(|t| t.model.as_str()).unwrap_or(raw)
    } else {
        raw
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn template(model: &str) -> RequestTemplate {
        RequestTemplate {
            model: model.to_string(),
            temperature: 0.0,
            max_completion_tokens: 0,
        }
    }

    #[test]
    fn non_empty_raw_wins() {
        let t = template("template-default");
        // Non-empty raw is returned regardless of whether a template is set.
        assert_eq!(
            resolve_request_model("user-supplied", Some(&t)),
            "user-supplied"
        );
        assert_eq!(
            resolve_request_model("user-supplied", None),
            "user-supplied"
        );
    }

    #[test]
    fn empty_raw_falls_back() {
        let t = template("template-default");
        // With a template, empty raw resolves to the template's model.
        assert_eq!(resolve_request_model("", Some(&t)), "template-default");
        // Without a template, empty raw stays empty so callers can detect
        // the missing-model condition downstream.
        assert_eq!(resolve_request_model("", None), "");
    }
}
