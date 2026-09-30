// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

const MAX_QUESTIONS: usize = 128;
const MAX_CHOICES: usize = 255;
const MAX_SCORE_LEVELS: usize = 10;
const MAX_NAME_CHARS: usize = 128;

#[derive(Debug, thiserror::Error, PartialEq)]
pub enum SystemOneError {
    #[error("{0}")]
    Validation(String),
    #[error("backend candidate scores are invalid: {0}")]
    CandidateScores(String),
}

#[derive(Clone, Debug, Deserialize)]
pub struct SystemOneRequest {
    pub model: String,
    pub state: Value,
    pub questions: IndexMap<String, SystemOneQuestion>,
    #[serde(default)]
    pub chat_template_kwargs: Option<Map<String, Value>>,
    #[serde(default)]
    pub temperature: Option<Value>,
    #[serde(default)]
    pub prompt_format_version: Option<Value>,
    #[serde(default)]
    pub return_prompt_token_ids: Option<Value>,
}

impl SystemOneRequest {
    pub fn validate(&self) -> Result<(), SystemOneError> {
        validate_state(&self.state)?;
        if self.questions.is_empty() || self.questions.len() > MAX_QUESTIONS {
            return Err(validation(format!(
                "questions must contain between 1 and {MAX_QUESTIONS} entries"
            )));
        }
        for field in [
            ("temperature", &self.temperature),
            ("prompt_format_version", &self.prompt_format_version),
            ("return_prompt_token_ids", &self.return_prompt_token_ids),
        ] {
            if field.1.is_some() {
                return Err(validation(format!(
                    "{} is not supported by /v1/systemone",
                    field.0
                )));
            }
        }
        if self.model.trim().is_empty() {
            return Err(validation("model must not be blank"));
        }
        if self.model.contains(':') {
            return Err(validation("model must not name a LoRA adapter"));
        }
        if let Some(kwargs) = self.chat_template_kwargs.as_ref() {
            for (name, value) in kwargs {
                let normalized = name.to_ascii_lowercase();
                if (normalized.contains("think") || normalized.contains("reason"))
                    && !matches!(value, Value::Bool(false))
                    && !value.as_str().is_some_and(|value| {
                        matches!(value.to_ascii_lowercase().as_str(), "disabled" | "none")
                    })
                {
                    return Err(validation(format!(
                        "chat_template_kwargs.{name} must disable reasoning for /v1/systemone"
                    )));
                }
            }
        }
        for (id, question) in &self.questions {
            question.validate(id)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum SystemOneQuestion {
    Noul {
        #[serde(default)]
        instructions: Option<Value>,
        #[serde(default)]
        criteria: Option<NoulCriteria>,
    },
    Choice {
        #[serde(default)]
        instructions: Option<Value>,
        criteria: IndexMap<String, Value>,
    },
    Score {
        #[serde(default)]
        instructions: Option<Value>,
        criteria: Vec<Value>,
    },
}

impl SystemOneQuestion {
    pub fn validate(&self, id: &str) -> Result<(), SystemOneError> {
        match self {
            Self::Noul {
                instructions,
                criteria,
            } => {
                validate_optional_description(instructions.as_ref(), id, "instructions")?;
                let has_instructions = instructions.as_ref().is_some_and(is_present_description);
                let has_criteria = criteria.as_ref().is_some_and(NoulCriteria::has_description);
                if !has_instructions && !has_criteria {
                    return Err(validation(format!(
                        "questions.{id}: noul requires instructions or true/false criteria"
                    )));
                }
                if let Some(criteria) = criteria {
                    criteria.validate(id)?;
                }
            }
            Self::Choice {
                instructions,
                criteria,
            } => {
                validate_optional_description(instructions.as_ref(), id, "instructions")?;
                if criteria.is_empty() || criteria.len() > MAX_CHOICES {
                    return Err(validation(format!(
                        "questions.{id}.criteria must contain between 1 and {MAX_CHOICES} options"
                    )));
                }
                let mut seen = HashSet::with_capacity(criteria.len());
                for (name, description) in criteria {
                    validate_name(name, id)?;
                    let canonical = name.trim().to_lowercase();
                    if !seen.insert(canonical) {
                        return Err(validation(format!(
                            "questions.{id}.criteria option {name:?} repeats another option"
                        )));
                    }
                    validate_description(description, id, "criteria description")?;
                }
            }
            Self::Score {
                instructions,
                criteria,
            } => {
                validate_optional_description(instructions.as_ref(), id, "instructions")?;
                if criteria.is_empty() || criteria.len() > MAX_SCORE_LEVELS {
                    return Err(validation(format!(
                        "questions.{id}.criteria must contain between 1 and {MAX_SCORE_LEVELS} levels"
                    )));
                }
                for level in criteria {
                    validate_description(level, id, "score level")?;
                    if !is_present_description(level) {
                        return Err(validation(format!(
                            "questions.{id}.criteria contains a blank or null level"
                        )));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn candidate_count(&self) -> usize {
        match self {
            Self::Noul { .. } => 2,
            Self::Choice { criteria, .. } => criteria.len(),
            Self::Score { criteria, .. } => criteria.len(),
        }
    }

    pub fn choice_names(&self) -> Option<impl Iterator<Item = &str>> {
        match self {
            Self::Choice { criteria, .. } => Some(criteria.keys().map(String::as_str)),
            _ => None,
        }
    }

    pub fn score_levels(&self) -> Option<&[Value]> {
        match self {
            Self::Score { criteria, .. } => Some(criteria),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct NoulCriteria {
    #[serde(default, rename = "true")]
    pub(crate) r#true: Option<Value>,
    #[serde(default, rename = "false")]
    pub(crate) r#false: Option<Value>,
}

impl NoulCriteria {
    fn has_description(&self) -> bool {
        self.r#true.as_ref().is_some_and(is_present_description)
            || self.r#false.as_ref().is_some_and(is_present_description)
    }

    fn validate(&self, id: &str) -> Result<(), SystemOneError> {
        validate_optional_description(self.r#true.as_ref(), id, "criteria.true")?;
        validate_optional_description(self.r#false.as_ref(), id, "criteria.false")
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct SystemOneResponse {
    pub model: String,
    pub answers: IndexMap<String, SystemOneAnswer>,
    pub usage: SystemOneUsage,
}

#[derive(Clone, Debug, Serialize)]
pub struct SystemOneUsage {
    pub input_tokens: usize,
    pub output_tokens: usize,
}

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum SystemOneAnswer {
    Noul(NoulAnswer),
    Choice(ChoiceAnswer),
    Score(ScoreAnswer),
}

#[derive(Clone, Debug, Serialize)]
pub struct NoulAnswer {
    pub noul: f64,
    pub x_label_mass: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ChoiceAnswer {
    pub choice: String,
    pub probabilities: IndexMap<String, f64>,
    pub confidence: f64,
    pub x_label_mass: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct ScoreAnswer {
    pub score: f64,
    pub probabilities: IndexMap<String, f64>,
    pub confidence: f64,
    pub legend: IndexMap<String, Value>,
    pub x_label_mass: f64,
}

fn validate_state(state: &Value) -> Result<(), SystemOneError> {
    match state {
        Value::Null | Value::String(_) | Value::Array(_) | Value::Object(_) => Ok(()),
        _ => Err(validation("state must be a string, object, array, or null")),
    }
}

fn validate_name(name: &str, id: &str) -> Result<(), SystemOneError> {
    if name.trim().is_empty() || name.chars().count() > MAX_NAME_CHARS {
        return Err(validation(format!(
            "questions.{id}.criteria option names must be non-blank and at most {MAX_NAME_CHARS} characters"
        )));
    }
    if name
        .chars()
        .any(|character| character.is_control() || matches!(character, '\u{2028}' | '\u{2029}'))
    {
        return Err(validation(format!(
            "questions.{id}.criteria option {name:?} contains a control or line break character"
        )));
    }
    Ok(())
}

fn validate_optional_description(
    value: Option<&Value>,
    id: &str,
    field: &str,
) -> Result<(), SystemOneError> {
    if let Some(value) = value {
        validate_description(value, id, field)?;
    }
    Ok(())
}

fn validate_description(value: &Value, id: &str, field: &str) -> Result<(), SystemOneError> {
    match value {
        Value::Null | Value::String(_) | Value::Array(_) | Value::Object(_) => Ok(()),
        _ => Err(validation(format!(
            "questions.{id}.{field} must be a string, object, array, or null"
        ))),
    }
}

fn is_present_description(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::String(value) => !value.trim().is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
        _ => false,
    }
}

fn validation(message: impl Into<String>) -> SystemOneError {
    SystemOneError::Validation(message.into())
}
