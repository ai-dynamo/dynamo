// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::Value;

use super::{SystemOneError, SystemOneQuestion};

#[derive(Clone, Debug, PartialEq)]
pub struct QuestionPrompt {
    pub content: String,
    pub labels: Vec<String>,
}

/// Render SGLang prompt format version 1 before the model chat template is applied.
pub fn render_question_prompt(
    state: &Value,
    question: &SystemOneQuestion,
) -> Result<QuestionPrompt, SystemOneError> {
    let state = render_text(state);
    let (lines, labels) = match question {
        SystemOneQuestion::Choice {
            instructions,
            criteria,
        } => {
            let labels = choice_labels(criteria.len());
            let mut lines = question_line(instructions.as_ref());
            for ((name, description), label) in criteria.iter().zip(&labels) {
                let description = render_text(description);
                lines.push(if description.is_empty() {
                    format!("{label}: {name}")
                } else {
                    format!("{label}: {name} - {description}")
                });
            }
            lines.push("Answer with the letter of one option only.".to_string());
            (lines, labels)
        }
        SystemOneQuestion::Score {
            instructions,
            criteria,
        } => {
            let labels = (0..criteria.len())
                .map(|level| level.to_string())
                .collect::<Vec<_>>();
            let mut lines = question_line(instructions.as_ref());
            lines.extend(
                labels
                    .iter()
                    .zip(criteria)
                    .map(|(label, level)| format!("{label}: {}", render_text(level))),
            );
            lines.push("Answer with the number of one level only.".to_string());
            (lines, labels)
        }
        SystemOneQuestion::Noul {
            instructions,
            criteria,
        } => {
            let question = instructions
                .as_ref()
                .filter(|value| !is_blank(value))
                .map(render_text);
            let mut lines = vec![match question {
                Some(question) => format!("Is the following true? {question}"),
                None => "Is the following true?".to_string(),
            }];
            if let Some(criteria) = criteria {
                for (label, description) in [
                    ("yes", criteria.r#true.as_ref()),
                    ("no", criteria.r#false.as_ref()),
                ] {
                    if let Some(description) = description.filter(|value| !is_blank(value)) {
                        lines.push(format!("{label}: {}", render_text(description)));
                    }
                }
            }
            lines.push("Answer with yes or no only.".to_string());
            (lines, vec!["yes".to_string(), "no".to_string()])
        }
    };

    Ok(QuestionPrompt {
        content: std::iter::once(state)
            .chain(std::iter::once(String::new()))
            .chain(lines)
            .collect::<Vec<_>>()
            .join("\n"),
        labels,
    })
}

fn question_line(instructions: Option<&Value>) -> Vec<String> {
    instructions
        .filter(|value| !is_blank(value))
        .map(|value| vec![format!("Question: {}", render_text(value))])
        .unwrap_or_default()
}

fn choice_labels(count: usize) -> Vec<String> {
    (0..count)
        .map(|index| {
            if count <= 26 {
                char::from(b'A' + index as u8).to_string()
            } else {
                format!(
                    "{}{}",
                    char::from(b'A' + (index / 26) as u8),
                    char::from(b'A' + (index % 26) as u8)
                )
            }
        })
        .collect()
}

fn is_blank(value: &Value) -> bool {
    matches!(value, Value::Null) || value.as_str().is_some_and(|value| value.trim().is_empty())
}

fn render_text(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(value) => value.clone(),
        value => serde_json::to_string(value).expect("serde_json::Value must serialize"),
    }
}
