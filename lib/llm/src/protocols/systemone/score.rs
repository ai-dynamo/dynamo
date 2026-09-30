// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use indexmap::IndexMap;

use super::types::{
    ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneAnswer, SystemOneError, SystemOneQuestion,
};

pub fn answer_from_logprobs(
    question: &SystemOneQuestion,
    logprobs: &[f64],
) -> Result<SystemOneAnswer, SystemOneError> {
    if logprobs.len() != question.candidate_count() {
        return Err(candidate_error(format!(
            "expected {} values, received {}",
            question.candidate_count(),
            logprobs.len()
        )));
    }
    if logprobs.iter().any(|value| !value.is_finite()) {
        return Err(candidate_error("all values must be finite"));
    }

    let probabilities = stable_softmax(logprobs);
    let label_mass = logprobs.iter().map(|value| value.exp()).sum();
    match question {
        SystemOneQuestion::Noul { .. } => Ok(SystemOneAnswer::Noul(NoulAnswer {
            noul: probabilities[0],
            x_label_mass: label_mass,
        })),
        SystemOneQuestion::Choice { .. } => {
            let names: Vec<_> = question.choice_names().expect("choice names").collect();
            let selected = argmax(&probabilities);
            let probability_map = names
                .iter()
                .zip(probabilities.iter().copied())
                .map(|(name, probability)| ((*name).to_string(), probability))
                .collect();
            Ok(SystemOneAnswer::Choice(ChoiceAnswer {
                choice: names[selected].to_string(),
                confidence: choice_confidence(&probabilities),
                probabilities: probability_map,
                x_label_mass: label_mass,
            }))
        }
        SystemOneQuestion::Score { .. } => {
            let levels = question.score_levels().expect("score levels");
            let score = probabilities
                .iter()
                .enumerate()
                .map(|(index, probability)| index as f64 * probability)
                .sum();
            let probability_map = probabilities
                .iter()
                .copied()
                .enumerate()
                .map(|(index, probability)| (index.to_string(), probability))
                .collect();
            let legend: IndexMap<_, _> = levels
                .iter()
                .cloned()
                .enumerate()
                .map(|(index, level)| (index.to_string(), level))
                .collect();
            Ok(SystemOneAnswer::Score(ScoreAnswer {
                score,
                confidence: score_confidence(&probabilities),
                probabilities: probability_map,
                legend,
                x_label_mass: label_mass,
            }))
        }
    }
}

fn stable_softmax(logprobs: &[f64]) -> Vec<f64> {
    let max = logprobs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let weights: Vec<_> = logprobs.iter().map(|value| (value - max).exp()).collect();
    let total: f64 = weights.iter().sum();
    weights.into_iter().map(|value| value / total).collect()
}

fn argmax(values: &[f64]) -> usize {
    values
        .iter()
        .enumerate()
        .reduce(|best, candidate| {
            if candidate.1 > best.1 {
                candidate
            } else {
                best
            }
        })
        .map(|(index, _)| index)
        .expect("validated non-empty probabilities")
}

fn choice_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() == 1 {
        return 1.0;
    }
    let uniform = 1.0 / probabilities.len() as f64;
    (probabilities.iter().copied().fold(0.0, f64::max) - uniform) / (1.0 - uniform)
}

fn score_confidence(probabilities: &[f64]) -> f64 {
    if probabilities.len() == 1 {
        return 1.0;
    }
    let mode = argmax(probabilities);
    let distance_from_mode: f64 = probabilities
        .iter()
        .enumerate()
        .map(|(index, probability)| probability * index.abs_diff(mode) as f64)
        .sum();
    let center = (probabilities.len() - 1) as f64 / 2.0;
    let uniform_mean_absolute_deviation: f64 = (0..probabilities.len())
        .map(|index| (index as f64 - center).abs())
        .sum::<f64>()
        / probabilities.len() as f64;
    (1.0 - distance_from_mode / uniform_mean_absolute_deviation).max(0.0)
}

fn candidate_error(message: impl Into<String>) -> SystemOneError {
    SystemOneError::CandidateScores(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_matches_typesafe_adapter_formulas() {
        assert!((choice_confidence(&[0.5, 0.5]) - 0.0).abs() < f64::EPSILON);
        assert!((choice_confidence(&[1.0, 0.0]) - 1.0).abs() < f64::EPSILON);
        assert!((score_confidence(&[0.0, 1.0, 0.0]) - 1.0).abs() < f64::EPSILON);
        assert!((score_confidence(&[0.5, 0.0, 0.5]) - 0.0).abs() < f64::EPSILON);
    }
}
