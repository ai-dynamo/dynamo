// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mock classifier for testing semantic routing architecture
//!
//! This classifier uses simple keyword heuristics to simulate classification.
//! It's useful for:
//! - Testing the semantic router architecture without ONNX dependencies
//! - Development and debugging
//! - Quick demos

use super::Classifier;
use anyhow::Result;
use std::collections::HashMap;

/// Mock classifier using keyword-based heuristics
///
/// This is a simple classifier for testing that categorizes text based on keywords:
/// - "reasoning": Keywords like prove, calculate, derive, theorem
/// - "code": Keywords like function, class, implement, algorithm
/// - "math": Keywords like equation, integral, derivative, polynomial
/// - "pii": Keywords like SSN, credit card, email, phone
///
/// For binary mode, it focuses on reasoning vs non-reasoning.
pub struct MockClassifier {
    /// Whether to use binary (reasoning/non-reasoning) or multi-class mode
    binary_mode: bool,
}

impl MockClassifier {
    /// Create a new mock classifier
    ///
    /// # Arguments
    /// - `binary_mode`: If true, returns only "reasoning" and "non-reasoning" labels
    pub fn new(binary_mode: bool) -> Self {
        tracing::info!(
            "Initialized MockClassifier (binary_mode={})",
            binary_mode
        );
        Self { binary_mode }
    }
}

impl Classifier for MockClassifier {
    fn classify(&self, text: &str) -> Result<HashMap<String, f32>> {
        let text_lower = text.to_lowercase();

        if self.binary_mode {
            // Binary classification: reasoning vs non-reasoning
            let reasoning_score = if text_lower.contains("prove")
                || text_lower.contains("calculate")
                || text_lower.contains("derive")
                || text_lower.contains("theorem")
                || text_lower.contains("solve")
                || text_lower.contains("explain why")
                || text_lower.contains("step by step")
            {
                0.85
            } else if text_lower.contains("what is")
                || text_lower.contains("define")
                || text_lower.contains("capital")
                || text_lower.contains("hello")
            {
                0.15
            } else {
                0.5 // Neutral
            };

            Ok([
                ("reasoning".to_string(), reasoning_score),
                ("non-reasoning".to_string(), 1.0 - reasoning_score),
            ]
            .into())
        } else {
            // Multi-class classification
            let mut scores: HashMap<String, f32> = [
                ("reasoning".to_string(), 0.1),
                ("code".to_string(), 0.1),
                ("math".to_string(), 0.1),
                ("pii".to_string(), 0.1),
                ("summarize".to_string(), 0.1),
                ("qa".to_string(), 0.5), // Default
            ]
            .into();

            // Adjust scores based on keywords
            if text_lower.contains("function")
                || text_lower.contains("class")
                || text_lower.contains("implement")
                || text_lower.contains("algorithm")
                || text_lower.contains("python")
                || text_lower.contains("rust")
            {
                *scores.get_mut("code").unwrap() = 0.7;
                *scores.get_mut("qa").unwrap() = 0.1;
            }

            if text_lower.contains("equation")
                || text_lower.contains("integral")
                || text_lower.contains("derivative")
                || text_lower.contains("polynomial")
                || text_lower.contains("matrix")
            {
                *scores.get_mut("math").unwrap() = 0.7;
                *scores.get_mut("reasoning").unwrap() = 0.3;
                *scores.get_mut("qa").unwrap() = 0.1;
            }

            if text_lower.contains("ssn")
                || text_lower.contains("social security")
                || text_lower.contains("credit card")
                || text_lower.contains("password")
            {
                *scores.get_mut("pii").unwrap() = 0.9;
                *scores.get_mut("qa").unwrap() = 0.05;
            }

            if text_lower.contains("prove")
                || text_lower.contains("why")
                || text_lower.contains("explain")
                || text_lower.contains("reasoning")
            {
                *scores.get_mut("reasoning").unwrap() = 0.7;
                *scores.get_mut("qa").unwrap() = 0.1;
            }

            if text_lower.contains("summarize")
                || text_lower.contains("tldr")
                || text_lower.contains("brief")
            {
                *scores.get_mut("summarize").unwrap() = 0.7;
                *scores.get_mut("qa").unwrap() = 0.1;
            }

            // Normalize scores to sum to 1.0
            let sum: f32 = scores.values().sum();
            for score in scores.values_mut() {
                *score /= sum;
            }

            Ok(scores)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_reasoning() {
        let classifier = MockClassifier::new(true);
        let result = classifier.classify("Prove that the square root of 2 is irrational").unwrap();

        assert!(result["reasoning"] > 0.7);
        assert!(result["non-reasoning"] < 0.3);
    }

    #[test]
    fn test_binary_non_reasoning() {
        let classifier = MockClassifier::new(true);
        let result = classifier.classify("What is the capital of France?").unwrap();

        assert!(result["reasoning"] < 0.3);
        assert!(result["non-reasoning"] > 0.7);
    }

    #[test]
    fn test_multiclass_code() {
        let classifier = MockClassifier::new(false);
        let result = classifier.classify("Write a Python function to sort a list").unwrap();

        assert!(result["code"] > 0.5);
    }

    #[test]
    fn test_multiclass_pii() {
        let classifier = MockClassifier::new(false);
        let result = classifier.classify("My SSN is 123-45-6789").unwrap();

        assert!(result["pii"] > 0.8);
    }
}

