// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::Classifier;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

#[cfg(feature = "onnx-classifier")]
use ndarray_ort::Array2;
#[cfg(feature = "onnx-classifier")]
use ort::{
    Allocator, Environment, GraphOptimizationLevel, Session, Tensor, Value,
};
#[cfg(feature = "onnx-classifier")]
use std::{path::Path, sync::Arc};
#[cfg(feature = "onnx-classifier")]
use tokenizers::Tokenizer;

/// Optimized for CodeIsAbstract/ReasoningTextClassifier (0=non-reasoning, 1=reasoning)
#[cfg(feature = "onnx-classifier")]
pub struct OnnxClassifier {
    tokenizer: Tokenizer,
    session: Session,
    allocator: Allocator,
    max_len: usize,
    labels: Vec<String>,
}

#[cfg(feature = "onnx-classifier")]
impl OnnxClassifier {
    pub fn new(onnx_path: &str, tokenizer_path: &str, max_len: usize) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(tokenizer_path).map_err(|e| anyhow!("tokenizer load: {e}"))?;

        let env = Arc::new(Environment::builder().with_name("semrouter").build()?);

        let session = env
            .new_session_builder()?
            .with_optimization_level(GraphOptimizationLevel::All)?
            .with_intra_op_num_threads(2)?
            .with_model_from_file(Path::new(onnx_path))?;

        let allocator = Allocator::default();

        tracing::info!(
            "Initialized ONNX classifier with {} labels, max_len={}",
            2,
            max_len
        );

        Ok(Self {
            tokenizer,
            session,
            allocator,
            max_len,
            labels: vec!["non-reasoning".into(), "reasoning".into()],
        })
    }
}

#[cfg(feature = "onnx-classifier")]
impl Classifier for OnnxClassifier {
    fn classify(&self, text: &str) -> Result<HashMap<String, f32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("encode: {e}"))?;

        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let mut mask: Vec<i64> = enc.get_attention_mask().iter().map(|&x| x as i64).collect();

        if ids.len() > self.max_len {
            ids.truncate(self.max_len);
            mask.truncate(self.max_len);
        }
        if ids.len() < self.max_len {
            let pad = self.max_len - ids.len();
            ids.extend(std::iter::repeat(0).take(pad));
            mask.extend(std::iter::repeat(0).take(pad));
        }

        let ids_arr = Array2::from_shape_vec((1, self.max_len), ids)?;
        let mask_arr = Array2::from_shape_vec((1, self.max_len), mask)?;

        // Build ONNX inputs from ndarray
        let ids_val = Value::from_array(&self.allocator, &ids_arr)?;
        let mask_val = Value::from_array(&self.allocator, &mask_arr)?;

        // Run (order must match model inputs)
        let outputs = self.session.run(vec![ids_val, mask_val])?;

        // Extract logits (float32)
        let logits: Tensor<f32> = outputs
            .get(0)
            .ok_or_else(|| anyhow!("no logits output"))?
            .try_extract()?;

        let view = logits.view();
        let shape = view.shape();
        if shape.len() != 2 || shape[0] != 1 {
            return Err(anyhow!("unexpected logits shape: {:?}", shape));
        }
        let c = shape[1];
        if c != self.labels.len() {
            return Err(anyhow!(
                "label count mismatch: logits dim={} labels={}",
                c,
                self.labels.len()
            ));
        }

        // Stable softmax
        let mut max_logit = f32::NEG_INFINITY;
        for i in 0..c {
            max_logit = max_logit.max(view[[0, i]]);
        }
        let mut exps = vec![0f32; c];
        let mut sum = 0f32;
        for i in 0..c {
            let e = (view[[0, i]] - max_logit).exp();
            exps[i] = e;
            sum += e;
        }

        let mut m = HashMap::with_capacity(c);
        for i in 0..c {
            m.insert(self.labels[i].clone(), exps[i] / sum);
        }
        Ok(m)
    }
}

#[cfg(not(feature = "onnx-classifier"))]
pub struct OnnxClassifier;

#[cfg(not(feature = "onnx-classifier"))]
impl OnnxClassifier {
    pub fn new(_onnx_path: &str, _tokenizer_path: &str, _max_len: usize) -> Result<Self> {
        Err(anyhow!("OnnxClassifier requires 'onnx-classifier' feature"))
    }
}

#[cfg(not(feature = "onnx-classifier"))]
impl Classifier for OnnxClassifier {
    fn classify(&self, _text: &str) -> Result<HashMap<String, f32>> {
        Err(anyhow!("OnnxClassifier requires 'onnx-classifier' feature"))
    }
}
