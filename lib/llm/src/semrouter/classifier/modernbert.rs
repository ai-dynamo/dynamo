use super::MultiClassifier;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

#[cfg(feature = "onnx-classifier")]
use ndarray::Array2;
#[cfg(feature = "onnx-classifier")]
use ort::{environment::Environment, session::SessionBuilder, value::Value};
#[cfg(feature = "onnx-classifier")]
use std::{path::Path, sync::Arc};
#[cfg(feature = "onnx-classifier")]
use tokenizers::Tokenizer;

#[cfg(feature = "onnx-classifier")]
pub struct ModernBertClassifier {
    tokenizer: Tokenizer,
    session: ort::Session,
    max_len: usize,
    input_ids: String,
    attention_mask: String,
    logits_out: String,
    labels: Vec<String>,
}

#[cfg(feature = "onnx-classifier")]
impl ModernBertClassifier {
    pub fn new(
        onnx: &str,
        tokenizer_json: &str,
        labels: Vec<String>,
        max_len: usize,
    ) -> Result<Self> {
        let tokenizer =
            Tokenizer::from_file(tokenizer_json).map_err(|e| anyhow!("tokenizer load: {e}"))?;
        let env = Arc::new(
            Environment::builder()
                .with_name("semrouter-modernbert")
                .build()?,
        );
        let session = SessionBuilder::new(&env)?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(2)?
            .with_model_from_file(Path::new(onnx))?;

        Ok(Self {
            tokenizer,
            session,
            max_len,
            input_ids: "input_ids".into(),
            attention_mask: "attention_mask".into(),
            logits_out: "logits".into(),
            labels,
        })
    }
}

#[cfg(feature = "onnx-classifier")]
impl MultiClassifier for ModernBertClassifier {
    fn probs(&self, text: &str) -> Result<HashMap<String, f32>> {
        let enc = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("encode: {e}"))?;
        let mut ids: Vec<i64> = enc.get_ids().iter().map(|&x| x as i64).collect();
        let mut mask: Vec<i64> = enc
            .get_attention_mask()
            .iter()
            .map(|&x| x as i64)
            .collect();

        if ids.len() > self.max_len {
            ids.truncate(self.max_len);
            mask.truncate(self.max_len);
        }
        if ids.len() < self.max_len {
            let pad = self.max_len - ids.len();
            ids.extend(std::iter::repeat(0).take(pad));
            mask.extend(std::iter::repeat(0).take(pad));
        }

        let ids = Array2::from_shape_vec((1, self.max_len), ids)?;
        let mask = Array2::from_shape_vec((1, self.max_len), mask)?;
        let inputs = vec![
            (
                self.input_ids.as_str(),
                Value::from_array(self.session.allocator(), &ids)?,
            ),
            (
                self.attention_mask.as_str(),
                Value::from_array(self.session.allocator(), &mask)?,
            ),
        ];

        let outputs = self.session.run(inputs)?;
        let logits = outputs
            .get(self.logits_out.as_str())
            .or_else(|| outputs.iter().next().map(|(_, v)| v))
            .ok_or_else(|| anyhow!("no logits"))?;
        let arr = logits.try_extract::<ndarray::ArrayD<f32>>()?;
        let v = arr.view();
        let [_, c] = *v.shape() else {
            return Err(anyhow!("unexpected logits shape"));
        };
        if c != self.labels.len() {
            return Err(anyhow!("label count mismatch"));
        }

        let mut exps = vec![0f32; c];
        let mut sum = 0f32;
        for i in 0..c {
            exps[i] = v[[0, i]].exp();
            sum += exps[i];
        }
        let mut m = HashMap::with_capacity(c);
        for i in 0..c {
            m.insert(self.labels[i].clone(), exps[i] / sum);
        }
        Ok(m)
    }
}

#[cfg(not(feature = "onnx-classifier"))]
pub struct ModernBertClassifier;

#[cfg(not(feature = "onnx-classifier"))]
impl ModernBertClassifier {
    pub fn new(
        _onnx: &str,
        _tokenizer_json: &str,
        _labels: Vec<String>,
        _max_len: usize,
    ) -> Result<Self> {
        Err(anyhow!("ModernBertClassifier requires 'onnx-classifier' feature"))
    }
}

#[cfg(not(feature = "onnx-classifier"))]
impl MultiClassifier for ModernBertClassifier {
    fn probs(&self, _text: &str) -> Result<HashMap<String, f32>> {
        Err(anyhow!("ModernBertClassifier requires 'onnx-classifier' feature"))
    }
}

