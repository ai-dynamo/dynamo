// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend-neutral output of model-specific media preprocessing.

use std::collections::BTreeMap;

use anyhow::{Result, ensure};

use crate::types::{FieldLayout, Modality};

#[derive(Debug)]
pub enum ProcessedValue {
    F32Tensor { data: Vec<f32>, shape: Vec<usize> },
    I64Tensor { data: Vec<i64>, shape: Vec<usize> },
    F64Tensor { data: Vec<f64>, shape: Vec<usize> },
}

impl ProcessedValue {
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32Tensor { shape, .. }
            | Self::I64Tensor { shape, .. }
            | Self::F64Tensor { shape, .. } => shape,
        }
    }

    fn element_count(&self) -> usize {
        match self {
            Self::F32Tensor { data, .. } => data.len(),
            Self::I64Tensor { data, .. } => data.len(),
            Self::F64Tensor { data, .. } => data.len(),
        }
    }
}

#[derive(Debug)]
pub struct ProcessedField {
    pub value: ProcessedValue,
    pub layout: FieldLayout,
    /// Keep metadata on the host when the backend supports placement hints.
    pub keep_on_host: bool,
    /// False for layout helper fields that are not model forward arguments.
    pub forward: bool,
}

#[derive(Debug)]
pub struct ProcessedMedia {
    pub modality: Modality,
    pub fields: BTreeMap<String, ProcessedField>,
    pub feature_token_counts: Vec<usize>,
    pub original_sizes: Vec<(u32, u32)>,
}

impl ProcessedMedia {
    pub fn validate(&self) -> Result<()> {
        ensure!(!self.fields.is_empty(), "processed media has no fields");
        ensure!(
            self.feature_token_counts.len() == self.original_sizes.len(),
            "feature-token and original-size item counts differ"
        );
        for (name, field) in &self.fields {
            ensure!(
                !field.value.shape().is_empty(),
                "field {name} must have at least one dimension"
            );
            let expected = field.value.shape().iter().try_fold(1usize, |acc, dim| {
                acc.checked_mul(*dim)
                    .ok_or_else(|| anyhow::anyhow!("field {name} shape overflows"))
            })?;
            ensure!(
                expected == field.value.element_count(),
                "field {name} shape describes {expected} elements but contains {}",
                field.value.element_count()
            );
            match &field.layout {
                FieldLayout::Batched => ensure!(
                    field.value.shape()[0] == self.feature_token_counts.len(),
                    "batched field {name} has the wrong item count"
                ),
                FieldLayout::Flat { sizes_key } => {
                    let sizes = self.fields.get(sizes_key).ok_or_else(|| {
                        anyhow::anyhow!("field {name} references missing sizes field {sizes_key}")
                    })?;
                    let ProcessedValue::I64Tensor { data, .. } = &sizes.value else {
                        anyhow::bail!("sizes field {sizes_key} must be an I64 tensor");
                    };
                    ensure!(
                        data.len() == self.feature_token_counts.len(),
                        "sizes field {sizes_key} has the wrong item count"
                    );
                    let flat_size = data.iter().try_fold(0usize, |sum, value| {
                        usize::try_from(*value)
                            .ok()
                            .and_then(|value| sum.checked_add(value))
                    });
                    ensure!(
                        flat_size == Some(field.value.shape()[0]),
                        "sizes field {sizes_key} does not partition flat field {name}"
                    );
                }
                FieldLayout::Shared => {}
            }
        }
        Ok(())
    }

    pub fn field(&self, name: &str) -> Option<&ProcessedField> {
        self.fields.get(name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_layout_requires_sizes_to_partition_first_dimension() {
        let mut fields = BTreeMap::new();
        fields.insert(
            "values".to_string(),
            ProcessedField {
                value: ProcessedValue::F32Tensor {
                    data: vec![0.0; 6],
                    shape: vec![3, 2],
                },
                layout: FieldLayout::flat("sizes"),
                keep_on_host: false,
                forward: true,
            },
        );
        fields.insert(
            "sizes".to_string(),
            ProcessedField {
                value: ProcessedValue::I64Tensor {
                    data: vec![2],
                    shape: vec![1],
                },
                layout: FieldLayout::Batched,
                keep_on_host: true,
                forward: false,
            },
        );
        let media = ProcessedMedia {
            modality: Modality::Video,
            fields,
            feature_token_counts: vec![1],
            original_sizes: vec![(1, 1)],
        };

        assert!(
            media
                .validate()
                .unwrap_err()
                .to_string()
                .contains("partition")
        );
    }
}
