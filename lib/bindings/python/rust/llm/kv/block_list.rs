// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Silence warnings about deprecated features (like pyo3::IntoPy::into_py)
#![allow(deprecated)]

use super::*;

use pyo3::{Python, PyResult, types::PyList};
use std::{sync::{Arc, Mutex}};

#[pyclass]
pub struct BlockList {
    inner: Vec<Arc<Mutex<block::BlockType>>>,
}

impl BlockList {
    pub fn from_rust(block_list: Vec<block::BlockType>) -> Self {
        Self {
            inner: block_list.into_iter().map(|b| Arc::new(Mutex::new(b))).collect(),
        }
    }
}

#[pymethods]
impl BlockList {
    fn to_list(&self) -> PyResult<Py<PyList>> {
        let py_list = Python::with_gil(|py| {
            let blocks: Vec<block::Block> = self.inner.iter().map(|b| block::Block::from_rust(b.clone())).collect();
            PyList::new(py, blocks).unwrap().unbind()
        });
        Ok(py_list)
    }
}
