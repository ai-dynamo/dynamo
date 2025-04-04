// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use pyo3::{types::IntoPyDict, Python};
use std::env;
use std::ffi::CString;
use std::path::{Path, PathBuf};

use dynamo_llm::engines::MultiNodeConfig;

const PY_START_ENGINE: &str = include_str!("vllm_inc.py");

/// Start the Python vllm engine that listens on zmq socket
/// This is called by running `<bin> --internal-vllm-process
/// This does not return until vllm exits.
pub fn run_subprocess(
    socket_id: &str,
    model_path: &Path,
    node_config: MultiNodeConfig,
    tp_size: u32,
    extra_engine_args: Option<PathBuf>,
) -> anyhow::Result<()> {
    pyo3::prepare_freethreaded_python(); // or enable feature "auto-initialize"
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        let _ = Python::with_gil(|py| crate::fix_venv(venv, py));
    }
    let model_path_str = model_path.display().to_string();
    let extra_engine_args_str = &extra_engine_args
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    Python::with_gil(|py| {
        let locals = [
            ("socket_id", socket_id),
            ("model_path", model_path_str.as_str()),
            ("tp_size_str", &tp_size.to_string()),
            ("nnodes_str", &node_config.num_nodes.to_string()),
            ("extra_engine_args", extra_engine_args_str),
        ]
        .into_py_dict(py)
        .unwrap();
        if let Err(err) = py.run(CString::new(PY_START_ENGINE)?.as_ref(), None, Some(&locals)) {
            anyhow::bail!("vllm engine run error: {err}");
        }
        tracing::info!("vllm subprocess exit");
        Ok(())
    })
}
