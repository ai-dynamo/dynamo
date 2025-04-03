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
use std::{
    env,
    ffi::CString,
    os::fd::RawFd,
    path::{Path, PathBuf},
};

use dynamo_llm::engines::MultiNodeConfig;

const PY_START_ENGINE: &str = include_str!("sglang_inc.py");

/// Start the Python sglang engine that listens on zmq socket
/// This is called by running `nio --internal-sglang-process
/// This does not return until the subprocess exits.
pub fn run_subprocess(
    // The prefix to put on the zmq socket names
    socket_id: &str,
    // Directory containing an HF repo with safetensors files, tokenizer, etc
    model_path: &Path,
    // The write half of a pipe, where sglang will signal when it's ready
    notify_pipe_fd: RawFd,
    // Multi node. Usually Default::default
    node_config: MultiNodeConfig,
    // Multi GPU. Usually Default::default
    gpu_config: super::MultiGPUConfig,
    // Allow passing any arguments to sglang
    extra_engine_args: Option<PathBuf>,
) -> anyhow::Result<()> {
    pyo3::prepare_freethreaded_python(); // or enable feature "auto-initialize"
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        let _ = Python::with_gil(|py| crate::fix_venv(venv, py));
    }
    let dir = model_path.display().to_string();
    let extra_engine_args_str = &extra_engine_args
        .map(|p| p.display().to_string())
        .unwrap_or_default();
    Python::with_gil(|py| {
        let locals = [
            ("socket_id", socket_id),
            ("model_path", dir.as_str()),
            ("pipe_fd", &notify_pipe_fd.to_string()),
            // to_string because slice must all be the same type
            ("tp_size_str", &gpu_config.tp_size.to_string()),
            ("tp_rank_str", &gpu_config.tp_rank.to_string()),
            ("gpu_id_str", &gpu_config.gpu_id.to_string()),
            ("nnodes_str", &node_config.num_nodes.to_string()),
            ("node_rank_str", &node_config.node_rank.to_string()),
            ("dist_init_addr", &node_config.leader_addr),
            ("extra_engine_args", extra_engine_args_str),
        ]
        .into_py_dict(py)
        .unwrap();
        if let Err(err) = py.run(CString::new(PY_START_ENGINE)?.as_ref(), None, Some(&locals)) {
            anyhow::bail!("sglang engine run error: {err}");
        }
        tracing::info!("sglang subprocess exit");
        Ok(())
    })
}
