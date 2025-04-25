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
    with_kv_routing: bool,
) -> anyhow::Result<()> {
    if with_kv_routing {
        set_kv_routing_vars()?;
    }
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
            ("enable_prefix_caching", &with_kv_routing.to_string()),
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

// These environment variables trigger our vllm patch to emit KV routing events
fn set_kv_routing_vars() -> anyhow::Result<()> {
    let exe = env::current_exe()?;
    let exe_dir = exe
        .parent()
        .ok_or(anyhow::anyhow!("Current binary has no directory"))?;
    let mut lib = PathBuf::from(exe_dir);
    lib.set_file_name("libdynamo_llm_capi.so");
    let vars = [
        // Path to the C API Library
        ("VLLM_KV_CAPI_PATH", lib.display().to_string()),
        // Identifiers to publish KV related information
        ("VLLM_KV_NAMESPACE", "dynamo".to_string()),
        ("VLLM_KV_COMPONENT", "vllm".to_string()),
        // Worker ID used for identifying workers in distributed settings
        ("VLLM_WORKER_ID", "0".to_string()),
    ];
    for (kvar, default_v) in vars {
        if env::var(kvar).is_err() {
            env::set_var(kvar, default_v);
        }
    }
    Ok(())
}
