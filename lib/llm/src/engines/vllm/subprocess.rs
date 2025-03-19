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
use std::path::Path;

use crate::engines::MultiNodeConfig;

const PY_START_ENGINE: &std::ffi::CStr = cr#"
import multiprocessing
import signal

from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.usage.usage_lib import UsageContext

engine_args = AsyncEngineArgs(
    model=f"{model_path}",
    served_model_name=None,
    tokenizer=f"{tokenizer_path}",
    task='generate',
    tokenizer_mode='auto',
    seed=0,
    max_model_len=8192,
    max_seq_len_to_capture=8192,
    tensor_parallel_size = int(tp_size_str),
    pipeline_parallel_size = int(nnodes_str),
)

ipc_path = f"ipc:///tmp/{socket_id}";

engine_alive = multiprocessing.Value('b', True, lock=False)
run_mp_engine(engine_args, UsageContext.OPENAI_API_SERVER, ipc_path, engine_alive)
"#;

/// Start the Python vllm engine that listens on zmq socket
/// This is called by running `<bin> --internal-vllm-process
/// This does not return until vllm exits.
pub fn run_subprocess(
    socket_id: &str,
    model_card_path: &Path,
    model_path: &Path,
    node_config: MultiNodeConfig,
    tp_size: u32,
) -> anyhow::Result<()> {
    pyo3::prepare_freethreaded_python(); // or enable feature "auto-initialize"
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        let _ = Python::with_gil(|py| crate::engines::fix_venv(venv, py));
    }
    let card = model_card_path.display().to_string();
    let model_path_str = model_path.display().to_string();
    Python::with_gil(|py| {
        let locals = [
            ("socket_id", socket_id),
            ("tokenizer_path", card.as_str()),
            ("model_path", model_path_str.as_str()),
            ("tp_size_str", &tp_size.to_string()),
            ("nnodes_str", &node_config.num_nodes.to_string()),
        ]
        .into_py_dict(py)
        .unwrap();
        if let Err(err) = py.run(PY_START_ENGINE, None, Some(&locals)) {
            anyhow::bail!("vllm engine run error: {err}");
        }
        tracing::info!("vllm subprocess exit");
        Ok(())
    })
}
