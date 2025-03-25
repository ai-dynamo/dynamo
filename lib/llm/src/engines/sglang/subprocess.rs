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
use std::{env, os::fd::RawFd, path::Path};

use crate::engines::MultiNodeConfig;

const PY_START_ENGINE: &std::ffi::CStr = cr#"
from multiprocessing.connection import Connection
import signal
import tempfile
import logging

from sglang.srt.server_args import ServerArgs, PortArgs
import sglang as sgl
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.entrypoints.engine import _set_envs_and_config


server_args = ServerArgs(
    model_path=f"{model_path}",
    enable_metrics = False,
    log_level = "debug",
    log_requests = True,
    tp_size = int(tp_size_str),
    # Multi-node
    dist_init_addr = dist_init_addr if dist_init_addr != "" else None,
    nnodes = int(nnodes_str),
    node_rank = int(node_rank_str),
)
logging.basicConfig(
    level="DEBUG",
    force=True,
    datefmt="%Y-%m-%d %H:%M:%S",
    format=f"[%(asctime)s] %(message)s",
)
_set_envs_and_config(server_args)

logging.debug(server_args)

ipc_path = f"ipc:///tmp/{socket_id}";
# These must match worker.rs zmq_sockets, which is the other side
port_args = PortArgs(
    # we don't use this one so use anything
    tokenizer_ipc_name=f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}",
    # Us -> sglang
    scheduler_input_ipc_name=f"{ipc_path}_input_socket",
    # sglang -> us
    detokenizer_ipc_name=f"{ipc_path}_output_socket",
    # The port for nccl initialization (torch.dist), which we don't use
    nccl_port=9876,
)

# Rank must be globally unique across nodes
tp_rank = int(tp_rank_str)

# See nvidia-smi for GPU IDs, they run 0,1,2,etc.
# In a single-node setup this is the same as rank
gpu_id = int(gpu_id_str)

pipe_fd_int = int(pipe_fd)
writer = Connection(handle=pipe_fd_int, readable=False, writable=True)

run_scheduler_process(server_args, port_args, gpu_id, tp_rank, None, writer)
"#;

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
) -> anyhow::Result<()> {
    pyo3::prepare_freethreaded_python(); // or enable feature "auto-initialize"
    if let Ok(venv) = env::var("VIRTUAL_ENV") {
        let _ = Python::with_gil(|py| crate::engines::fix_venv(venv, py));
    }
    let dir = model_path.display().to_string();
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
        ]
        .into_py_dict(py)
        .unwrap();
        if let Err(err) = py.run(PY_START_ENGINE, None, Some(&locals)) {
            anyhow::bail!("sglang engine run error: {err}");
        }
        tracing::info!("sglang subprocess exit");
        Ok(())
    })
}
