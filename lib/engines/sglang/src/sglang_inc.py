# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

#
# This file is included as a string in subprocess.rs. Most work should be done in the Rust caller.
#

import json
import logging
import tempfile
from multiprocessing.connection import Connection

from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.managers.scheduler import run_scheduler_process
from sglang.srt.server_args import PortArgs, ServerArgs

logging.basicConfig(
    level="DEBUG",
    force=True,
    datefmt="%Y-%m-%d %H:%M:%S",
    format="[%(asctime)s] %(message)s",
)

# These can all be overridden by --extra-engine-args json file
arg_map = {
    "model_path": f"{model_path}",
    "enable_metrics": False,
    "log_level": "debug",
    "log_requests": True,
    "tp_size": int(tp_size_str),
    # Multi-node
    "dist_init_addr": dist_init_addr if dist_init_addr != "" else None,
    "nnodes": int(nnodes_str),
    "node_rank": int(node_rank_str),
}
json_map = {}
if extra_engine_args != "":
    # extra_engine_args is a filename
    try:
        with open(extra_engine_args) as f:
            json_map = json.load(f)
    except FileNotFoundError:
        logging.debug(f"File {extra_engine_args} not found.")
    except json.JSONDecodeError as e:
        logging.debug(f"Invalid JSON in {extra_engine_args}: {e}")
    logging.debug(f"Adding extra engine arguments: {json_map}")
    arg_map = {**arg_map, **json_map}  # json_map gets precedence

server_args = ServerArgs(**arg_map)
_set_envs_and_config(server_args)
logging.debug(server_args)

ipc_path = f"ipc:///tmp/{socket_id}"
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
