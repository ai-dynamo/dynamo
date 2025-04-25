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
import multiprocessing

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.multiprocessing.engine import run_mp_engine
from vllm.usage.usage_lib import UsageContext

arg_map = {
    "model": f"{model_path}",
    "served_model_name": None,
    "task": "generate",
    "skip_tokenizer_init": True,
    "seed": 0,
    "max_model_len": 8192,
    "max_seq_len_to_capture": 8192,
    "tensor_parallel_size": int(tp_size_str),
    "pipeline_parallel_size": int(nnodes_str),
    "enable_prefix_caching": enable_prefix_caching.lower() == "true",
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

engine_args = AsyncEngineArgs(**arg_map)
ipc_path = f"ipc:///tmp/{socket_id}"

engine_alive = multiprocessing.Value("b", True, lock=False)

run_mp_engine(engine_args, UsageContext.OPENAI_API_SERVER, ipc_path, engine_alive)
