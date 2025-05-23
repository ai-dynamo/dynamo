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

import os
import socket
import random
import logging

logger = logging.getLogger(__name__)

def _find_free_port():
    """Find a free port on the local machine."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('0.0.0.0', 0))
        return s.getsockname()[1]

def get_host_port():
    """Gets host and port from environment variables. 
    
    Defaults to 0.0.0.0:8000.
    If DYNAMO_LOCAL_SERVE is set, uses a random free port.
    """
    host = os.environ.get("DYNAMO_HOST", "0.0.0.0")
    
    # When running in local development mode, use a random port
    if os.environ.get("DYNAMO_LOCAL_SERVE"):
        port = _find_free_port()
        logger.info(f"DYNAMO_LOCAL_SERVE detected, using random port {port}")
        # Store the selected port in the environment for reference
        os.environ["DYNAMO_PORT"] = str(port)
    else:
        port = int(os.environ.get("DYNAMO_PORT", 8000))
    
    return host, port
