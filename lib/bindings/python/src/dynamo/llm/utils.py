# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import socket
import time


def maybe_sleep():
    """
    Maybe sleep for the duration specified in the environment variable if it is set.
    """
    sleep_duration = int(os.environ.get("DYN_KVBM_SLEEP", "5"))
    if sleep_duration > 0:
        print(f"Sleeping {sleep_duration} seconds to avoid metrics port conflict")
        time.sleep(sleep_duration)


# TODO(keiven|ziqi|rihuo): Auto port selection to be done in Rust
def find_and_set_available_system_port_for_leader(env_var="DYN_KVBM_METRICS_PORT"):
    """
    Find an available port from the environment variable for kvbm leader.
    """
    system_metrics = os.environ.get("DYN_SYSTEM_KVBM_METRICS", "0")
    if system_metrics not in ("1", "true", "True", "TRUE"):
        print("system kvbm metrics disabled.")
        return

    os.environ["DYN_SYSTEM_ENABLED"] = "true"
    port = int(os.environ.get(env_var, "0"))
    if port == 0:
        port = 6881  # default metrics port number is 6881

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        # Port is available
        s.bind(("127.0.0.1", port))
        s.close()
        os.environ["DYN_SYSTEM_PORT"] = str(port)
        print(f"Port {port} is available, setting env var DYN_SYSTEM_PORT to {port}")
    except Exception as e:
        raise RuntimeError(f"Error bind to port {port}: {e}")
