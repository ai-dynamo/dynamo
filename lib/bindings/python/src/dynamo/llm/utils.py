# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import socket


# TODO(keiven|ziqi): Auto port selection to be done in Rust
def find_and_set_available_port_from_env(env_var="DYN_SYSTEM_PORT"):
    """
    Find an available port from the environment variable.
    """
    port = int(os.environ.get(env_var, "0"))
    if port == 0:
        # No port specified, let system pick
        pass
    while True:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            # Port is available
            s.bind(("127.0.0.1", port))
            s.close()
            os.environ[env_var] = str(port)
            print(f"Port {port} is available, setting env var {env_var} to {port}")
            break
        except OSError:
            # Port is in use, try next
            port += 1
            s.close()
        except Exception as e:
            raise RuntimeError(f"Error finding available port: {e}")


# TODO(rihuo): Auto setup standalone metrics based on if Distributed Runtime is provided or not.
# remove this when ETCD can be removed from the leader-worker sync
def is_standslone_kvbm_metrics_enabled() -> bool:
    """
    Return True if DYN_KVBM_METRICS_STANDALONE is set to '1' or any case-variant of 'true'.
    """
    val = os.environ.get("DYN_KVBM_METRICS_STANDALONE")
    if val is None:
        return False
    v = val.strip().lower()
    return v == "1" or v == "true"
