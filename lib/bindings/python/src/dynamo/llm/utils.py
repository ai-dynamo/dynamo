# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import socket
import time


def maybe_sleep():
    """
    Maybe sleep for the duration specified in the environment variable if it is set.
    """
    sleep_duration = int(os.environ.get("DYN_KVBM_SLEEP", "0"))
    if sleep_duration > 0:
        print(f"Sleeping {sleep_duration} seconds to avoid metrics port conflict")
        time.sleep(sleep_duration)


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
