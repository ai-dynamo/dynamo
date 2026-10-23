# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the inherited TRT-LLM NIXL stack without a global preload.

A second NIXL/UCX installation can import successfully yet conflict with
TRT-LLM's native wrapper. Check real CPU transfers and loaded library identities
in both load orders, in the built image's unmodified loader environment.
TRT-LLM's native wrapper requires the NVIDIA driver even for a CPU transfer;
this test loads no model and allocates no GPU tensors.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [
    pytest.mark.trtllm,
    pytest.mark.core,
    pytest.mark.pre_merge,
    pytest.mark.post_merge,
    pytest.mark.gpu_1,
    pytest.mark.integration,
    pytest.mark.timeout(90),
    pytest.mark.skipif(
        importlib.util.find_spec("tensorrt_llm") is None,
        reason="requires the built TRT-LLM image",
    ),
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Linux only"),
]


def _probe(order):
    # This function also runs as a standalone subprocess to isolate native state.
    import ctypes
    import os
    import time

    def mapped(name):
        return {
            str(Path(line.split()[-1]).resolve())
            for line in Path("/proc/self/maps").read_text().splitlines()
            if Path(line.split()[-1]).name == name
            or Path(line.split()[-1]).name.startswith(name + ".")
        }

    assert not mapped("libnixl.so"), "ordinary processes must not preload NIXL"
    spec = importlib.util.find_spec("tensorrt_llm")
    libs = Path(spec.origin).parent / "libs"

    def load_native():
        ctypes.CDLL(str(libs / "libtensorrt_llm.so"), mode=ctypes.RTLD_GLOBAL)
        ctypes.CDLL(
            str(libs / "libtensorrt_llm_nixl_wrapper.so"), mode=ctypes.RTLD_GLOBAL
        )

    if order == "native-first":
        load_native()
    from nixl._api import nixl_agent

    if order == "python-first":
        load_native()
    sender = nixl_agent(f"packaging-{os.getpid()}-sender")
    receiver = nixl_agent(f"packaging-{os.getpid()}-receiver")
    if os.environ.get("EFA_VERSION"):
        assert "LIBFABRIC" in sender.get_plugin_list(), (
            "EFA plugin must be discoverable in both native/Python load orders"
        )
    source = ctypes.create_string_buffer(bytes(range(256)) * 2048)
    target = ctypes.create_string_buffer(len(source))
    local = [(ctypes.addressof(source), len(source), 0, "")]
    remote = [(ctypes.addressof(target), len(target), 0, "")]
    source_registration = sender.register_memory(local, "DRAM")
    target_registration = receiver.register_memory(remote, "DRAM")
    handle = None
    try:
        sender.add_remote_agent(receiver.get_agent_metadata())
        # Registration includes metadata; transfer descriptors contain only
        # the address, length, and device ID.
        handle = sender.initialize_xfer(
            "WRITE",
            sender.get_xfer_descs([entry[:3] for entry in local], "DRAM"),
            receiver.get_xfer_descs([entry[:3] for entry in remote], "DRAM"),
            receiver.name,
        )
        state = sender.transfer(handle)
        deadline = time.monotonic() + 15
        while state != "DONE":
            assert state in ("PROC", "IN_PROG"), state
            assert time.monotonic() < deadline, "CPU transfer did not complete"
            time.sleep(0.001)
            state = sender.check_xfer_state(handle)
        assert source.raw == target.raw, "NIXL CPU transfer corrupted data"
        for name in ("libnixl.so", "libucp.so", "libucs.so", "libuct.so"):
            paths = mapped(name)
            assert len(paths) == 1, f"expected one loaded {name}, got {paths}"
    finally:
        if handle is not None:
            sender.release_xfer_handle(handle)
        sender.deregister_memory(source_registration)
        receiver.deregister_memory(target_registration)


@pytest.mark.parametrize("order", ["native-first", "python-first"])
def test_native_and_python_nixl_share_a_working_stack(order):
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), order],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"NIXL compatibility failed in {order} load order:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


if __name__ == "__main__":
    _probe(sys.argv[1])
