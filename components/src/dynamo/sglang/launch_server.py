# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Drop-in replacement for ``sglang.launch_server`` with Dynamo's compat shims.

The Dynamo SGLang sidecar drives SGLang's native gRPC server, which in SGLang
0.5.17 and 0.5.18 fails every streaming request (see
``_compat.ensure_sglang_grpc_bridge_batch_size``). The gRPC bridge runs in the
same process as the engine, so the fix has to be installed there rather than in
the sidecar. Launching the engine through this module installs the override and
then hands off to SGLang unchanged.

Every argument is passed through untouched::

    python3 -m dynamo.sglang.launch_server --model-path <model> --grpc-port 30001

Use plain ``sglang.launch_server`` when the engine runs from a stock SGLang
image with no Dynamo Python available; that requires an SGLang release which
fixes the ordering upstream.
"""

import runpy

from dynamo.sglang._compat import ensure_sglang_grpc_bridge_batch_size


def main() -> None:
    """Install Dynamo's SGLang compat overrides, then run SGLang's launcher."""
    # Deliberately unguarded. The override already returns quietly for a release
    # that does not need it or that restructured the bridge, so anything raising
    # here is a real failure -- and launching anyway would start an engine whose
    # every streaming request fails with HTTP 500, the defect this module exists
    # to avoid. Failing at startup reports that far better than a served 500.
    ensure_sglang_grpc_bridge_batch_size()

    # run_module, not import: sglang.launch_server does its work under
    # `if __name__ == "__main__"`, so this reproduces `python -m` exactly.
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
