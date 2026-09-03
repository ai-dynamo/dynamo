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

import logging
import runpy

from dynamo.sglang._compat import ensure_sglang_grpc_bridge_batch_size

logger = logging.getLogger(__name__)


def main() -> None:
    """Install Dynamo's SGLang compat overrides, then run SGLang's launcher."""
    try:
        ensure_sglang_grpc_bridge_batch_size()
    except Exception:
        # Starting the engine matters more than the override. On a release that
        # does not need it, or one that restructured the bridge, log and hand
        # off anyway rather than failing a launch that would otherwise work.
        logger.warning(
            "Could not install the SGLang gRPC bridge compatibility override; "
            "continuing to launch the engine",
            exc_info=True,
        )

    # run_module rather than an import: sglang.launch_server does its work under
    # `if __name__ == "__main__"`, so this reproduces `python -m sglang.launch_server`
    # exactly, including its sys.argv handling.
    runpy.run_module("sglang.launch_server", run_name="__main__")


if __name__ == "__main__":
    main()
