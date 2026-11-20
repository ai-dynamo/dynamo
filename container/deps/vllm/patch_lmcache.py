# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Adapt the image's LMCache 0.5.6 to vLLM 0.34.

LMCache 0.5.6 reads ``CachedRequestData.resumed_req_ids``, which vLLM 0.34
removed (vllm-project/vllm#63373), so the engine fails on its first decode
step. vLLM 0.34 resends a resumed request in full as a new request, so no
cached request is a resumed one. Remove this script once the image moves to an
LMCache release that supports vLLM 0.34.
"""

import importlib.metadata
import importlib.util
import py_compile
from pathlib import Path

LMCACHE_VERSION = "0.5.6"
NOTICE = "# Modified by NVIDIA Dynamo: resumed_req_ids is optional for vLLM 0.34.\n"
PATCHES = {
    "vllm_v1_adapter.py": (
        """            else:
                # This case should not be reached with supported vLLM versions.
                # Raising an error is safer than assuming not preempted.
                raise AttributeError(
                    f"Unable to determine preemption status for request {req_id}. "
                    f"This might be due to an unsupported vLLM version."
                )
""",
        """            else:
                preempted = False
""",
    ),
    "lmcache_mp_connector.py": (
        "            if request_id not in cached_reqs.resumed_req_ids:\n",
        '            if request_id not in getattr(cached_reqs, "resumed_req_ids", ()):\n',
    ),
}


def main() -> None:
    version = importlib.metadata.version("lmcache")
    if version != LMCACHE_VERSION:
        raise SystemExit(
            f"lmcache {version} is installed, but this patch targets "
            f"{LMCACHE_VERSION}; update or remove it"
        )
    spec = importlib.util.find_spec("lmcache")
    package = Path(spec.submodule_search_locations[0]) / "integration" / "vllm"
    for name, (old, new) in PATCHES.items():
        path = package / name
        text = path.read_text()
        if text.count(old) != 1:
            raise SystemExit(
                f"{path}: patched code not found once; update or remove this patch"
            )
        path.write_text(NOTICE + text.replace(old, new))
        py_compile.compile(str(path), doraise=True)


if __name__ == "__main__":
    main()
