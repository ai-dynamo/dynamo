# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Placeholder-side gate: resolve actual claim, validate artifacts and all servers."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from resolve_plan import resolve

root = Path("/snapshot-app")
plan = resolve(
    *(
        json.loads((root / name).read_text())
        for name in ["capture.json", "claim.json", "slices.json"]
    )
)
assert plan["cuda_device_map"] == os.environ["SNAPSHOT_CUDA_DEVICE_MAP"]
path = Path("/gms/restore-plan.json")
path.write_text(json.dumps(plan))
while not all(Path(f"/gms/published-{r['rank']}").exists() for r in plan["ranks"]):
    time.sleep(0.01)
subprocess.run(
    [sys.executable, str(root / "verify_publication.py"), str(path)], check=True
)
Path("/gms/all-ready").write_text(str(time.time()))
print(json.dumps({"event": "all_ranks_verified", "epoch": time.time()}), flush=True)
while True:
    time.sleep(60)
