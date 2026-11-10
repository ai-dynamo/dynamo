# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream the real Job output without changing its command or callbacks."""

import json
import sys
import time
from pathlib import Path

import yaml

from dynamo.aisimulate.output.dgdr_run.adapter import candidate_id_for

directory = Path(sys.argv[1])
previous = None
deadline = time.monotonic() + 540

# Observe files without replacing callbacks, changing the Job, or delaying its search.
while time.monotonic() < deadline:
    try:
        content = (directory / "dgdr_run_snapshot.yaml").read_text()
    except FileNotFoundError:
        content = None
    if content is not None and content != previous:
        snapshot = yaml.safe_load(content)
        if not snapshot["run"]["terminal"] and snapshot["progress"]["evaluated"]:
            print(json.dumps({"snapshot": snapshot}), flush=True)
        previous = content
    try:
        report = json.loads((directory / "recommendation.json").read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        time.sleep(0.01)
        continue
    break
else:
    raise TimeoutError("AISimulate did not write recommendation.json")

# Emit evidence promptly; do not race container exit while computing the test oracle.
records = [
    {
        "id": candidate_id_for(record["config"]),
        "status": record["status"],
        "objectives": record["objectives"],
    }
    for record in report["candidates"]
]
by_id = {record["candidate_id"]: record for record in report["candidates"]}
terminal_front = [
    candidate_id_for(by_id[key]["config"]) for key in report["views"]["pareto_front"]
]
print(json.dumps({"records": records, "terminalFront": terminal_front}), flush=True)

