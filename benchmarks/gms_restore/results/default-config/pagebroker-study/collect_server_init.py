# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce first-trial server startup measurements from archived events."""

import json
from pathlib import Path

study = Path(__file__).resolve().parent
case = study.parent / "pagebroker-native16-1"
rows = []
for rank in range(8):
    events = {}
    for line in (case / f"gms-{rank}.txt").read_text().splitlines():
        if line.startswith("{"):
            event = json.loads(line)
            events.setdefault(event["event"], event)
    online = events["online"]
    rows.append(
        {
            "rank": rank,
            "cuinit_s": events["cuinit_complete"]["epoch"]
            - events["cuinit_start"]["epoch"],
            "logged_start_to_online_s": online["epoch"]
            - events["daemon_start"]["epoch"],
            "script_to_online_s": online["online_epoch"]
            - online["daemon_started_epoch"],
        }
    )
(study / "server-init.json").write_text(json.dumps(rows, indent=2) + "\n")
