# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workloads for the learned-routing campaign: trace acquisition, transforms, cells and splits.

Modules:

- :mod:`.common`: hashing, canonical JSON, JSONL IO and campaign-root resolution.
- :mod:`.agentx`: AgentX (Weka) play parsing and the 128K complete-play selection.
- :mod:`.synthetic`: seeded multi-turn Mooncake session generator.
- :mod:`.acquire`: ``python -m learned_routing.workloads.acquire`` writes ``CR/traces`` and its
  ``MANIFEST.json``.
- :mod:`.transform`: ``python -m learned_routing.workloads.transform`` materializes derived traces.
- :mod:`.cells`: ``python -m learned_routing.workloads.cells`` emits candidate cells and the split
  manifest.
"""
