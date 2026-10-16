# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU power control for Dynamo deployments, driven by NVIDIA EPD.

The package has two command-line entry points:

* ``python -m dynamo.power.node_agent`` -- one process per GPU group (the GPUs
  on one node that belong to one power pool). Every rank writes EPD's energy
  snapshots; rank 0 of a pool also runs EPD's controller and the figure-of-merit
  (FoM) provider, and publishes ``power_decision.json`` through EPD's ``file``
  actuator.
* ``python -m dynamo.power.applier`` -- one process per node, on the bare host
  under SLURM. It reads the decision file and makes the caps real
  (``sudo -n nvidia-smi -pl``, ``dcgmi`` or NVML), verifies them through
  read-only NVML and falls back to the pool's uniform budget when the
  controller goes quiet.

EPD itself is an optional dependency that is installed separately; only
:mod:`dynamo.power.node_agent` imports it, and only :mod:`dynamo.power.applier`
and :mod:`dynamo.power.node_agent` import ``pynvml``. The command-line parsing
(:mod:`dynamo.power.node_agent_cli`), the applier's state machine
(:mod:`dynamo.power.applier_core`), the configuration contract
(:mod:`dynamo.power.config`), the FoM providers (:mod:`dynamo.power.fom`) and
the decision-file reader (:mod:`dynamo.power.decision`) need neither, so they
are unit-tested without GPUs. This module imports nothing so the applier stays
runnable on a host that has only Python and ``nvidia-ml-py``.
"""
