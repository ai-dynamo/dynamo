# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PR B — _make_actuator(args, metrics) factory tests.

The factory is the single source of truth for translating the
operator's `--actuator` choice into a concrete Actuator instance.
These tests assert:

  * `nvml` → NvmlActuator with the metrics object plumbed through.
  * `dcgm` → DcgmActuator with host/port/enforce plumbed through.
  * An unknown value raises ValueError loudly (not silently fall back).

We do NOT call `init()` on the resulting actuators here — that would
require a live hostengine for the DCGM path. Construction itself is
the unit under test.

Coverage parity with argparse's `choices=` guard
-----------------------------------------------
argparse rejects unknown `--actuator` values at parse time, but the
factory is also defensive against a future caller building an
`argparse.Namespace` programmatically (or a refactor that loosens
choices). The `test_unknown_actuator_raises_value_error` test guards
that boundary.
"""

import argparse
import unittest
from unittest.mock import MagicMock

import power_agent
from actuator import DcgmActuator, NvmlActuator


def _make_args(actuator="nvml", dcgm_host=None, dcgm_port=None, dcgm_enforce=False):
    """Build an argparse.Namespace with the actuator-related fields."""
    return argparse.Namespace(
        actuator=actuator,
        dcgm_host=dcgm_host or DcgmActuator.DEFAULT_HOST,
        dcgm_port=dcgm_port or DcgmActuator.DEFAULT_PORT,
        dcgm_enforce=dcgm_enforce,
    )


class TestMakeActuatorNvml(unittest.TestCase):
    def test_nvml_actuator_returned(self):
        metrics = MagicMock()
        actuator = power_agent._make_actuator(_make_args("nvml"), metrics)
        self.assertIsInstance(actuator, NvmlActuator)
        self.assertEqual(actuator.name, "nvml")

    def test_nvml_actuator_receives_metrics(self):
        """Without this, NvmlActuator.apply_cap would raise at runtime
        (PR A's test_apply_cap_requires_metrics covers the symptom; this
        test prevents the factory from being the cause)."""
        metrics = MagicMock()
        actuator = power_agent._make_actuator(_make_args("nvml"), metrics)
        # NvmlActuator stores metrics on _metrics — private but stable
        # across PR A + PR B (it's the contract this factory wires up).
        self.assertIs(actuator._metrics, metrics)


class TestMakeActuatorDcgm(unittest.TestCase):
    def test_dcgm_actuator_returned(self):
        metrics = MagicMock()
        actuator = power_agent._make_actuator(_make_args("dcgm"), metrics)
        self.assertIsInstance(actuator, DcgmActuator)
        self.assertEqual(actuator.name, "dcgm")

    def test_dcgm_actuator_plumbs_host_port_enforce(self):
        """The whole point of the CLI flags — they must reach the actuator."""
        args = _make_args(
            "dcgm",
            dcgm_host="other-host",
            dcgm_port=9999,
            dcgm_enforce=True,
        )
        actuator = power_agent._make_actuator(args, MagicMock())
        self.assertEqual(actuator._host, "other-host")
        self.assertEqual(actuator._port, 9999)
        self.assertTrue(actuator._enforce)

    def test_dcgm_actuator_defaults_match_gpu_operator(self):
        args = _make_args("dcgm")  # no overrides
        actuator = power_agent._make_actuator(args, MagicMock())
        self.assertEqual(actuator._host, "nvidia-dcgm.gpu-operator.svc.cluster.local")
        self.assertEqual(actuator._port, 5555)
        # Enforce defaults to false (matches values.yaml default and
        # the v1.3 design decision: opt-in re-assertion).
        self.assertFalse(actuator._enforce)

    def test_dcgm_actuator_receives_metrics(self):
        metrics = MagicMock()
        actuator = power_agent._make_actuator(_make_args("dcgm"), metrics)
        self.assertIs(actuator._metrics, metrics)


class TestMakeActuatorInvalid(unittest.TestCase):
    def test_unknown_actuator_raises_value_error(self):
        """Defends against a future caller building a Namespace
        programmatically and bypassing argparse's choices= guard."""
        with self.assertRaises(ValueError) as ctx:
            power_agent._make_actuator(_make_args("typo"), MagicMock())
        self.assertIn("typo", str(ctx.exception))
        # Surface the valid options in the error so the operator
        # doesn't have to grep the source.
        self.assertIn("nvml", str(ctx.exception))
        self.assertIn("dcgm", str(ctx.exception))

    def test_empty_actuator_raises_value_error(self):
        with self.assertRaises(ValueError):
            power_agent._make_actuator(_make_args(""), MagicMock())


if __name__ == "__main__":
    unittest.main()
