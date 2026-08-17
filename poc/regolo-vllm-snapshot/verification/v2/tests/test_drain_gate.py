import concurrent.futures
import unittest

from _support import load_harness


class DrainGateTests(unittest.TestCase):
    def setUp(self):
        self.harness = load_harness()

    def drained_gate(self):
        gate = self.harness.DrainGate()
        gate.close_admission()
        gate.set_harness_inflight(0)
        gate.observe_vllm({"epoch": "drain-7", "sample_seq": 1, "observed_monotonic_ns": 1_000_000_000, "running": 0, "waiting": 0})
        gate.observe_vllm({"epoch": "drain-7", "sample_seq": 2, "observed_monotonic_ns": 2_000_000_000, "running": 0, "waiting": 0})
        return gate

    def test_drain_requires_closed_admission_zero_ledger_and_two_distinct_zero_samples(self):
        self.assertTrue(self.drained_gate().is_drained)
        gate = self.harness.DrainGate()
        gate.set_harness_inflight(0)
        gate.observe_vllm({"epoch": "drain-7", "sample_seq": 1, "observed_monotonic_ns": 1_000_000_000, "running": 0, "waiting": 0})
        gate.observe_vllm({"epoch": "drain-7", "sample_seq": 2, "observed_monotonic_ns": 2_000_000_000, "running": 0, "waiting": 0})
        self.assertFalse(gate.is_drained)
        gate.close_admission()
        gate.set_harness_inflight(1)
        self.assertFalse(gate.is_drained)

    def test_replay_nonmonotonic_or_cross_epoch_samples_cannot_prove_drain(self):
        gate = self.harness.DrainGate()
        gate.close_admission()
        gate.set_harness_inflight(0)
        gate.observe_vllm({"epoch": "drain-7", "sample_seq": 2, "observed_monotonic_ns": 2_000_000_000, "running": 0, "waiting": 0})
        with self.assertRaises(ValueError):
            gate.observe_vllm({"epoch": "drain-7", "sample_seq": 2, "observed_monotonic_ns": 2_000_000_000, "running": 0, "waiting": 0})
        with self.assertRaises(ValueError):
            gate.observe_vllm({"epoch": "drain-7", "sample_seq": 1, "observed_monotonic_ns": 1_000_000_000, "running": 0, "waiting": 0})
        self.assertFalse(gate.is_drained)
        gate = self.harness.DrainGate()
        gate.close_admission()
        gate.set_harness_inflight(0)
        gate.observe_vllm({"epoch": "drain-7", "sample_seq": 1, "observed_monotonic_ns": 1_000_000_000, "running": 0, "waiting": 0})
        gate.observe_vllm({"epoch": "drain-8", "sample_seq": 2, "observed_monotonic_ns": 2_000_000_000, "running": 0, "waiting": 0})
        self.assertFalse(gate.is_drained)

    def test_samples_are_thread_safe_and_replay_is_rejected_under_contention(self):
        gate = self.harness.DrainGate()
        gate.close_admission()
        gate.set_harness_inflight(0)
        samples = [
            {"epoch": "drain-7", "sample_seq": 1, "observed_monotonic_ns": 1_000_000_000, "running": 0, "waiting": 0},
            {"epoch": "drain-7", "sample_seq": 2, "observed_monotonic_ns": 2_000_000_000, "running": 0, "waiting": 0},
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            list(executor.map(gate.observe_vllm, samples))
        self.assertTrue(gate.is_drained)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(gate.observe_vllm, samples[1]) for _ in range(2)]
            self.assertTrue(all(future.exception() is not None for future in futures))


if __name__ == "__main__":
    unittest.main()
