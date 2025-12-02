"""
XID 79 E2E Test - Persistent Hardware Failure Simulation

This tests realistic node-level GPU failure with persistent hardware fault:

Phase 0: Natural pod distribution across nodes (realistic!)
Phase 1: Fault injection on target node (persistent via hostPath)
  - Fault marker written to /var/lib/cuda-fault-test on node (survives pod restarts)
  - Pods on faulty node crash-loop indefinitely (realistic!)
  - Pods on other nodes stay healthy
  - Inference partially degraded

Phase 2: NVSentinel response
  - Cordons faulty node
  - Waits for drain/eviction (NVSentinel policy-dependent)
  - Crashed pods reschedule to healthy nodes where fault marker absent
  - Pods on healthy nodes recover automatically (no fault file there)

Phase 3: Recovery
  - All pods on healthy nodes (no fault marker)
  - Inference fully recovered
  - Test cleanup removes fault marker from node

This simulates: Persistent GPU hardware failure requiring rescheduling
(not transient failures that restart-recovery can fix)

"""

import pytest


@pytest.mark.xid79
@pytest.mark.nvsentinel
@pytest.mark.slow
def test_xid79_full_automation(xid79_test, expect_full_automation):
    """XID 79 with full NVSentinel automation (detection → cordon → drain → remediate → uncordon)."""
    xid79_test(gpu_id=0, expect=expect_full_automation)


@pytest.mark.xid79
@pytest.mark.nvsentinel
def test_xid79_cordon_drain_only(xid79_test, expect_cordon_and_drain):
    """XID 79 with cordon + drain (no auto-remediation)."""
    xid79_test(gpu_id=0, expect=expect_cordon_and_drain)


@pytest.mark.xid79
@pytest.mark.parametrize("gpu_id", [0, 1, 2, 3])
def test_xid79_all_gpus(xid79_test, expect_cordon_and_drain, gpu_id):
    """Test XID 79 on each GPU."""
    xid79_test(gpu_id=gpu_id, expect=expect_cordon_and_drain)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s", "-m", "xid79"])

