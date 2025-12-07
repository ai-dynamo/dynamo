# Fault Tolerance Tests

End-to-end tests for GPU failure scenarios with automated validation.

## Quick Start

```bash
# Set your deployment
export TEST_NAMESPACE="your-namespace"
export TARGET_DEPLOYMENT="your-deployment"

# Run XID 79 test (GPU fell off bus)
pytest test_xid79_minimal.py::test_xid79_cordon_drain_only -v -s
```

## Available Fixtures

### Test Fixtures

- **`xid79_test`** - GPU fell off bus (XID 79)
- **`xid74_test`** - NVLink failure (XID 74)

### Expectation Fixtures

- **`expect_cordon_and_drain`** - Node cordoned + pods evicted (5-10 min)
- **`expect_full_automation`** - Full NVSentinel automation (cordon → drain → remediate → uncordon)
- **`expect_cordon_only`** - Only cordon, no drain

## Example Test

```python
@pytest.mark.xid79
def test_xid79_cordon_drain_only(xid79_test, expect_cordon_and_drain):
    """Test GPU failure with NVSentinel cordon and drain."""
    xid79_test(gpu_id=0, expect=expect_cordon_and_drain)
```

## What Gets Validated

- Fault injection (pods crash on target node)
- NVSentinel response (cordon/drain)
- Pod recovery (reschedule to healthy nodes)
- Inference recovery (service restored)
- Latency impact analysis (baseline → degraded → recovered)

## Requirements

- NVSentinel deployed with `DeleteAfterTimeout` mode
- Multi-node cluster with GPU nodes
- Target deployment with 2+ worker pods

## Troubleshooting

**Test fails with "Pods did not recover" after 15+ minutes?**

NVSentinel is likely in `AllowCompletion` mode, which never evicts crash-looping pods.

**Fix:**
```bash
kubectl edit configmap node-drainer-config -n nvsentinel
```

Change:
```toml
deleteAfterTimeoutMinutes = 5     # From 60
[[userNamespaces]]
name = "*"
mode = "DeleteAfterTimeout"       # From "AllowCompletion"
```

Restart:
```bash
kubectl rollout restart deployment nvsentinel-node-drainer -n nvsentinel
```

Test will now complete in ~5-10 minutes with proper pod eviction and recovery.

