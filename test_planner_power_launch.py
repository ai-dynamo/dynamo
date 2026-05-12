"""Launch planner with power awareness enabled to validate the full
DPP loop on the live cluster:

    planner -> patch_pod_annotation(`dynamo.nvidia.com/gpu-power-limit`)
            -> Power Agent reconcile
            -> nvmlDeviceSetPowerManagementLimit
            -> worker GPU enforced_W reflects the planner's value.

Distinctive value `decode_engine_gpu_power_limit=320` (not 280 [safe default],
not 350 [previous manual annotation], not 400 [NVML default]) lets us prove
the cap on the worker GPU was driven by the planner, not by something else.

Env vars (DYN_*, POD_*, NATS_SERVER, etc.) are injected by the dev-pod spec.
"""
import asyncio
import sys

sys.path.insert(0, "/workspace/repo/components/src")
sys.path.insert(0, "/workspace/repo")

from dynamo.planner.config.planner_config import PlannerConfig

config = PlannerConfig(
    environment="kubernetes",
    mode="agg",
    backend="vllm",
    optimization_target="throughput",
    advisory=True,  # don't scale workers, only annotate
    live_dashboard_port=0,
    metric_reporting_prometheus_port=0,
    # --- Power-awareness knobs ---
    enable_power_awareness=True,
    total_gpu_power_limit=2000,  # 2 GPUs × ~1000W headroom
    decode_engine_gpu_power_limit=320,  # distinctive value (not 280/350/400)
    power_agent_safe_default_watts=280,
)
print(f"Config: env={config.environment} mode={config.mode}")
print(f"        advisory={config.advisory} power_aware={config.enable_power_awareness}")
print(f"        decode_cap={config.decode_engine_gpu_power_limit}W")

from dynamo.planner.__main__ import init_planner  # noqa: E402
from dynamo.runtime import DistributedRuntime, dynamo_worker  # noqa: E402


@dynamo_worker()
async def worker(runtime: DistributedRuntime):
    print("DistributedRuntime connected; starting planner (45 s ceiling)…")
    try:
        await asyncio.wait_for(init_planner(runtime, config), timeout=45)
    except asyncio.TimeoutError:
        print("\n[OK] Planner ran for 45s in advisory + power-aware mode.")
    except Exception as e:
        print(f"\n[ERR] Planner init: {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()


asyncio.run(worker())
