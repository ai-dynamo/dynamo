"""Print real watt values returned by the planner's DCGM methods.

Run inside the dev pod:
    NS=<your-namespace>
    kubectl cp scripts/inspect_planner_dcgm_methods.py $NS/planner-dev:/tmp/
    kubectl exec -n $NS planner-dev -- python3 /tmp/inspect_planner_dcgm_methods.py
"""
import os

from dynamo.planner.monitoring.traffic_metrics import PrometheusAPIClient

PROM_URL = "http://kube-prometheus-stack-prometheus.monitoring.svc.cluster.local:9090"
ns = os.environ["DYN_PARENT_DGD_K8S_NAMESPACE"]
dgd = os.environ["DYN_PARENT_DGD_K8S_NAME"]
dyn_ns = os.environ.get("DYN_NAMESPACE", f"{ns}-{dgd}")

prom = PrometheusAPIClient(PROM_URL, dyn_ns)
print(f"k8s_namespace={ns!r}  dgd={dgd!r}  dyn_namespace={dyn_ns!r}")

total = prom.get_total_dgd_power(k8s_namespace=ns, dgd_name=dgd)
per_comp = prom.get_avg_per_gpu_power_by_component(
    interval="5m",
    k8s_namespace=ns,
    dgd_name=dgd,
    component="agg",
    service_key="VllmWorker",
)
print(f"get_total_dgd_power           -> {total} W")
print(f"get_avg_per_gpu_power_by_comp -> {per_comp} W (avg per GPU, VllmWorker)")
