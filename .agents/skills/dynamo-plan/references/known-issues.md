# Planning Known Issues

Stable issue patterns relevant to pre-deployment planning. Per the
6-element shape from.

---

### `planner-profile-data` ConfigMap missing in `rapid` mode

**Symptom:** A DGDR run with `searchStrategy: rapid` and `features.planner` enabled reaches `Ready` but the planner-profile-data ConfigMap that the Planner reads at runtime is not created. The generated DGD's Planner pod cannot scale because it has no curves to follow.

**Root cause:** `rapid` mode uses the AIC simulator and does not produce the same interpolation curves that `thorough` mode produces from real-GPU sweeps. The Planner integration relies on those curves.

**Affected:** Dynamo 1.2.x, any release where `rapid` is the default. Tracked as DYN-3063 / NVBug 6189270.

**Fix:** For SLA-bounded production with the Planner enabled, prefer `searchStrategy: thorough`. For iteration without the Planner, `rapid` is fine.

**Verify:**

```bash
kubectl get configmap -n <ns> | grep planner-profile-data
```

A non-empty result confirms the ConfigMap exists; an empty result on a `rapid` DGDR confirms this issue.

Source: (lifecycle map), upstream Dynamo source.

---

### PCIe SKU profiler falls back to defaults

**Symptom:** A DGDR with `hardware.gpuSku` set to `h100_pcie`, `a100_pcie`, or `v100_pcie` runs successfully but the recommended config is generic — it looks like the AIC simulator didn't use SKU-specific data.

**Root cause:** The CRD admits PCIe SKUs but the profiler does not yet ship AIC training data for them. The schema accepts the value; the simulator falls back to defaults during config selection.

**Affected:** All Dynamo releases that include the PCIe SKU enum (per).

**Fix:** Either accept the default config and validate via benchmark, or skip planning for PCIe SKUs and author the DGD by hand using a similar-SXM-tier recipe as a starting point.

**Verify:** Compare the DGDR's `.status.profilingResults.selectedConfig` to a recipe for the matching SXM SKU; if they're identical, you got defaults rather than profiled data.

Source:.

---

### Latency-mode Planner stalls at `num_workers=1`

**Symptom:** Planner-driven DGD in `optimizationType: latency` mode never scales up past 1 worker even though the TTFT measurement is well above the SLA target.

**Root cause:** Latency-mode prefill scale-up triggers a Grove pod-gang roll that erases the scale-up state, leaving the Planner at `num_workers=1`. Without Grove (or with Grove misconfigured), the roll fails silently.

**Affected:** Dynamo 1.2.x with Grove sub-chart. Tracked as DYN-2879 / NVBug 6109874.

**Fix:** Either (a) confirm Grove is installed (`helm list | grep grove`) and the gang-scheduler is working, or (b) switch to `optimizationType: throughput` for this deployment which uses a different scale-up path.

**Verify:**

```bash
kubectl get pods -n <ns> -l app.kubernetes.io/name=grove
kubectl logs <planner-pod> -n <ns> | grep -i 'scale-up'
```

Source: (GHCR sub-charts), (versioning).
