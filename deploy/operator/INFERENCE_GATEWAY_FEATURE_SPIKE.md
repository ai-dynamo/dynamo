# Spike: `features.inferenceGateway` (DGDR/DGD → operator wires EPP + GAIE)

Phase-1 of the IGW × global-router × planner integration. Goal: a single DGDR field stands up the
gateway path that today is wired by hand (the EPP note's Fix-1/Fix-4). **Status: CRD field added;
reconcile + DGDR→DGD wiring sketched below. Not yet built (`make generate`/`go build` pending — no Go
toolchain on the authoring box).**

## Phase-1 code landed (unbuilt — needs a Go box)
- `api/.../dynamographdeploymentrequest_types.go`: `Features.InferenceGateway` field + enum (DONE).
- `internal/dynamo/epp/httproute.go` + `httproute_test.go`: **typed** gateway-api HTTPRoute generator
  (mirrors the dynamo-gaie `http-router.yaml`), not unstructured — per sttts/tmonty12 typed-over-convenience.
- `internal/controller/dynamographdeployment_controller.go`: least-privilege `+kubebuilder:rbac` for
  `gateway.networking.k8s.io/httproutes` (HTTPRoute is namespaced → rides the existing controller-gen
  Role/ClusterRole path; no dedicated ClusterRole needed, unlike tmonty12's node-read case).
- `v1alpha1/...conversion.go`: listed `Features.InferenceGateway` as a v1beta1-only field (conversion contract).

**Required to build/land (do on a Go box):**
1. `go get sigs.k8s.io/gateway-api@<ver matched to inference-extension v1.2.0>` + `go mod tidy`
   (core gateway-api is NOT yet a module dep — this is the typed-dep cost we accepted).
2. `make generate` (deepcopy for InferenceGatewayFeature) + `make manifests` (CRD YAML + RBAC role.yaml).
3. Wire the reconcile hook (§3) + DGDR→DGD EPP injection (§2); add controller `_test.go` + run envtest +
   the DGDR conversion **fuzz** test.

## 1. CRD field — DONE
`api/v1beta1/dynamographdeploymentrequest_types.go`: added `FeaturesSpec.InferenceGateway
*InferenceGatewayFeature` + `InferenceGatewayFeature{Enabled, RoutingProfile, GatewayClassName,
GatewayName}` + `RoutingProfile` enum (`throughput|latency|balanced`). Mirror onto the DGD spec if the
feature should also be settable directly on a hand-written DGD (today the EPP is a DGD *component*, so
a DGD author already has the `services.Epp` path; the DGDR field is the new ergonomic entry point).

**Follow-up:** `make generate` (deepcopy for the new types) + `make manifests` (CRD YAML). The existing
`zz_generated.deepcopy.go` won't copy the new pointer field until regenerated.

## 2. DGDR → DGD generation: inject the EPP component (closes Fix-1)
The DGDR profiler generates the DGD (`ProfilingPhaseGeneratingDGD`; Python side in
`components/src/dynamo/profiler`). When `spec.features.inferenceGateway.enabled`, the generated DGD must get:
- a `services.Epp` component (`componentType: epp` + `eppConfig`) — the operator already turns this into
  ConfigMap + Service + InferencePool (`reconcileEPPResources`, controller L1913).
- the frontend/worker on `--router-mode direct` (frontend sidecar) so the EPP owns endpoint selection.
- `eppConfig` scorer weights derived from `RoutingProfile` (see §4).

This is the same shape the working `5.a` DGD-component path uses (validated end-to-end in the EPP note);
the new work is *auto-emitting* it from the DGDR feature instead of hand-authoring `services.Epp`.

## 3. Operator reconcile: also emit the HTTPRoute (closes Fix-4)
`reconcileEPPResources` already syncs ConfigMap (step 1) + InferencePool (step 2). Add **step 2.5** to
emit the `HTTPRoute` binding the public model to the InferencePool — the one GAIE resource still manual.

Dependency note: the operator module deps `gateway-api-inference-extension` (InferencePool/`gaiev1`) but
**not** core `sigs.k8s.io/gateway-api` (HTTPRoute). To avoid a new typed dep, emit the HTTPRoute as an
`unstructured.Unstructured` (GVK `gateway.networking.k8s.io/v1`, Kind `HTTPRoute`). Sketch:

```go
// deploy/operator/internal/dynamo/epp/httproute.go  (new)
func GenerateHTTPRoute(dgd *v1beta1.DynamoGraphDeployment, componentName string,
    eppConfig *v1beta1.EPPConfig, igw *v1beta1.InferenceGatewayFeature) *unstructured.Unstructured {
    poolName := GetPoolName(dgd.Name, eppConfig)
    gwName := igw.GatewayName            // if empty, operator-created Gateway (separate step)
    hr := &unstructured.Unstructured{}
    hr.SetGroupVersionKind(schema.GroupVersionKind{
        Group: "gateway.networking.k8s.io", Version: "v1", Kind: "HTTPRoute"})
    hr.SetName(dgd.Name + "-route"); hr.SetNamespace(dgd.Namespace)
    _ = unstructured.SetNestedMap(hr.Object, map[string]any{
        "parentRefs": []any{map[string]any{"name": gwName}},
        "rules": []any{map[string]any{
            "backendRefs": []any{map[string]any{
                "group": "inference.networking.k8s.io", "kind": "InferencePool",
                "name": poolName, "port": int64(8000)}}}},
    }, "spec")
    return hr
}
```
Hook in `reconcileEPPResources` (after the InferencePool sync, controller L1959):
```go
if igw := dgd.GetInferenceGatewayFeature(); igw != nil && igw.Enabled {
    hr := epp.GenerateHTTPRoute(dgd, componentName, eppService.EPPConfig, igw)
    if _, _, err := commoncontroller.SyncResource(ctx, r, dgd,
        func(ctx context.Context) (*unstructured.Unstructured, bool, error) { return hr, false, nil }); err != nil {
        return fmt.Errorf("failed to sync EPP HTTPRoute: %w", err)
    }
}
```
(Optionally emit a `Gateway` of `GatewayClassName` when `GatewayName` is empty — gate behind CRD presence,
like the existing Istio DestinationRule step 3.)

## 4. RoutingProfile → EPP scorer env
Map the enum to the EPP env the note documented (`DYN_OVERLAP_SCORE_WEIGHT`, `DYN_ROUTER_TEMPERATURE`,
`DYN_DECODE_FALLBACK`), set on the injected EPP component:
- `throughput` → high overlap weight (max KV reuse).
- `latency` → low overlap weight / load-dominant (min queueing; **also the spill-to-cold-replica
  behavior the planner needs — see risk below**).
- `balanced` → middle (default).

## 5. Build / validate plan
1. `make generate manifests` (operator dir) → deepcopy + CRD YAML; `go build ./...`; `go vet`.
2. Unit: extend `epp/inference_pool_test.go` with an `httproute_test.go` (golden object).
3. e2e (reuse the validated microk8s EPP path): a DGDR with `features.inferenceGateway.enabled: true`
   → assert the operator creates Epp component + InferencePool + HTTPRoute, and a `chat/completions`
   through the gateway returns 200 with `DynDecodeScorer` in the EPP log.

## Risks carried from measurements
- **Version-matched images** (operator↔EPP arg convention) — Phase-0 prereq; without it the EPP CrashLoops.
- **Autoscaling × KV-affinity**: when the planner scales a pool, the EPP must spill to the cold replica
  or the scale-up yields no goodput (measured: 100% to warm worker → 0 benefit). The `latency` profile
  (load-dominant) is the lever; Phase-4 wires the planner↔EPP loop and tests new-replica utilization.
- **HTTPRoute as unstructured** trades a typed dep for weaker compile-time checking — acceptable for the
  spike; revisit if the operator later needs richer gateway-api types.

---

## Design principles to follow (sttts + tmonty12) — and corrections to the above

Reviewed how **sttts** (apimachinery/conversion) and **tmonty12 / T. Montfort** (operator
correctness) write operator code; conforming this plan:

**From sttts (API discipline):**
- **Conversion-first.** v1beta1 is the DGDR hub; `Features.InferenceGateway` is a v1beta1-only field →
  must round-trip via sparse `annDGDRSpec` and be covered by `dynamographdeploymentrequest_legacy_fuzz_test.go`.
  *Done:* added it to the v1beta1-only list in `v1alpha1/dynamographdeploymentrequest_conversion.go`'s
  header contract. *Still required:* `make generate` (deepcopy) + run the conversion + fuzz tests; per
  `api/AGENTS.md`/`CONVERSION.md` an API change is incomplete without them.
- Typed conversion fns named after the type; constants centralized in `internal/consts/consts.go`;
  downgrade-compat annotations with `TODO(...)`+removal-version markers.

**From tmonty12 (operator correctness) — these change the plan:**
- **CORRECTION: emit the HTTPRoute (and Gateway) as TYPED objects, not `unstructured`.** He adds typed
  APIs (#9767/#9768) and avoids unstructured. → add `sigs.k8s.io/gateway-api` to the operator module and
  use typed `HTTPRoute`; pay the dep cost rather than weaken type-safety.
- **Gate the feature as experimental first.** New/risky APIs go under an `experimental` surface
  (cf. `spec.experimental.kvTransferPolicy`, `DynamoGraphDeploymentExperimentalSpec`). Mark
  `InferenceGateway` experimental until the planner↔EPP loop is proven.
- **Minimal, generated RBAC, both modes.** Emitting InferencePool/HTTPRoute/Gateway needs new verbs →
  add the *least-privilege* rules to `config/rbac/role.yaml` **and** a dedicated helm template that works
  in cluster-wide **and** `namespaceRestriction.enabled=true` modes (cf. #9879 `topology-label-rbac.yaml`).
- **Tight watch predicates.** The DGD reconcile should enqueue EPP/HTTPRoute work only when actionable
  (EPP component added/changed, InferenceGateway toggled), not on every pod event.
- **Test-paired + envtest.** Every new file gets a `_test.go`; validate exactly as his PRs do:
  `make manifests` · `KUBEBUILDER_ASSETS=… go test ./internal/controller ./internal/dynamo ./cmd` ·
  `go vet ./...` · `helm template … --show-only <new rbac>.yaml`.
- **Conventional, scoped commits** (`feat(operator): …`), conversion+tests **in the same PR**.

**Net correction to §3:** drop the `unstructured` HTTPRoute; use typed gateway-api + minimal RBAC +
experimental gating + tight predicates + envtest, conversion & fuzz landed together.


---

## Phase-1b update (this pass) — reconcile hook DONE + validated; injection scoped

**Reconcile hook — implemented + validated on Go 1.25.0:**
- `cmd/main.go`: register core gateway-api scheme `gatewayv1.Install(crdScheme)` (only `gaiev1` was
  registered — typed HTTPRoute sync would have failed at runtime without this; caught pre-build).
- `internal/consts/consts.go`: `KubeAnnotationInferenceGatewayName` — profiler→operator handoff
  (same pattern as `nvidia.com/current-worker-hash`).
- `internal/dynamo/epp/httproute.go`: `GenerateHTTPRoute(dgd, eppConfig, gatewayName string)`
  (refactored to take the gateway name from the annotation, not the DGDR type).
- `reconcileEPPResources`: step 2.5 emits the HTTPRoute via `SyncResource` when the annotation is set
  (absent ⇒ InferencePool still created, no route — safe for hand-authored EPP DGDs).
- Validated: `go build ./...` · `go vet ./internal/... ./cmd/...` · `go test ./internal/dynamo/epp/...` — green.

## Phase-2 update — DGDR→DGD producer IMPLEMENTED + unit-tested

This is the piece that makes `inferenceGateway.enabled: true` actually emit the
GAIE resources end-to-end (the consumer of the annotation the reconcile hook reads).
**Lives on the stacked branch `feat/dgdr-inference-gateway-producer` (off PR #9910), not in #9910 itself.**

**`profiler/utils/profile_common.py`:**
- `EPP_IMAGE_NAME` + `derive_epp_image()` (mirrors `derive_planner_image`).
- `is_inference_gateway_enabled(dgdr)` feature gate (mirrors `is_planner_enabled`).

**`profiler/utils/dgd_generation.py` — `add_inference_gateway_to_config(dgdr, config_dict)`**,
wired into `assemble_final_config` (gated on `is_inference_gateway_enabled`). Mutates the
v1alpha1 services-dict DGD in place:
  1. **`Epp` service** (`componentType: epp`) with inline `eppConfig.config` **EndpointPickerConfig**,
     built agg-vs-disagg from worker `subComponentType` (`disagg-profile-handler` + `label-filter`(s)
     + `max-score-picker` + `dyn-decode-scorer` [+ `prefill-filter`/`dyn-prefill-scorer` for disagg]);
     EPP image via `derive_epp_image`; env `DYN_MODEL_NAME` / `DYN_ENFORCE_DISAGG` /
     `DYN_KV_CACHE_BLOCK_SIZE` (block size read from worker `--block-size`, else agg/disagg default).
  2. **`frontendSidecar`** (`-m dynamo.frontend --router-mode direct`) added to **every** worker; the
     standalone `Frontend` service is dropped (gateway+EPP now own routing).
  3. **Annotation** `nvidia.com/inference-gateway-name` set when `gatewayName` is supplied (drives the
     reconcile hook). No name ⇒ EPP InferencePool only, no HTTPRoute (logged) — the operator does not
     yet create a Gateway from `gatewayClassName`.
  4. **`RoutingProfile` is now a real consumer** — maps to Dynamo KV-router env knobs the EPP's FFI
     scorer reads (`lib/bindings/c/src/lib.rs`): `throughput` → high
     `DYN_ROUTER_KV_OVERLAP_SCORE_CREDIT` (1.0) + low `DYN_ROUTER_PREFILL_LOAD_SCALE` (0.5) [pack onto
     cache-warm workers]; `latency` → 0.5 / 2.0 [spread to cut queueing]; `balanced` → router defaults
     (no env). Magnitudes are initial heuristics, not tuned.
- Also gated the vLLM self-benchmark step on `planner` (its only consumer) so an IGW-only deployment
  doesn't spuriously set `DYN_BENCHMARK_MODE`.

**Tests:** `tests/unit/test_dgd_generation_inference_gateway.py` — 13 cases (agg/disagg EPP config,
routing-profile env, sidecar conversion + Frontend removal, annotation handoff, disabled = no-op).
All green; no regression in the existing profiler unit suite.

**Controller coverage (this pass):** `dynamographdeployment_epp_route_test.go` — fake-client tests of
`reconcileEPPResources` itself (the repo's dominant controller-test style; the Ginkgo envtest suite
lacks the gateway-api/gaie CRDs). Asserts: annotation set ⇒ HTTPRoute emitted, bound to the named
(cross-namespace) Gateway, and owner-referenced by the DGD; annotation absent ⇒ InferencePool created
but no HTTPRoute. Green on Go 1.25.0.

**Still deferred:** real-silicon EPP e2e (needs the `epp-image` + GPUs); cross-namespace gateway via a
DGDR field (operator supports it, DGDR doesn't expose a gateway namespace yet); operator-side Gateway
*creation* from `gatewayClassName`.
