# External plugin — reference implementation

A minimal, self-contained Python external plugin server that
implements the four stage contracts (Predict / Propose / Reconcile /
Constrain) defined in `plugins/proto/v1/plugin.proto`. It is the
canonical starting point for users writing their own external plugin
and is also used as the cross-process fixture for K8s smoke
validation.

## What you get

`reference_runner.py` exposes one stage per invocation:

| `--stage`   | Servicer                       | Fixed response                       |
|-------------|--------------------------------|--------------------------------------|
| `predict`   | `PredictPluginServicer`        | `PredictionData(num_req, isl, osl)`  |
| `propose`   | `ProposePluginServicer`        | `OverrideResult(prefill, decode)`    |
| `reconcile` | `ReconcilePluginServicer`      | `OverrideResult(prefill, decode)`    |
| `constrain` | `ConstrainPluginServicer`      | `AT_MOST(prefill, decode)` ceilings  |

The fixed values are configurable via CLI flags so a single binary
can stand in for any stage in a smoke deployment.

## Run locally

```bash
python -m dynamo.planner.examples.external_plugin.reference_runner \
    --listen=0.0.0.0:9099 \
    --stage=predict \
    --plugin-id=ext-predict \
    --predict-num-req=4242 \
    --predict-isl=1024 \
    --predict-osl=256
```

The server logs `listening on 0.0.0.0:9099` and waits for the
planner to dial it. SIGTERM and SIGINT shut down cleanly.

## Run in K8s

The intended K8s deployment shape is one Pod per stage, each running
this binary with the appropriate `--stage`, and the planner registering
all four via the static `external_plugins:` config block — exercising
the full PREDICT / PROPOSE / RECONCILE / CONSTRAIN pipeline over real
cross-pod gRPC. The ready-to-`kubectl apply` fixture is deferred to a
follow-up PR; the four `--stage` invocations above are the building
blocks.

## Forking to a real plugin

For each stage you want to serve:

1. Copy `reference_runner.py` to your package.
2. Replace the fixed response inside the corresponding
   `_Deterministic{Predict,Propose,Reconcile,Constrain}Plugin`
   class with your real logic (model inference, policy evaluation,
   historical lookups, etc.).
3. Keep the proto contract unchanged — return the same Pydantic
   types in the same shape.
4. Build a container image and deploy under the same
   `external_plugins:` config block in your planner deployment.

The proto schema (`plugins/proto/v1/plugin.proto`) and Pydantic
mirror (`plugins/types.py`) define everything you may return.
`plugins/proto/v1/README.md` covers the schema-evolution policy and
the contract invariants per stage (such as the spec-ignored
`final` flag on CONSTRAIN responses, or chain-augment ordering for
PREDICT).

## Why a reference is shipped at all

The plugin framework is contract-driven (proto), so any plugin in
any language that speaks the proto can register. We ship the Python
reference because (a) it doubles as the cross-process fixture for
the framework's own K8s smoke validation and (b) most early users
will start in Python — having a known-good baseline to diff against
makes the first day a lot shorter.
