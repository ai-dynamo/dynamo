# Cluster Env Design

`clusterenv` supports controller tests that need a complete Kubernetes control
plane or need to run Pods and Jobs. It connects to an externally managed
cluster. It does not create clusters, install application images, or deploy the
operator chart.

## Cluster Selection

Cluster tests may create cluster-scoped webhook registrations and temporarily
rewrite conversion webhooks. A caller must therefore explicitly unlock the
kubeconfig context:

```text
DYNAMO_CLUSTERTEST_CONTEXT=kind-dynamo-clustertest-v136
```

The value must be non-empty and name an existing context. `clusterenv` loads
exactly that context and never falls back to kubeconfig's `current-context`.
This check happens once, when an `Env` first loads its REST configuration.

## Package And Test Lifecycle

An `Env` is normally shared by one test package:

```go
var clusterEnv = clusterenv.New(clusterenv.Options{
    Scheme:        scheme,
    SetupWebhooks: setupWebhooks,
})

func TestMain(m *testing.M) {
    os.Exit(clusterEnv.RunM(m))
}

func TestController(t *testing.T) {
    env := clusterEnv.RunT(t)
    env.BlockWorkloads()
    env.StartManager(func(mgr ctrl.Manager) error {
        return controller.SetupDynamoGraphDeployment(mgr, options)
    })
}
```

`RunM` owns the package-wide cluster connection and webhook runtime. They start
lazily on the first `RunT`, so packages with no selected cluster tests do not
touch a cluster. `RunT` creates a unique namespace and one test environment.
`StartManager` starts only the controller or contiguous controller chain
explicitly selected by that test and limits its cache to the test namespace.

The client is intentionally not namespace restricted. Tests must put all
namespaced fixtures in `TestEnv.Namespace()` and must not mutate unrelated
cluster-scoped resources. During cleanup, the selected controller remains
active while its namespace is deleted so it can process deletion and remove its
own finalizers. The manager stops after namespace deletion.

## In-Process Webhooks

The shared webhook manager runs production admission and conversion handlers in
the Go test process. Because the cluster API server cannot normally connect to
the developer machine or CI runner, `clusterenv` creates a small proxy Pod and
Service in a dedicated namespace. A client-go port-forward and reverse TCP
tunnels connect that Service to the local TLS webhook server.

Admission registrations are rendered from the production operator Helm chart.
The harness changes only their names, service references, and CA bundles.
Conversion references in the checked-in Dynamo CRDs are temporarily pointed at
the same proxy and restored when `RunM` finishes. `AdditionalAdmission` supports
dependency webhooks; the controller suite uses the production LWS handler and a
focused registration generated from its kubebuilder markers. Dependency
adapters live under `internal/testing/mocks`: `mocks/lws` wraps the production
LWS handlers, while `mocks/grove` temporarily provides transparent
PodCliqueSet handlers at Grove's production paths. Grove CRD schema validation
and OpenAPI defaulting still run in the API server; the transparent handlers
are replaced by Grove's production `Setup` API once it is released.

The webhook manager is independent of all per-test controller managers. No
operator controller runs merely because a test uses admission or conversion.
The proxy image is configurable with `WebhookProxyImage`; the controller suite
reads `DYNAMO_CLUSTERTEST_WEBHOOK_PROXY_IMAGE` and otherwise uses
`python:3.12-alpine`. Making that image available to the cluster remains
external setup.

## Manifest Boundary

Manifest contracts cover the full relevant Dynamo controller chain through the
terminal workload API. They include both intermediate Dynamo resources and the
resulting Deployment, LeaderWorkerSet, or Grove manifests in one golden file;
checking only the next controller boundary makes the effective workload harder
to review. Tests install the LWS, Grove, and Volcano PodGroup CRDs but do not run
their controllers.

`BlockWorkloads` creates a namespace-scoped `ResourceQuota` with zero allowances
for ReplicaSets and Pods. `BlockReplicaSets` blocks only ReplicaSets so a
Job-backed test can still run its Pods. Both wait for the quota controller to
publish the active limits before returning.

This lets the Dynamo controller store Deployments with their real replica
counts, LeaderWorkerSets, and Grove resources while preventing downstream
actuation. Tests that intentionally run Jobs or Pods, such as the DGDR profiler
test, use `BlockReplicaSets` instead.

## External Setup

`make setup-clustertest` creates or reuses the named Kind cluster, writes a
dedicated kubeconfig under `bin/`, installs the Dynamo, LWS, Grove, and Volcano
PodGroup CRDs, and builds and loads the local profiler image. New clusters use
the pinned Kubernetes 1.36 node image. The profiler image layers the current
profiler, planner, common, and deployment utility sources over the matching
released planner image so the real AIC profiler runs without rebuilding its
unchanged runtime dependencies.

```bash
make setup-clustertest
KUBECONFIG=bin/clustertest.kubeconfig \
  DYNAMO_CLUSTERTEST_CONTEXT=kind-dynamo-clustertest-v136 \
  DYNAMO_CLUSTERTEST_PROFILER_IMAGE=dynamo-planner:clustertest \
  make clustertest
```

`make envtest` runs the API-only suite, `make clustertest` runs the Kind-backed
suite, and `make integration` runs both after preparing the standard Kind
cluster and its profiler image. The operator pre-merge job runs
`make integration` directly.
