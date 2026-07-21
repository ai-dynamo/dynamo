# Operator Env Design

`operatorenv` provides a Kubernetes API server for operator tests. It can run
production or test-specific admission and conversion webhook handlers, but no
controllers until an individual test starts one.

## Lifecycle

`Env` owns either a shared envtest process or an isolated process:

```go
var sharedEnv = operatorenv.New(operatorenv.Options{
    Admission: operatorenv.AdmissionWebhooks{
        Mutating:   true,
        Validating: true,
    },
    SetupWebhooks: setupProductionWebhooks,
})

func TestMain(m *testing.M) {
    os.Exit(sharedEnv.RunM(m))
}

func TestSomething(t *testing.T) {
    env := sharedEnv.ForTest(t)
    // ...
}
```

`RunM` starts the shared API server lazily, on the first `ForTest` call, and
stops it after the package test run. Each `ForTest` creates and cleans up a
unique namespace. This keeps API-server startup cheap without sharing test
objects. `Client` remains cluster-wide; tests use `Namespace` for their
namespaced fixtures, while `StartManager` restricts its cache to that namespace.

`RunT` starts an isolated API server for a test that cannot share a cluster:

```go
func TestIsolated(t *testing.T) {
    env := operatorenv.New(operatorenv.Options{
        Admission:     operatorenv.AdmissionWebhooks{Validating: true},
        SetupWebhooks: setupTestWebhooks,
    }).RunT(t)
    // ...
}
```

## Webhooks

The envtest API server always configures CRD conversion webhooks from the
registered API versions. It also renders the production Helm webhook
configuration and installs the selected mutating and validating webhook
objects. A dedicated webhook manager invokes the required `SetupWebhooks`
function, which normally registers the production handlers via
`webhooksetup.Setup`, including conversion endpoints. Therefore normal `Client`
CRUD reaches the API server, CRD CEL validation, and production webhook code.

The webhook manager is separate from controller managers. It runs for the
lifetime of the environment and is the only always-on manager.

## Controller Managers

Tests start only the controller they need:

```go
env.StartManager(func(mgr ctrl.Manager) error {
    return controller.SetupDynamoGraphDeployment(mgr, options)
})
```

`StartManager` limits its cache to the test namespace and stops the manager at
test cleanup. Controller setup functions belong to `internal/controller`; the
test package assembles their dependencies from `TestEnv` to avoid an import
cycle.

## Scope

`operatorenv` owns API-server, namespace, webhook, and controller-manager
lifecycle. It does not own YAML fixture loading or expected-manifest matching.
Those APIs should be introduced with the first controller test that uses them.
