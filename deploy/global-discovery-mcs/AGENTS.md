<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# AGENTS.md - global-discovery-mcs

The Multi-Cluster Services (KEP-1645) layer of the Dynamo Global Router:
`mcs-registry`, `mcs-controller` and `mcs-admin`. The Global Discovery
Service (`lib/global-discovery-service`) consumes the EndpointSlices the
controller imports. Read `README.md` for the layout, flags and operations,
`docs/api.md` for the API, `docs/consistency.md` for the view,
write-through and lease model, and `docs/mcs-controller.md` for the
controller.

## Build and test

Run from this directory; the Makefile sets `GOWORK=off` and installs every
tool into `bin/` at a pinned version.

```bash
make build             # bin/mcs-registry, bin/mcs-controller, bin/mcs-admin
make test              # unit and envtest suites, -race; coverage of ./internal/... must stay at or above 85 percent
make envtest           # the mcs-controller envtest suite alone
make test-integration  # DynamoDB store and mcs-admin against dynamodb-local; needs Docker
make e2e               # kind end-to-end test (test/e2e, see its AGENTS.md); needs Docker, kubectl, helm, jq
make vet
make lint              # golangci-lint v2, integration tests included; make lint-fix applies fixes
make proto             # after editing api/proto
make proto-check       # buf lint and a diff of api/gen against a fresh generation
make manifests         # after changing +kubebuilder:rbac markers
make sync-crds         # after changing the sigs.k8s.io/mcs-api version
make tidy-check        # fails when go mod tidy would change go.mod or go.sum
make check             # tidy-check, manifests, proto, sync-crds and check-chart-crds, then fails on a dirty tree
go run ./cmd/mcs-registry --log-level=debug   # memory store on :9090, :8080, :9091
```

Run `make lint test check` before committing. `make test-integration` skips
without Docker; with colima it takes `DOCKER_HOST` from the docker context.

## API contract

- Edit `api/proto/dynamo/mcs/v1/*.proto`, run `make proto` and commit the
  regenerated `api/gen`. Never edit `api/gen` by hand. `make proto` adds the
  SPDX header to the generated Go files.
- Protos follow the repository's clang-format style (pre-commit formats
  them). `api/proto/google/api` is vendored; leave it alone.
- Changes are additive only: new fields, messages, enum values and RPCs.
  Never renumber, rename or remove. A breaking change needs a `v2` package.
- `buf lint` (STANDARD) must pass. Every RPC has its own request and response
  message.
- Keep `docs/api.md` and `api/gateway_test.go` in step with REST paths.
- Label, annotation, finalizer, reason and limit values in `api/` are a
  wire contract with deployments and with consumers of the imported
  EndpointSlices, among them the Global Discovery Service. Change every
  side together.

## Registry code

- Handlers in `internal/api` validate, write to the store, then apply the
  stored record to the view (`view.Apply*`) before responding. Reads come
  from the view only. Return errors through the helpers in
  `internal/api/errors.go` so every error has an ErrorInfo.
- Derived objects in `internal/view` are immutable once published. Never
  modify a message returned by the view; clone it first. When a proto
  message gains a field, update `liteImport` (a test fails otherwise).
- Merge rules live in `internal/registry/merge` with their test
  cases; keep them in step with the `sigs.k8s.io/mcs-api` reasons.
- Time comes from `internal/clock`; tests use `clock.NewFake` and
  `BlockUntil` before advancing.
- A new store backend registers its opener in `internal/storefactory` and
  its admin commands with `registerCommand` in a new file of
  `cmd/mcs-admin`.

## Store

- Every `Store` implementation must pass `storetest.Run`. Add a conformance
  case for any new store behaviour instead of testing one implementation.
- Store record types mirror the proto messages but use Kubernetes spellings
  for enums. `internal/convert` maps them.
- Writes are conditional; see the package comment of `internal/store`.
- DynamoDB store: `docs/storage.md` is the item catalogue and condition
  table. Change it together with `keys.go`, `codec.go` and `store.go`.
  Table definition changes go to `schema.go`,
  `internal/store/dynamodb/testdata/create-table.json` and
  `docs/storage.md` together.
- The unit tests run the DynamoDB store against an in-memory fake of the
  DynamoDB API (`fake_test.go`). It rejects unused expression placeholders,
  as DynamoDB does, but it does not prove DynamoDB semantics: run
  `make test-integration` after any change to a request or condition.
- Integration tests carry the build tag `integration`; `make lint` and
  `make vet` include it.

## Client and fake registry

- `pkg/client` is a public API that other modules may import. Keep changes
  additive and keep `pkg/client/README.md` in step.
- `pkg/fakeregistry` is the test server for code that uses the client. When
  the API semantics change (error reasons and details, watch event shape),
  update the fake and its tests in the same change.
- Client tests run against `pkg/fakeregistry` over loopback listeners and
  bufconn with the race detector. The coverage gate of `make test` covers
  `./internal/...` only; keep `pkg/` tests thorough anyway.

## mcs-controller

`cmd/mcs-controller` and `internal/mcs/*`; behaviour, flags, RBAC and
metrics are in `docs/mcs-controller.md`. Keep that page in step with the
code.

- Run `make envtest` after changing the export or import logic. The
  controller-runtime fake client differs from an API server in ways that
  matter here: it fills apply configurations with the answer of `Apply`
  (the real client does not), it does not separate status from main
  resource field ownership under server-side apply, and it allocates no
  ClusterIPs or UIDs. Unit tests alone hide bugs of that kind.
- The envtest suite (`internal/mcs/integration`) skips without
  `KUBEBUILDER_ASSETS`; `make test` and `make envtest` set it from
  setup-envtest (`ENVTEST_K8S_VERSION` in the Makefile).
- `config/rbac/role.yaml` is generated: change the `+kubebuilder:rbac`
  markers, run `make manifests` and commit `config/rbac/role.yaml` and the
  chart copy `../helm/charts/global-discovery-mcs/files/controller-role.yaml`.
  The chart test `tests/rbac_test.yaml` pins the rules as a golden list so
  RBAC changes get reviewed; update it in the same change. Leader election
  RBAC belongs to the chart.
- CRDs are not generated here: ServiceExport and ServiceImport are copied
  from `sigs.k8s.io/mcs-api` by `make sync-crds` (a unit test fails while
  the copies differ from the `go.mod` version), and the ClusterProperty CRD
  is kept by hand in `config/crd/bases`. `config/crd` embeds the directory.
  The chart keeps a copy under `files/crds`; after `make sync-crds` run
  `make -C ../helm/charts/global-discovery-mcs sync-crds` (`make check`
  fails while the copies differ).
- Events go through the `events.k8s.io` API (`mgr.GetEventRecorder`); tests
  use `events.NewFakeRecorder`. Requeue with `RequeueAfter`, never
  `Requeue`.
- Label, annotation, finalizer, field manager and condition values come
  from `api/labels.go` and `api/errors.go`.
- Everything that writes to the registry or applies imports is
  leader-only. One controller owns all export records of its cluster id: it
  deletes records without a local ServiceExport on every resync.
- Use `internal/mcs/mcstest` (a fake registry with a client, a fake
  Kubernetes client with the controller scheme, object builders, metric
  reads) in tests.

## Conventions

- Names: binaries, metrics, flags and help text say registry, MCS or
  mcs-controller. "Global Discovery Service" means only the Rust service in
  `lib/global-discovery-service`.
- Metrics: `dynamo_mcs_registry_*` (`internal/metrics`),
  `dynamo_mcs_controller_*` (`internal/mcs/metrics`) and
  `dynamo_mcs_client_*` (`pkg/client`), with pre-initialised counters. Add
  new label values to the initialisation of the package.
- Spans: `mcs.store.*`, `mcs.view.*` and `mcs-controller.*`, with the
  attributes `dynamo.clusterset` and `dynamo.cluster.id`.
- Flags use the standard `flag` package. Every `mcs-registry` and
  `mcs-admin` store flag has an `MCS_` environment fallback
  (`internal/config`). The controller's flags (`internal/mcs/config`) read
  only `MCS_REGISTRY_URLS` and `POD_NAMESPACE`. A new flag needs a row in
  the `README.md` flag tables.
- Logs: zap JSON. Do not log and return the same error; the gRPC logging
  interceptor logs every failed call. The controller logs through
  controller-runtime's logr (zap); controller-runtime logs reconcile
  errors.
- Timestamps given to the store are server time; leases are evaluated at
  read time with `registry.LeaseExpired`.
- Every file starts with the dated two-line SPDX header (`goheader` checks
  Go files); imports are grouped standard, third party, then this module
  (`gci`). Markdown stays plain ASCII.
