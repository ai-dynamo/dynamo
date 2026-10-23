# Operator upgrade contract: immutable historical baselines

This suite implements historical operator takeover checks from #16624 / #16623
and the fixture coverage tracked in #16650. Each candidate restores the saved
resource graph, runs production admission/controllers, and checks that an
operator-only upgrade preserves existing workloads.

## Immutable inputs and separate expected outputs

`history/<origin>/.../input.yaml` is a frozen observation of an older operator's
persisted resources. Passing an upgrade check must never replace that input with
the candidate's output. Every future candidate starts from the same saved state.
Adding a 1.6 baseline does not route the existing 1.4 and 1.5 inputs through 1.6
first or remove their direct takeover coverage.

| Historical input | Origin | Operator that produced the saved state | Fixed runtime |
|---|---|---|---|
| sglang-1.4 | 1.4.0 | 1.5.0, after the original bootstrap takeover | official 1.4.0 |
| sglang-1.5-custom | 1.4.0 | 1.5.0, after the original bootstrap takeover | custom v1.5.0-9aa3-repro |
| sglang-1.5 | 1.5.0 | 1.5.0 | official 1.5.0 |
| 13 new 1.6 matrix scenarios | 1.6.0 | published 1.6.0 | fixed per fixture |

The original three inputs and their goldens remain byte-for-byte unchanged by
this capture. The first two were already reconciled by 1.5 in the original
bootstrap; this does not certify the known-bad custom-tag transition from
1.4 to 1.5 documented in #16623. No production hash migration is implemented here.

`output.yaml` is the separately reviewed expected candidate result. A golden
mismatch writes `output.yaml.new`. Updating an expected output cannot authorize
changes to the independent workload safety invariants or to historical inputs.
`latest/` exercises ordinary CREATE admission in the development operator and
retains its development inputs and goldens when a release is captured.

## Independent safety and complete resource checks

| Check | Purpose | Can updating output.yaml make it pass? |
|---|---|---|
| Complete owned resource golden | Review persisted lowering differences, including new resource kinds | Yes, after semantic review |
| Upgrade safety assertion | Preserve generation selection, active identity, effective templates and rollout intent | No |

Before controllers start, the suite reads the production transition's
`hashChanged` decision. After a completed reconciliation, it compares identities,
DGD origin, rollout intent, and complete effective Grove clique templates,
including the frontend. Native Component scenarios additionally protect workload
generation and spec. The test name referring to worker templates does not exclude
frontend cliques. An intentional worker edit must still reach a new generation.

The captured ownership closure includes DGD, PCS, Services, ServiceAccount, Role,
RoleBinding, and any other operator-owned kinds discovered from the API. Native
Component scenarios include DCDs and their immediate workloads. Expected output
does not determine which kinds are observed. Kubernetes/Grove descendants are
excluded. Server metadata, Service addresses and condition timestamps are
normalized only as documented by the matcher; historical origins, worker hashes
and namespace suffixes remain concrete.

## Storage and production lifecycle

Run against an exclusively owned cluster without installed controllers or
admission webhooks. The `upgradecontract` build tag provides a separate TestMain.

1. Create historical namespaces with their saved names, which affect hashing.
2. Temporarily install permissive schemas for historical custom resources.
3. Restore unstructured specs/status, reconstruct owners, and reallocate only
   server-assigned metadata and Service addresses.
4. Read back the restored state before takeover.
5. Restore production CRDs and start production conversion/admission webhooks.
6. Start namespace-scoped production DGD/DCD controllers and wait for completed
   reconciliation, including the DCD completion barrier for Component scenarios.
7. Compare the complete owned graph and independent safety invariants.

Runtime restoration changes UIDs and Service addresses on the temporary API
server; none of those observations are written back to existing input files.
ResourceQuota blocks Pod and ReplicaSet creation. Grove workload actuation is
absent. This is CPU admission/controller/API-server evidence, not GPU serving,
Pod-replacement or live-traffic evidence.

## Running

Go runs the operator webhooks/controllers in process. The clusterenv Python proxy
connects the API server to those webhooks. Helm, kubectl, Go, Docker/Kind and
PyYAML are prerequisites; model or operator container builds are unnecessary.

```bash
make -C deploy/operator clustertest-upgrade-contract
```

The wrapper creates a uniquely named Kind cluster with the pinned node image,
uses a temporary kubeconfig, installs CRDs, and removes only its own cluster on
completion, failure or interruption. It refuses to reuse an existing cluster.
For an already prepared exclusively owned cluster, set `KUBECONFIG` and
`DYNAMO_CLUSTERTEST_CONTEXT` explicitly and use
`clustertest-upgrade-contract-in-cluster`. The in-cluster targets never select
an implicit current context. CI uses the same replay and retains logs and `.new`
differences.

## Provenance and release boundaries

`bootstrap.json` records the original 1.4/1.5 capture sources and exact input
hashes. `RECONCILED_VERSION` remains 1.5.0 and describes only that original
bootstrap state. It is not advanced by later capture or verification runs.

`history.lock.json` contains:

- `bootstrap`: the unchanged original lock, including its 1.5 source and hashes.
- `captures`: separate published-release tag/SHA and input hashes for each newly
  captured baseline. A capture owns only the new origin's input paths.
- `inputs`: the complete hash inventory, equal to the disjoint union of the
  original bootstrap inputs and all subsequently appended captures.

Checks reject changed input bytes, overlapping ownership, missing provenance,
and provisional candidates published as formal history. The next-minor
appVersion bump requires a captured baseline from the immediately previous minor;
it does not require overwriting older inputs with that minor's reconcile output.
All historical origins continue to be replayed directly. Same-minor recapture,
skipped-minor bumps and major bumps remain unsupported.

`history/N` must be observed using an actual published stable upstream tag for
operator N. Main's chart reporting N is insufficient because its production
code may differ from the release. Formal capture verifies the local tag against
the published, non-draft, non-prerelease GitHub release and rejects tracked or
untracked production changes. `gh` access is required for capture, not ordinary
checks. The selected module runs with `GOWORK=off`.

## Scoped v1.6.0 capture

`capture-v1.6.0.json` freezes 19 input paths and hashes from main revision
`75d0686fb7def243ab5a07ec69159427b169675c`: the three immutable existing history
inputs and 16 latest inputs. The latter cover 13 of the 27 canonical matrix paths
plus three legacy variants used only for validation. The 14 missing canonical
paths are explicitly listed.

This exact partial inventory is accepted only with stable tag `v1.6.0`, release
revision `701da25292ecca8f64d770cc0b17d389200019a1`, chart version 1.6.0, and the
original bootstrap marker 1.5.0. The capture records the scope under
`captures["1.6.0"].captureInventory`. Existing historical inputs are replayed for
safety; their temporary reconciled snapshots are discarded. `promotedScenarios`
explicitly selects only the 13 available canonical matrix cases for publication.
The three legacy latest cases (`sglang-1.4`, `sglang-1.5`, `sglang-1.5-custom`)
remain CREATE coverage; they are not duplicated under the new origin. Only 13
newly admitted matrix graphs and their goldens are appended. Future captures
likewise publish only scenarios in the release matrix catalog.

This does not certify missing cases. Later releases and provisional captures
still require the complete matrix from `required-matrix.json`: 24 component/Grove
cases plus three multi-node DisaggregatedSet cases. A subsequent 1.7 bump can
consume the captured 1.6 baseline; a 1.7 capture still requires complete coverage,
and a 1.8 bump still requires a published 1.7 baseline. Keep #16650 open.

To repeat the scoped capture, prepare a dedicated published-release checkout and
carry only the reviewed test harness. Start with the frozen pre-capture inventory:

```bash
harness=$(git rev-parse HEAD)
inventory=75d0686fb7def243ab5a07ec69159427b169675c
release_tree=../dynamo-release-capture
git fetch origin tag v1.6.0
git worktree add --detach "$release_tree" v1.6.0
git -C "$release_tree" restore --source="$harness" -- \
  deploy/operator/Makefile deploy/operator/hack \
  deploy/operator/internal/controller/clusterenv_helpers_test.go \
  deploy/operator/internal/controller/clusterenv_test.go \
  deploy/operator/internal/controller/clusterenv_main_test.go \
  deploy/operator/internal/controller/upgrade_clustertest_test.go \
  deploy/operator/internal/controller/upgrade_contract_test.go \
  deploy/operator/internal/controller/upgrade_component_clustertest_test.go \
  deploy/operator/internal/controller/upgrade_component_reconcile_test.go \
  deploy/operator/internal/controller/upgrade_component_test.go \
  deploy/operator/internal/controller/upgrade_grove_matrix_test.go \
  deploy/operator/internal/testing/clusterenv \
  deploy/operator/internal/testing/golden
git -C "$release_tree" restore --source="$inventory" -- \
  deploy/operator/internal/controller/testdata/upgrade
git -C "$release_tree" restore --source="$harness" -- \
  deploy/operator/internal/controller/testdata/upgrade/capture-v1.6.0.json
git -C "$release_tree" apply deploy/operator/hack/v1.6.0-capture-harness.patch
make -C "$release_tree/deploy/operator" clustertest-upgrade-fixtures-regen \
  UPGRADE_RELEASE_TAG=v1.6.0
```

The release-only patch omits the post-cut v1beta2 test scheme, uses v1.6.0's DCD
watch/index wiring with the existing completed-reconcile recorder, and removes
the frontend startup probe absent from that release from 13 release-checkout
latest goldens. Review of the release run showed precisely that frontend probe
and its dependent PCS spec hash difference; worker templates were unchanged.
The PCS hash binding is calculated from the reviewed golden, never wildcarded.
Production sources, API types, CRDs, chart, dependencies and safety assertions
remain at the selected release. Do not apply the patch to the development tree.

Publication is atomic and happens only after all release replay, CREATE and
safety checks pass. It preserves every existing input and golden byte-for-byte
and keeps the bootstrap marker. Copy back only `history/1.6.0/` and the updated
`history.lock.json`, then replay the development operator. Do not copy temporary
recaptures over older origins. Development `latest` inputs/goldens are unchanged.

Candidate evidence can be collected using a full `UPGRADE_CANDIDATE_SHA` and a
new `UPGRADE_CANDIDATE_OUTPUT` directory outside the fixture tree. It records
`sourceKind=release-candidate` and never publishes baseline history, modifies the
lock, or changes the bootstrap marker.

## Original bootstrap reproduction

The original sources are 1.4.0 `03014943323e78feb5bd672ef08b72caea0918ac`
and 1.5.0 `b83b1d9304ebfc624709ac46db32b1b6f1ff1615`. The bootstrap retained the
same API server while 1.5 took over 1.4, using production admission/controllers.
The release-specific setup adaptation supplies the ScaleClient needed by 1.4;
it does not synthesize lowering or worker identity.

With PyYAML and envtest assets installed:

```bash
KUBEBUILDER_ASSETS=/path/to/envtest/binaries \
  python3 deploy/operator/hack/bootstrap_upgrade_fixtures.py --output /new/staging/directory
```

This independently writes staging evidence and never replaces committed history.
Normal CI does not fetch or compile old operators.

## Known development takeover failure

At development revision `75d0686fb7def243ab5a07ec69159427b169675c`, #15581 adds
a frontend startup probe for runtime 1.6.0 and later, absent from the published
1.6 operator's captured state. The 12 new canonical Grove histories trigger the
independent template-safety assertion when main adds this probe. The new native
Component history also fails to converge after a frontend Deployment update.
These findings do not establish that inference worker Pods were replaced: Grove
actuation and serving Pods are absent from this suite.

Keep the captured inputs and safety assertions intact. Accepting new goldens
cannot authorize this template mutation. Resolve the production compatibility
policy separately before merging this capture and the subsequent version bump.
