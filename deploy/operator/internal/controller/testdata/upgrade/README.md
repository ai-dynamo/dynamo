# Operator upgrade contract: successive historical takeover

This suite implements historical operator takeover checks from #16624 / #16623
and the fixture coverage tracked in #16650. Each candidate restores the saved
resource graph, runs production admission/controllers, and checks that an
operator-only upgrade preserves existing workloads.

## Fixed origins and separate expected outputs

`history/<origin>/.../input.yaml` records an origin's persisted resources after
reconciliation by the last released operator. A formal release capture advances
all active histories through that release, preserving their origin and runtime.
Development candidate checks read these inputs without replacing them, so each
candidate starts from the same release-reconciled state.

| Historical input | Origin | Operator that produced the saved state | Fixed runtime |
|---|---|---|---|
| sglang-1.4 | 1.4.0 | published 1.6.0, after the 1.5 bootstrap | official 1.4.0 |
| sglang-1.5-custom | 1.4.0 | published 1.6.0, after the 1.5 bootstrap | custom v1.5.0-9aa3-repro |
| sglang-1.5 | 1.5.0 | published 1.6.0 | official 1.5.0 |
| 13 new 1.6 matrix scenarios | 1.6.0 | published 1.6.0 | fixed per fixture |

The first two inputs had already been reconciled by 1.5 in the original
bootstrap. Advancing those saved states through 1.6 does not certify the known-bad
custom-tag transition from 1.4 to 1.5 documented in #16623. No production hash
migration is implemented here.

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
excluded. Expected outputs normalize server metadata, Service addresses and condition
timestamps as documented by the matcher; historical origins, worker hashes and
namespace suffixes remain concrete.

## Multinode Component and DisaggregatedSet coverage

The canonical suite includes aggregated and disaggregated multinode Component
scenarios for vLLM, SGLang, and TensorRT-LLM, plus one disaggregated multinode
DisaggregatedSet scenario per engine. Component scenarios stop at DCD-owned
LeaderWorkerSets; DisaggregatedSet scenarios stop at the DisaggregatedSet itself
and the frontend's auxiliary DCD/Deployment. The LWS/DS controllers and serving
Pods do not run. Dependency CRDs and production admission handlers come from the
pinned LWS module already used by the shared clustertest setup.

Independent checks enforce the selected provider, component/role inventory and
multinode shape. Historical safety also protects the complete DisaggregatedSet
spec, identity and generation. Intentional worker edits must reach every edited
component and both leader/worker templates, including embedded DS role templates.
The populated graph replay controls cover Deployment, LWS and DS graphs without
claiming that these freshly captured test graphs came from an older release.
Formal released history still advances only through the pinned-release workflow.

Canonical matrix completeness and the missing-origin historical case are tracked
in [#16650](https://github.com/ai-dynamo/dynamo/issues/16650). The required
release-matrix gate remains enforced as the remaining fixtures are added.

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
server. Development checks do not write those observations back to input files;
formal release capture publishes the resulting graphs after all checks pass.
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

## Stable capture metadata

The Go capture writer uses the same YAML formatting for recapture and promotion.
Before input hashes are computed, it replaces only top-level `metadata.uid` and
owner-reference UIDs with `00000000-0000-0000-0000-000000000000`,
`metadata.resourceVersion` with the string `"1"`, and `metadata.creationTimestamp`
with `"2000-01-01T00:00:00Z"`. It omits `metadata.managedFields`. These are concrete
values, not golden directives. Generation, generation annotations, origins,
hashes, workload templates and status remain unchanged by recording.

Snapshot restoration identifies owners by namespace, API version, kind and name,
then replaces their placeholder UIDs with the referenced objects' actual API
UIDs. Missing owners and duplicate object identities are rejected. Several
objects may share the same recorded UID without losing their ownership graph.

## Provenance and release boundaries

`bootstrap.json` records the original 1.4/1.5 capture sources. Git retains those
initial snapshots; active `history/<origin>/input.yaml` files advance through
each intervening released operator without changing their origin or runtime.

`RECONCILED_VERSION` identifies the released controller that last processed
**every** active historical input. `history.lock.json` records that same version,
the published release tag and SHA, and hashes covering the entire active history.
A new origin alone cannot advance this gate: formal capture must publish the
release-reconciled results of all existing origins as well as the selected new
origins. An ordinary replay never rewrites historical inputs.

Checks reject changed input bytes, incomplete provenance, and provisional
candidates published as formal history. The next-minor appVersion bump requires
all history to have passed through the immediately previous release. Same-minor
recapture, skipped-minor bumps and major bumps remain unsupported.

`history/N` must be observed using an actual published stable upstream tag for
operator N. Main's chart reporting N is insufficient because its production
code may differ from the release. Formal capture verifies the local tag against
the published, non-draft, non-prerelease GitHub release and rejects tracked or
untracked production changes. `gh` access is required for capture, not ordinary
checks. The selected module runs with `GOWORK=off`.

## Scoped v1.6.0 capture

`capture-v1.6.0.json` freezes 19 input paths and hashes from main revision
`75d0686fb7def243ab5a07ec69159427b169675c`: the three existing history
inputs and 16 latest inputs. The latter cover 13 of the 27 canonical matrix paths
plus three legacy variants used only for validation. The 14 missing canonical
paths are explicitly listed.

This exact partial inventory is accepted only with stable tag `v1.6.0`, release
revision `701da25292ecca8f64d770cc0b17d389200019a1`, chart version 1.6.0, and the
predecessor reconciler 1.5.0. The capture records this pre-capture scope under
`history.lock.json`'s `captureInventory`. All three existing historical inputs
are reconciled by the released 1.6 operator and their resulting graphs replace
the active inputs and goldens under their original origin directories.
`promotedScenarios` additionally selects the 13 available canonical matrix cases
for publication under origin 1.6. The three legacy latest variants remain CREATE
coverage; they are not duplicated under the new origin. Future captures likewise
advance every existing origin and promote only release-matrix catalog scenarios.

This does not certify missing cases. Later releases and provisional captures
still require the complete matrix from `required-matrix.json`: 24 component/Grove
cases plus three multi-node DisaggregatedSet cases. Main at 1.7 consumes the
captured 1.6 baseline; a 1.7 capture still requires complete coverage, and a 1.8
bump still requires a published 1.7 baseline. Keep #16650 open.

To repeat the scoped capture, prepare a dedicated published-release checkout and
carry only the reviewed test harness. Pin that harness as well as the inputs so
later development-matrix additions do not change this scoped reproduction:

```bash
harness=515e4515d141e1d12e31d7d33641022b39e81d09
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
safety checks pass. Copy back the complete regenerated `history/`, including the
older-origin results, plus `history.lock.json` and `RECONCILED_VERSION` (1.6.0).
Review that all historical origin annotations and runtime images are preserved,
then replay the development operator against those released-controller states.
Development `latest` inputs/goldens are unchanged.

Candidate evidence can be collected using a full `UPGRADE_CANDIDATE_SHA` and a
new `UPGRADE_CANDIDATE_OUTPUT` directory outside the fixture tree. It records
`sourceKind=release-candidate` and never publishes baseline history, modifies the
lock, or advances `RECONCILED_VERSION`.

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

## Development takeover compatibility

The released 1.6 operator did not generate a frontend startup probe. Current
main preserves those existing templates through the runtime-and-origin gate in
#17182: only newly created workloads with operator origin 1.7 or later and a
supported frontend runtime receive the generated default. Historical origins
remain concrete and unchanged during replay.

Main already reports operator 1.7.0. The complete released-1.6 history replaces the
one-time 1.7-on-1.5 history exception from #17153; subsequent minor transitions
must use the immediately preceding published baseline. Do not hand-edit captured inputs
or relax safety assertions; verify takeover with the development operator
before accepting this capture.

Latest origin annotations use `$var:operator-version`, bound exactly to the
operator chart's `appVersion` by the golden matcher. Worker hashes, the PCS
spec digest and the DS role revision also depend on origin: their exact bindings
are computed from the reviewed golden DGD/PCS/DS specifications with that version, never from candidate
output. The independent CREATE origin assertion remains active. All other
lowering fields remain exact, and historical origins and hashes remain concrete. Accepting a golden never bypasses rollout
safety assertions.

CI detects operator-affecting changes before running the suite. It retains test
output and .new differences, and deletes its dedicated cluster on completion.
