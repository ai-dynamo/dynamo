# Operator upgrade contract: initial Grove slice

This suite implements the historical Grove portion of #16624 / #16623, aligned
with the storage and adjacent-upgrade design in #16650. It covers three SGLang
scenarios, not the full engine/topology/workload-program matrix proposed there.

| Scenario | Origin | Last reconciler | Candidate | Runtime image |
|---|---|---|---|---|
| sglang-1.4 | 1.4.0 | 1.5.0 | operator chart appVersion | fixed official 1.4.0 |
| sglang-1.5-custom | 1.4.0 | 1.5.0 | operator chart appVersion | fixed custom v1.5.0-9aa3-repro |
| sglang-1.5 | 1.5.0 | 1.5.0 | operator chart appVersion | fixed official 1.5.0 |

The first two inputs originated in 1.4 but have already passed through production
1.5 admission/controllers on the same API server. This makes the candidate edge
1.5 → 1.6. It does not certify the known-bad custom-tag transition from 1.4 → 1.5;
that transition and its hash drift remain documented in #16623. No production
hash migration is implemented by this test PR.

## Two separate checks

| Check | Purpose | Can updating output.yaml make it pass? |
|---|---|---|
| Complete owned resource golden | Review all persisted lowering differences, including new resource kinds | Yes, after semantic review |
| Upgrade safety assertion | Preserve generation selection, active identity, effective worker templates and rollout intent | No |

The safety assertion reads the production Grove transition's `hashChanged`
decision before controllers start. After a completed production reconciliation,
it separately compares historical identities, DGD origin, and effective
Grove worker templates. It does not equate every internal comparison annotation
with the active worker identity. PCS-level metadata defaults may therefore be
accepted as golden changes, while worker environment, labels, annotations and
PodSpec changes remain protected. The independent detector test rejects a new
generation even if the output is identical. An intentional worker env edit must
still reach a new workload generation after takeover.

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

These cases complete 22 of the 27 canonical matrix entries. The five remaining
single-node Component cases and the missing-origin historical case remain in
[#16650](https://github.com/ai-dynamo/dynamo/issues/16650); the incomplete
release-matrix gate stays enforced.

## Storage and production lifecycle

Run this suite only against a dedicated cluster with no installed controllers or
admission webhooks. The `upgradecontract` build tag gives it an exclusive TestMain;
it cannot run the ordinary clusterenv suite concurrently.

1. Create all historical namespaces, retaining their names because namespace
   participates in worker hashing.
2. Temporarily install permissive schemas for the historical custom resources.
3. Load unstructured YAML; remove server-assigned metadata, recreate owner UIDs,
   and restore status. Only allocated Service addresses are reallocated.
4. Read back exact historical custom-resource specs/status before takeover.
5. Restore production CRDs, then start production conversion/admission webhooks.
6. Start namespace-scoped production DGD/DCD controllers and wait for an actual
   completed reconciliation, rather than matching untouched historical objects.
7. Compare the full owned graph and independent safety invariants.

The captured graph includes DGD, PCS, Services, ServiceAccount, Role and RoleBinding.
API discovery finds the ownership closure through immediate workloads and Services;
Kubernetes/Grove descendants are excluded. Expected output does not determine
which kinds are observed. Snapshot writers omit top-level `metadata.managedFields`
because replay discards it. Only server metadata, Service addresses and condition
timestamps are wildcarded. Worker hashes and namespace suffixes are literal
historical values, never recomputed to build an expected historical result.

`latest/` exercises ordinary CREATE admission, including replacement of a supplied
origin annotation. `history/<origin>/` retains the origin separately from the
`RECONCILED_VERSION` marker. The storage regression test also uses a field absent
from the candidate schema, checks owner/status restoration, and verifies cleanup
on failed preparation. Kubernetes may normalize historical fields after the
candidate schemas become active; the harness must not normalize them earlier.

The suite inspects immediate workloads. Grove workload actuation is absent and
Pod/ReplicaSet creation is blocked with ResourceQuota. It is not a GPU serving test.

## Running

A dedicated Kind cluster is sufficient; model or operator container builds are
not needed. Go runs the candidate webhooks/controllers in process, and clusterenv
uses its existing Python proxy to connect the API server to those webhooks.
Helm, kubectl, Go and Docker/Kind are prerequisites.

```bash
make -C deploy/operator clustertest-upgrade-contract
```

The wrapper creates a uniquely named Kind cluster with the pinned node image,
uses a temporary kubeconfig, installs the shared clustertest CRDs, and removes
only its own cluster on completion, failure, or interruption. It refuses to
reuse an existing cluster. CI calls the same target. Runtime images are not
needed by this suite.

For an already prepared, exclusively owned cluster, set `KUBECONFIG` and
`DYNAMO_CLUSTERTEST_CONTEXT` explicitly, then run
`clustertest-upgrade-contract-in-cluster`. The `-in-cluster` targets fail if the
context is unset; they never select the current context implicitly.

A golden mismatch writes `output.yaml.new`. Review the difference and replace
only `output.yaml` when it is intentional. Never update historical input or relax
the safety assertion just to turn CI green.

## One-time historical bootstrap

The initial post-1.5 history was observed using these official commits:

- 1.4.0: `03014943323e78feb5bd672ef08b72caea0918ac`
- 1.5.0: `b83b1d9304ebfc624709ac46db32b1b6f1ff1615`

`bootstrap.json` records release SHA and capture-source provenance;
`history.lock.json` verifies exact frozen input bytes and inventory. Bootstrap
uses production CREATE admission, production DGD/DCD controllers, and API reads.
The only release-specific setup adaptation supplies the ScaleClient required by
1.4's controller setup. No release adapter synthesizes annotations, hashes,
suffixes, defaults or generated workloads. The production hash helper is used
only to wait for the controller's persisted state.

To independently repeat the capture, install PyYAML and envtest assets, then run:

```bash
KUBEBUILDER_ASSETS=/path/to/envtest/binaries \
  python3 deploy/operator/hack/bootstrap_upgrade_fixtures.py --output /new/staging/directory
```

It never overwrites committed history. The 1.4 process retains its API server and
etcd while the 1.5 process takes over. Each snapshot records the concrete resource
graph after 1.5 has persisted it. Normal CI does not fetch or compile old operators.

## Release boundary

`history/N` must come from an actual published stable upstream tag for operator N,
not from `main` merely because its chart still reports N. `main` may already contain
post-cut changes that no N user runs. The next-minor bump on `main` remains blocked
until the released predecessor history has been captured and committed. One
exception exists: `main` moved to 1.7 while history was still reconciled by 1.5, so
`HISTORY_EXCEPTIONS` in `hack/generate_upgrade_fixtures.py` lets 1.7 replay 1.5
history. Capture `history/1.6.0` from the `v1.6.0` tag before the 1.8 bump, then
delete the exception. Regeneration always advances exactly one minor, so 1.6 must
be captured from its own release tree before 1.7.

Use a dedicated checkout of the selected release tag. Fetch the tag from upstream,
and carry only the reviewed test harness and fixture inventory into that checkout;
keep its production operator code, API types, CRDs, chart and Go dependencies at
the selected release revision. For example, after `v1.6.0` is published:

```bash
harness=$(git rev-parse HEAD)  # reviewed fixture/harness commit, including the complete matrix
release_tree=../dynamo-release-capture
git fetch origin tag v1.6.0
git worktree add --detach "$release_tree" v1.6.0
git -C "$release_tree" restore --source="$harness" -- \
  deploy/operator/Makefile deploy/operator/hack \
  deploy/operator/internal/controller/clusterenv_helpers_test.go \
  deploy/operator/internal/controller/clusterenv_test.go \
  deploy/operator/internal/controller/clusterenv_main_test.go \
  ':(glob)deploy/operator/internal/controller/upgrade_*_test.go' \
  deploy/operator/internal/controller/testdata/upgrade \
  deploy/operator/internal/testing/clusterenv \
  deploy/operator/internal/testing/golden \
  deploy/operator/internal/testing/mocks
make -C "$release_tree/deploy/operator" clustertest-upgrade-fixtures-regen \
  UPGRADE_RELEASE_TAG=v1.6.0
```

Formal capture requires `gh` access to the upstream GitHub repository. It verifies
that the tag is a published, non-draft, non-prerelease GitHub release and resolves
to the same commit as the local tag. Before running Go, it rejects a mismatched
chart version, tracked production changes (including committed post-cut main
changes), and untracked production files. Only test/harness paths and documentation
are excluded from this source comparison; there are no production lowering adapters.
The selected module runs with `GOWORK=off`.

The command reconciles history and admits/promotes the latest scenario inputs using
the selected release's production controllers. All goldens and independent safety
assertions must pass. If a release differs from the development goldens, review
those differences in the capture checkout; never relax the historical safety assertions.
On success, commit/copy back **only** `history/`, `history.lock.json` and
`RECONCILED_VERSION` to the development tree, then run its ordinary replay against
the new history. Review any candidate-output golden differences there separately.
`latest` inputs and goldens continue to describe development lowering and are not
rewritten by release capture. The lock records `sourceKind=release-tag`, the tag
and its full commit SHA. Failure does not publish history or advance the marker.
Regeneration at the same minor is rejected.

During release testing, a fixed full candidate SHA may be used instead of a tag.
Prepare the checkout and reviewed harness in the same way, then run:

```bash
make -C "$release_tree/deploy/operator" clustertest-upgrade-fixtures-regen \
  UPGRADE_CANDIDATE_SHA=<full-40-character-candidate-sha> \
  UPGRADE_CANDIDATE_OUTPUT=/absolute/path/to/new-candidate-evidence
```

Candidate evidence is provisional: `capture-source.json` explicitly records
`sourceKind=release-candidate` and the SHA. This mode writes to a new directory
outside the fixture tree and never changes committed history or
`RECONCILED_VERSION`. Once the stable tag is published, capture it from the same
unchanged predecessor inputs for the formal release boundary. Candidates cannot
silently become the published baseline. Ordinary checks require no GitHub access.

Ordinary checks discover every available scenario recursively. The required
release matrix is declared in `required-matrix.json`: 24 component/Grove cases
(three engines × four deployment shapes × two programs), plus three supported
multi-node disaggregated DisaggregatedSet cases. Additional scenarios are valid.
Regeneration rejects a missing required scenario. A minor `appVersion` bump
relative to `--base` requires predecessor history, which only a complete-matrix
regeneration produces. The initial three cases can therefore land before
broader coverage, but cannot advance the release boundary alone. Keep #16650
open for the complete matrix, missing-origin history and rollout safety across
all workload programs.

After the released-source capture commit is brought back to `main`, bump the operator subchart's **appVersion** to N+1. The check
rejects premature, skipped-minor and major bumps, apart from the one 1.7 exception
above. CI checks the PR/merge-group base
version as well as current consistency; pre-commit runs the inexpensive local
consistency check. Patch/prerelease changes do not add history generations.
Promotion derives a namespace from `history/<version>/<complete-scenario-path>`
and runs CREATE admission and reconciliation there before capturing the new
history. Existing history retains its original namespace. Latest namespaces
are derived separately from `latest/<complete-scenario-path>` and stay stable
across releases; namespace collisions across all roles are rejected. Long
namespace names include a digest to remain within Kubernetes' length limit.

Latest origin annotations use `$var:operator-version`, bound exactly to the
operator chart's `appVersion` by the golden matcher. Worker hashes, the PCS
spec digest and the DS role revision also depend on origin: their exact bindings
are computed from the reviewed golden DGD/PCS/DS specifications with that version, never from candidate
output. The independent CREATE origin assertion remains active. All other
lowering fields remain exact, and historical origins and hashes remain concrete. Accepting a golden never bypasses rollout
safety assertions.

CI detects operator-affecting changes before running the suite. It retains test
output and .new differences, and deletes its dedicated cluster on completion.
