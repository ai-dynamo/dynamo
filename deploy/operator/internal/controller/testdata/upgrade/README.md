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
which kinds are observed. Only server metadata, Service addresses and condition
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
export KUBECONFIG="$(mktemp)"
export DYNAMO_CLUSTERTEST_CONTEXT=kind-operator-upgrade
make -C deploy/operator kind
deploy/operator/bin/kind create cluster --name operator-upgrade --kubeconfig "$KUBECONFIG" \
  --image kindest/node:v1.36.1@sha256:3489c7674813ba5d8b1a9977baea8a6e553784dab7b84759d1014dbd78f7ebd5
make -C deploy/operator install-upgrade-crds
make -C deploy/operator upgrade-contract
deploy/operator/bin/kind delete cluster --name operator-upgrade
```

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

While the chart still reports version N, run:

```bash
make -C deploy/operator regen-upgrade-fixtures
```

The command validates the predecessor marker, reconciles every historical and
latest scenario with the current checkout, requires both checks to pass, stages
concrete API captures, promotes latest to history/N, and publishes the new fixture
tree with `RECONCILED_VERSION=N`. Failure leaves the committed history and marker
unchanged. Regeneration at the same minor is rejected. This initial suite applies
that lifecycle to its three Grove scenarios only; broader DEP coverage remains
follow-up work.

After that commit, bump the operator subchart's **appVersion** to N+1. The check
rejects premature, skipped-minor and major bumps. CI checks the PR/merge-group base
version as well as current consistency; pre-commit runs the inexpensive local
consistency check. Patch/prerelease changes do not add history generations.
Give latest scenarios fresh namespaces when advancing to a new origin so they
cannot collide with older promoted scenarios; review their resulting golden
outputs through current CREATE admission.

CI detects operator-affecting changes before running the suite. It retains test
output and .new differences, and deletes its dedicated cluster on completion.
