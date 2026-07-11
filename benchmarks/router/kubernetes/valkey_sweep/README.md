# HA Valkey router Kubernetes sweep

This directory builds one ARM64 image and deploys an isolated benchmark stack
in `bis-rl-3`. The stack uses two independent Valkey replication groups behind
the same three Sentinel witnesses:

- the router group loads `dynkv.so`, uses `noeviction`, persists AOF data to
  separate RWO volumes, and requires one replica acknowledgement;
- the tokenizer group uses `allkeys-lru`, has no persistence, and can evict L2
  tokenization entries without evicting router metadata;
- every frontend and mocker receives the same `DYN_ROUTER_VALKEY_CONFIG` JSON;
- the benchmark client addresses each ready frontend pod directly and gives
  AIPerf all URLs with round-robin URL selection.

Sentinel rewrites its configuration on three independent RWO volumes. Data
pods use a role-neutral startup helper: on every restart they accept a
Sentinel majority as authoritative and only use the bootstrap role before a
Sentinel majority exists. A persistent per-identity marker makes that fallback
one-shot only after the local server answers PING and a Sentinel majority adopts
it; an initialized node fails closed and waits indefinitely during a Sentinel
outage. Three disruption budgets retain one member of each data group and two
Sentinel witnesses during voluntary disruptions.

All resources carry `app.kubernetes.io/part-of=valkey-router-sweep`. Dynamo's
current Valkey transport does not support TLS or ACL credentials, so the
committed NetworkPolicies isolate every plaintext data, Sentinel, request,
discovery, and benchmark path by component. The launchers and driver also
reject non-canonical Sentinel master hosts. No sweep pod receives a Kubernetes
service-account token, and the policies do not alter unrelated resources. A
preflight proves that the benchmark client can reach the protected endpoints
while an otherwise identical unlabeled pod cannot. The campaign also validates
and records every Sentinel's effective quorum, down-after, failover-timeout,
and parallel-syncs settings, so stale configuration on a retained PVC fails
the run instead of silently changing the HA policy.

These policies are a namespace traffic boundary, not workload authentication:
an identity that can create or relabel pods in `bis-rl-3` can spoof their label
selectors. Run the sweep only when pod create/patch permissions in the namespace
are restricted to trusted operators; use a dedicated namespace when that RBAC
boundary cannot be guaranteed. Valkey ACL/TLS support is required before this
design can defend against an untrusted namespace tenant.

## Matrix and methodology

The committed driver runs exactly 36 points:

- frontend pods `M = [1, 2, 4]`;
- closed-loop concurrency `C = [4096, 8192, 16384, 32768]`;
- one-worker mocker pods `N = [10, 40, 80]`;
- omitted `--request-rate`, which is effectively infinite offered rate;
- Qwen/Qwen3-0.6B, configured ISL/OSL 1024/1024, mocker speedup 100000;
- `max(16384, 4*C)` measured requests after one complete `C`-request warmup
  wave.

The driver creates a distinct Dynamo namespace for every `(M,C,N)` point,
scales the prior frontends and mockers to zero, flushes and replicates both the
abandoned router index and tokenizer L2, starts fresh workers/frontends, and
warms one full concurrency wave. It waits for exactly `N` worker-rank
registrations in `DYNKV.STATS`, waits for all frontend readiness probes, and
only then measures the point. It checks the exact worker-rank count again after
AIPerf completes.
This makes the matrix independent of execution order rather than allowing KV,
L1, or L2 state from a lower-concurrency point to warm a later point.
Each attempt first persists a random generation and includes it in the Dynamo
namespace. Retrying an interrupted point therefore cannot reuse stale router
indexes, worker leases, or discovery registrations from the failed attempt.
The source-bound core revision, immutable image IDs, topology, RPS, latency,
TTFT, ITL, actual ISL/OSL, errors, and selected Valkey telemetry are retained.
The immutable campaign manifest fingerprints the clean driver commit, every
methodology file, and the fully rendered stack; its revision must match the
core embedded in the image. Resume refuses source, image, rendered-manifest,
matrix, Sentinel-policy, network-proof, or failover-proof drift. An attempt
left in `starting` state is never trusted, even when a result exists;
resume gives it a new generation and reruns the point. Complete attempt records
hash the result and raw AIPerf summary, and resume revalidates the raw metric
schema. This detects accidental/local modification;
use signed build provenance and append-only or externally signed result storage
when defending against an adversary who can rewrite all campaign files.
The client pod expands its namespaced ephemeral-port range to `1024-65535`.
Every point records and requires enough file descriptors and source ports for
`C` plus 25%/4096-connection headroom, preventing the `C=32768, M=1` point from
silently becoming a client-port benchmark.
AIPerf 0.10 records per-request data while calculating its summary. The driver
copies a successful aggregate JSON locally, validates and hashes it, and always
deletes that attempt's remote record directory—even after a failed validation
or AIPerf process—so the 36-point campaign cannot fill the client's isolated
10 GiB `emptyDir`. The namespace's shared `data` PVC is never used.

## Build and push from `biswa-dind`

Push the signed source branch to Gitea first. In the `biswa-dind` pod, clone or
fast-forward `/build/valkey-router-src` from the in-cluster Gitea service. The
pod exposes `GITEA_USERNAME` and `GITEA_PASSWORD`; use a temporary credential
helper and never place either value in a URL, image layer, build argument, or
log.

On the trusted host, bind the clean local commit to the authenticated remote
branch before passing the full revision into DinD:

```bash
BRANCH=bis/valkey-router-k8s-sweep
REVISION=$(git rev-parse HEAD)
test -z "$(git status --porcelain)"
test "$(git ls-remote ssh://git@localhost:2224/biswa/dynamo2.git \
  "refs/heads/$BRANCH" | awk '{print $1}')" = "$REVISION"
IMAGE="nvcr.io/nvidian/dynamo-dev/biswa:valkey-router-${REVISION:0:12}-arm64"
```

Build in the existing DinD pod from an authenticated Git bundle plus a tarred
checkout of the same clean commit. The Dockerfile clones the bundle, requires
its HEAD to equal `GIT_REVISION`, runs `git fsck --strict`, and builds Dynamo
and `dynkv.so` only from that verified object graph. The checkout tar supplies
the Dockerfile and bundle, but the Dockerfile copies only the bundle into build
stages; no source outside the verified clone is used by a build stage:

```bash
kubectl exec -n bis-rl-3 biswa-dind -- env \
  REVISION="$REVISION" IMAGE="$IMAGE" sh -lc '
    cd /build/valkey-router-src
    test "$(git rev-parse HEAD)" = "$REVISION"
    test -z "$(git status --porcelain)"
    context=$(mktemp -d)
    trap "rm -rf \"$context\"" EXIT
    mkdir "$context/root"
    git archive --format=tar HEAD | tar -x -C "$context/root"
    git bundle create "$context/root/source.bundle" HEAD
    tar -C "$context/root" -cf "$context/context.tar" .
    docker buildx build --load \
      --platform linux/arm64 \
      --file benchmarks/router/kubernetes/valkey_sweep/Dockerfile \
      --build-arg GIT_REVISION="$REVISION" \
      --tag "$IMAGE" - <"$context/context.tar"
    DOCKER_CONFIG=/run/secrets/nvcr docker push "$IMAGE"
  '
```

The build fails unless `GIT_REVISION` exactly matches the bundled commit and
the bundle passes Git object validation. The Python extension embeds that
identity and a clean flag. The final image also contains Valkey server/CLI,
the loadable module, and AIPerf 0.10.0. All base images are digest-pinned,
Python is fixed at ARM64 3.12.13, Rust is fixed at 1.96.1, Maturin is installed
from its locked 1.9.4 crate, and
AIPerf's ARM64/Python 3.12 dependency graph and Python build backend are
URL/hash locked in `pylock.aiperf.toml` and `pylock.build.toml`. Debian packages
are resolved from the fixed `20260623T000000Z` snapshot. Crick's only available
artifact is a source distribution, so it is built in the builder stage with
`pylock.crick-build.toml`; the final AIPerf install uses `--no-build`. Regenerate
the lock files only with:

```bash
uv pip compile benchmarks/router/kubernetes/valkey_sweep/aiperf.in \
  --python-version 3.12 --python-platform aarch64-manylinux_2_28 \
  --format pylock.toml --no-emit-index-url --no-annotate --no-header \
  --output-file benchmarks/router/kubernetes/valkey_sweep/pylock.aiperf.toml

uv pip compile benchmarks/router/kubernetes/valkey_sweep/build.in \
  --python-version 3.12 --python-platform aarch64-manylinux_2_28 \
  --format pylock.toml --no-emit-index-url --no-annotate --no-header \
  --output-file benchmarks/router/kubernetes/valkey_sweep/pylock.build.toml

uv pip compile benchmarks/router/kubernetes/valkey_sweep/crick-build.in \
  --python-version 3.12 --python-platform aarch64-manylinux_2_28 \
  --format pylock.toml --no-emit-index-url --no-annotate --no-header \
  --output-file \
    benchmarks/router/kubernetes/valkey_sweep/pylock.crick-build.toml
```

Verify before deployment:

```bash
kubectl exec -n bis-rl-3 biswa-dind -- env IMAGE="$IMAGE" sh -lc '
  docker run --rm "$IMAGE" python -c \
    "import os,dynamo._core as c; print(c.__build_git_revision__, c.__build_git_dirty__, os.environ[\"VALKEY_IMAGE_GIT_REVISION\"])"
  docker run --rm "$IMAGE" valkey-server --version
  docker run --rm "$IMAGE" /opt/aiperf-venv/bin/aiperf --version
'
```

The driver records and rejects an image whose embedded Valkey revision is not
`5b690cefd6cad707a748879c2bab6b72e18efcb7`. This guards accidental build-arg
drift; it does not replace signed image provenance. The DinD builder and NVCR
repository remain trusted because a malicious builder or registry can forge
both binaries and self-reported labels/environment variables.

Record the registry digest from `docker image inspect` or the push output. Do
not pass the tag to the sweep: the driver accepts only a registry digest
reference. Resolve it from the pushed image:

```bash
DIGEST_IMAGE=$(kubectl exec -n bis-rl-3 biswa-dind -- env IMAGE="$IMAGE" \
  DOCKER_CONFIG=/run/secrets/nvcr sh -lc \
  'docker image inspect --format "{{index .RepoDigests 0}}" "$IMAGE"')
test -n "$DIGEST_IMAGE"
```

## Deploy and run

Prerequisites in `bis-rl-3` are the existing `etcd` Service,
`shared-model-cache` PVC, and `nvcr-pull-secret`. The Qwen tokenizer
must be present in the shared Hugging Face cache because the pods run with
`TRANSFORMERS_OFFLINE=1`.

Run the tested driver from a host with `kubectl` access:

```bash
CAMPAIGN="valkey-$(date -u +%Y%m%d-%H%M%S)"
python -m benchmarks.router.kubernetes.valkey_sweep.sweep \
  --image "$DIGEST_IMAGE" \
  --campaign "$CAMPAIGN" \
  --results "/tmp/$CAMPAIGN" \
  --apply-stack \
  --prove-failover
```

Resume an interrupted campaign without rerunning points whose `result.json`
has `status: ok`:

```bash
python -m benchmarks.router.kubernetes.valkey_sweep.sweep \
  --image "$DIGEST_IMAGE" \
  --campaign "$CAMPAIGN" \
  --results "/tmp/$CAMPAIGN" \
  --resume
```

For a smoke point, repeat `--frontends`, `--concurrencies`, and `--mockers` with
allowed values, for example `--frontends 1 --concurrencies 4096 --mockers 10`.
The full matrix is the default when no filters are present.

With `--prove-failover`, the driver first starts one frontend and ten mockers.
It sends a unique chat canary before failover, while each old primary remains
down after promotion, again after each old-primary restart, and after all
Sentinels restart. Every canary
must return a model response, retain exactly ten router registrations, and grow
the centralized tokenizer L2. Direct Valkey replication canaries are retained
as lower-level evidence alongside these Dynamo-path reconnect checks.
Each measured point also captures Valkey counters immediately before and after
AIPerf and reports per-identity deltas over the measured interval; post-load
instantaneous rates are retained only as telemetry, not presented as event
volume during the run.

## Inspect and clean up

```bash
kubectl get all,pvc,networkpolicy -n bis-rl-3 \
  -l app.kubernetes.io/part-of=valkey-router-sweep
```

Delete only runtime resources after copying the compact results. This command
intentionally leaves all router and Sentinel PVCs intact:

```bash
kubectl delete -n bis-rl-3 \
  configmap,service,statefulset,deployment,poddisruptionbudget,networkpolicy \
  -l app.kubernetes.io/part-of=valkey-router-sweep
```

Only when the AOF and Sentinel election history are no longer needed, delete
PVCs carrying the same label after inspecting their names. Never delete the
`bis-rl-3` namespace or the shared `shared-model-cache` PVC.
