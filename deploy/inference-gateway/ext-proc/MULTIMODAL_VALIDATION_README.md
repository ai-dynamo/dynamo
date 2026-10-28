<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime EPP multimodal validation

Status: **CPU and GPU validation passed on 2026-10-28**. EPP compiles with `mm-routing`; all 168 EPP
library tests passed in a read-only, externally disconnected container with
local loopback sockets and an empty temporary HOME. The PNG regression passed
separately in the stricter offline sandbox. The final KV-router suite passed
(1,112 passed, 3 ignored), the native EPP executable built, and workspace
formatting passed. Full service startup requires discovery and worker services.
Real Gateway tests passed in aggregated and disaggregated serving, including
image identity controls, prefill cache credit, completion bookkeeping, and
raw/tokenized/multi-prompt completions. Measured results are below.
This procedure is for a separate, operator-controlled validation environment.
Standalone EPP render-sidecar mode and device-aware embedding-cache selection are
outside this change.

## Hardware and prerequisites

Use an NVIDIA GPU with at least 24 GiB and a CPU-only EPP pod. One worker
is enough to verify cache credit; two GPUs with one worker each additionally
exercise worker selection. Record actual GPU model/count, driver/CUDA, CPU model, EPP CPU
requests/limits, image digests, model revision and backend version in the results.
Use Qwen/Qwen3-VL-2B-Instruct, block size 16, max model length 8192, prefix
caching and KV events. Keep all these settings fixed between runs.

The test cluster needs the matching Dynamo operator, NATS, Gateway API, GAIE and a
Gateway named `inference-gateway` in `agentgateway-system`, with routes allowed
from the test namespace. Follow the repository Gateway setup guide first:
`docs/fern/pages/kubernetes/kv-aware-routing/gateway-api.mdx`.
Configure the operator with `discoveryBackend=kubernetes`. In the tested revision,
etcd discovery does not create the EPP Service required by the InferencePool;
verify `service/qwen-epp` exists and the pool has `ResolvedRefs=True` before probes.
Use prepared images and ensure both EPP and workers can read the public model
configuration (a mounted cache or Hugging Face access). Do not use production traffic.

## CPU and build checks

From the repository root, in a prepared Rust development environment:

```bash
export CARGO_BUILD_JOBS=1
export CARGO_PROFILE_DEV_DEBUG=0 CARGO_PROFILE_TEST_DEBUG=0 CARGO_INCREMENTAL=0
mkdir -p .oss-checks/tmp .oss-checks/assets
export TMPDIR="$PWD/.oss-checks/tmp"
cargo fetch --locked
curl --fail --location --retry 2 \
  https://github.com/swagger-api/swagger-ui/archive/refs/tags/v5.33.1.zip \
  --output .oss-checks/assets/swagger-ui-v5.33.1.zip
export SWAGGER_UI_DOWNLOAD_URL="file://$PWD/.oss-checks/assets/swagger-ui-v5.33.1.zip"
cargo metadata --offline --locked --format-version 1 --manifest-path Cargo.toml
cargo --config 'env.SWAGGER_UI_DOWNLOAD_URL.force=false' test --offline --locked -p dynamo-kv-router --lib --no-default-features
cargo --config 'env.SWAGGER_UI_DOWNLOAD_URL.force=false' test --offline --locked -p dynamo-kv-router --lib --no-default-features \
  protocols::tests::multimodal_routing_preserves_image_prefix_credit -- --exact
cargo --config 'env.SWAGGER_UI_DOWNLOAD_URL.force=false' test --offline --locked -p dynamo-ext-proc --lib \
  epp::tests::multimodal_routing_view_matches_frontend -- --exact
cargo --config 'env.SWAGGER_UI_DOWNLOAD_URL.force=false' test --offline --locked -p dynamo-ext-proc --lib --features mm-routing \
  epp::tests::chat_images_match_frontend_routing_without_expanding_worker_tokens -- --exact
cargo --config 'env.SWAGGER_UI_DOWNLOAD_URL.force=false' test --offline --locked -p dynamo-ext-proc --lib --features mm-routing
cargo --config 'env.SWAGGER_UI_DOWNLOAD_URL.force=false' build --offline --locked --release -p dynamo-ext-proc --features mm-routing
cargo fmt --all -- --check
```

The host needs `protobuf-compiler`, `libprotobuf-dev`, `libclang-dev`,
`libzmq3-dev`, a C/C++ toolchain and pkg-config (the Dockerfile installs its own
prerequisites). Fetch locked Cargo dependencies before using `--offline`.
Build the patched image without publishing it:

```bash
make -C deploy/inference-gateway/ext-proc image-load \
  IMAGE_TAG=dynamo/dynamo-epp:mm-test
```

Use the recorded base commit for the baseline, not `--no-default-features` on
the patched source. Add the same whole-pick timer to that baseline so inference
latency is excluded from the comparison:

```bash
git worktree add --detach ../dynamo-epp-mm-baseline 2e96c98ba75372091b6f2bdc1f5b08e13f5a5dc5
python3 - <<'PY'
from pathlib import Path
p = Path('../dynamo-epp-mm-baseline/deploy/inference-gateway/ext-proc/src/epp.rs')
s = p.read_text()
start = ') -> Result<PickResult, PickError> {'
end = '            "Picked endpoint"'
assert s.count(start) == s.count(end) == 1
s = s.replace(start, start + '\n        let pick_start = std::time::Instant::now();')
s = s.replace(end, '            pick_seconds = pick_start.elapsed().as_secs_f64(),\n' + end)
p.write_text(s)
PY
make -C ../dynamo-epp-mm-baseline/deploy/inference-gateway/ext-proc image-load \
  IMAGE_TAG=dynamo/dynamo-epp:mm-baseline
```

Both images must be loaded into the cluster's container runtime (or transferred
through your test registry). Keep the worker and sidecar images identical for
both runs, built from the same recorded Dynamo revision with `mm-routing`.

If matching worker images are not already available, the repository's standard
vLLM runtime build is below. Rendering requires PyYAML and Jinja2. This also
provides the Frontend sidecar; record the resulting image digest. This build
downloads backend dependencies and model weights are downloaded separately at
deployment time. The current default runtime uses CUDA 13.0, so the GPU node's
driver must support that image. The commands below target Linux amd64.

```bash
python3 -m venv .oss-checks/render-env
.oss-checks/render-env/bin/python -m pip install PyYAML Jinja2
.oss-checks/render-env/bin/python container/render.py \
  --framework vllm --target runtime --platform linux/amd64 --output-short-filename
docker build -t dynamo/vllm-runtime:mm-test -f container/rendered.Dockerfile .
docker image inspect --format '{{.Id}}' dynamo/vllm-runtime:mm-test
nvidia-smi
```

Both EPP variants, the worker/sidecar runtime and matching operator were built
from the recorded source revision and exercised on the GPU node. The node setup
used Minikube 1.37.0, Kubernetes 1.34.3, GAIE 1.2.1 and agentgateway 1.0.0.

## Launch through the Gateway

Prepare an isolated namespace and an adapted copy of the existing runtime-mode
GAIE example. The following commands require Python with PyYAML in the validation
environment. Set image names to images already loaded into the cluster. The EPP
image must launch the newly built `/epp` binary; worker and sidecar images must
contain the production Frontend with `mm-routing` enabled.

```bash
export TEST_NS=epp-mm-validation
export EPP_IMAGE=dynamo/dynamo-epp:mm-test
export WORKER_IMAGE=dynamo/vllm-runtime:mm-test
export WORKER_REPLICAS=1  # set 2 when two GPUs are available
kubectl create namespace "$TEST_NS"
python3 - <<'PY'
import os, yaml
with open('examples/backends/vllm/deploy/gaie/agg.yaml') as f:
    dgd = yaml.safe_load(f)
for component in dgd['spec']['components']:
    component['runtimeVersionOverride'] = '1.7.0'  # custom tags are not semver
    containers = component['podTemplate']['spec']['containers']
    for container in containers:
        container.pop('envFrom', None)  # public, locally provisioned model
        container['image'] = os.environ['EPP_IMAGE' if component['type'] == 'epp' else 'WORKER_IMAGE']
        container['imagePullPolicy'] = 'IfNotPresent'
        for env in container.get('env', []):
            if env['name'] in ('DYN_MODEL_NAME', 'MODEL_PATH', 'SERVED_MODEL_NAME'):
                env['value'] = 'Qwen/Qwen3-VL-2B-Instruct'
            if env['name'] == 'DYN_KV_CACHE_BLOCK_SIZE':
                env['value'] = '16'
    if component['type'] == 'epp':
        containers[0]['command'] = ['/epp']
        containers[0]['env'] += [
            {'name': 'DYN_EPP_MODE', 'value': 'dynamo'},
            {'name': 'RUST_LOG', 'value': 'info,dynamo_ext_proc::epp=debug,mm_routing=debug,dynamo_llm::kv_router=debug'}]
    else:
        component['replicas'] = int(os.environ['WORKER_REPLICAS'])
        containers[0]['args'][0] = containers[0]['args'][0].replace(
            '--block-size 128', '--enable-multimodal --block-size 16 --max-model-len 8192')
with open('epp-mm-dgd.yaml', 'w') as f:
    yaml.safe_dump(dgd, f)
PY
kubectl apply -n "$TEST_NS" -f epp-mm-dgd.yaml
kubectl wait -n "$TEST_NS" --for=condition=Ready dynamographdeployment/qwen --timeout=20m
kubectl apply -n "$TEST_NS" -f examples/backends/vllm/deploy/gaie/http-route.yaml
kubectl get inferencepool,httproute,pods -n "$TEST_NS"
kubectl get service/qwen-epp -n "$TEST_NS"
kubectl wait -n "$TEST_NS" inferencepool/qwen-pool \
  --for=jsonpath='{.status.parents[0].conditions[?(@.type=="ResolvedRefs")].status}'=True --timeout=120s
```

Confirm the EPP discovers the same model card as the Frontend and logs exact MM
routing as available. A run that silently falls back to text-only routing is a
failure. Ensure the worker-side direct Frontend performs the same preprocessing
and supplies `mm_hashes` to vLLM; the EPP forwards worker placeholders, not its
expanded/padded routing sequence. Confirm all workers publish KV events before
sending probes. Port-forward the Gateway service (find its name with `kubectl
get svc -n agentgateway-system`) to localhost:8000:

```bash
kubectl port-forward -n agentgateway-system service/inference-gateway 8000:80
```

If the generated service has a different name, use that exact service name.

## Reproduction and expected results

Generate two distinct local PNG files `image-a.png` and `image-b.png` using the
standard library. Repeat at 224x224, 1024x1024 and 2048x2048, recording encoded byte sizes and actual expanded
image-token counts. Inline data URLs avoid external image-fetch variability.
Generate requests with text both before and after the image:

```bash
export IMAGE_SIZE=224
python3 - <<'PY'
import os, pathlib, struct, zlib
size = int(os.environ['IMAGE_SIZE'])
def chunk(kind, data):
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
for label, rgb in [('a', (255, 0, 0)), ('b', (0, 255, 0))]:
    png = b'\x89PNG\r\n\x1a\n'
    png += chunk(b'IHDR', struct.pack('>2I5B', size, size, 8, 2, 0, 0, 0))
    png += chunk(b'IDAT', zlib.compress((b'\0' + bytes(rgb)*size)*size))
    png += chunk(b'IEND', b'')
    pathlib.Path(f'image-{label}.png').write_bytes(png)
PY
python3 - <<'PY'
import base64, json
for label in ('a', 'b'):
    with open(f'image-{label}.png', 'rb') as f:
        image = 'data:image/png;base64,' + base64.b64encode(f.read()).decode()
    for suffix in ('Describe this image.', 'Count the objects.'):
        body = {'model': 'Qwen/Qwen3-VL-2B-Instruct', 'max_tokens': 4,
                'messages': [{'role': 'user', 'content': [
                    {'type': 'text', 'text': 'Inspect the following image carefully. '},
                    {'type': 'image_url', 'image_url': {'url': image}},
                    {'type': 'text', 'text': suffix}]}]}
        with open(f'request-{label}-{len(suffix)}.json', 'w') as f:
            json.dump(body, f)
PY
for request in request-a-20.json request-a-20.json request-b-20.json request-a-18.json; do
  curl --fail-with-body -sS http://localhost:8000/v1/chat/completions \
    -H 'Content-Type: application/json' --data-binary "@$request"
  sleep 5  # Allow KV event ingestion before the next control.
done
kubectl logs -n "$TEST_NS" -l nvidia.com/dynamo-component-type=epp \
  --all-containers --prefix --tail=-1 > epp-mm.log
```

Allow KV event ingestion between the first and second identical request. Verify
observed cache hits in backend usage/metrics AND router overlap in logs; merely
choosing the same worker is insufficient. Expected patched result: identical
image identity receives credit through image blocks; changing the image retains
only the text prefix before the image; changing suffix text retains image-block
credit. Changing the text before the image must invalidate the subsequent chain.
The CPU PNG regression above compares EPP's actual preprocessing output with
the Frontend's `block_mm_routing_info()` and tests repeated versus different
image identities. The synthetic parity test also hashes the two views. Keep
the model card, cache namespace, block size and worker runtime configuration
identical in the live deployment; selecting the same destination is not proof
of parity. `Picked endpoint` logs expose `routing_token_count` and
`overlap_blocks` in addition to the worker-placeholder `token_count`.
The baseline should reproduce missing credit beyond the first image boundary.

Also exercise text-only chat and these completion forms. Expect one choice for
raw/tokenized input and two indexed choices for the two-prompt batch:

```bash
curl --fail-with-body --max-time 300 -sS http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"Qwen/Qwen3-VL-2B-Instruct","prompt":"The capital of France is","max_tokens":4,"temperature":0}' > completion-raw.json
curl --fail-with-body --max-time 300 -sS http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"Qwen/Qwen3-VL-2B-Instruct","prompt":[791,3363,315,9822,374],"max_tokens":4,"temperature":0}' > completion-tokens.json
curl --fail-with-body --max-time 300 -sS http://localhost:8000/v1/completions \
  -H 'Content-Type: application/json' \
  --data '{"model":"Qwen/Qwen3-VL-2B-Instruct","prompt":["The capital of France is","The capital of Italy is"],"max_tokens":4,"temperature":0}' > completion-multiple.json
```

Repeat the image and completion controls with the
runtime disaggregated GAIE example (same model and block-size substitutions) to
cover prefill reservations and decode bookkeeping.
Enable `--enable-multimodal` on both disaggregated workers and use the same
runtime version/model/block-size substitutions. The tested CPU-buffer NIXL
configuration required `UCX_TLS=tcp,self,cuda_copy`: NIXL 1.5 rejects `tcp,self`
when GPUs are present, even with CPU transfer buffers. Keep `UCX_RNDV_THRESH=inf`.
On H100s, GPU utilization 0.25 per role kept CPU mirror buffers within the test
cluster's 64 GiB RAM; this was a functional check, not an aggregated/disaggregated
performance comparison. Wait for PrefillRouter activation and require every
`Picked endpoint` to have `is_disaggregated=true`. Check prefill `Selected worker`
logs (`worker_type=prefill`, `effective_cached_blocks`); the final pick's
`overlap_blocks` describes decode, which need not publish KV events.

On the two-H100 test environment, prepare that deployment as follows. Archive
aggregate results before replacing `qwen`; wait for its old pods and discovery
metadata to disappear so EPP cannot initialize from the previous model card.

```bash
python3 - <<'PY'
import os, pathlib, yaml
dgd = yaml.safe_load(pathlib.Path('examples/backends/vllm/deploy/gaie/disagg.yaml').read_text())
for component in dgd['spec']['components']:
    component['runtimeVersionOverride'] = '1.7.0'
    component['replicas'] = 1
    spec = component['podTemplate']['spec']
    spec['terminationGracePeriodSeconds'] = 10
    containers = spec['containers']
    for container in containers:
        container.pop('envFrom', None)
        container['image'] = os.environ['EPP_IMAGE' if component['type'] == 'epp' else 'WORKER_IMAGE']
        container['imagePullPolicy'] = 'IfNotPresent'
        for env in container.get('env', []):
            if env['name'] in ('DYN_MODEL_NAME', 'MODEL_PATH', 'SERVED_MODEL_NAME'):
                env['value'] = 'Qwen/Qwen3-VL-2B-Instruct'
            if env['name'] == 'UCX_TLS':
                env['value'] = 'tcp,self,cuda_copy'
    if component['type'] == 'epp':
        containers[0]['command'] = ['/epp']
        containers[0]['env'] += [
            {'name': 'DYN_EPP_MODE', 'value': 'dynamo'},
            {'name': 'RUST_LOG', 'value': 'info,dynamo_ext_proc::epp=debug,dynamo_llm::kv_router=debug,dynamo_kv_router=debug'}]
    else:
        containers[0]['args'][0] = containers[0]['args'][0].replace(
            '--gpu-memory-utilization 0.50', '--gpu-memory-utilization 0.25').replace(
            '--max-model-len 16384', '--max-model-len 8192') + ' --enable-multimodal'
pathlib.Path('epp-mm-disagg.yaml').write_text(yaml.safe_dump(dgd))
PY
kubectl delete -n "$TEST_NS" dgd/qwen --wait=true
kubectl wait -n "$TEST_NS" pod -l nvidia.com/dynamo-graph-deployment-name=qwen \
  --for=delete --timeout=300s
kubectl get -n "$TEST_NS" pods,dynamoworkermetadatas
# Continue only once the previous workers and their metadata are gone.
kubectl apply -n "$TEST_NS" -f epp-mm-disagg.yaml
kubectl wait -n "$TEST_NS" dgd/qwen --for=condition=Ready --timeout=20m
kubectl apply -n "$TEST_NS" -f examples/backends/vllm/deploy/gaie/http-route.yaml
kubectl logs -n "$TEST_NS" -l nvidia.com/dynamo-component-type=epp --tail=-1 \
  | grep 'Prefill worker discovered, activating PrefillRouter'
```

## Pick latency and results

Run baseline and patched images sequentially against identical workers. Restart
workers between cold-cache runs, wait for readiness/event subscriptions, and use
at least 30 cold observations and 200 warm observations per size. For warm runs,
prime the image first, wait for KV events, then repeat the same request serially;
repeat at concurrency 8. Run text-only controls too. Keep CPU limits and logging
levels identical. Alternate baseline/patched run order to reduce drift.

Patched EPP debug logs contain `preprocessing_seconds` for each successful chat
preprocessing call; `Picked endpoint` includes `pick_seconds` for successful
body-bearing picks, including selection/bookkeeping. These are seconds, not
milliseconds. Parse each field into samples and compute median/p95/p99. Baseline
has no new timing fields: use identical local timing instrumentation around
baseline `tokenize` and `pick`, or a gRPC pick client measuring both versions.
HTTP latency alone includes inference and cannot establish added pick cost.
Report sample counts, errors/timeouts, baseline and patched distributions, and
absolute/percentage deltas for whole-pick latency. Patched preprocessing timing
is an additional diagnostic; the baseline only needs the whole-pick timer above.
Record cold media-processing and warm media-cache conditions separately from KV
cache state. Do not present unexecuted numbers as measurements.

For one warm serial sample set, prime and then collect 200 picks. Run this once
with each EPP image, saving logs as `baseline.log` and `patched.log` respectively.
Wait for EPP readiness after changing the image; workers must remain unchanged.

```bash
export CONCURRENCY=1
python3 - <<'PY'
import concurrent.futures, datetime, json, os, pathlib, time, urllib.request
body = pathlib.Path('request-a-20.json').read_bytes()
def probe(_):
    request = urllib.request.Request('http://localhost:8000/v1/chat/completions',
        data=body, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=300) as response:
        result = json.load(response)
    assert 'error' not in result, result
    return result
concurrency = int(os.environ['CONCURRENCY'])
probe(0)
with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
    warmup = list(pool.map(probe, range(max(8, concurrency * 4))))
pathlib.Path('warmup.json').write_text(json.dumps(warmup, indent=2))
time.sleep(5)
start = datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')
pathlib.Path('sample-start.txt').write_text(start)
with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
    responses = list(pool.map(probe, range(200)))
pathlib.Path('responses.json').write_text(json.dumps(responses, indent=2))
for response in responses:
    usage = response['usage']
    assert usage['prompt_tokens_details']['cached_tokens'] >= (usage['prompt_tokens'] - 1) // 16 * 16
PY
START=$(cat sample-start.txt)
kubectl logs -n "$TEST_NS" -l nvidia.com/dynamo-component-type=epp \
  --all-containers --prefix --tail=-1 --since-time="$START" > patched.log
```

Rename the last output to `baseline.log` for the baseline image. Confirm usage
contains `prompt_tokens_details.cached_tokens`: for identical warm requests it
should reach the complete reusable prompt blocks, including the image. Compare
this with EPP's `overlap_blocks`, multiplied by block size 16. Backend caching
alone is insufficient: an old EPP can still get backend hits while receiving no
image-block routing credit. Save the negative-control responses as well.
Repeat with `CONCURRENCY=8`, and regenerate both images/requests at each
`IMAGE_SIZE`. Keep each size/concurrency/cache condition in a separate result
directory. Concurrent warmup is necessary: one serial prime can leave the second
worker cold. Retain failed warmup/sample sets separately and repeat after priming;
do not silently discard individual slow requests. For cold samples, restart
workers between variants and send 30 distinct new PNGs at each size/concurrency.
For example, generate RGB `(i + 1, concurrency, 17)` for `i` in `range(30)` with
the PNG generator above. Use fractional RFC3339 log timestamps; whole-second
boundaries can include picks from the previous condition. If a large image exceeds the node's memory or model token budget,
record that condition as blocked and share the error; do not count it as a pass.

Calculate the observed whole-pick delta (logs must come from matching workloads):

```bash
python3 - <<'PY'
import json, math, pathlib, re
def summarize(path):
    text = re.sub(r'\x1b\[[0-9;]*m', '', pathlib.Path(path).read_text())
    samples = sorted(float(x) * 1000 for x in re.findall(r'pick_seconds[=:]\s*([0-9.eE+-]+)', text))
    assert len(samples) >= 200, f'{path}: only {len(samples)} picks; inspect errors and logging'
    return {'n': len(samples), **{f'p{p}_ms': samples[math.ceil(len(samples)*p/100)-1]
                                for p in (50, 95, 99)}}
base, patched = summarize('baseline.log'), summarize('patched.log')
report = {'baseline': base, 'patched': patched,
          'delta_ms': {k: patched[k]-base[k] for k in base if k != 'n'}}
pathlib.Path('pick-latency.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
PY
```

## Recorded validation (2026-10-28)

Source base: `2e96c98ba75372091b6f2bdc1f5b08e13f5a5dc5`. Two H100 80GB workers;
driver 580.95.05, CUDA 13.0, vLLM 0.32.0, Torch 2.13.0+cu130, NIXL 1.5.0.
Final worker-cache model revision: `89644892e4d85e24eaac8bacfd4f463576704203`.
Launches requested the public model's `main` revision; that resolved revision
was collected from the final workers, rather than explicitly pinned at launch.
The isolated Minikube container had 12 CPUs and 64 GiB RAM. Aggregate EPP had
no explicit CPU limit; baseline and patched used the same settings/log levels.
Solid-color PNG sizes were 729, 5,588 and 17,289 bytes at 224/1024/2048 pixels.

All 14 paired conditions passed: 30 cold requests per size at concurrency 1 and
8, 200 warm requests per size at both concurrencies, and 200 warm text controls
at both concurrencies. There were zero errors in accepted sets; every accepted
warm response had full reusable prompt-block cache credit. Whole-pick image
median overhead was 0.167–0.338 ms warm and 0.240–0.570 ms cold. Warm image p99
overhead was 0.425–0.681 ms; the largest cold p99 delta was 1.302 ms. These are
deterministic PNG fixtures, not a production SLO or an inference-throughput result.

At 224 pixels, repeated-image routing credit changed from a legacy query of
22 tokens (unable to span the image blocks) to an expanded 96-token view and
5 cached blocks/80 backend cached tokens. Changing the image or preceding text
removed credit; changing only the suffix retained 4 blocks/64 tokens. The same
controls passed at 1024 and 2048 pixels. Original worker placeholders remained
22 tokens, while backend prompt lengths were 85/1045/4117 tokens.

All three completion formats returned 200 in baseline, patched and disaggregated
serving; multi-prompt requests returned both choices. Disaggregated controls
passed at all sizes, plus 200 concurrent warm requests. All 254 observed prefill
booking IDs, including priming and completion controls, matched completion
callbacks, with no bookkeeping-registration warnings. Prefill cache credit,
not decode overlap, was checked for this topology.

The test record retains initial setup failures (missing runtime version,
EPP Service/discovery mismatch, missing multimodal flag, UCX CUDA support) and
the initial concurrent sets that encountered an unprimed worker. Those sets are
not counted as accepted warm measurements. Full commands, raw logs/responses,
image/model provenance and the complete percentile table are retained in the
validation handoff; no latency outliers were trimmed.

## Cleanup

Stop the port-forward with Ctrl-C, archive logs/results, then remove only the
isolated test resources:

```bash
kubectl delete namespace "$TEST_NS"
rm -f epp-mm-dgd.yaml request-a-20.json request-a-18.json request-b-20.json request-b-18.json
```

Restore the baseline image if a reusable test deployment was used. Retain the
original local images and result files until review is complete.
