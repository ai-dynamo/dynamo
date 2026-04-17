---
name: kvbm-v2-trtllm-test
description: Run end-to-end KVBM v2 TRT-LLM connector test verifying offload (G1→G2) and onboard (G2→G1) paths
user-invocable: true
---

# KVBM v2 TRT-LLM Connector — End-to-End Test

This skill runs the full offload + onboard verification for the KVBM v2 TRT-LLM connector.

## Prerequisites

- Docker with GPU support
- A container image with KVBM v2 wheel (built with `v2` feature) and TRT-LLM installed
- NATS and etcd infrastructure

Before starting, ask the user which container image to use. The image must have:
- `kvbm` wheel with v2 support (`python3 -c 'import kvbm; print(kvbm.v2.is_available())'` returns `True`)
- `kvbm.v2.trtllm` module installed
- TRT-LLM with the KV connector API (`KVConnectorOutput`, `update_state_after_alloc` with `num_external_tokens`)
- NIXL library

## Step 1: Start Infrastructure

```bash
cd /home/oandreeva/Code/dynamo
docker compose -f deploy/docker-compose.yml up -d
```

Verify NATS is reachable on port 4222 and etcd on port 2379.

## Step 2: Start Container

Replace `<IMAGE>` with the image provided by the user:

```bash
docker run -d --gpus all --ipc=host --ulimit memlock=-1 --name kvbm-v2-test \
  <IMAGE> \
  sleep infinity
```

## Step 3: Create Config

```bash
docker exec -u root kvbm-v2-test bash -c "
cat > /tmp/kvbm_v2_test.yaml <<'YAML'
backend: pytorch
cuda_graph_config: null
kv_cache_config:
  enable_partial_reuse: false
  max_tokens: 200
kv_connector_config:
  connector_module: kvbm.v2.trtllm
  connector_scheduler_class: TrtllmConnectorScheduler
  connector_worker_class: TrtllmConnectorWorker
YAML
"
```

`max_tokens: 200` limits G1 to ~6 blocks (192 tokens at block_size=32). This is intentionally
small so 10 requests exhaust G1 and force eviction, enabling G2 cache hit testing.

## Step 4: Start Server

```bash
docker exec -d kvbm-v2-test bash -c "
export NATS_SERVER=nats://172.17.0.1:4222
export ETCD_ENDPOINTS=http://172.17.0.1:2379
export CUDA_VISIBLE_DEVICES=0
export DYN_KVBM_CPU_CACHE_GB=4
export DYN_KVBM_NIXL_BACKEND_UCX=true
export DYN_LOG=debug

python3 -m dynamo.frontend --http-port 8000 2>/dev/null &
sleep 5
python3 -m dynamo.trtllm \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --extra-engine-args /tmp/kvbm_v2_test.yaml 2>&1 | tee /tmp/trtllm.log
"
```

Wait ~90 seconds for model load. Server is ready when you see:
```
Registered endpoint 'dynamo.tensorrt_llm.generate'
```

### Verify startup in logs

Check for these lines:

```bash
docker exec kvbm-v2-test bash -c "grep -E 'layout=FullyContiguous|created_layouts|Registered endpoint' /tmp/trtllm.log"
```

Expected:
- `layout=FullyContiguous` — G1 registered with single stacked tensor
- `created_layouts=[G1, G2]` — NIXL registration complete
- `Registered endpoint` — server accepting requests

## Step 5: Send 10 Unique Requests + Repeat First

```bash
docker exec kvbm-v2-test bash -c '
PROMPTS=(
  "Please explain in detail how compiled programming languages such as C plus plus and Rust differ from interpreted ones like Python and JavaScript including their runtime behavior"
  "Can you describe the fundamental architectural differences between relational databases such as PostgreSQL and document stores like MongoDB including query capabilities and scaling"
  "What are the primary advantages and disadvantages of microservices architecture compared to monolithic application design patterns in terms of deployment and maintenance overhead"
  "How does the TCP IP networking protocol stack work from the physical layer through transport and application layers with examples of common protocols at each level"
  "Explain the concept of eventual consistency in distributed systems and how it contrasts with strong consistency models used in traditional relational database management systems"
  "What are the key differences between symmetric and asymmetric encryption algorithms and how are they typically combined in modern secure communication protocols like TLS"
  "Describe how garbage collection works in managed runtime environments like the Java Virtual Machine compared to manual memory management in languages like C and Rust"
  "How do modern CPU cache hierarchies with L1 L2 and L3 caches affect software performance and what programming patterns help maximize cache utilization and minimize misses"
  "Explain the differences between process based and thread based concurrency models and how async await patterns provide an alternative approach to handling concurrent operations"
  "What are the fundamental principles behind containerization technologies like Docker and how do they differ from traditional virtual machine based deployment strategies in practice"
)

for i in $(seq 0 9); do
  echo "=== R$((i+1)) ==="
  curl -s --max-time 60 localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPTS[$i]}\"}],\"stream\":false,\"max_tokens\":3}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(f\"prompt={r[\"usage\"][\"prompt_tokens\"]}, completion={r[\"usage\"][\"completion_tokens\"]}\")" 2>&1
done

echo "=== R11 (repeat R1 — should hit G2 cache) ==="
curl -s --max-time 60 localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPTS[0]}\"}],\"stream\":false,\"max_tokens\":3}" | python3 -c "import sys,json; r=json.load(sys.stdin); print(f\"prompt={r[\"usage\"][\"prompt_tokens\"]}, completion={r[\"usage\"][\"completion_tokens\"]}\")" 2>&1

echo "=== Done ==="
'
```

All 11 requests should return ~31-39 prompt tokens and 3 completion tokens.

## Step 6: Verify Offload and Onboard in Logs

### Verify offload (first request, G1→G2)

```bash
docker exec kvbm-v2-test bash -c "grep 'blocks_to_offload=1' /tmp/trtllm.log | head -1"
```

Expected:
```
offload: queuing blocks req_id="2048" blocks_to_offload=1
```

### Verify G2 cache hit + onboard (11th request, G2→G1)

```bash
docker exec kvbm-v2-test bash -c "grep -E 'Some\(32\), true|Onboarding completed|onboard complete' /tmp/trtllm.log"
```

Expected:
```
get_num_new_matched_tokens: return=Ok((Some(32), true)) request_id="XXXX" num_computed_tokens=0
Onboarding completed successfully
Worker received onboard complete request_id=XXXX
```

- `Some(32)` = 32 tokens matched from G2 (1 full block)
- `true` = async load (G2→G1 transfer)
- `Onboarding completed successfully` = transfer done

## Step 7: (Optional) Test Intra-Pass Onboarding

The default test uses **inter-pass** onboarding (async G2→G1 transfer between forward passes).
To test **intra-pass** onboarding (sync layer-by-layer transfer during the forward pass),
restart the server with `KVBM_ONBOARD_MODE=intra`.

### Stop existing server

```bash
docker exec kvbm-v2-test bash -c "pkill -9 -f 'dynamo.trtllm'; pkill -9 -f 'dynamo.frontend'"
```

### Restart with intra-pass mode

```bash
docker exec -d kvbm-v2-test bash -c "
export NATS_SERVER=nats://172.17.0.1:4222
export ETCD_ENDPOINTS=http://172.17.0.1:2379
export CUDA_VISIBLE_DEVICES=0
export DYN_KVBM_CPU_CACHE_GB=4
export DYN_KVBM_NIXL_BACKEND_UCX=true
export DYN_LOG=debug
export KVBM_ONBOARD_MODE=intra

python3 -m dynamo.frontend --http-port 8000 2>/dev/null &
sleep 5
python3 -m dynamo.trtllm \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --extra-engine-args /tmp/kvbm_v2_test.yaml 2>&1 | tee /tmp/trtllm_intra.log
"
```

Wait ~90 seconds. Verify startup shows:
```
ConnectorLeader initialized with onboard mode: Intra
```

### Re-run the 11-request test

Use the same requests from Step 5 (copy the full script).

### Verify intra-pass onboard in logs

```bash
docker exec kvbm-v2-test bash -c "grep -E 'onboard_mode.*Intra|Intra pass load: [1-9]|Some\(32\), false|intra.*pass.*onboard' /tmp/trtllm_intra.log"
```

Expected differences from inter-pass:

| | Inter-pass | Intra-pass |
|---|---|---|
| Onboard mode | `onboard mode: Inter` | `onboard mode: Intra` |
| `get_num_new_matched_tokens` return | `(Some(32), true)` | `(Some(32), false)` |
| Metadata | `Intra pass load: 0` | `Intra pass load: N` (non-zero) |
| Transfer timing | Between forward passes (async) | During forward pass, per-layer (sync) |

- `false` in `(Some(32), false)` means intra-pass — KV data loaded layer-by-layer during the forward pass
- `Intra pass load: N` in metadata summary means N blocks queued for intra-pass H2D transfer
- `start_load_kv` triggers the per-layer DMA, `wait_for_layer_load` synchronizes before attention

## Step 8: (Optional) Test Decode Block Offload + Onboard

This tests that KV cache blocks generated DURING decode (not just prefill)
are offloaded to G2 and can be onboarded back. Requires `KVBM_DECODE_OFFLOAD=true`.

### Stop existing server and restart with decode offload

```bash
docker exec kvbm-v2-test bash -c "pkill -9 -f 'dynamo.trtllm'; pkill -9 -f 'dynamo.frontend'"
```

```bash
docker exec -d kvbm-v2-test bash -c "
export NATS_SERVER=nats://172.17.0.1:4222
export ETCD_ENDPOINTS=http://172.17.0.1:2379
export CUDA_VISIBLE_DEVICES=0
export DYN_KVBM_CPU_CACHE_GB=4
export DYN_KVBM_NIXL_BACKEND_UCX=true
export DYN_LOG=debug
export KVBM_DECODE_OFFLOAD=true

python3 -m dynamo.frontend --http-port 8000 2>/dev/null &
sleep 5
python3 -m dynamo.trtllm \
  --model-path Qwen/Qwen3-0.6B \
  --served-model-name Qwen/Qwen3-0.6B \
  --extra-engine-args /tmp/kvbm_v2_test.yaml 2>&1 | tee /tmp/trtllm_decode.log
"
```

Wait ~90 seconds for server ready.

### Send R1 with 40 OSL (generates decode blocks)

Copy and run this Python script inside the container:

```python
import json, urllib.request

def send(messages, max_tokens):
    data = json.dumps({"model": "Qwen/Qwen3-0.6B", "messages": messages,
                       "stream": False, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request("http://localhost:8000/v1/chat/completions",
                                data=data, headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=120).read())

PROMPT = ("Please explain in great detail the complete history and evolution of "
          "programming languages from the earliest assembly languages and FORTRAN "
          "in the 1950s through the structured programming revolution with C and "
          "Pascal in the 1970s to the rise of object oriented languages like C++ "
          "Java and Python in the 1980s and 1990s and finally the modern era of "
          "Rust Go Kotlin and TypeScript. Discuss the key innovations that each "
          "generation brought")

# R1: 40 decode tokens — should offload 3 prefill + 1 decode block
print("=== R1 ===")
r1 = send([{"role": "user", "content": PROMPT}], 40)
print(f"prompt={r1['usage']['prompt_tokens']}, completion={r1['usage']['completion_tokens']}")
response_text = r1["choices"][0]["message"]["content"]

# 10 fillers to evict R1 blocks from G1
FILLERS = [
    "Can you provide a comprehensive overview of distributed systems architecture including the CAP theorem and its implications for system design the differences between strong and eventual consistency",
    "What are the fundamental principles behind modern machine learning including supervised unsupervised and reinforcement learning approaches explain how neural networks work from basic perceptrons",
    "Describe the complete software development lifecycle from requirements gathering and system design through implementation testing deployment and maintenance cover agile methodologies like Scrum",
    "Explain the principles of computer networking from the physical layer through the application layer including Ethernet WiFi TCP IP UDP DNS HTTP HTTPS TLS and modern protocols like QUIC",
    "What are the key differences between symmetric and asymmetric encryption algorithms and how are they typically combined in modern secure communication protocols like TLS and SSH",
    "Describe how garbage collection works in managed runtime environments like the Java Virtual Machine compared to manual memory management in languages like C and Rust with examples",
    "How do modern CPU cache hierarchies with L1 L2 and L3 caches affect software performance and what programming patterns help maximize cache utilization and minimize cache misses",
    "Explain the differences between process based and thread based concurrency models and how async await patterns provide an alternative approach to handling concurrent operations efficiently",
    "What are the fundamental principles behind containerization technologies like Docker and how do they differ from traditional virtual machine based deployment strategies in modern cloud computing",
    "Describe the evolution of web development from static HTML pages through server side rendering with PHP and Rails to modern single page applications with React Vue and Angular frameworks",
]
for i, f in enumerate(FILLERS):
    print(f"=== Filler {i+2} ===")
    r = send([{"role": "user", "content": f}], 40)
    print(f"prompt={r['usage']['prompt_tokens']}, completion={r['usage']['completion_tokens']}")

# R12: prompt + response + continue — should match 128 tokens (3 prefill + 1 decode block)
print("\n=== R12: prompt + response (should match 128 tokens from G2) ===")
r12 = send([
    {"role": "user", "content": PROMPT},
    {"role": "assistant", "content": response_text},
    {"role": "user", "content": "Please continue your explanation"}
], 3)
print(f"prompt={r12['usage']['prompt_tokens']}, completion={r12['usage']['completion_tokens']}")
print("Done")
```

### Verify decode block offload + onboard in logs

```bash
# R1 should offload 4 blocks (3 prefill + 1 decode)
docker exec kvbm-v2-test bash -c "sed 's/\x1b\[[0-9;]*m//g' /tmp/trtllm_decode.log | grep 'req_id=\"2048\".*blocks_to_offload' | grep -v 'blocks_to_offload=0' | sort -u"
```

Expected (two offload events):
```
blocks_to_offload=3 iteration=1    ← prefill blocks
blocks_to_offload=1 iteration=29   ← decode block
```

```bash
# R12 should match 128 tokens (4 blocks) from G2
docker exec kvbm-v2-test bash -c "sed 's/\x1b\[[0-9;]*m//g' /tmp/trtllm_decode.log | grep 'get_num_new_matched_tokens.*return' | tail -1"
```

Expected:
```
return=Ok((Some(128), true))   ← 128 tokens = 3 prefill + 1 decode block from G2
```

Without decode offload this would be `Some(96)` (3 prefill blocks only).

## Step 9: Cleanup

```bash
docker stop kvbm-v2-test && docker rm kvbm-v2-test
cd /home/oandreeva/Code/dynamo && docker compose -f deploy/docker-compose.yml down
```

## Pass Criteria

### Inter-pass (default)

| Check | Expected |
|-------|----------|
| All 11 requests return valid responses | prompt=31-39, completion=3 |
| First request offloads 1 block | `blocks_to_offload=1` in logs |
| 11th request gets G2 cache hit | `Some(32), true)` in logs |
| Onboard completes | `Onboarding completed successfully` |
| No `Sampling failed` errors | `grep 'Sampling failed' /tmp/trtllm.log` returns nothing |

### Intra-pass (optional)

| Check | Expected |
|-------|----------|
| Onboard mode is intra | `onboard mode: Intra` in startup logs |
| All 11 requests return valid responses | prompt=31-39, completion=3 |
| 11th request gets G2 cache hit (sync) | `Some(32), false)` in logs |
| Metadata has intra-pass load | `Intra pass load: N` with N > 0 |
| No `Sampling failed` errors | `grep 'Sampling failed' /tmp/trtllm_intra.log` returns nothing |

### Decode offload (optional)

| Check | Expected |
|-------|----------|
| R1 offloads 4 blocks (3 prefill + 1 decode) | Two offload events: `blocks_to_offload=3` then `blocks_to_offload=1` |
| Decode block offloaded at iteration ~29 | `evaluated_blocks_after=4` at iteration ~29 |
| R12 matches 128 tokens from G2 | `Some(128), true)` (inter) or `Some(128), false)` (intra) |
| No errors | 0 `Sampling failed`, 0 `Precondition timeout` |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `NATS: Connection refused` | Infrastructure not running | `docker compose -f deploy/docker-compose.yml up -d` |
| `NIXL agent not found` | Missing env var | `export DYN_KVBM_NIXL_BACKEND_UCX=true` |
| `No cache tier configured` | Missing env var | `export DYN_KVBM_CPU_CACHE_GB=4` |
| `blocks_to_offload=0` on all requests | Prompts too short (<32 tokens) | Use provided prompts (35+ tokens) |
| `Some(0), false)` on repeat request | G1 not exhausted | Send more unique requests before repeating |
| No offload/onboard logs | Log level too low | `export DYN_LOG=debug` |
| `Sampling failed` | Stale forward_pass_completion_active | Ensure Rust fix is applied (reset flag in bind_connector_metadata) |
| `Precondition timeout after 30s` | Piecewise CUDA graphs enabled | Disable `enable_piecewise_cuda_graph` — incompatible with KVBM v2 offload (`save_kv_layer` doesn't fire during graph replay) |
