### Baseline TP4 OSS



```bash
vllm serve neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic \
    --tensor-parallel-size 4
```


### Head node setup

```
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379
nats-server -js -p 4222 -m 8222
```


```bash
dynamo serve --service-name Frontend components.frontend:Frontend --Frontend.served_model_name=neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic --Frontend.endpoint=dynamo.Processor.chat/completions --Frontend.port=8000
```

```bash
dynamo serve --service-name Processor components.processor:Processor --Processor.model=neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic --Processor.router=round-robin
```mv


### Head node with TP4DP2 generate

Set environment variables on the compute node:

```bash
export ETCD_ENDPOINTS=http://<your head node>:2379
export NATS_SERVER=nats://<your head node>:4222
```

Token generator configuration:

```yaml
VllmWorker:
  ServiceArgs:
    workers: 2
    resources:
      gpu: 4

  # vLLM engine arguments
  model: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
  kv-transfer-config: '{"kv_connector":"DynamoNixlConnector"}'
  max-model-len: 3500
  load-format: auto
  disable-log-requests: true
  tensor-parallel-size: 4

  # Dynamo arguments
  remote-prefill: true
  conditional-disagg: false
  router: round-robin
```

```bash
dynamo serve --service-name VllmWorker components.worker:VllmWorker -f /workspace/examples/llm/benchmarks/disagg_generate_tp4dp2.yaml
```
# Compute nodes with TP1DP8 prefill

You need two of such nodes to have the best performance.

Set environment variables on the compute node:

```bash
export ETCD_ENDPOINTS=http://<your head node>:2379
export NATS_SERVER=nats://<your head node>:4222
```

Create a yaml file for the prefill worker:

```yaml
# "PrefillWorker" is for the prefill stage of the pipeline.
PrefillWorker:
  ServiceArgs:
    workers: 8
    resources:
      gpu: 1

  # vLLM engine arguments
  model: neuralmagic/DeepSeek-R1-Distill-Llama-70B-FP8-dynamic
  kv-transfer-config: '{"kv_connector":"DynamoNixlConnector"}'
  max-model-len: 3500
  max-num-batched-tokens: 3500
  max-num-seqs: 2
  load-format: auto
  gpu-memory-utilization: 0.95
  disable-log-requests: true
  tensor-parallel-size: 1
```

```bash
dynamo serve --service-name PrefillWorker components.prefill_worker:PrefillWorker -f /workspace/examples/llm/benchmarks/disagg_prefill_tp1dp8.yaml
```

### Run the benchmark

You should use the following command to run the benchmark for OSS vLLM for aggregated prefill and decode stages:

```bash
bash benchmarks/perf_for_graph.sh vll_tp4dp1
```

You should name the benchmark with the following format to indicate how many TP and DP are used for disaggregated prefill and decode stages:

```
bash benchmarks/perf_for_graph.sh prefill_tp1dp16_decode_tp4dp2_disaggregated
```

### Graphs generation

You need mathplotlib and seaborn installed to generate the graphs:

```bash
python3 examples/llm/multinode_graphs/process_results.py \
    <your folder with results> \
    "N70B FP8 NVIDIA H100 ISL/OSL 3000/150" \
    p90 \
    --footnote "The results shown in this graph are provided as examples only. Actual performance and outcomes may vary depending on your specific hardware and software configuration" \
    --output-tokens-per-request 150
```

The script