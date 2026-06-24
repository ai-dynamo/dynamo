# Dynamo Megatron Backend

This package is a Dynamo worker that proxies streaming generation requests to
a separately-launched [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
`DataParallelInferenceCoordinator`.

**Status:** [WIP] : aggregated decode, streaming tokens. No KV events, no
metrics publishing, no disaggregated prefill/decode split

## Layout

```
dynamo/megatron/
  __init__.py
  __main__.py             # `python -m dynamo.megatron`
  args.py                 # CLI Config
  engine_client.py        # Async wrapper around megatron InferenceClient
  handlers.py             # DecodeWorkerHandler — request translation + streaming
  main.py                 # worker() — endpoint registration + serve loop
```

## Dependencies

- `megatron-core` must be importable in this worker's Python environment. The
  worker imports `megatron.core.inference.inference_client.InferenceClient`
  and `megatron.core.inference.sampling_params.SamplingParams` directly.
- `pyzmq`, `msgpack` — pulled in transitively by `megatron-core`.
- The Dynamo runtime (`dynamo.runtime`, `dynamo.llm`) — already a peer
  dependency for any Dynamo backend.

## End-to-end flow

1. Launch the Megatron coordinator on the model node(s):

   ```
   torchrun --nproc-per-node 2 tools/run_dynamic_text_generation_server.py \
       --frontend dynamo \
       --inference-coordinator-port 5555 \
       --tensor-model-parallel-size 2 \
       --load <megatron-checkpoint> \
       --tokenizer-type HuggingFaceTokenizer \
       --tokenizer-model <hf-model-id>
       # ... other Megatron flags
   ```

   The launcher prints `MEGATRON_COORDINATOR_ADDR=tcp://<host>:<port>` to
   stdout. Capture it.

2. Launch the Dynamo Megatron worker against that coordinator:

   ```
   python -m dynamo.megatron \
       --coordinator-addr "$MEGATRON_COORDINATOR_ADDR" \
       --model <hf-model-id> \
       --context-length 4096
   ```

3. Launch a Dynamo frontend and test:

   ```
   curl -N http://<frontend>/v1/chat/completions \
       -H 'content-type: application/json' \
       -d '{"model": "<hf-model-id>", "messages": [...], "stream": true}'
   ```

   Tokens should stream back. The path is: frontend tokenizes → tokens to
   `dynamo.backend.generate` → Megatron worker handler → InferenceClient
   `add_request_streaming` → Megatron coordinator → engine emits
   `ENGINE_REPLY_PARTIAL` per step → handler yields `{token_ids: [...]}` per
   chunk back to the frontend.