This directory contains configs used for the DeepEP branch from SGLang. You can use it from inside of the Dockerfile.sglang-deepep container.

Say you'd like to run 1P1D (2 nodes for prefill, 2 nodes for decode). Once you are in the deepep container - you can run the following commands:

# Setup
```bash
# nagivate to the sglang directory
cd dynamo/examples/sglang
```

# Prefill Worker
```bash
# node 1
# ensure you are swapping the dist-init-addr to the correct address 
dynamo serve graphs.agg:Frontend -f configs/head.yaml
```

```bash
# node 2
# ensure you are swapping the dist-init-addr to the correct address 
dynamo serve graphs.agg:Frontend -f configs/head.yaml --service-name SGLangWorker
```

# Decode Worker
```bash
# node 3
# ensure you are swapping the dist-init-addr to the correct address 
dynamo serve graphs.disagg:Frontend -f configs/head.yaml --service-name SGLangDecodeWorker
```

```bash
# node 4
# ensure you are swapping the dist-init-addr to the correct address 
dynamo serve graphs.agg:Frontend -f configs/head.yaml --service-name SGLangDecodeWorker
```