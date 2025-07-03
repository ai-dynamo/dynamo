dynamo-ctl
----------

```bash
sudo rm -rf ./target
sudo rm -rf /tmp/nixl/nixl_src
sudo ./container/build.sh
./container/run.sh -it --mount-workspace
```

```bash
nats-server -js &
etcd --listen-client-urls http://0.0.0.0:2379 --advertise-client-urls http://0.0.0.0:2379 --data-dir /tmp/etcd &

cd examples/llm
dynamo serve graphs.agg:Frontend -f configs/agg.yaml

cargo run --bin dynamo-ctl

curl localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [
    {
        "role": "user",
        "content": "Hello, how are you?"
    }
    ],
    "stream":false,
    "max_tokens": 300
  }' | jq
```

TODO:

- Preload existing keys on startup
