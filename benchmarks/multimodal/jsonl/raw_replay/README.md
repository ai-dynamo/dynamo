# Custom multi-turn JSONL Request Generator

Generates a directory of .jsonl files for [aiperf](https://github.com/ai-dynamo/aiperf) to use with Raw Payload Replay. The intended usecase is for multi-turn chats that need customization at each turn. More details can be found in the AIPerf documentation [here](https://github.com/ai-dynamo/aiperf/blob/6627dca940d33ad167c3764106d38ea84137092e/docs/tutorials/raw-payload-replay.md)

## Usage

```bash
python generate_raw_replay.py \
    --config raw_replay.yaml \
    --output-dir test_payload \
    --num-conversations 5 \
    --image-pool-size 20 \
    --image-mode http \
    --seed 42
```

Output is a directory of per-session JSONL files, e.g. `test_payload/session_000001.jsonl`.

## Running with aiperf

```bash
aiperf profile \
    --input-file test_payload \
    --model test \
    --custom-dataset-type raw_payload \
    --endpoint-type raw \
    --url localhost:8000/v1/chat/completions \
    --concurrency 5
```