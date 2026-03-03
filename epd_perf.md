aiperf profile -m 'Qwen/Qwen3-VL-30B-A3B-Instruct-FP8' --endpoint-type 'chat' -u 'localhost:8000' --streaming --request-count 100 --warmup-request-count 2 --concurrency 16 --osl 500 --input-file '/tmp/data_small.jsonl'
--custom-dataset-type 'single_turn' --ui None --no-server-metrics


Start server

