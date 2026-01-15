#!/bin/bash
#DYN_SYSTEM_PORT=9090 python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --reasoning-parser deepseek-r1 --tool-call-parser qwen
python -m dynamo.frontend --router-mode kv &
#DYN_SYSTEM_PORT=9090 python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --page-size 16 &
DYN_SYSTEM_PORT=9090 python -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --reasoning-parser deepseek-r1 --tool-call-parser qwen --page-size 16 &
wait
