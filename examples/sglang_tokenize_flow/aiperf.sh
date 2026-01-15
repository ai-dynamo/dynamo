#!/bin/bash
aiperf profile --model "Qwen/Qwen3-0.6B" --url "localhost:8000" --endpoint-type "chat" --streaming --isl
8000
