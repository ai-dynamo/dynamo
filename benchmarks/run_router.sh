#!/bin/bash

python -m dynamo.frontend \
    --router-mode kv \
    --kv-cache-block-size 64 \
    --http-port 8080