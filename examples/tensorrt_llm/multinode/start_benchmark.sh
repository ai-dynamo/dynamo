#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


bash -x /benchmarks/llm/perf.sh \
 --mode $MODE \
 --deployment-kind $KIND \
 --tensor-parallelism $TP \
 --data-parallelism $DP \
 --prefill-tensor-parallelism $PREFILL_TP \
 --prefill-data-parallelism $PREFILL_DP \
 --decode-tensor-parallelism $DECODE_TP \
 --decode-data-parallelism $DECODE_DP \
 --model $SERVED_MODEL_NAME \
 --input-sequence-length $INPUT_SEQ_LEN \
 --output-sequence-length $OUTPUT_SEQ_LEN \
 --url http://${HEAD_NODE_IP}:8000 \
 --concurrency $CONCURRENCY \
 --artifacts-root-dir $ARTIFACTS_ROOT_DIR
