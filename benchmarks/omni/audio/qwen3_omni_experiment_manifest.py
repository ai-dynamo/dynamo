#!/usr/bin/env python3
# Generate Qwen3-Omni single-pod async-disagg experiment manifests.

from __future__ import annotations

import argparse
import math
import shlex
from pathlib import Path

IMAGE = "nvcr.io/nvidian/dynamo-dev/dynamo:vllm-runtime-ptarasiewicz-dyn3068-qwen3-async-38584b4-r20-current-omni-20260602"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
BENCH_CONFIGMAP = "qwen3-omni-audio-benchmark-script"


def app_name(thinkers: int, talkers: int, code2wav: int) -> str:
    return f"qwen3-omni-async-{thinkers}h{talkers}t{code2wav}c"


def stage_lines(
    thinkers: int,
    talkers: int,
    code2wav: int,
    share_code2wav_with_last_talker: bool = False,
) -> str:
    lines: list[str] = []
    gpu = 0
    port = 8081
    for index in range(thinkers):
        lines.append(f"    start_stage 0 thinker-{index} {gpu} {port}")
        gpu += 1
        port += 1
    lines.append("")
    for index in range(talkers):
        lines.append(f"    start_stage 1 talker-{index} {gpu} {port}")
        gpu += 1
        port += 1
    lines.append("")
    for index in range(code2wav):
        if share_code2wav_with_last_talker:
            if talkers < 1:
                raise SystemExit("cannot share code2wav without a talker")
            code_gpu = thinkers + talkers - 1
        else:
            code_gpu = gpu
            gpu += 1
        lines.append(f"    start_stage 2 code2wav-{index} {code_gpu} {port}")
        port += 1
    return "\n".join(lines)


def deployment_yaml(
    thinkers: int,
    talkers: int,
    code2wav: int,
    share_code2wav_with_last_talker: bool = False,
) -> str:
    total_gpus = (
        thinkers + talkers + (0 if share_code2wav_with_last_talker else code2wav)
    )
    if total_gpus > 8:
        raise SystemExit("single-node H200 experiments support at most 8 GPUs")
    if min(thinkers, talkers, code2wav) < 1:
        raise SystemExit("each stage count must be at least 1")

    name = app_name(thinkers, talkers, code2wav)
    mem_request = math.ceil(total_gpus * 112.5)
    mem_limit = total_gpus * 175
    dshm = total_gpus * 8
    stages = stage_lines(thinkers, talkers, code2wav, share_code2wav_with_last_talker)

    return f"""# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Generated Qwen3-Omni async disaggregated experiment: {thinkers} thinker, {talkers} talker, {code2wav} code2wav.
apiVersion: v1
kind: ConfigMap
metadata:
  name: {name}-launcher
data:
  launch.sh: |
    #!/usr/bin/env bash
    set -euo pipefail
    APP_NAME={name}
    trap "echo [${{APP_NAME}}] cleaning up; kill 0" EXIT

    MODEL="${{MODEL:-{MODEL}}}"
    BASE_STAGE_CONFIG="/usr/local/lib/python3.12/dist-packages/vllm_omni/deploy/qwen3_omni_moe.yaml"
    STAGE_CONFIG_DIR="/tmp/qwen3_omni_stage_configs"

    export HF_HOME=/home/dynamo/.cache/huggingface
    export FLASHINFER_DISABLE_VERSION_CHECK=1
    export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
    export OMP_NUM_THREADS="${{OMP_NUM_THREADS:-1}}"
    export MKL_NUM_THREADS="${{MKL_NUM_THREADS:-1}}"
    export OPENBLAS_NUM_THREADS="${{OPENBLAS_NUM_THREADS:-1}}"
    export NUMEXPR_NUM_THREADS="${{NUMEXPR_NUM_THREADS:-1}}"
    export RAYON_NUM_THREADS="${{RAYON_NUM_THREADS:-1}}"
    export TOKENIZERS_PARALLELISM="${{TOKENIZERS_PARALLELISM:-false}}"
    export DYN_NAMESPACE="${{DYN_NAMESPACE:-{name}}}"
    export DYN_HEALTH_CHECK_ENABLED="${{DYN_HEALTH_CHECK_ENABLED:-false}}"
    export DYN_DISCOVERY_BACKEND="${{DYN_DISCOVERY_BACKEND:-file}}"
    export DYN_FILE_KV="${{DYN_FILE_KV:-/tmp/dynamo_store_kv}}"
    export DYN_REQUEST_PLANE="${{DYN_REQUEST_PLANE:-tcp}}"
    export DYN_EVENT_PLANE="${{DYN_EVENT_PLANE:-zmq}}"

    OMNI_FLAGS=(
      --model "$MODEL"
      --output-modalities text audio
      --media-output-fs-url file:///tmp/dynamo_media
      --trust-remote-code
    )

    make_stage_config() {{
      local stage_id="$1"
      local gpu_id="$2"
      local out="${{STAGE_CONFIG_DIR}}/stage_${{stage_id}}_${{gpu_id}}.yaml"
      mkdir -p "${{STAGE_CONFIG_DIR}}"
      python3 - "${{BASE_STAGE_CONFIG}}" "${{out}}" "${{stage_id}}" "${{gpu_id}}" <<PY
    import sys
    import yaml

    src, out, stage_id, gpu_id = sys.argv[1:5]
    stage_id = int(stage_id)
    with open(src, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for stage in cfg.get("stages", []):
        if int(stage.get("stage_id")) == stage_id:
            stage["devices"] = str(gpu_id)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    PY
      printf "%s\\\\n" "${{out}}"
    }}

    start_stage() {{
      local stage_id="$1"
      local stage_name="$2"
      local gpu_id="$3"
      local system_port="$4"
      echo "[${{APP_NAME}}] starting stage ${{stage_id}} ${{stage_name}} on GPU ${{gpu_id}}"
      local stage_config
      stage_config="$(make_stage_config "${{stage_id}}" "${{gpu_id}}")"
      CUDA_VISIBLE_DEVICES="${{gpu_id}}" DYN_SYSTEM_PORT="${{system_port}}" \
        python3 -m dynamo.vllm.omni --stage-id "${{stage_id}}" --stage-configs-path "${{stage_config}}" "${{OMNI_FLAGS[@]}}" &
      sleep "${{STAGE_START_DELAY:-15}}"
    }}

    echo "[${{APP_NAME}}] namespace=${{DYN_NAMESPACE}}"
    echo "[${{APP_NAME}}] base_stage_config=${{BASE_STAGE_CONFIG}}"
{stages}

    echo "[${{APP_NAME}}] starting omni router"
    DYN_SYSTEM_PORT=8090 python3 -m dynamo.vllm.omni --omni-router --stage-configs-path "${{BASE_STAGE_CONFIG}}" "${{OMNI_FLAGS[@]}}" &
    sleep 5

    echo "[${{APP_NAME}}] starting frontend on :8000"
    python3 -m dynamo.frontend &

    wait -n
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: {name}
spec:
  replicas: 1
  selector:
    matchLabels:
      app: {name}
  template:
    metadata:
      labels:
        app: {name}
    spec:
      securityContext:
        runAsUser: 1000
        runAsGroup: 0
        fsGroup: 1000
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H200
      imagePullSecrets:
        - name: nvcr-dynamo-dev-pullsecret
        - name: nvcr-imagepullsecret
      containers:
        - name: dynamo-omni
          image: {IMAGE}
          imagePullPolicy: Always
          command: ["bash", "/etc/dynamo/launcher/launch.sh"]
          ports:
            - name: http
              containerPort: 8000
          envFrom:
            - secretRef:
                name: hf-token-secret
          env:
            - name: HF_HOME
              value: /home/dynamo/.cache/huggingface
            - name: DYN_NAMESPACE
              value: {name}
            - name: DYN_HEALTH_CHECK_ENABLED
              value: "false"
            - name: DYN_DISCOVERY_BACKEND
              value: file
            - name: DYN_FILE_KV
              value: /tmp/dynamo_store_kv
            - name: DYN_REQUEST_PLANE
              value: tcp
            - name: DYN_EVENT_PLANE
              value: zmq
            - name: OMP_NUM_THREADS
              value: "1"
            - name: MKL_NUM_THREADS
              value: "1"
            - name: OPENBLAS_NUM_THREADS
              value: "1"
            - name: NUMEXPR_NUM_THREADS
              value: "1"
            - name: RAYON_NUM_THREADS
              value: "1"
            - name: TOKENIZERS_PARALLELISM
              value: "false"
          resources:
            requests:
              memory: {mem_request}Gi
              nvidia.com/gpu: "{total_gpus}"
            limits:
              memory: {mem_limit}Gi
              nvidia.com/gpu: "{total_gpus}"
          volumeMounts:
            - name: model-cache
              mountPath: /home/dynamo/.cache/huggingface
            - name: launcher
              mountPath: /etc/dynamo/launcher
              readOnly: true
            - name: dshm
              mountPath: /dev/shm
            - name: media
              mountPath: /tmp/dynamo_media
      volumes:
        - name: model-cache
          persistentVolumeClaim:
            claimName: model-cache
        - name: launcher
          configMap:
            name: {name}-launcher
            defaultMode: 0755
        - name: dshm
          emptyDir:
            medium: Memory
            sizeLimit: {dshm}Gi
        - name: media
          emptyDir: {{}}
---
apiVersion: v1
kind: Service
metadata:
  name: {name}
spec:
  type: ClusterIP
  selector:
    app: {name}
  ports:
    - name: http
      port: 8000
      targetPort: 8000
"""


def benchmark_job_yaml(
    thinkers: int,
    talkers: int,
    code2wav: int,
    *,
    tag: str,
    prompt: str,
    max_tokens: int,
    concurrency: str,
    request_timeout_s: int,
    requests_multiplier: int,
    min_requests: int,
    warmup: int,
    voice: str,
) -> str:
    name = app_name(thinkers, talkers, code2wav)
    job_name = f"{name}-bench-{tag}"[:63].rstrip("-")
    result_path = "/tmp/benchmark-result.json"
    args = [
        "python3",
        "/bench/chat_audio_benchmark.py",
        "--url",
        f"http://{name}:8000",
        "--prompt",
        prompt,
        "--max-tokens",
        str(max_tokens),
        "--concurrency",
        concurrency,
        "--request-timeout-s",
        str(request_timeout_s),
        "--requests-multiplier",
        str(requests_multiplier),
        "--min-requests",
        str(min_requests),
        "--warmup",
        str(warmup),
        "--voice",
        voice,
        "--output-json",
        result_path,
    ]
    cmd = " ".join(shlex.quote(arg) for arg in args)
    cmd = f"{cmd}; status=$?; echo __RESULT_JSON__; cat {result_path} || true; exit $status"
    return f"""apiVersion: batch/v1
kind: Job
metadata:
  name: {job_name}
spec:
  backoffLimit: 0
  template:
    metadata:
      labels:
        app: {job_name}
    spec:
      restartPolicy: Never
      nodeSelector:
        nvidia.com/gpu.product: NVIDIA-H200
      tolerations:
        - key: nvidia.com/gpu
          operator: Exists
          effect: NoSchedule
      imagePullSecrets:
        - name: nvcr-dynamo-dev-pullsecret
        - name: nvcr-imagepullsecret
      containers:
        - name: benchmark
          image: {IMAGE}
          imagePullPolicy: IfNotPresent
          command:
            - /bin/bash
            - -lc
            - {cmd!r}
          resources:
            requests:
              cpu: "2"
              memory: 4Gi
            limits:
              cpu: "8"
              memory: 16Gi
          volumeMounts:
            - name: benchmark-script
              mountPath: /bench
              readOnly: true
      volumes:
        - name: benchmark-script
          configMap:
            name: {BENCH_CONFIGMAP}
            defaultMode: 0755
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--thinkers", type=int, required=True)
    parser.add_argument("--talkers", type=int, required=True)
    parser.add_argument("--code2wav", type=int, required=True)
    parser.add_argument("--kind", choices=["deployment", "job"], required=True)
    parser.add_argument("--share-code2wav-with-last-talker", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tag", default="heavy")
    parser.add_argument("--prompt", default="Say the word benchmark in a clear voice.")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrency", default="64,128")
    parser.add_argument("--request-timeout-s", type=int, default=600)
    parser.add_argument("--requests-multiplier", type=int, default=2)
    parser.add_argument("--min-requests", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--voice", default="Chelsie")
    args = parser.parse_args()

    if args.kind == "deployment":
        text = deployment_yaml(
            args.thinkers,
            args.talkers,
            args.code2wav,
            args.share_code2wav_with_last_talker,
        )
    else:
        text = benchmark_job_yaml(
            args.thinkers,
            args.talkers,
            args.code2wav,
            tag=args.tag,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency,
            request_timeout_s=args.request_timeout_s,
            requests_multiplier=args.requests_multiplier,
            min_requests=args.min_requests,
            warmup=args.warmup,
            voice=args.voice,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
