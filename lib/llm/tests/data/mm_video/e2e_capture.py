# SPDX-License-Identifier: Apache-2.0
"""Capture vLLM KV events for a video request, for Dynamo MM-routing parity.

Runs a real vLLM engine with ZMQ KV events, submits one Qwen2.5-VL video
request with a fixed multi_modal_uuid (the frontend-forwarded mm_hash), and
dumps the raw event payloads plus everything the Rust parity test needs to
rebuild the frontend routing stream independently.
"""
import base64
import json
import os
import threading

import numpy as np
import zmq

MODEL = "Qwen/Qwen2.5-VL-3B-Instruct"
# vLLM's publisher binds only when the endpoint contains a wildcard
# (kv_events.py "bind if wildcard" heuristic); the subscriber connects.
PUB_ENDPOINT = "tcp://*:56797"
SUB_ENDPOINT = "tcp://127.0.0.1:56797"
MM_HASH_U64 = 0xABCDEF0123456789
UUID_HEX = f"{MM_HASH_U64:016x}".ljust(64, "0")
T, H, W = 8, 224, 224
BLOCK_SIZE = 16
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "e2e_capture.json")

payloads = []
stop = threading.Event()


def subscriber():
    ctx = zmq.Context()
    s = ctx.socket(zmq.SUB)
    s.connect(SUB_ENDPOINT)
    s.setsockopt_string(zmq.SUBSCRIBE, "")
    s.setsockopt(zmq.RCVTIMEO, 500)
    while not stop.is_set():
        try:
            parts = s.recv_multipart()
        except zmq.Again:
            continue
        payloads.append(base64.b64encode(parts[-1]).decode())
    s.close()
    ctx.term()


def main():
    sub = threading.Thread(target=subscriber, daemon=True)
    sub.start()

    from huggingface_hub import snapshot_download
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from vllm.config import KVEventsConfig

    local = snapshot_download(MODEL, allow_patterns=["*.json", "*.txt", "*.model"])
    proc = AutoProcessor.from_pretrained(local)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video"},
                {"type": "text", "text": "Describe the video."},
            ],
        }
    ]
    text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    pre_ids = proc.tokenizer(text, add_special_tokens=False).input_ids

    llm = LLM(
        model=MODEL,
        max_model_len=4096,
        gpu_memory_utilization=0.5,
        limit_mm_per_prompt={"video": 1},
        block_size=BLOCK_SIZE,
        kv_events_config=KVEventsConfig(
            enable_kv_cache_events=True,
            publisher="zmq",
            endpoint=PUB_ENDPOINT,
        ),
    )

    import time

    # Warm-up text request: vLLM starts its ZMQ publisher thread lazily on the
    # first request, and a fresh SUB socket drops messages published during
    # the join window (slow joiner). Let the publisher come up, then wait for
    # the subscription to propagate before the request we actually capture.
    llm.generate(
        [{"prompt": "Warm up."}], SamplingParams(max_tokens=1, temperature=0.0)
    )
    time.sleep(3)

    rng = np.random.default_rng(42)
    video = rng.integers(0, 255, size=(T, H, W, 3), dtype=np.uint8)
    req = {
        "prompt": text,
        "multi_modal_data": {"video": video},
        "multi_modal_uuids": {"video": [UUID_HEX]},
    }
    out = llm.generate([req], SamplingParams(max_tokens=8, temperature=0.0))
    prompt_token_ids = list(out[0].prompt_token_ids)

    time.sleep(3)  # let the publisher flush
    stop.set()
    sub.join(timeout=5)

    cfg = json.load(open(os.path.join(local, "config.json")))
    json.dump(
        {
            "model_id": MODEL,
            "local_dir": local,
            "mm_hash_u64": MM_HASH_U64,
            "uuid_hex": UUID_HEX,
            "num_frames": T,
            "height": H,
            "width": W,
            "kv_block_size": BLOCK_SIZE,
            "video_token_id": cfg["video_token_id"],
            "image_token_id": cfg.get("image_token_id"),
            "pre_expansion_input_ids": pre_ids,
            "engine_prompt_token_ids": prompt_token_ids,
            "raw_event_payloads_b64": payloads,
        },
        open(OUT, "w"),
    )
    n_video = sum(1 for x in prompt_token_ids if x == cfg["video_token_id"])
    print("engine prompt len:", len(prompt_token_ids), "video tokens:", n_video)
    print("captured payloads:", len(payloads))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
