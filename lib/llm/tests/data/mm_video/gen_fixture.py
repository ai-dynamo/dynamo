# SPDX-License-Identifier: Apache-2.0
"""Generate MM-routing video fixtures from the real HF processors.

For each model and (T, H, W) case, runs the HF video pipeline the backend
(vLLM) would run and records everything the Rust `VideoTokenCounter` +
expansion must reproduce:
  - total video token count (count of video_token_id in expanded input_ids)
  - the full expanded input_ids for a 1-video chat prompt
  - the pre-expansion input_ids (placeholder triple / single token intact)
  - video_grid_thw, per-frame timestamps handed to the processor

Torch CPU only; downloads config/tokenizer JSONs (no weights).
"""
import json
import os
import sys

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoProcessor

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

CASES = [(8, 224, 224), (5, 224, 224), (16, 360, 640), (12, 480, 854), (64, 1080, 1920)]
FPS = 2.0  # sampled fps we pretend our frontend decoder used


def frames(t, h, w):
    rng = np.random.default_rng(1234)
    return rng.integers(0, 255, size=(t, h, w, 3), dtype=np.uint8)


def chat_text(proc, num_videos=1):
    content = [{"type": "video"}] * num_videos + [{"type": "text", "text": "Describe."}]
    messages = [{"role": "user", "content": content}]
    return proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_model(model_id, out_name):
    local = snapshot_download(model_id, allow_patterns=["*.json", "*.txt", "*.model"])
    proc = AutoProcessor.from_pretrained(local)
    cfg = json.load(open(os.path.join(local, "config.json")))
    video_token_id = cfg["video_token_id"]
    fixture = {
        "model_id": model_id,
        "local_dir": local,
        "video_token_id": video_token_id,
        "vision_start_token_id": cfg.get("vision_start_token_id"),
        "vision_end_token_id": cfg.get("vision_end_token_id"),
        "transformers_version": __import__("transformers").__version__,
        "fps": FPS,
        "cases": [],
    }
    text = chat_text(proc)
    pre_ids = proc.tokenizer(text, add_special_tokens=False).input_ids
    fixture["pre_expansion_input_ids"] = pre_ids

    for (t, h, w) in CASES:
        video = frames(t, h, w)
        ts = [i / FPS for i in range(t)]
        metadata = {
            "fps": FPS,
            "total_num_frames": t,
            "duration": t / FPS,
            "frames_indices": list(range(t)),
            "video_backend": "opencv",
        }
        kwargs = dict(
            text=[text],
            videos=[torch.from_numpy(video).permute(0, 3, 1, 2)],
            video_metadata=[metadata],
            return_tensors="pt",
            do_sample_frames=False,
        )
        try:
            out = proc(**kwargs)
        except TypeError:
            kwargs.pop("do_sample_frames")
            out = proc(**kwargs)
        input_ids = out["input_ids"][0].tolist()
        grid = out["video_grid_thw"][0].tolist()
        n_video_tokens = sum(1 for x in input_ids if x == video_token_id)
        fixture["cases"].append(
            {
                "num_frames": t,
                "height": h,
                "width": w,
                "sampled_timestamps": ts,
                "video_grid_thw": grid,
                "n_video_tokens": n_video_tokens,
                "expanded_input_ids": input_ids,
            }
        )
        print(f"{model_id} T={t} {h}x{w}: grid={grid} video_tokens={n_video_tokens}")

    path = os.path.join(OUT_DIR, out_name)
    json.dump(fixture, open(path, "w"))
    print("wrote", path)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("all", "qwen25"):
        run_model("Qwen/Qwen2.5-VL-3B-Instruct", "fixture_qwen2_5_vl.json")
    if which in ("all", "qwen3"):
        for mid in ("Qwen/Qwen3-VL-2B-Instruct", "Qwen/Qwen3-VL-4B-Instruct"):
            try:
                run_model(mid, "fixture_qwen3_vl.json")
                break
            except Exception as e:
                print(f"{mid} failed: {e}", file=sys.stderr)
