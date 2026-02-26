# Multimodal JSONL Request Generator

Generates `.jsonl` benchmark files for [aiperf](https://github.com/NVIDIA/aiperf) with single-turn multimodal requests (text + images).

## Key concept: image pool reuse

Each request samples images from a fixed pool. A smaller pool relative to total
image slots produces more cross-request image reuse — useful for benchmarking
embedding cache hit rates.

For example, 500 requests x 3 images each = 1500 image slots. With
`--images-pool 200`, many requests will share the same images.

## Image modes

| Mode | `--image-mode` | What goes in the JSONL | Who fetches the image |
|------|---------------|------------------------|----------------------|
| base64 (default) | `base64` | Absolute file paths to local PNGs | aiperf reads and base64-encodes before sending |
| HTTP | `http` | COCO test2017 URLs | The LLM server downloads images itself |

For `http` mode, download COCO annotations first:
```bash
mkdir -p annotations && cd annotations
wget http://images.cocodataset.org/annotations/image_info_test2017.zip
unzip image_info_test2017.zip
```

## Usage

```bash
# Defaults: 500 requests, 3 images each, all unique, base64 mode
python main.py

# HTTP mode with COCO URLs
python main.py --image-mode http

# Control reuse: 200 requests, pool of 100 unique images
python main.py -n 200 --images-pool 100

# More images per request
python main.py -n 100 --images-per-request 20 --images-pool 500
```

Output filename encodes the parameters, e.g. `500req_3img_200pool_300word_http.jsonl`.

## Running with aiperf

```bash
aiperf profile \
  --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 \
  --input-file 500req_3img_200pool_300word_http.jsonl \
  --custom-dataset-type single_turn \
  --shared-system-prompt-length 1000 \
  --extra-inputs "max_tokens:500" \
  --extra-inputs "min_tokens:500" \
  --extra-inputs "ignore_eos:true"
```

Note: the JSONL contains actual content (text + image references), not token
counts. Do not pass `--isl` — it only applies to synthetic data generation.
