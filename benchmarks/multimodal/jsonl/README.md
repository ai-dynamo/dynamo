# Multimodal JSONL Request Generator

Generates `.jsonl` benchmark files for [aiperf](https://github.com/ai-dynamo/aiperf) with multimodal requests (text + images or text + videos).

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

## Video modes

Video workloads use the same pool-reuse idea as images, but emit aiperf's
`videos` field instead of `images`. `--videos-pool` is the number of unique
videos to sample from, and `--videos-per-request` controls how many video slots
each request gets. A smaller pool relative to total video slots produces more
cross-request video reuse.

Video modes generate deterministic local synthetic MP4 files under
`--synthetic-video-dir`. The generated content is derived from `--seed`, the
video index, and the synthetic video parameters, so rerunning the same command
on the same machine reuses equivalent clips. This is the simplest mode for
measuring embedding-cache reuse from repeated local video inputs.

`video-single-turn` measures cross-request reuse from a shared video pool.

To generate the video inputs referenced by
`benchmarks/multimodal/sweep/experiments/embedding_cache/sglang_e_pd.yaml`:

```bash
python benchmarks/multimodal/jsonl/main.py video-single-turn \
  -n 100 \
  --videos-per-request 1 \
  --videos-pool 20 \
  --user-text-tokens 300 \
  --synthetic-video-dir /tmp/bench_videos \
  -o benchmarks/multimodal/jsonl/100req_1vid_20pool_300word_local.jsonl

python benchmarks/multimodal/jsonl/main.py video-single-turn \
  -n 100 \
  --videos-per-request 1 \
  --videos-pool 80 \
  --user-text-tokens 300 \
  --synthetic-video-dir /tmp/bench_videos \
  -o benchmarks/multimodal/jsonl/100req_1vid_80pool_300word_local.jsonl

python benchmarks/multimodal/jsonl/main.py video-single-turn \
  -n 100 \
  --videos-per-request 2 \
  --videos-pool 160 \
  --user-text-tokens 300 \
  --synthetic-video-dir /tmp/bench_videos \
  -o benchmarks/multimodal/jsonl/100req_2vid_160pool_300word_local.jsonl
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

# Video workload with repeated clips
python main.py video-single-turn \
  -n 100 \
  --videos-per-request 1 \
  --videos-pool 20 \
  --synthetic-video-dir /tmp/bench_videos \
  --seed 1

# Multiple videos per request with the same pool-reuse semantics
python main.py video-single-turn \
  -n 100 \
  --videos-per-request 2 \
  --videos-pool 160

```

Output filename encodes the parameters, e.g. `500req_3img_200pool_300word_http.jsonl`
or `100req_1vid_20pool_300word_local.jsonl`.

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
