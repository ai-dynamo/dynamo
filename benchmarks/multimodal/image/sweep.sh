for concurrency in 1 2 4; do
    bash aiperf_30_image_150_osl.sh --model Qwen/Qwen3-VL-30B-A3B-Instruct-FP8 --concurrency $concurrency
done