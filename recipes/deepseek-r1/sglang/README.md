# DeepSeek R1 SGLang Recipe

This recipe is for running DeepSeek R1 with SGLang in disaggregated mode. It is based on the WideEP recipe from the SGLang team.

## Container

Build the container using the `build.sh` script:

```bash
./container/build.sh --framework SGLANG
```

Dynamo commits after `1b3eed4b6a0e735d4ecec6681f4c0b89f2112167` (Sep 18, 2025) are required.

## Hardware

The two deployment recipes are for 16x H200 (disagg-8gpu) and 32x H200 (disagg-16gpu). The folder names refer to GPUs per worker type (8 or 16), with separate prefill and decode workers each using that many GPUs. It should also work for other GPU SKUs. Change the TP and EP size accordingly to match the GPU capacity.

If you see NCCL errors when sending requests to the engines, it is usually caused by OOM error. Try to reduce `--mem-fraction-static` in both prefill and decode engines.

