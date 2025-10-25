# Container

Use the Dockerfile in `container/Dockerfile.sglang-wideep` to build the container, or

```bash
./container/build.sh --framework sglang-wideep
```

Dynamo commits after `1b3eed4b6a0e735d4ecec6681f4c0b89f2112167` (Sep 18, 2025) are required.

# Hardware

The two deployment recipes are for 8xH200 and 16xH200. It should also work for other GPU SKUs. Change the TDP and DEP size accordingly to match the GPU capacity.

If you see NCCL errors when sending requests to the engines, it is usually caused by OOM error. Try to reduce `--mem-fraction-static` in both prefill and decode engines.

