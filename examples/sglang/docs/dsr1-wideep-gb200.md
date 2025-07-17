<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Running DeepSeek-R1 Disaggregated with WideEP on GB200s

Dynamo supports SGLang's GB200 implementation of wide expert parallelism and large scale P/D for DeepSeek-R1! You can read their blog post [here](https://lmsys.org/blog/2025-06-16-gb200-part-1/) for more details. Full end to end optimization is still a work in progress but you can get this up and running with the following steps.

## Instructions

1. Build the SGLang DeepEP container on an ARM64 machine.

> [!NOTE]
> This sglang side branch is based on an open [PR](https://github.com/sgl-project/sglang/pull/7721/files) to SGLang that allows their main dockerfile to be built for aarch64. Once that PR is merged in, we can add the gb200 dockerfile to the main sglang repo.

```bash
git clone https://github.com/kyleliang-nv/sglang.git
cd sglang
git checkout sglang_gb200_wideep_docker
docker build -f docker/Dockerfile -t sgl-blackwell-wideep --build-arg BUILD_TYPE=blackwell --build-arg CUDA_VERSION=12.8.1 .
```

2. Build the Dynamo container

> [!NOTE]
> This is a side branch that contains all of the scripts to run on GB200s. Once the PR is merged in, we can switch to the main branch.

```bash
cd $DYNAMO_ROOT
git checkout ishan/more-slurm-targets # temporary
docker build -f container/Dockerfile.sglang-gb200 . -t dynamo-wideep-gb200 --no-cache
```

3. In your SLURM cluster, clone dynamo and switch to this side branch.

```bash
git clone https://github.com/ai-dynamo/dynamo.git
git checkout ishan/more-slurm-targets
cd examples/sglang/slurm_jobs
```

4. Ensure you have the proper paths that you can use to mount things to the container

- The path to the DSR1 model which should be mounted to the `--model-dir` flag and `--config-dir` flag 

5. Run the following command to submit the job

```bash
python3 submit_job_script.py \
  --template job_script_template.j2 \
  --model-dir <path-to-dsr1-model> \
  --container-image <image-from-step-2> \
  --account <your-account> \
  --gpus-per-node 4 \
  --config-dir <path-to-dsr1-model> \
  --network-interface enp138s0f0np0 \
  --gpu-type gb200 \
  --use-sglang-commands \
  --prefill-nodes 2 \
  --decode-nodes 12
```

**Note**: if you want to spin up dynamo, you can remove the `--use-sglang-commands` flag.

6. This will create a logs directory in the `examples/sglang/slurm_jobs` directory. You can `cd` into the directory, cd into your job id, and then run `tail -f *_prefill.err *_decode.err` or `tail -f *_prefill.out *_decode.out` to see the logs.