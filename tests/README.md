# Release Testing

## Installation

### Pip install

#### Build


Build:

```
docker run -it --name dynamo_pip_install -v$PWD/container/deps:/deps ubuntu:24.04 bash -c "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv libucx0 && python3 -m venv /venv && /venv/bin/pip install --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-dl-triton-pypi/simple ai-dynamo[all] && /venv/bin/pip install -r /deps/requirements.test.txt " && docker commit dynamo_pip_install dynamo:latest-pip && docker rm dynamo_pip_install
```

Run:

```
./container/run.sh --image dynamo:latest-pip --mount-workspace -it -- bash -c "source /venv/bin/activate && bash"
```

#### Run

### Docker

#### vLLM

Build:

```
./container/build.sh
```

Run:

```
./container/run.sh --mount-workspace -it
```

#### TRT-LLM

Build:

```
./container/build_trtllm_base_image.sh
./container/build.sh --framework tensorrtllm
```

Run:

```
./container/run.sh --mount-workspace -it --framework tensorrtllm
```

## Local Serving

### Dynamo Run

After running container

```
cd qa_test_plan
pytest -s -v test_dynamo_run.py
```

### Dynamo Serve

##### vllm

Follow instructions:

[vllm](../examples/llm/README.md)

```
cd qa_test_plan
pytest -s -v test_dynamo_serve.py
```

##### tensorrtllm

[tensorrtllm](../examples/tensorrt_llm/README.md)

### Benchmarks

##### vllm

[vllm](../examples/llm/benchmarks/README.md)

##### planner

[planner](../docs/guides/planner_benchmark/benchmark_planner.md)

## Dynamo Deploy

### Helm chart

Uses microk8s

[manual](../docs/guides/dynamo_deploy/manual_helm_deployment.md)

### Deploy Operator

Uses minikube

[operator](../docs/guides/dynamo_deploy/operator_deployment.md)

