# Release Testing

## Installation

There are two primary ways of installing and running the dynamo framework.

Installation via pip wheel and installation via docker.

### Testing Pip Installation

To test installing and running via pip we install the ai-dynamo wheel
with only required packages and install it with all optional packages
in a minimal ubuntu:24.04 image.

#### Build `dynamo_pip_all`

```
docker run -it --name dynamo_pip_install -v$PWD/container/deps:/deps ubuntu:24.04 bash -c "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv libucx0 && python3 -m venv /venv && /venv/bin/pip install --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-dl-triton-pypi/simple ai-dynamo[all] && /venv/bin/pip install -r /deps/requirements.test.txt " && docker commit dynamo_pip_install dynamo:latest-pip-all && docker rm dynamo_pip_install
```


#### Build `dynamo_pip`


```
docker run -it --name dynamo_pip_install -v$PWD/container/deps:/deps ubuntu:24.04 bash -c "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv libucx0 && python3 -m venv /venv && /venv/bin/pip install --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-dl-triton-pypi/simple ai-dynamo && /venv/bin/pip install -r /deps/requirements.test.txt " && docker commit dynamo_pip_install dynamo:latest-pip-base && docker rm dynamo_pip_install
```


#### Build `dynamo_pip_sglang`


```
docker run -it --name dynamo_pip_install -v$PWD/container/deps:/deps ubuntu:24.04 bash -c "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -yq python3-dev python3-pip python3-venv libucx0 && python3 -m venv /venv && /venv/bin/pip install --extra-index-url https://urm.nvidia.com/artifactory/api/pypi/sw-dl-triton-pypi/simple ai-dynamo && /venv/bin/pip install -r /deps/requirements.test.txt && /venv/bin/pip install sglang[all]==0.4.6.post2 " && docker commit dynamo_pip_install dynamo:latest-pip-sglang && docker rm dynamo_pip_install
```

#### Run (Replace `X` with `sglang`, `all` or `base`)

```
./container/run.sh --image dynamo:latest-pip-X --mount-workspace -it -- bash -c "source /venv/bin/activate && bash"
```

### Testing Docker

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


#### SGLang

Build:

```
./container/build.sh
```

Run:

```
./container/run.sh --mount-workspace -it
```

Install SGlang:

```
uv pip install "sglang[all]==0.4.6.post2"
```

## Test Cases

### Local Serving

#### Dynamo Run

After running container

```
cd tests
pytest -s -v test_dynamo_run.py
```

#### Dynamo Serve

##### vllm

HW Requirements: H100 x 2

Follow instructions:

[vllm](../examples/llm/README.md)


Optional: Automated agg testing:

After running container

```
cd tests
pytest -s -v test_dynamo_serve.py -k "[agg] or [agg_router]"
```

##### sglang

HW Requirements: H100 x 1

Follow instructions:

[sglang](../examples/sglang/README.md)

After running container

Optional: Automated agg testing:

```
uv pip install "sglang[all]==0.4.6.post2"
cd tests
pytest -s -v test_dynamo_serve.py -k "[sglang]"
```

##### multimodal

HW Requirements: H100 x 2

Follow instructions:

[multimodal](../examples/multimodal/README.md)


Optional: Automated agg testing:

After running container

```
cd tests
pytest -s -v test_dynamo_serve.py -k "multimodal"
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

