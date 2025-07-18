# Profiler

## Setup

From within the dynamo container:
```bash
./k8s.sh  # install binaries, auth into aks cluster
cd benchmarks/profiler
python -m profile_sla --config ../../examples/vllm/deploy/disagg.yaml --namespace mo-dyn-cloud # run the profiler
```