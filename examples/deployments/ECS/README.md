# Dynamo on Amazon ECS

Task definition JSON files for the ECS deployment guide.

**Full guide:** [docs/kubernetes/cloud-providers/ecs/ecs.md](../../../docs/kubernetes/cloud-providers/ecs/ecs.md) (published at [docs.nvidia.com/dynamo](https://docs.nvidia.com/dynamo/kubernetes/cloud-providers/ecs/ecs))

## Contents

| Path | Description |
|------|-------------|
| `task_definition_etcd_nats.json` | ETCD and NATS services on Fargate |
| `task_definition_frontend.json` | Dynamo frontend on EC2 |
| `task_definition_prefillworker.json` | vLLM prefill worker on EC2 |
