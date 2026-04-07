# 3.1 Deployment & Integration with NVIDIA Dynamo

In this blog, DeepSeek-R1 deployment on GB300 NVL72 is orchestrated with NVIDIA Dynamo, which provides the control plane for disaggregated prefill-decode (PD) serving at cluster scale. Dynamo keeps the serving graph stable under long-context pressure while preserving SGLang kernel-level performance optimizations.

**Low-Overhead Orchestration:**
Dynamo’s primary performance contribution is efficient request steering between prefill and decode workers with minimal scheduling overhead. In 128K/8K workloads, where decode is memory-bound and PD handoff is frequent, this keeps orchestration from becoming a bottleneck and allows GB300 memory bandwidth to be used for model work rather than control-path stalls.

Dynamo Router further improves this path by separating control decisions from worker execution and supporting disaggregated routing policies (including decode-first / KV-aware strategies where applicable). In practice, this improves queue stability under bursty long prompts and reduces prefill-induced interference on active decode traffic.

**Production-Ready Scaling:**
Dynamo provides a robust abstraction for worker discovery, worker coordination, request migration, and lifecycle management across multi-node deployments. On Kubernetes, worker discovery is handled natively through the Operator-managed discovery path (`DYN_DISCOVERY_BACKEND=kubernetes`), while request migration allows in-flight generations to continue on healthy workers during worker failures or restarts.

This keeps prefill/decode disaggregation operationally stable as concurrency increases, including Wide-EP-heavy decode settings used in this study.

**Ease of Deployment from Benchmark to Production:**
To accelerate iteration, we use `srt-slurm` recipes for rapid benchmarking on Slurm-managed GB200/GB300 environments, then promote the same PD topology to Kubernetes using the Dynamo Operator. This keeps model-serving semantics consistent while changing only the orchestration surface:
- On Slurm: fast, recipe-driven performance bring-up and sweep automation.
- On Kubernetes: declarative `DynamoGraphDeployment` with Operator-managed reconciliation, multinode runtime injection, and native K8s discovery (`DYN_DISCOVERY_BACKEND=kubernetes`).

This provides a practical path from one-command experimentation to production deployment without re-architecting the serving stack.

Reproduction and deployment references:
- Dynamo Operator: `docs/pages/kubernetes/dynamo-operator.md`
- Kubernetes install guide: `docs/pages/kubernetes/installation-guide.md`
- Multinode deployment details: `docs/pages/kubernetes/deployment/multinode-deployment.md`
- DeepSeek recipe index: `recipes/deepseek-r1/README.md`
- DeepSeek GB200 Wide-EP deployment example: `recipes/deepseek-r1/trtllm/disagg/wide_ep/gb200/deploy.yaml`
- SGLang Slurm DeepSeek example: `examples/backends/sglang/slurm_jobs/README.md`
