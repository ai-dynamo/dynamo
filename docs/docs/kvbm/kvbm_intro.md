---
title: "KV Block Manager"
---

# KV Block Manager

The Dynamo KV Block Manager (KVBM) is a scalable runtime component
designed to handle memory allocation, management, and remote sharing of
Key-Value (KV) blocks for inference tasks across heterogeneous and
distributed environments. It acts as a unified memory layer for
frameworks like vLLM, SGLang, and TRT-LLM.

It offers:

- A **unified memory API** that spans GPU memory(in future) , pinned
  host memory, remote RDMA-accessible memory, local or distributed pool
  of SSDs and remote file/object/cloud storage systems.
- Support for evolving **block lifecycles** (allocate → register →
  match) with event-based state transitions that storage can subscribe
  to.
- Integration with **NIXL**, a dynamic memory exchange layer used for
  remote registration, sharing, and access of memory blocks over
  RDMA/NVLink.

The Dynamo KV Block Manager serves as a reference implementation that
emphasizes modularity and extensibility. Its pluggable design enables
developers to customize components and optimize for specific
performance, memory, and deployment needs.

||
||
||
||
||
||
||
||
||

<div class="toctree" hidden="">

Overview \<self\> Motivation \<kvbm_motivation.md\> Architecture
\<kvbm_architecture.md\> Components \<kvbm_components.md\> Design Deep
Dive \<kvbm_design_deepdive.md\> Integrations \<kvbm_integrations.md\>
KVBM in vLLM \<vllm-setup.md\> KVBM in TRTLLM \<trtllm-setup.md\>
LMCache Integration \<../backends/vllm/LMCache_Integration\> Further
Reading \<kvbm_reading.md\>

</div>
