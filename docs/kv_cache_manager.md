# Dynamo Distributed KV Cache Manager

Calculating LLM KV values for user requests is resource-intensive and thus expensive. Leveraging KV cache to minimize the need for its recomputation is common practice. However, as AI demand increases, solely relying on GPU memory for KV cache would not be sustainable to meet SLA under fixed budget. It poses a significant demand to a more effective KV cache reuse management mechanism.

The Dynamo KV Cache Manager feature addresses this challenge by enabling the offloading of older or less frequently accessed KV cache blocks to more cost-effective memory and storage solutions, such as CPU memory, local storage or networked object or file storage. This capability enables organizations to store up to petabytes of KV cache data at a fraction of the cost of keeping it in GPU memory. By offloading KV cache to alternative memory hierarchies, developers can free up valuable GPU resources while still retaining and reusing historical KV cache to reduce inference computation costs.

<figure>
    <img src='images/kv_cache_mgr.png' alt='missing' />
    <p>Figure 1. Dynamo Distributed KV Cache Manager offloads less frequently accessed KV cache to more economical memory hierarchies </p>
</figure>

The Dynamo KV Cache Manager uses advanced caching policies that prioritize placing frequently accessed data in GPU memory, while less accessed data is moved to shared CPU memory, SSDs, or networked object storage. It incorporates eviction policies that strike a balance between over-caching (which can introduce lookup latencies) and under-caching (which leads to missed lookups and KV cache re-computation).
Additionally, this feature can manage KV cache across multiple GPU nodes, supporting both distributed and disaggregated inference serving, and offers hierarchical caching capabilities, creating offloading strategies at the GPU, node, and cluster levels.

The Dynamo KV Cache Manager is designed to be framework-agnostic to support various backends, including TensorRT-TLLM, vLLM, and SGLang, and to facilitate the scaling of KV cache storage across large, distributed clusters using NVLink, NVIDIA Quantum switches, and NVIDIA Spectrum switches. It integrates with [NIXL](https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md) to enable data transfers across different worker instances and storage backends.

## Design

- Separation of Mechanism and Policy
    - Mechanism: Manages memory allocation, caching hierarchy, and data flow.
    - Policy: Determines caching strategies, including the choice of data structures (e.g., radix tree, distributed hash tables) and eviction algorithms.

  This separation ensures that the underlying infrastructure can evolve without disrupting the caching logic. This design decision was created to enable each customer to come up with their own policies and mechanisms to manage memory that fits their access pattern.

- Hierarchical caching
    - A radix tree provides a clean, structured approach for organizing KV storage in distributed inference. A local tree can be built per node, with a global tree  at the cluster level, ensuring an efficient abstraction.
    - The hierarchy spans HBM, local node KV stores, and external storage, with each layer caching data for the next to optimize lookups. Data movements across the tiers are handled using NIXL APIs for seamless communication. The data flow is fully asynchronous and is transparent to worker instances.
    - Multiple backends are supported as long as they are compatible with KV manager APIs.
    - RDMA transfers are preferred for optimal performance.

- Registration with runtimes
    - Distributed KV manager registers with inference engine runtimes to enable KV offloading to the pool.
    - Registration creates a two-way communication queue between the runtime and the pool.

- Management and transfer granularities
    - KV blocks are managed in block level (group of tokens) however transfer of KV states can be performed at layer level.
    - If multiple tokens are needed to be fetched, then these layer transfers are parallelized to ensure maximum throughput from the KV pool.

## V1 Implementation

Dynamo Distributed KV Manager has two implementations: V1 and V2. V1 serves as a proof-of-concept design, providing a lightweight KV offloading framework with simple, asynchronous APIs — GET() and PUT(), allowing inference engines to offload KV caches efficiently. These APIs are designed to be fully asynchronous, enabling seamless overlap with inference computation.

<figure>
    <img src='images/kv_cache_mgr_design.png' alt='missing' />
    <p>Figure 2. Design of Dynamo KV manager V1 </p>
</figure>


The left section of Figure 2 illustrates the execution timeline and data movement sequence in the V1 architecture. Inference engines like vLLM can initiate asynchronous operations with flexible access granularity, enabling various overlapping strategies to optimize execution based on whether the priority is throughput or latency.

The right section of Figure 2 depicts data flow within the runtime. At present, we do not allocate any portion of the GPU's high-bandwidth memory (HBM) beyond what is required by the inference engine, ensuring its exclusive utilization for inference tasks. Within the inference runtime, GPU device memory can either be fully dedicated to key-value (KV) storage or partially allocated for prefix KV caches, which are dynamically managed by the inference engine—similar to vLLM.

When the inference engine determines that some entries in the KV cache should be evicted from GPU memory, it invokes the put_async() API to offload them by the KV manager, which updates its index and transfers the data to the appropriate storage tier (CPU memory or a combination of CPU and SSD). Conversely, if the inference engine fails to locate a required KV entry in its self-managed prefix cache, it issues a get_async() request to the KV manager. If the KV entry already exists, retrieval via get_async() will significantly reduces recomputation overhead, ensuring efficient KV management, optimized memory utilization, and improved inference performance.

In the V1 implementation, CPU memory functions as a cache layer for SSD storage. If a required KV entry resides in CPU memory, the system bypasses SSD access, reducing transfer latency. Asynchronous APIs like  get_async() or put_async() also enable transfers such that it does not impact system performance.
A key aspect of our implementation is the introduction of multiple parallel queues (or pipelines) for critical operations, including:
- Index matching, updates, and block allocation/free operations
- Data transfers between GPU and CPU
- Data transfers between CPU and SSD

This multi-queue design is crucial because it:

- Enables true asynchronous execution by decoupling blocking operations.
- Maximizes parallelism, allowing multiple requests to be processed concurrently.
- Fully utilizes different hardware resources, such as CPU, GPU, and storage, avoid bottlenecks.
- Decouples slow operations (e.g., SSD writes) from the critical path of responding to user queries, to improve responsiveness.
- Ensures the correctness of index updates and data transfers, even under high-throughput, concurrent workloads.

Looking ahead, V1 architecture will integrate with NIXL to enable KV reuse across multiple nodes. Additionally, we will add GPUDirect Storage capabilities to reduce the get_async() latency and minimize the CPU overhead while facilitating direct data transfers between GPU memory and SSD. These enhancements will be made available post-GTC.

V1 architecture is an excellent design for quick enablement and execution. However, it does not offer much finer control on memory management and interactions with the NVIDIA Dynamo ecosystem. To address this, we are parallelly implementing V2 architecture providing a notion of distributed KV pool across workers and storage. V2 architecture will be released in coming weeks.

## V2 Implementation

The V2 implementation introduces a distributed KV pool across worker instances and storage, incorporating all features outlined in the design. Development is still in progress, and we welcome collaborators to share their feedback. This documentation aims to offer a high-level overview of the V2 implementation and gather input.

The V2 BlockManager changes the ownership patterns to RAII objects. The primary object will be a KvBlock object which defines the contents of the tokens in the block and the unique sequence hash associated with that block. In Rust, the KvBlock is a generic KvBlock<S: BlockStorage>.  This means each KvBlock is strongly typed to the storage type (S) which must conform to the behavior defined in the BlockStorage trait.

KvBlocks are allocated and ownership is transferred to a ReusePool object. The ReusePool object is used to provide free blocks in user defined priority. This specialized Pool is a compound collective, so it can also lookup and extract blocks from the pool by matching on the sequence hash.

When acquired from the ReusePool, the object is a PoolItem<KvBlock<BST>>. PoolItem is the object that is the RAII object that when it goes out of scope (Drop), it will be returned to the pool.

A PoolItem<KvBlock<BST> which is typedef’ed as `UniqueBlock<BST>` is a uniquely owned and mutable block. In order to make the block shareable and discoverable, the UniqueBlock<BST> must be registered with the ReservedBlockRegistry.  Upon registration, a RegisteredBlock<BST> is returned – this block is shared, immutable and discoverable.

- Immutable - should only provide a const pointer to the storage
- Shared - internally atomically referenced counted object and therefore Cloneable.
    - When refcount → 0, the block is unregistered and the backing UniqueBlock<BST> is returned to the ReusePool.
- Discoverable/Reserved
    - Incoming requests can be matched to blocks by sequence hash which will return  a list of RegisteredBlock<BST> clones for the matching blocks.
    - Registered block state changes are emitted as events allowing the KV Aware Router to add/remove the block from the radix tree.

All data movement requires either Shared or Unique block ownership that is owned for the scope of the TransferEngines operation.

For example,

```bash
pub async fn copy_blocks<D, S>(dst: &[KvBlock<D>], src: &[KvBlock<S>]) -> Result<()>;
```

This allows us to specialize implementations on D and S which will be compiler matched.  For python, this will be dynamically dispatched and mismatched types will be raised as exceptions.

The underlying Storage is layer-aware.  This allows for us to expose a layer-wise trigger.

```bash
pub async fn copy_blocks_by_layer<D, S>(dst: &[KvBlock<D>], src: &[KvBlock<S>, layers: &[usize]) -> Result<()>;
```

To coordinate layer-wise chaining of transfers, say from GPU -> CPU -> Storage we will provide TransferCoordinator can pipeline layer transfers from to the next storage.  Example, the moment a layer or set of layers arrives in CPU memory from GPU, we can trigger those layers begin CPU -> Storage. This allows the secondary transfers to have layer-wise overlap with the primary transfers.





