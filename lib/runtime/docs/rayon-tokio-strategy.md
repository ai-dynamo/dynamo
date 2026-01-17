# Rayon-Tokio Integration Strategy

## Overview

This document describes the integration strategy for combining Tokio's asynchronous runtime with Rayon's data-parallel compute capabilities in the Dynamo runtime. The core philosophy is simple:

- **Tokio** handles I/O-bound operations, waiting, and coordination
- **Rayon** handles CPU-bound operations and data parallelism
- Multiple async tasks can concurrently submit different types of work to the shared Rayon thread pool

## Architecture

```text
+---------------------------------------------------------------+
|                     Tokio Runtime                             |
|  +-----------+    +-----------+    +-----------+             |
|  | Async     |    | Async     |    | Async     |             |
|  | Task 1    |    | Task 2    |    | Task 3    |             |
|  |           |    |           |    |           |             |
|  | Receives  |    | Processes |    | Handles   |             |
|  | requests  |    | streams   |    | batches   |             |
|  +-----+-----+    +-----+-----+    +-----+-----+             |
|        |                |                |                   |
|        +----------------+----------------+                   |
|                         |                                    |
|                  tokio_rayon::spawn                          |
|                         |                                    |
+-------------------------+------------------------------------+
                          |
                          v
+---------------------------------------------------------------+
|                    Rayon Thread Pool                         |
|                                                               |
|  +----------------------------------------------------------+ |
|  |         Work-Stealing Thread Pool (N threads)           | |
|  |                                                          | |
|  |  +---------+  +-----------+  +------------------+       | |
|  |  | scope() |  | par_iter()|  | join()           |       | |
|  |  | tasks   |  | chunks    |  | computations     |       | |
|  |  +---------+  +-----------+  +------------------+       | |
|  |                                                          | |
|  |  All patterns share the same thread pool                | |
|  +----------------------------------------------------------+ |
+---------------------------------------------------------------+
```

## When to Use Tokio vs Rayon

### Use Tokio (async/await) when
- **Waiting for I/O**: Network requests, file I/O, database queries
- **Coordinating tasks**: Channels, synchronization, signaling
- **Stream processing**: Items arrive over time with delays
- **Resource pooling**: Connection pools, async locks
- **Service orchestration**: Managing component lifecycles

### Use Rayon (compute pool) when
- **Batch processing**: You have all data ready for parallel processing
- **CPU-intensive work**: Computation takes >1ms per item
- **Data transformation**: Tokenization, serialization, compression
- **Parallel algorithms**: Matrix operations, sorting, searching
- **Map-reduce patterns**: Aggregations over large datasets

### Decision Thresholds
- Use Rayon when processing **≥10 items** in parallel
- Use `spawn_blocking` when CPU work takes **>1ms**
- Keep Tokio for operations with **>100μs waits** between items
- Use Rayon when you can **saturate multiple CPU cores**

### Overhead Considerations
Based on benchmarks, the async bridge between Tokio and Rayon has:
- **~25μs overhead** for small tasks (due to channel communication)
- **~4% overhead** for tasks taking >2ms
- **Negligible overhead** for tasks taking >10ms

For minimal overhead when using Rayon from async context:
- **Small tasks (<100μs)**: Run directly on Tokio
- **Medium tasks (100μs-1ms)**: Use `spawn_blocking` + `pool.execute_sync()`
- **Large tasks (>1ms)**: Use `pool.execute()` for convenience

## Concurrent Usage Patterns

The key insight is that multiple async tasks can concurrently use the same Rayon thread pool with different parallelization patterns. Rayon's work-stealing scheduler efficiently distributes work regardless of the pattern used.

### Pattern 1: Concurrent Scope and ParIter

```rust,ignore
use std::sync::Arc;
use dynamo_runtime::compute::ComputePool;

async fn concurrent_compute_tasks(pool: Arc<ComputePool>) {
    // Task 1: Using scope for dynamic task spawning
    let task1 = tokio::spawn({
        let pool = pool.clone();
        async move {
            pool.execute_scoped(|scope| {
                // Dynamically spawn tasks based on runtime conditions
                for i in 0..num_tasks {
                    scope.spawn(move |_| {
                        expensive_computation(i)
                    });
                }
            }).await
        }
    });

    // Task 2: Using parallel iterators for batch processing
    let task2 = tokio::spawn({
        let pool = pool.clone();
        async move {
            pool.install(|| {
                // Process data in parallel chunks
                data.par_chunks(100)
                    .map(|chunk| transform_chunk(chunk))
                    .collect::<Vec<_>>()
            }).await
        }
    });

    // Task 3: Using join for binary parallelism
    let task3 = tokio::spawn({
        let pool = pool.clone();
        async move {
            pool.join(
                || compute_left_branch(),
                || compute_right_branch(),
            ).await
        }
    });

    // All three tasks run concurrently, sharing the Rayon thread pool
    let (r1, r2, r3) = tokio::join!(task1, task2, task3);
}
```

### Pattern 2: Stream Processing with Batch Compute

```rust,no_run
# use futures::StreamExt;
# use rayon::prelude::*;
# use std::sync::Arc;
# struct Data;
# fn process_item(_: &Data) -> i32 { 0 }
# async fn send_results(_: Vec<i32>) {}
# use dynamo_runtime::compute::ComputePool;
# use futures::stream::Stream;

/// Example: Process async stream with CPU-intensive batch operations
async fn stream_with_compute(
    pool: Arc<ComputePool>,
    stream: impl Stream<Item = Vec<Data>>,
) {
    // Use for_each_concurrent for proper stream consumption
    stream.for_each_concurrent(4, |batch| {
        let pool = pool.clone();
        async move {
            // Process batch using parallel iterators
            let result = pool.install(move || {
                batch.par_iter()
                    .map(|item| process_item(item))
                    .collect::<Vec<_>>()
            }).await.unwrap();

            // Async I/O to send results
            send_results(result).await;
        }
    }).await;
}
```

### Pattern 3: Mixed Workload Service

```rust,ignore
/// Real-world example: LLM service with mixed workloads
struct LLMService {
    runtime: Arc<Runtime>,
    tokenizer: Arc<Tokenizer>,
}

impl LLMService {
    async fn run(&self) {
        let pool = self.runtime.compute_pool()
            .expect("Compute pool required");

        // Tokenization service - uses parallel iterators
        let tokenization_task = {
            let pool = pool.clone();
            let tokenizer = self.tokenizer.clone();
            tokio::spawn(async move {
                loop {
                    // Async I/O: receive batch from network
                    let texts = receive_tokenization_batch().await;

                    // CPU-bound: parallel tokenization
                    let tokens = pool.install(move || {
                        texts.par_iter()
                            .map(|text| tokenizer.encode(text))
                            .collect::<Vec<_>>()
                    }).await.unwrap();

                    // Async I/O: send results
                    send_tokens(tokens).await;
                }
            })
        };

        // Embedding service - uses scope for multi-stage computation
        let embedding_task = {
            let pool = pool.clone();
            tokio::spawn(async move {
                loop {
                    // Async I/O: receive request
                    let request = receive_embedding_request().await;

                    // CPU-bound: multi-stage parallel computation
                    let embeddings = pool.execute_scoped(|scope| {
                        let mut text_emb = None;
                        let mut context_emb = None;

                        scope.spawn(|_| {
                            text_emb = Some(compute_text_embedding(&request.text));
                        });

                        scope.spawn(|_| {
                            context_emb = Some(compute_context_embedding(&request.context));
                        });

                        // Scope waits for both to complete
                        combine_embeddings(text_emb.unwrap(), context_emb.unwrap())
                    }).await.unwrap();

                    // Async I/O: send results
                    send_embeddings(embeddings).await;
                }
            })
        };

        // Batch inference service - uses nested parallelism
        let inference_task = {
            let pool = pool.clone();
            tokio::spawn(async move {
                loop {
                    let batch = receive_inference_batch().await;

                    let results = pool.execute_scoped(|scope| {
                        let mut results = Vec::with_capacity(batch.len());

                        // Spawn a task for each item
                        for item in batch {
                            scope.spawn(move |s2| {
                                // Within each task, use parallel iterators
                                let preprocessed = item.data
                                    .par_chunks(10)
                                    .map(|chunk| preprocess(chunk))
                                    .collect::<Vec<_>>();

                                // Can spawn more tasks within nested scope
                                let mut stages = vec![];
                                for p in preprocessed {
                                    s2.spawn(move |_| {
                                        stages.push(run_inference(p));
                                    });
                                }

                                results.push(merge_stages(stages));
                            });
                        }

                        results
                    }).await.unwrap();

                    send_inference_results(results).await;
                }
            })
        };

        // All services run concurrently, sharing the compute pool
        tokio::join!(tokenization_task, embedding_task, inference_task);
    }
}
```

## How It Works: Thread Pool Sharing

Rayon's work-stealing scheduler ensures efficient resource utilization even when different async tasks submit different types of work:

1. **Work Queues**: Each Rayon thread has a local deque (double-ended queue)
2. **Local Execution**: Threads prefer executing their own tasks (LIFO for cache locality)
3. **Work Stealing**: Idle threads steal tasks from busy threads (FIFO from the other end)
4. **No Interference**: Different parallelization patterns (scope, par_iter) coexist peacefully

This means:
- A `scope` task spawning many small tasks works alongside `par_chunks` processing large batches
- The thread pool automatically balances load between different types of work
- No manual coordination needed between different async tasks using the pool

## Performance Considerations

### Thread Pool Sizing

```toml
[runtime]
# Tokio threads: optimize for concurrent async tasks
num_worker_threads = 8  # Usually number of cores

# Rayon threads: optimize for CPU saturation
compute_threads = 4     # Often cores/2 to avoid oversubscription
```

### Avoiding Oversubscription

Total threads = Tokio workers + Rayon threads + System threads

**Recommendation**: Keep total ≤ 1.5 × physical cores

### Monitoring Pool Utilization

```rust,ignore
// Check pool metrics
let metrics = pool.metrics();
println!("Active tasks: {}", metrics.tasks_active());
println!("Average duration: {:.2}ms", metrics.avg_task_duration_us() / 1000.0);
println!("Slow tasks (>100ms): {}", metrics.slow_tasks());

// Adjust pool size if consistently over/under utilized
if metrics.tasks_active() > pool.num_threads() * 2 {
    // Consider increasing compute_threads
}
```

## Common Patterns and Best Practices

### DO: Batch Collection Before Processing

```rust,ignore
// ✅ Good: Collect async items, then process in parallel
let items = stream.take(100).collect::<Vec<_>>().await;
let processed = pool.install(|| {
    items.par_iter().map(|item| process(item)).collect()
}).await?;
```

### DON'T: Mix Async and Compute in Tight Loops

```rust,ignore
// ❌ Bad: Alternating between async and compute
for item in items {
    let data = fetch_data(item).await;  // Async
    let result = pool.execute(|| compute(data)).await?;  // Compute
    store_result(result).await;  // Async
}

// ✅ Good: Batch operations
let all_data = futures::future::join_all(
    items.iter().map(|item| fetch_data(item))
).await;

let all_results = pool.install(|| {
    all_data.par_iter().map(|data| compute(data)).collect()
}).await?;

futures::future::join_all(
    all_results.iter().map(|result| store_result(result))
).await;
```

### DO: Use Scope for Dynamic Parallelism

```rust,ignore
// ✅ Good: When you don't know the parallelism level upfront
pool.execute_scoped(|scope| {
    while let Some(work) = find_more_work() {
        scope.spawn(move |_| {
            process_work(work);
        });
    }
}).await?;
```

### DO: Use ParIter for Data Parallelism

```rust,ignore
// ✅ Good: When processing collections
pool.install(|| {
    data.par_chunks(optimal_chunk_size())
        .map(|chunk| process_chunk(chunk))
        .reduce(|| initial_value(), |a, b| combine(a, b))
}).await?;
```

## Troubleshooting

### Issue: High Latency Despite Low CPU Usage
**Cause**: Too few Rayon threads for the workload
**Solution**: Increase `compute_threads` configuration

### Issue: System Feels Sluggish
**Cause**: Thread oversubscription
**Solution**: Reduce total thread count (Tokio + Rayon)

### Issue: Uneven Work Distribution
**Cause**: Poor chunk size selection
**Solution**: Use smaller chunks or dynamic scheduling with `scope`

### Issue: Deadlock or Hanging
**Cause**: Nested `install()` calls or blocking in Rayon threads
**Solution**: Use `execute()` instead of `install()` for simple tasks

## Configuration Examples

### High-Throughput Service
```toml
# Many concurrent requests, moderate compute per request
[runtime]
num_worker_threads = 16
compute_threads = 8
compute_stack_size = "4MB"
```

### Batch Processing System
```toml
# Few concurrent tasks, heavy compute per batch
[runtime]
num_worker_threads = 4
compute_threads = 12
compute_stack_size = "8MB"
```

### Mixed Workload
```toml
# Balance between async I/O and compute
[runtime]
num_worker_threads = 8
compute_threads = 6
compute_stack_size = "2MB"
```

## Summary

The Rayon-Tokio integration provides a powerful model for handling mixed workloads:

1. **Tokio** manages async I/O and coordination
2. **Rayon** provides a shared compute thread pool
3. Multiple async tasks can concurrently use different Rayon patterns
4. Work-stealing ensures efficient resource utilization
5. Clear separation between I/O-bound and CPU-bound work

This architecture enables building high-performance services that efficiently handle both network I/O and CPU-intensive computations without manual thread management or complex synchronization.