// // SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// // SPDX-License-Identifier: Apache-2.0

// //! Integration tests for leader-worker connector coordination.
// //!
// //! This test demonstrates the complete flow:
// //! 1. Leader starts and creates cohort
// //! 2. Workers discover leader and join cohort
// //! 3. Leader broadcasts layout creation request
// //! 4. Workers create layouts and register with TransportManager
// //! 5. Leader exports layouts metadata from all workers

// use dynamo_am::runtime::cohort::CohortType;
// use dynamo_kvbm::v2::distributed::cohort::*;
// use dynamo_kvbm::v2::physical::layout::LayoutConfig;
// use std::sync::Arc;
// use tokio_util::sync::CancellationToken;
// use tracing::{debug, info};

// #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
// async fn test_leader_worker_coordination() -> anyhow::Result<()> {
//     // Initialize tracing for test visibility
//     let _ = tracing_subscriber::fmt()
//         .with_env_filter("debug")
//         .with_test_writer()
//         .try_init();

//     info!("Starting leader-worker coordination test");

//     // Configuration
//     const WORLD_SIZE: usize = 2;

//     // =========================================================================
//     // PHASE 1: Leader Setup
//     // =========================================================================
//     info!("Phase 1: Creating leader");
//     let leader_cancel = CancellationToken::new();
//     let (leader, leader_addr) =
//         Leader::new(CohortType::FixedSize(WORLD_SIZE), leader_cancel.clone()).await?;

//     info!(
//         "Leader created at {}",
//         leader_addr.primary_endpoint().unwrap_or("unknown")
//     );

//     // =========================================================================
//     // PHASE 2: Worker Setup (Parallel)
//     // =========================================================================
//     info!("Phase 2: Creating {} workers", WORLD_SIZE);

//     let discovery = Arc::new(StaticLeaderDiscovery::new(leader_addr.clone()));

//     // Create workers with different ranks
//     let mut worker_tasks = Vec::new();

//     for rank in 0..WORLD_SIZE {
//         let discovery_clone = discovery.clone();
//         let task = tokio::spawn(async move {
//             debug!("Worker {} starting", rank);

//             // Create worker
//             let worker_cancel = CancellationToken::new();
//             let (worker, worker_addr) =
//                 Worker::new(discovery_clone, Some(rank), WORLD_SIZE, worker_cancel).await?;

//             debug!(
//                 "Worker {} created at {}",
//                 rank,
//                 worker_addr.primary_endpoint().unwrap_or("unknown")
//             );

//             // Phase 2a: Register create_layout handler early
//             worker.register_create_layout_handler().await?;
//             debug!("Worker {} registered create_layout handler", rank);

//             // Phase 2b: Join cohort
//             debug!("Worker {} attempting to join cohort", rank);
//             let position = worker.join_cohort().await?;
//             info!("Worker {} joined cohort at position {}", rank, position);

//             Ok::<_, anyhow::Error>((worker, rank, position))
//         });

//         worker_tasks.push(task);
//     }

//     // Wait for all workers to join
//     let mut workers = Vec::new();
//     for task in worker_tasks {
//         let (worker, rank, position) = task.await??;
//         info!(
//             "Worker {} joined successfully at position {}",
//             rank, position
//         );
//         workers.push((worker, rank));
//     }

//     // =========================================================================
//     // PHASE 3: Wait for Cohort Complete
//     // =========================================================================
//     info!("Phase 3: Waiting for cohort to be complete");
//     leader.await_cohort_complete().await?;
//     info!("Cohort is complete with {} workers", WORLD_SIZE);

//     // =========================================================================
//     // PHASE 4: Broadcast Layout Creation
//     // =========================================================================
//     info!("Phase 4: Broadcasting layout creation request");

//     let layout_config = LayoutConfig::builder()
//         .num_blocks(10)
//         .num_layers(4)
//         .outer_dim(2)
//         .page_size(16)
//         .inner_dim(128)
//         .dtype_width_bytes(2)
//         .build()?;

//     let handles = leader
//         .broadcast_create_layout(
//             layout_config,
//             LayoutType::FullyContiguous,
//             MemoryType::System,
//             "test_layout".to_string(),
//         )
//         .await?;
//     info!(
//         "All workers successfully created layouts, received {} handles",
//         handles.len()
//     );
//     assert_eq!(handles.len(), WORLD_SIZE, "Should have handle per worker");

//     // =========================================================================
//     // PHASE 5: Register Export Handlers and Export Layouts
//     // =========================================================================
//     info!("Phase 5: Registering export handlers on workers");
//     for (worker, rank) in &workers {
//         worker.register_export_layouts_handler().await?;
//         debug!("Worker {} registered export_layouts handler", rank);
//     }

//     info!("Phase 5: Broadcasting export layouts request");
//     let exported_metadata = leader.broadcast_export_layouts().await?;

//     info!(
//         "Collected exported metadata from {} workers",
//         exported_metadata.len()
//     );

//     // Verify we got metadata from all workers
//     assert_eq!(
//         exported_metadata.len(),
//         WORLD_SIZE,
//         "Should have metadata from all workers"
//     );

//     // Verify each metadata is non-empty
//     for (idx, metadata) in exported_metadata.iter().enumerate() {
//         debug!(
//             "Worker {} returned {} bytes of metadata",
//             idx,
//             metadata.len()
//         );
//         assert!(
//             !metadata.is_empty(),
//             "Worker {} metadata should not be empty",
//             idx
//         );
//     }

//     // =========================================================================
//     // PHASE 7: Cleanup
//     // =========================================================================
//     info!("Phase 7: Cleaning up");

//     // Shutdown workers
//     for (worker, rank) in workers {
//         worker.shutdown().await?;
//         debug!("Worker {} shut down", rank);
//     }

//     // Shutdown leader
//     leader.shutdown().await?;
//     leader_cancel.cancel();
//     info!("Leader shut down");

//     info!("Test completed successfully");
//     Ok(())
// }

// #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
// async fn test_worker_rejection_duplicate_rank() -> anyhow::Result<()> {
//     // Initialize tracing
//     let _ = tracing_subscriber::fmt()
//         .with_env_filter("debug")
//         .with_test_writer()
//         .try_init();

//     info!("Testing worker rejection for duplicate rank");

//     const WORLD_SIZE: usize = 2;

//     // Create leader
//     let leader_cancel = CancellationToken::new();
//     let (leader, leader_addr) =
//         Leader::new(CohortType::FixedSize(WORLD_SIZE), leader_cancel.clone()).await?;

//     let discovery = Arc::new(StaticLeaderDiscovery::new(leader_addr.clone()));

//     // Create first worker with rank 0
//     let worker1_cancel = CancellationToken::new();
//     let (worker1, _) = Worker::new(discovery.clone(), Some(0), WORLD_SIZE, worker1_cancel).await?;

//     worker1.register_create_layout_handler().await?;
//     let position1 = worker1.join_cohort().await?;
//     info!("Worker 1 joined at position {}", position1);

//     // Create second worker with SAME rank 0 (should be rejected)
//     let worker2_cancel = CancellationToken::new();
//     let (worker2, _) = Worker::new(
//         discovery.clone(),
//         Some(0), // Duplicate rank!
//         WORLD_SIZE,
//         worker2_cancel,
//     )
//     .await?;

//     worker2.register_create_layout_handler().await?;

//     // This should fail due to duplicate rank
//     let result = worker2.join_cohort().await;
//     assert!(
//         result.is_err(),
//         "Worker with duplicate rank should be rejected"
//     );

//     info!("Worker 2 was correctly rejected for duplicate rank");

//     // Cleanup
//     worker1.shutdown().await?;
//     worker2.shutdown().await?;
//     leader.shutdown().await?;
//     leader_cancel.cancel();

//     Ok(())
// }

// #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
// async fn test_cohort_with_no_ranks() -> anyhow::Result<()> {
//     // Initialize tracing
//     let _ = tracing_subscriber::fmt()
//         .with_env_filter("debug")
//         .with_test_writer()
//         .try_init();

//     info!("Testing cohort formation without ranks");

//     const WORLD_SIZE: usize = 2;

//     // Create leader
//     let leader_cancel = CancellationToken::new();
//     let (leader, leader_addr) =
//         Leader::new(CohortType::FixedSize(WORLD_SIZE), leader_cancel.clone()).await?;

//     let discovery = Arc::new(StaticLeaderDiscovery::new(leader_addr.clone()));

//     // Create workers WITHOUT ranks (None)
//     let mut worker_tasks = Vec::new();
//     for i in 0..WORLD_SIZE {
//         let discovery_clone = discovery.clone();
//         let task = tokio::spawn(async move {
//             let worker_cancel = CancellationToken::new();
//             let (worker, _) = Worker::new(
//                 discovery_clone,
//                 None, // No rank provided
//                 WORLD_SIZE,
//                 worker_cancel,
//             )
//             .await?;

//             // Don't register layout handler for rank-less workers (not a realistic scenario)
//             let position = worker.join_cohort().await?;
//             info!("Rank-less worker {} joined at position {}", i, position);

//             Ok::<_, anyhow::Error>(worker)
//         });
//         worker_tasks.push(task);
//     }

//     // Wait for all workers
//     let mut workers = Vec::new();
//     for task in worker_tasks {
//         workers.push(task.await??);
//     }

//     // Wait for cohort complete
//     leader.await_cohort_complete().await?;
//     info!("Cohort complete with rank-less workers");

//     // Cleanup
//     for worker in workers {
//         worker.shutdown().await?;
//     }
//     leader.shutdown().await?;
//     leader_cancel.cancel();

//     Ok(())
// }
