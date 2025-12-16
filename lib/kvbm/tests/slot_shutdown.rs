// // SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// // SPDX-License-Identifier: Apache-2.0

// use std::sync::Arc;

// use dynamo_kvbm::v2::integrations::connector::slot::{
//     PlannedTransfer, SlotActions, SlotState, TransferCandidateMetadata,
// };
// use dynamo_kvbm::v2::integrations::connector::{MockBlockManager, TransferDirection};
// use dynamo_kvbm::v2::logical::executor::testing::{
//     ConsoleEvent, EngineCommandKind, MockLeader, MockTransferEngine, MockWorker,
//     RecordingConsoleHook, SlotFinishStatus, sample_block_ids, sample_token_blocks,
// };
// use dynamo_kvbm::v2::logical::executor::{
//     BroadcastSlotStateBroadcaster, SlotExecutor, TransferPipelineRuntime,
// };
// use tokio::sync::broadcast;
// use tokio::time::{Duration, sleep};

// #[test]
// fn test_multi_slot_shutdown_flow() {
//     let recording = Arc::new(RecordingConsoleHook::new());
//     let leader = MockLeader::new(Some(recording.as_hook()));

//     let slot_a = leader.create_slot("req-a", 128);
//     let slot_b = leader.create_slot("req-b", 128);

//     slot_a.set_state(SlotState::Prefilling);
//     slot_b.set_state(SlotState::Prefilling);

//     let worker0 = MockWorker::new(0, TransferDirection::DeviceToHost);
//     let worker1 = MockWorker::new(1, TransferDirection::DeviceToHost);

//     let op_a0 = worker0.start_operation(slot_a.as_ref(), 4);
//     let op_a1 = worker1.start_operation(slot_a.as_ref(), 3);
//     let op_b0 = worker0.start_operation(slot_b.as_ref(), 2);

//     match slot_a.request_finish() {
//         SlotFinishStatus::Pending { outstanding, .. } => assert_eq!(outstanding.len(), 2),
//         other => panic!("slot_a expected pending status, got {other:?}"),
//     }

//     match slot_b.request_finish() {
//         SlotFinishStatus::Pending { outstanding, .. } => assert_eq!(outstanding.len(), 1),
//         other => panic!("slot_b expected pending status, got {other:?}"),
//     }

//     assert!(
//         !worker0.finish_operation(slot_a.as_ref(), op_a0),
//         "first slot_a completion should not finish the slot"
//     );
//     assert_eq!(slot_a.state(), SlotState::Finishing);
//     assert_eq!(slot_b.state(), SlotState::Finishing);

//     assert!(
//         worker1.finish_operation(slot_a.as_ref(), op_a1),
//         "second slot_a completion should finish the slot"
//     );
//     assert_eq!(slot_a.state(), SlotState::Finished);
//     assert_eq!(slot_b.state(), SlotState::Finishing);

//     assert!(
//         worker0.finish_operation(slot_b.as_ref(), op_b0),
//         "slot_b completion should finish the slot"
//     );
//     assert_eq!(slot_b.state(), SlotState::Finished);

//     let events = recording.events();

//     let events_for = |request: &str| -> Vec<&'static str> {
//         events
//             .iter()
//             .filter_map(|event| classify_event(event, request))
//             .collect()
//     };

//     assert_eq!(
//         events_for("req-a"),
//         vec![
//             "slot_created",
//             "state_prefilling",
//             "operation_registered",
//             "operation_registered",
//             "state_finishing",
//             "finish_started",
//             "operation_completed",
//             "operation_completed",
//             "state_finished",
//             "slot_finished",
//         ],
//         "slot_a events captured unexpected ordering: {events:?}"
//     );

//     assert_eq!(
//         events_for("req-b"),
//         vec![
//             "slot_created",
//             "state_prefilling",
//             "operation_registered",
//             "state_finishing",
//             "finish_started",
//             "operation_completed",
//             "state_finished",
//             "slot_finished",
//         ],
//         "slot_b events captured unexpected ordering: {events:?}"
//     );

//     let mut counts = events
//         .iter()
//         .filter_map(|event| match event {
//             ConsoleEvent::FinishStarted {
//                 request_id,
//                 outstanding,
//             } => Some((request_id.as_str(), *outstanding)),
//             _ => None,
//         })
//         .collect::<Vec<_>>();
//     counts.sort_by_key(|(req, _)| *req);

//     assert_eq!(
//         counts,
//         vec![("req-a", 2), ("req-b", 1)],
//         "unexpected outstanding counts at finish"
//     );
// }

// #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
// async fn test_pipeline_dispatches_offloads_multi_slot() {
//     let recording = Arc::new(RecordingConsoleHook::new());
//     let leader = MockLeader::new(Some(recording.as_hook()));

//     let block_manager = Arc::new(MockBlockManager::new());
//     let runtime = TransferPipelineRuntime::<String>::new(block_manager);

//     let engine = MockTransferEngine::spawn(
//         runtime.engine_receiver().expect("engine receiver"),
//         Arc::clone(&leader),
//     );

//     let (state_tx, _state_rx) = broadcast::channel(8);
//     let broadcaster = BroadcastSlotStateBroadcaster::new(state_tx);
//     let executor = SlotExecutor::new(runtime.dispatcher(), broadcaster);

//     let slot_a = leader.create_slot("req-a", 4);
//     let slot_b = leader.create_slot("req-b", 4);

//     runtime
//         .slots()
//         .insert("req-a".to_string(), slot_a.slot_handle_dyn());
//     runtime
//         .slots()
//         .insert("req-b".to_string(), slot_b.slot_handle_dyn());

//     slot_a.set_state(SlotState::Prefilling);
//     slot_b.set_state(SlotState::Prefilling);

//     let block_ids_a = sample_block_ids(1, 2);
//     let block_ids_b = sample_block_ids(21, 3);

//     let uuid_a = slot_a.plan_operation(0, TransferDirection::DeviceToHost, block_ids_a.len());
//     let uuid_b = slot_b.plan_operation(1, TransferDirection::DeviceToHost, block_ids_b.len());

//     slot_a.with_slot_mut(|slot| {
//         let block_size = slot.core().block_size();
//         slot.store_candidate_metadata(
//             uuid_a,
//             TransferCandidateMetadata::Offload {
//                 token_blocks: sample_token_blocks(block_size, block_ids_a.len()),
//             },
//         );
//         let actions = SlotActions::default().with_transfer(PlannedTransfer {
//             transfer_id: uuid_a,
//             direction: TransferDirection::DeviceToHost,
//             block_ids: block_ids_a.clone(),
//             token_blocks: sample_token_blocks(block_size, block_ids_a.len()),
//         });
//         executor.execute(slot, actions).unwrap();
//     });

//     slot_b.with_slot_mut(|slot| {
//         let block_size = slot.core().block_size();
//         slot.store_candidate_metadata(
//             uuid_b,
//             TransferCandidateMetadata::Offload {
//                 token_blocks: sample_token_blocks(block_size, block_ids_b.len()),
//             },
//         );
//         let actions = SlotActions::default().with_transfer(PlannedTransfer {
//             transfer_id: uuid_b,
//             direction: TransferDirection::DeviceToHost,
//             block_ids: block_ids_b.clone(),
//             token_blocks: sample_token_blocks(block_size, block_ids_b.len()),
//         });
//         executor.execute(slot, actions).unwrap();
//     });

//     sleep(Duration::from_millis(50)).await;

//     let status_a = slot_a.request_finish();
//     let status_b = slot_b.request_finish();
//     assert!(matches!(status_a, SlotFinishStatus::Ready));
//     assert!(matches!(status_b, SlotFinishStatus::Ready));

//     assert_eq!(slot_a.state(), SlotState::Finished);
//     assert_eq!(slot_b.state(), SlotState::Finished);

//     drop(runtime);
//     let events = engine.shutdown().await;

//     assert_eq!(events.len(), 2, "expected two engine events: {events:?}");

//     let mut per_request: Vec<(String, usize)> = events
//         .into_iter()
//         .map(|event| {
//             let blocks = match event.kind {
//                 EngineCommandKind::Offload { num_blocks } => num_blocks,
//                 EngineCommandKind::Onboard { num_blocks } => num_blocks,
//                 EngineCommandKind::Noop => 0,
//             };
//             (event.request_id, blocks)
//         })
//         .collect();
//     per_request.sort_by(|a, b| a.0.cmp(&b.0));

//     assert_eq!(
//         per_request,
//         vec![
//             ("req-a".to_string(), block_ids_a.len()),
//             ("req-b".to_string(), block_ids_b.len())
//         ],
//         "unexpected engine events per request"
//     );
// }

// fn classify_event(event: &ConsoleEvent, target: &str) -> Option<&'static str> {
//     match event {
//         ConsoleEvent::SlotCreated {
//             request_id,
//             state: SlotState::Initialized,
//         } if request_id == target => Some("slot_created"),
//         ConsoleEvent::StateTransition {
//             request_id,
//             from: SlotState::Initialized,
//             to: SlotState::Prefilling,
//         } if request_id == target => Some("state_prefilling"),
//         ConsoleEvent::StateTransition {
//             request_id,
//             from: SlotState::Prefilling,
//             to: SlotState::Finishing,
//         } if request_id == target => Some("state_finishing"),
//         ConsoleEvent::StateTransition {
//             request_id,
//             from: SlotState::Finishing,
//             to: SlotState::Finished,
//         } if request_id == target => Some("state_finished"),
//         ConsoleEvent::OperationRegistered { request_id, .. } if request_id == target => {
//             Some("operation_registered")
//         }
//         ConsoleEvent::OperationCompleted { request_id, .. } if request_id == target => {
//             Some("operation_completed")
//         }
//         ConsoleEvent::FinishStarted { request_id, .. } if request_id == target => {
//             Some("finish_started")
//         }
//         ConsoleEvent::SlotFinished { request_id } if request_id == target => Some("slot_finished"),
//         _ => None,
//     }
// }
