// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::mpsc;
    use std::sync::{Arc, Condvar, Mutex};
    use std::thread;
    use std::time::Duration;

    use dynamo_runtime::protocols::annotated::Annotated;
    use serde_json::json;

    use super::{EngineResponse, OwnedFrameSink, ShardedProcessor};

    #[derive(Default)]
    struct RecordingSink {
        frames: Mutex<Vec<Annotated<serde_json::Value>>>,
        closed: AtomicBool,
    }

    impl OwnedFrameSink for RecordingSink {
        fn send(&self, frame: Annotated<serde_json::Value>) -> Result<(), String> {
            self.frames
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(frame);
            Ok(())
        }

        fn close(&self) {
            self.closed.store(true, Ordering::Relaxed);
        }

        fn close_with_error(&self, _message: String) {
            self.close();
        }
    }

    struct GateSink {
        entered: mpsc::Sender<u64>,
        client_id: u64,
        released: Arc<(Mutex<bool>, Condvar)>,
    }

    impl GateSink {
        fn release(&self) {
            let (lock, ready) = &*self.released;
            *lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = true;
            ready.notify_all();
        }
    }

    impl OwnedFrameSink for GateSink {
        fn send(&self, _frame: Annotated<serde_json::Value>) -> Result<(), String> {
            self.entered
                .send(self.client_id)
                .expect("test receiver remains open");
            let (lock, ready) = &*self.released;
            let mut released = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            while !*released {
                released = ready
                    .wait(released)
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
            }
            Ok(())
        }

        fn close(&self) {}

        fn close_with_error(&self, _message: String) {
            self.close();
        }
    }

    #[test]
    fn rejects_zero_shards_and_zero_queue_depth() {
        assert!(ShardedProcessor::new(0, 1).is_err());
        assert!(ShardedProcessor::new(1, 0).is_err());
    }

    #[test]
    fn clients_on_different_shards_process_concurrently() {
        let processor = Arc::new(ShardedProcessor::new(2, 1).expect("valid processor"));
        let (entered_tx, entered_rx) = mpsc::channel();
        let sink_zero = Arc::new(GateSink {
            entered: entered_tx.clone(),
            client_id: 2,
            released: Arc::new((Mutex::new(false), Condvar::new())),
        });
        let sink_one = Arc::new(GateSink {
            entered: entered_tx,
            client_id: 3,
            released: Arc::new((Mutex::new(false), Condvar::new())),
        });
        processor
            .register(2, 0, 1, sink_zero.clone(), 0.0)
            .expect("register shard zero client");
        processor
            .register(3, 0, 1, sink_one.clone(), 0.0)
            .expect("register shard one client");

        let processing = {
            let processor = processor.clone();
            thread::spawn(move || {
                processor.process_batch(vec![
                    EngineResponse::tokens(2, vec![vec![1]], false),
                    EngineResponse::tokens(3, vec![vec![2]], false),
                ])
            })
        };

        let first = entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("one worker entered the sink");
        let second_before_release = entered_rx.recv_timeout(Duration::from_millis(250));
        sink_zero.release();
        sink_one.release();
        processing.join().expect("batch processor did not panic");

        assert_ne!(second_before_release, Err(mpsc::RecvTimeoutError::Timeout));
        assert_ne!(first, second_before_release.expect("second shard entered"));
    }

    #[test]
    fn same_client_responses_remain_fifo() {
        let processor = ShardedProcessor::new(4, 1).expect("valid processor");
        let sink = Arc::new(RecordingSink::default());
        processor
            .register(9, 2, 1, sink.clone(), 0.0)
            .expect("register client 9");

        let outcome = processor
            .process_batch(vec![
                EngineResponse::tokens(9, vec![vec![10]], false),
                EngineResponse::tokens(9, vec![vec![11]], true),
            ])
            .expect("process ordered batch");

        let frames = sink
            .frames
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(frames[0].data, Some(json!({"token_ids": [10], "index": 0})));
        assert_eq!(frames[1].data.as_ref().unwrap()["token_ids"], json!([11]));
        assert_eq!(outcome.completed_client_ids, vec![9]);
        assert!(sink.closed.load(Ordering::Relaxed));
    }

    #[test]
    fn cancellation_does_not_wait_for_a_backpressured_shard() {
        let processor = Arc::new(ShardedProcessor::new(2, 1).expect("valid processor"));
        let (entered_tx, entered_rx) = mpsc::channel();
        let sink = Arc::new(GateSink {
            entered: entered_tx,
            client_id: 4,
            released: Arc::new((Mutex::new(false), Condvar::new())),
        });
        processor
            .register(4, 0, 1, sink.clone(), 0.0)
            .expect("register client 4");

        let processing = {
            let processor = processor.clone();
            thread::spawn(move || {
                processor.process_batch(vec![EngineResponse::tokens(
                    4,
                    vec![vec![1]],
                    false,
                )])
            })
        };
        entered_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("worker entered blocked sink");

        let (cancelled_tx, cancelled_rx) = mpsc::channel();
        let cancelling = {
            let processor = processor.clone();
            thread::spawn(move || {
                cancelled_tx
                    .send(processor.cancel(4))
                    .expect("test receiver remains open");
            })
        };
        let cancelled = cancelled_rx.recv_timeout(Duration::from_millis(250));

        sink.release();
        processing.join().expect("batch processor did not panic");
        cancelling.join().expect("cancellation did not panic");

        assert_eq!(cancelled, Ok(true));
        assert_eq!(processor.active_requests(), 0);
    }
}
