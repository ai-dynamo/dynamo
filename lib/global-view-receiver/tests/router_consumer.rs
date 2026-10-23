// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::mpsc::{self, Receiver, SyncSender, TryRecvError};

use dynamo_global_view_receiver::{ApplyError, GlobalViewReceiver, StreamToken};

enum MockEvent {
    Snapshot(Vec<(&'static str, u64)>),
    Upsert(&'static str, u64),
    Remove(&'static str),
    Failure,
}

struct MockConnection {
    stream: StreamToken<&'static str>,
    events: Receiver<MockEvent>,
}

fn mock_connection(
    receiver: &mut GlobalViewReceiver<&'static str, &'static str, u64>,
    relay: &'static str,
) -> (SyncSender<MockEvent>, MockConnection) {
    let (sender, events) = mpsc::sync_channel(1);
    let stream = receiver.open(relay);
    (sender, MockConnection { stream, events })
}

impl MockConnection {
    fn next(
        &self,
        receiver: &mut GlobalViewReceiver<&'static str, &'static str, u64>,
    ) -> Result<(), ApplyError> {
        match self.events.try_recv() {
            Ok(MockEvent::Snapshot(pools)) => receiver.replace_snapshot(&self.stream, pools),
            Ok(MockEvent::Upsert(pool_id, view)) => receiver.upsert(&self.stream, pool_id, view),
            Ok(MockEvent::Remove(pool_id)) => receiver.remove(&self.stream, pool_id),
            Ok(MockEvent::Failure) | Err(TryRecvError::Disconnected) => {
                receiver.disconnect(&self.stream)
            }
            Err(TryRecvError::Empty) => panic!("mock connection has no pending event"),
        }
    }
}

#[test]
fn router_consumer_builds_a_live_pool_view_and_browns_out_after_last_relay_loss() {
    let mut receiver = GlobalViewReceiver::<&str, &str, u64>::new();
    let relay_a = receiver.open("relay-a");
    let relay_b = receiver.open("relay-b");
    let blue = "blue-dgd";
    let green = "green-dgd";

    receiver
        .replace_snapshot(&relay_a, [(blue, 10), (green, 20)])
        .unwrap();
    receiver.replace_snapshot(&relay_b, [(blue, 20)]).unwrap();
    assert_eq!(receiver.pool_ids().count(), 2);
    let mut views = receiver
        .pool_views(&blue)
        .map(|(_, value)| *value)
        .collect::<Vec<_>>();
    views.sort_unstable();
    assert_eq!(views, vec![10, 20]);
    assert_eq!(receiver.pool_views(&green).count(), 1);

    receiver.remove(&relay_a, blue).unwrap();
    assert_eq!(receiver.pool_views(&blue).count(), 1);
    receiver.upsert(&relay_a, blue, 11).unwrap();
    assert_eq!(receiver.pool_views(&blue).count(), 2);
    receiver.replace_snapshot(&relay_a, [(blue, 12)]).unwrap();
    assert_eq!(receiver.pool_views(&green).count(), 0);

    receiver.disconnect(&relay_a).unwrap();
    assert_eq!(receiver.pool_views(&blue).count(), 1);
    receiver.disconnect(&relay_b).unwrap();
    assert_eq!(receiver.pool_views(&blue).count(), 0);
    assert_eq!(receiver.pool_ids().count(), 0);

    let recovered = receiver.open("relay-a");
    assert_eq!(receiver.pool_views(&blue).count(), 0);
    receiver.replace_snapshot(&recovered, [(blue, 13)]).unwrap();
    assert_eq!(
        receiver.pool_views(&blue).next().map(|(_, value)| *value),
        Some(13)
    );
}

#[test]
fn mock_connections_apply_events_and_remove_views_on_eof_or_error() {
    let mut receiver = GlobalViewReceiver::<&str, &str, u64>::new();
    let (relay_a_sender, relay_a) = mock_connection(&mut receiver, "relay-a");
    let (relay_b_sender, relay_b) = mock_connection(&mut receiver, "relay-b");
    let blue = "blue-dgd";
    let green = "green-dgd";

    relay_a_sender
        .send(MockEvent::Snapshot(vec![(blue, 10), (green, 20)]))
        .unwrap();
    relay_a.next(&mut receiver).unwrap();
    relay_b_sender
        .send(MockEvent::Snapshot(vec![(blue, 11)]))
        .unwrap();
    relay_b.next(&mut receiver).unwrap();
    assert_eq!(receiver.pool_views(&blue).count(), 2);

    relay_a_sender.send(MockEvent::Upsert(blue, 12)).unwrap();
    relay_a.next(&mut receiver).unwrap();
    assert!(
        receiver
            .pool_views(&blue)
            .any(|(relay, view)| *relay == "relay-a" && *view == 12)
    );
    relay_a_sender.send(MockEvent::Remove(green)).unwrap();
    relay_a.next(&mut receiver).unwrap();
    assert_eq!(receiver.pool_views(&green).count(), 0);

    drop(relay_a_sender);
    relay_a.next(&mut receiver).unwrap();
    assert_eq!(receiver.pool_views(&blue).count(), 1);

    relay_b_sender.send(MockEvent::Failure).unwrap();
    relay_b.next(&mut receiver).unwrap();
    assert_eq!(receiver.pool_views(&blue).count(), 0);
    assert_eq!(receiver.pool_ids().count(), 0);

    let (reconnected_sender, reconnected) = mock_connection(&mut receiver, "relay-a");
    assert_eq!(receiver.pool_views(&blue).count(), 0);
    reconnected_sender
        .send(MockEvent::Snapshot(vec![(blue, 13)]))
        .unwrap();
    reconnected.next(&mut receiver).unwrap();
    assert_eq!(
        receiver.pool_views(&blue).next().map(|(_, view)| *view),
        Some(13)
    );
}

#[test]
fn old_mock_connection_cannot_change_or_disconnect_its_replacement() {
    let mut receiver = GlobalViewReceiver::<&str, &str, u64>::new();
    let blue = "blue-dgd";
    let (old_sender, old) = mock_connection(&mut receiver, "relay-a");
    old_sender
        .send(MockEvent::Snapshot(vec![(blue, 10)]))
        .unwrap();
    old.next(&mut receiver).unwrap();

    let (new_sender, new) = mock_connection(&mut receiver, "relay-a");
    assert_eq!(receiver.pool_views(&blue).count(), 0);
    new_sender
        .send(MockEvent::Snapshot(vec![(blue, 20)]))
        .unwrap();
    new.next(&mut receiver).unwrap();

    old_sender.send(MockEvent::Upsert(blue, 30)).unwrap();
    assert_eq!(old.next(&mut receiver), Err(ApplyError::StaleStream));
    old_sender.send(MockEvent::Failure).unwrap();
    assert_eq!(old.next(&mut receiver), Err(ApplyError::StaleStream));
    assert_eq!(
        receiver.pool_views(&blue).next().map(|(_, view)| *view),
        Some(20)
    );
}
