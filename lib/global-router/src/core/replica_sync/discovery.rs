// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Keeps a [`ReplicaHub`]'s peers matching a peer set published from
//! outside, such as the Global Frontend's poll of Global Discovery's
//! `/routers`. The hub does not know where the set comes from; it only
//! sees a [`watch`] channel of API addresses.

use std::collections::HashSet;
use std::sync::Arc;

use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use url::Url;

use super::{ReplicaHub, peer_url};

/// Keeps the hub's peers matching each set `peers` publishes, until
/// `cancel` fires or the sender drops. Run it as a task.
///
/// Peers the hub has when this starts are pinned: a static `--peer` stays
/// however the published set changes. Each set applies in full: addresses
/// new to the hub are added, and unpinned peers missing from it are
/// removed. An address the dialer cannot use (not `http://`, see
/// [`peer_url`]) is skipped with a warning.
pub async fn follow_peer_set(
    hub: Arc<ReplicaHub>,
    mut peers: watch::Receiver<Vec<Url>>,
    cancel: CancellationToken,
) {
    let pinned: HashSet<Url> = hub.peers().into_iter().collect();
    // Apply the current set first, then each change.
    peers.mark_changed();
    loop {
        tokio::select! {
            _ = cancel.cancelled() => return,
            changed = peers.changed() => if changed.is_err() {
                return;
            },
        }
        let wanted = peers.borrow_and_update().clone();
        apply_peer_set(&hub, &pinned, &wanted);
    }
}

fn apply_peer_set(hub: &ReplicaHub, pinned: &HashSet<Url>, wanted: &[Url]) {
    let mut keep = pinned.clone();
    for url in wanted {
        let peer = match peer_url(url.as_str()) {
            Ok(peer) => peer,
            Err(error) => {
                tracing::warn!(error = %error, "skipping a discovered replica the dialer cannot reach");
                continue;
            }
        };
        if keep.insert(peer.clone()) && hub.add_peer(peer.clone()) {
            tracing::info!(peer = %peer, "following a discovered replica");
        }
    }
    for peer in hub.peers() {
        if !keep.contains(&peer) && hub.remove_peer(&peer) {
            tracing::info!(peer = %peer, "a discovered replica left; no longer following it");
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn url(s: &str) -> Url {
        s.parse().unwrap()
    }

    /// Waits until the hub's peers are exactly `expected`, in any order.
    async fn assert_peers(hub: &ReplicaHub, expected: &[&str]) {
        let expected: HashSet<Url> = expected.iter().map(|s| url(s)).collect();
        tokio::time::timeout(Duration::from_secs(5), async {
            loop {
                let peers: HashSet<Url> = hub.peers().into_iter().collect();
                if peers == expected {
                    return;
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "peers never became {expected:?}, last seen {:?}",
                hub.peers()
            )
        });
    }

    #[tokio::test]
    async fn follows_the_published_set_and_keeps_pinned_peers() {
        let cancel = CancellationToken::new();
        let hub = ReplicaHub::new(cancel.clone());
        assert!(hub.add_peer(url("http://static:8000")));
        let (publish, peers) = watch::channel(vec![url("http://a:8000")]);
        tokio::spawn(follow_peer_set(hub.clone(), peers, cancel.clone()));

        assert_peers(&hub, &["http://static:8000/", "http://a:8000/"]).await;

        // A new set adds and removes; the pinned peer survives, even if the
        // set lists and then drops it.
        publish
            .send(vec![
                url("http://b:8000"),
                url("http://static:8000"),
                url("http://a:8000"),
            ])
            .unwrap();
        assert_peers(
            &hub,
            &["http://static:8000/", "http://a:8000/", "http://b:8000/"],
        )
        .await;
        publish.send(vec![url("http://b:8000")]).unwrap();
        assert_peers(&hub, &["http://static:8000/", "http://b:8000/"]).await;
        publish.send(Vec::new()).unwrap();
        assert_peers(&hub, &["http://static:8000/"]).await;
        cancel.cancel();
    }

    #[tokio::test]
    async fn skips_addresses_the_dialer_cannot_use() {
        let cancel = CancellationToken::new();
        let hub = ReplicaHub::new(cancel.clone());
        let (publish, peers) = watch::channel(vec![
            url("https://tls:8000"),
            url("tcp://mesh:5600"),
            url("http://a:8000"),
        ]);
        tokio::spawn(follow_peer_set(hub.clone(), peers, cancel.clone()));
        assert_peers(&hub, &["http://a:8000/"]).await;
        drop(publish);
        cancel.cancel();
    }
}
