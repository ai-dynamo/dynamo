// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! An instance reads the others directly even when its environment names a
//! proxy, as in clusters with an egress proxy. Its own test binary, because
//! it changes the environment.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use dynamo_global_discovery_service::backends::memory::{MemoryPoolStore, StaticReplicaDirectory};
use dynamo_global_discovery_service::pool_discovery::{
    Asker, Lease, PoolDirectory, PoolRegistrar, RegistrationSource, RouterReplica, RouterReplicaId,
};
use dynamo_global_discovery_service::service::{self, PoolDiscoveryBackend};
use dynamo_global_discovery_service::{HttpRegistrationSource, PeerSync};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;
use url::Url;
use uuid::Uuid;

#[tokio::test]
async fn other_instances_are_read_directly_when_the_environment_names_a_proxy() {
    // A proxy that counts the connections it gets and answers none.
    let proxy = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let proxy_url = format!("http://{}", proxy.local_addr().unwrap());
    let proxied = Arc::new(AtomicUsize::new(0));
    let counter = proxied.clone();
    tokio::spawn(async move {
        while let Ok((connection, _)) = proxy.accept().await {
            counter.fetch_add(1, Ordering::SeqCst);
            drop(connection);
        }
    });
    // SAFETY: this binary runs only this test, and no other thread reads the
    // environment while it changes.
    unsafe {
        for name in ["HTTP_PROXY", "http_proxy", "ALL_PROXY", "all_proxy"] {
            std::env::set_var(name, &proxy_url);
        }
        // REQUEST_METHOD marks a CGI process, whose proxy variables are ignored.
        for name in ["NO_PROXY", "no_proxy", "REQUEST_METHOD"] {
            std::env::remove_var(name);
        }
    }

    // An instance that shares registrations, served over HTTP.
    let lease = Lease::new(1, 3).unwrap();
    let store = Arc::new(MemoryPoolStore::new(lease));
    let (_, sharing) = PeerSync::new(
        Arc::new(StaticReplicaDirectory::new(Vec::new())),
        Arc::new(HttpRegistrationSource::new().unwrap()),
        store.clone(),
        lease,
    )
    .unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = Url::parse(&format!("http://{}/", listener.local_addr().unwrap())).unwrap();
    let pools: Arc<PoolDirectory> = store.clone();
    let registrar: Arc<dyn PoolRegistrar> = store;
    let stop = CancellationToken::new();
    tokio::spawn(service::serve(
        listener,
        PoolDiscoveryBackend {
            pools,
            routers: Arc::new(StaticReplicaDirectory::new(Vec::new())),
            registrar: Some(registrar),
            sharing: Some(Arc::new(sharing)),
            router_registrar: None,
        },
        None,
        stop.clone(),
    ));

    // Control: a client that follows the environment goes to the proxy, so
    // the check below can tell a direct read from a proxied one.
    let healthz = url.join("healthz").unwrap();
    assert!(reqwest::Client::new().get(healthz).send().await.is_err());
    assert_eq!(proxied.load(Ordering::SeqCst), 1);

    let instance = RouterReplica {
        replica: RouterReplicaId("a".to_owned()),
        url,
    };
    let asker = Asker {
        instance: Uuid::new_v4(),
        holding: false,
    };
    let reply = HttpRegistrationSource::new()
        .unwrap()
        .shared_registrations(&instance, asker)
        .await;
    assert!(reply.is_ok(), "{reply:?}");
    assert_eq!(proxied.load(Ordering::SeqCst), 1);
    stop.cancel();
}
