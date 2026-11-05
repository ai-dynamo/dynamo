// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The DynamoDB store against dynamodb-local. Every test but the last skips
//! unless `GDS_TEST_DYNAMODB_ENDPOINT` points to a running DynamoDB Local instance.
//! Each test makes its own table, and moves its own `ManualClock`.

mod common;
#[macro_use]
#[path = "common/dynamodb.rs"]
mod dynamodb_support;

use std::time::Duration;

use aws_sdk_dynamodb::types::AttributeValue;
use common::{conformance, find, key, lease, mocker, next_change};
use dynamo_global_discovery_service::pool_discovery::{
    Completeness, Directory, DiscoveryError, Lease, PoolEvent, PoolListing, PoolRegistrar,
    PoolRegistration, Snapshot, WatchEvent,
};
use dynamo_global_discovery_service::{DynamoDbPoolStore, SystemClock};
use dynamodb_support::{
    Fixture, POLL, REPLICATION_GRACE, SKEW, config, test_client_without_retries,
};
use futures::StreamExt;
use std::sync::Arc;
use tokio::time::{Instant, timeout_at};
use url::Url;

const S: fn(&str) -> AttributeValue = |value| AttributeValue::S(value.to_owned());

macro_rules! conforms {
    ($($case:ident),+ $(,)?) => {$(
        #[tokio::test]
        async fn $case() {
            let endpoint = skip_without_dynamodb!();
            let fixture = Fixture::new(&endpoint).await;
            let store = fixture.store();
            conformance::$case(store.clone(), store, fixture.clock_handle()).await;
        }
    )+};
}

conforms!(
    register_answers_the_configured_lease,
    a_registered_pool_is_listed_normalised,
    the_last_registration_wins,
    distinct_keys_are_distinct_pools,
    registering_again_only_renews,
    a_missed_lease_hides_then_removes_the_pool,
    only_the_pool_that_missed_its_lease_expires,
    lists_are_exact_once_a_lease_has_passed,
);

/// The attributes of the item for `dgd` of cluster `west`, read directly.
async fn raw_item(
    fixture: &Fixture,
    dgd: &str,
) -> Option<std::collections::HashMap<String, AttributeValue>> {
    fixture
        .client
        .get_item()
        .table_name(&fixture.table)
        .key("pk", S("pool"))
        .key("sk", S(&format!("west#dynamo#{dgd}")))
        .consistent_read(true)
        .send()
        .await
        .unwrap()
        .item
}

fn number(item: &std::collections::HashMap<String, AttributeValue>, name: &str) -> u64 {
    item[name].as_n().unwrap().parse().unwrap()
}

#[tokio::test]
async fn an_item_has_the_documented_shape() {
    let endpoint = skip_without_dynamodb!();
    let fixture = Fixture::new(&endpoint).await;
    let store = fixture.store();
    let _ = store.register(mocker("mocker", "m1")).await.unwrap();
    let first_ms = fixture.now_ms();

    let item = raw_item(&fixture, "mocker").await.expect("stored");
    assert_eq!(item["pk"].as_s().unwrap(), "pool");
    assert_eq!(item["sk"].as_s().unwrap(), "west#dynamo#mocker");
    assert_eq!(number(&item, "renewed_at"), first_ms);
    assert_eq!(number(&item, "created_at"), first_ms);
    assert_eq!(number(&item, "lease_s"), 15);
    // The region of the table replica the store writes through.
    assert_eq!(item["written_in"].as_s().unwrap(), "us-east-1");
    // The TTL is garbage collection only: a week after the last renewal, in
    // unix seconds.
    assert_eq!(number(&item, "expires_at"), first_ms / 1000 + 7 * 24 * 3600);
    // The registration is JSON with its lists sorted.
    let stored: PoolRegistration =
        serde_json::from_str(item["registration"].as_s().unwrap()).unwrap();
    assert_eq!(stored.capabilities, ["decode", "prefill"]);
    let hash = item["hash"].as_s().unwrap().clone();
    assert_eq!(hash.len(), 64, "{hash}");
    assert!(hash.chars().all(|c| c.is_ascii_hexdigit()), "{hash}");

    // A renewal moves `renewed_at` and the TTL, and keeps the rest.
    fixture.advance(Duration::from_secs(5));
    let _ = store.register(mocker("mocker", "m1")).await.unwrap();
    let renewed = raw_item(&fixture, "mocker").await.unwrap();
    assert_eq!(number(&renewed, "renewed_at"), first_ms + 5000);
    assert_eq!(number(&renewed, "created_at"), first_ms);
    assert_eq!(renewed["hash"].as_s().unwrap(), &hash);
    assert_eq!(
        number(&renewed, "expires_at"),
        (first_ms + 5000) / 1000 + 7 * 24 * 3600
    );

    // A change is a new hash.
    let _ = store.register(mocker("mocker", "m2")).await.unwrap();
    let changed = raw_item(&fixture, "mocker").await.unwrap();
    assert_ne!(changed["hash"].as_s().unwrap(), &hash);
    assert_eq!(number(&changed, "created_at"), first_ms);
}

#[tokio::test]
async fn two_stores_over_one_table_list_each_others_pools() {
    let endpoint = skip_without_dynamodb!();
    let fixture = Fixture::new(&endpoint).await;
    let (a, b) = (fixture.store(), fixture.store());

    let _ = a.register(mocker("from-a", "m1")).await.unwrap();
    let _ = b.register(mocker("from-b", "m1")).await.unwrap();
    for store in [&a, &b] {
        let snapshot = store.list().await.unwrap();
        // Until a lease has passed, a PoolRelay may still register elsewhere.
        assert_eq!(snapshot.completeness, Completeness::Estimated);
        let mut names: Vec<_> = snapshot
            .items
            .iter()
            .map(|listing| listing.key.dgd_name().to_owned())
            .collect();
        names.sort();
        assert_eq!(names, ["from-a", "from-b"]);
    }
    // A fresh replica lists every pool at once.
    let fresh = fixture.store();
    assert_eq!(fresh.list().await.unwrap().items.len(), 2);

    // A lease later, every replica is exact at its next poll, and the pools
    // are still live.
    fixture.advance(lease().duration());
    for store in [&a, &b, &fresh] {
        assert_eq!(exact_list(store).await.items.len(), 2);
    }
}

/// `store`'s list once it is `Exact`. Lists come from the store's view of
/// the table, so one that becomes complete does at the store's next poll.
async fn exact_list(store: &DynamoDbPoolStore) -> Snapshot<PoolListing> {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let snapshot = store.list().await.unwrap();
        if snapshot.completeness == Completeness::Exact {
            return snapshot;
        }
        assert!(Instant::now() < deadline, "the list never became exact");
        tokio::time::sleep(POLL).await;
    }
}

#[tokio::test]
async fn two_stores_see_each_other_through_the_watch() {
    let endpoint = skip_without_dynamodb!();
    let fixture = Fixture::new(&endpoint).await;
    let (a, b) = (fixture.store(), fixture.store());

    let _ = a.register(mocker("before", "m1")).await.unwrap();
    let mut watch = b.list_and_watch(None).await.unwrap();
    let Some(Ok(WatchEvent::Upserted(initial))) = watch.next().await else {
        panic!("expected the pool the other store registered first");
    };
    assert_eq!(initial.key, key("west", "dynamo", "before"));
    let Some(Ok(WatchEvent::Resync(snapshot))) = watch.next().await else {
        panic!("expected the first Resync");
    };
    // `Exact` only once a lease has passed since B was made.
    assert_eq!(snapshot.completeness, Completeness::Estimated);
    assert_eq!(snapshot.items.len(), 1);

    // Within a few polls, B learns what A registers.
    let _ = a.register(mocker("after", "m1")).await.unwrap();
    let PoolEvent::Upserted(listing) = next_change(&mut watch).await else {
        panic!("expected the new pool");
    };
    assert_eq!(listing.key, key("west", "dynamo", "after"));

    // And a change.
    let _ = a.register(mocker("after", "m2")).await.unwrap();
    let PoolEvent::Upserted(listing) = next_change(&mut watch).await else {
        panic!("expected the change");
    };
    assert_eq!(listing.model.as_deref(), Some("m2"));

    // And when it goes: past A's lease, with nobody renewing.
    fixture.advance(lease().duration() + SKEW + Duration::from_secs(1));
    let mut removed = Vec::new();
    while removed.len() < 2 {
        let PoolEvent::Removed(gone) = next_change(&mut watch).await else {
            panic!("expected the pools to be removed");
        };
        removed.push(gone.dgd_name().to_owned());
    }
    removed.sort();
    assert_eq!(removed, ["after", "before"]);
}

#[tokio::test]
async fn a_renewal_on_another_store_publishes_nothing() {
    let endpoint = skip_without_dynamodb!();
    let fixture = Fixture::new(&endpoint).await;
    let (a, b) = (fixture.store(), fixture.store());
    let _ = a.register(mocker("mocker", "m1")).await.unwrap();
    let mut watch = b.list_and_watch(None).await.unwrap();
    assert!(matches!(
        watch.next().await,
        Some(Ok(WatchEvent::Upserted(_)))
    ));
    assert!(matches!(
        watch.next().await,
        Some(Ok(WatchEvent::Resync(_)))
    ));

    // Renew for a while, over several polls.
    for _ in 0..4 {
        fixture.advance(Duration::from_secs(5));
        let _ = a.register(mocker("mocker", "m1")).await.unwrap();
        tokio::time::sleep(POLL).await;
    }
    // Nothing but, at most, a Resync.
    let deadline = Instant::now() + Duration::from_secs(1);
    while let Ok(next) = timeout_at(deadline, watch.next()).await {
        match next {
            Some(Ok(WatchEvent::Resync(_))) => {}
            Some(Ok(WatchEvent::Upserted(listing))) => {
                panic!("renewal published an upsert of {:?}", listing.key)
            }
            Some(Ok(WatchEvent::Removed(gone))) => panic!("renewal removed {gone:?}"),
            other => panic!("watch ended: {:?}", other.is_some()),
        }
    }
    assert!(find(&*b, "mocker").await.is_some());
}

#[tokio::test]
async fn a_pool_expires_once_the_lease_and_the_skew_have_passed() {
    let endpoint = skip_without_dynamodb!();
    let fixture = Fixture::new(&endpoint).await;
    let store = fixture.store();
    let _ = store.register(mocker("mocker", "m1")).await.unwrap();

    // Listed through the lease and the skew, and gone just after. (The
    // design makes the bound inclusive; the test stays a millisecond clear of
    // it.)
    let window = lease().duration() + SKEW;
    fixture.advance(window - Duration::from_millis(1));
    assert!(find(&*store, "mocker").await.is_some());
    fixture.advance(Duration::from_millis(2));
    assert!(find(&*store, "mocker").await.is_none());
    // The item is still there: the store never deletes, TTL does, a week on.
    assert!(raw_item(&fixture, "mocker").await.is_some());

    // A renewal brings it back with a fresh lease.
    let _ = store.register(mocker("mocker", "m1")).await.unwrap();
    assert!(find(&*store, "mocker").await.is_some());
}

#[tokio::test]
async fn a_reader_uses_the_lease_its_writer_granted() {
    let endpoint = skip_without_dynamodb!();
    let fixture = Fixture::new(&endpoint).await;
    let writer = fixture.store();
    // Configured with a far longer lease than the writer's 15 s.
    let reader = fixture.store_with_lease(Lease::new(30, 600).unwrap());
    let _ = writer.register(mocker("mocker", "m1")).await.unwrap();
    assert!(find(&*reader, "mocker").await.is_some());

    fixture.advance(Duration::from_secs(30));
    assert!(find(&*reader, "mocker").await.is_none());
}

/// With a Global Table, a renewal written through another region's replica
/// may arrive late, so a pool written there stays listed past its lease, as
/// `Estimated`, until the replication grace runs out. One written through
/// this store's region does not.
#[tokio::test]
async fn a_pool_written_in_another_region_stays_listed_until_the_grace() {
    let endpoint = skip_without_dynamodb!();
    let fixture = Fixture::new(&endpoint).await;
    // The fixture's stores read the us-east-1 replica.
    let store = fixture.store();
    let there = serde_json::to_string(&mocker("there", "m1")).unwrap();
    let here = serde_json::to_string(&mocker("here", "m1")).unwrap();
    put_raw(&fixture, "there", &there, Some(("written_in", "eu-west-1"))).await;
    put_raw(&fixture, "here", &here, Some(("written_in", "us-east-1"))).await;

    // A lease later both are still live, and the list is complete.
    fixture.advance(lease().duration());
    let snapshot = store.list().await.unwrap();
    assert_eq!(snapshot.items.len(), 2);
    assert_eq!(snapshot.completeness, Completeness::Exact);

    // Past the lease and the skew, this region's pool is gone. The other
    // region's stays, and the list may be wrong.
    fixture.advance(SKEW + Duration::from_secs(1));
    let snapshot = store.list().await.unwrap();
    assert_eq!(snapshot.completeness, Completeness::Estimated);
    assert!(find(&*store, "here").await.is_none());
    assert!(find(&*store, "there").await.is_some());

    // Past the grace too, it is gone at once, and the list is complete again
    // at the next poll.
    fixture.advance(REPLICATION_GRACE);
    assert!(store.list().await.unwrap().items.is_empty());
    assert!(exact_list(&store).await.items.is_empty());
}

/// An item written outside the store, with the attributes the store reads.
async fn put_raw(fixture: &Fixture, dgd: &str, registration: &str, extra: Option<(&str, &str)>) {
    let now = fixture.now_ms();
    let mut request = fixture
        .client
        .put_item()
        .table_name(&fixture.table)
        .item("pk", S("pool"))
        .item("sk", S(&format!("west#dynamo#{dgd}")))
        .item("registration", S(registration))
        .item("hash", S("0000"))
        .item("renewed_at", AttributeValue::N(now.to_string()))
        .item("lease_s", AttributeValue::N("15".to_owned()))
        .item("created_at", AttributeValue::N(now.to_string()))
        .item(
            "expires_at",
            AttributeValue::N((now / 1000 + 7 * 24 * 3600).to_string()),
        );
    if let Some((name, value)) = extra {
        request = request.item(name, S(value));
    }
    request.send().await.unwrap();
}

#[tokio::test]
async fn a_corrupt_item_is_skipped_while_the_others_list() {
    let endpoint = skip_without_dynamodb!();
    let fixture = Fixture::new(&endpoint).await;
    let store = fixture.store();
    let _ = store.register(mocker("good-1", "m1")).await.unwrap();
    put_raw(&fixture, "not-json", "this is not json", None).await;
    put_raw(&fixture, "wrong-shape", r#"{"key": 1}"#, None).await;
    // A registration the service would refuse: an invalid key.
    put_raw(
        &fixture,
        "bad-key",
        r#"{"key":{"cluster_id":" ","namespace":"n","dgd_name":"d"},"location":{"region":"r","availability_zone":null,"datacenter":null},"model":"m"}"#,
        None,
    )
    .await;
    let _ = store.register(mocker("good-2", "m1")).await.unwrap();

    let snapshot = store.list().await.unwrap();
    let mut names: Vec<_> = snapshot
        .items
        .iter()
        .map(|listing| listing.key.dgd_name().to_owned())
        .collect();
    names.sort();
    assert_eq!(names, ["good-1", "good-2"]);

    // The watch skips them too, and a registration over a corrupt item
    // replaces it.
    let mut watch = store.list_and_watch(None).await.unwrap();
    let mut initial = 0;
    loop {
        match watch.next().await {
            Some(Ok(WatchEvent::Upserted(_))) => initial += 1,
            Some(Ok(WatchEvent::Resync(_))) => break,
            other => panic!("unexpected {:?}", other.is_some()),
        }
    }
    assert_eq!(initial, 2);
    let _ = store.register(mocker("not-json", "m1")).await.unwrap();
    assert!(find(&*store, "not-json").await.is_some());
}

#[tokio::test]
async fn an_unknown_attribute_is_tolerated() {
    let endpoint = skip_without_dynamodb!();
    let fixture = Fixture::new(&endpoint).await;
    let store = fixture.store();
    // One written by a newer version, next to the attributes this one knows,
    // and one that gets one added after the fact.
    let registration = serde_json::to_string(&mocker("newer", "m1")).unwrap();
    put_raw(&fixture, "newer", &registration, Some(("added_later", "x"))).await;
    let _ = store.register(mocker("older", "m1")).await.unwrap();
    fixture
        .client
        .update_item()
        .table_name(&fixture.table)
        .key("pk", S("pool"))
        .key("sk", S("west#dynamo#older"))
        .update_expression("SET unknown_attribute = :v")
        .expression_attribute_values(":v", AttributeValue::M(Default::default()))
        .send()
        .await
        .unwrap();

    let snapshot = store.list().await.unwrap();
    assert_eq!(snapshot.items.len(), 2, "{:?}", snapshot.items);
    // The attribute is not in the way of a renewal or of a change.
    let _ = store.register(mocker("older", "m2")).await.unwrap();
    assert_eq!(
        find(&*store, "older").await.unwrap().model.as_deref(),
        Some("m2")
    );
}

/// More than one 1 MB page of items, so `list` has to paginate.
#[tokio::test]
async fn a_list_spans_pages() {
    let endpoint = skip_without_dynamodb!();
    let fixture = Fixture::new(&endpoint).await;
    let store = fixture.store();
    let pools = 450;
    futures::stream::iter(0..pools)
        .for_each_concurrent(32, |index| {
            let store = store.clone();
            async move {
                let mut registration = mocker(&format!("pool-{index}"), "m1");
                registration.frontends = (0..120)
                    .map(|frontend| {
                        format!("http://10.{}.{}.{frontend}:8000/", index / 250, index % 250)
                            .parse()
                            .unwrap()
                    })
                    .collect();
                let _ = store.register(registration).await.unwrap();
            }
        })
        .await;

    let snapshot = store.list().await.unwrap();
    assert_eq!(snapshot.items.len(), pools);
    let mut names: Vec<_> = snapshot
        .items
        .iter()
        .map(|listing| listing.key.dgd_name().to_owned())
        .collect();
    names.sort();
    names.dedup();
    assert_eq!(names.len(), pools);
}

/// `connect` builds its client from the default credential chain, so its
/// tests need `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` (any value
/// works for dynamodb-local). Says why it skips when they are not set.
fn without_credentials() -> bool {
    let missing = std::env::var_os("AWS_ACCESS_KEY_ID").is_none();
    if missing {
        eprintln!(
            "skipping: `connect` reads credentials from the environment; \
             export AWS_ACCESS_KEY_ID=test AWS_SECRET_ACCESS_KEY=test"
        );
    }
    missing
}

/// TTL is on `expires_at`, or being turned on.
async fn assert_ttl_is_on(client: &aws_sdk_dynamodb::Client, table: &str) {
    let ttl = client
        .describe_time_to_live()
        .table_name(table)
        .send()
        .await
        .unwrap();
    let ttl = ttl.time_to_live_description().unwrap();
    assert_eq!(ttl.attribute_name(), Some("expires_at"));
    assert!(
        matches!(
            ttl.time_to_live_status(),
            Some(
                aws_sdk_dynamodb::types::TimeToLiveStatus::Enabled
                    | aws_sdk_dynamodb::types::TimeToLiveStatus::Enabling
            )
        ),
        "{ttl:?}"
    );
}

/// `connect` creates the table, and its TTL, when `ensure_table` is set.
#[tokio::test]
async fn ensure_table_creates_the_table_and_its_ttl() {
    let endpoint = skip_without_dynamodb!();
    if without_credentials() {
        return;
    }
    let client = dynamodb_support::test_client(&endpoint);
    let table = dynamodb_support::fresh_table_name();

    // Without `ensure_table`, a missing table is a startup error that names it.
    let Err(error) = DynamoDbPoolStore::connect(lease(), config(&table, &endpoint)).await else {
        panic!("connected to a table that does not exist");
    };
    assert!(format!("{error:#}").contains(&table), "{error:#}");

    let mut ensure = config(&table, &endpoint);
    ensure.ensure_table = true;
    let store = DynamoDbPoolStore::connect(lease(), ensure.clone())
        .await
        .unwrap();
    // `connect` checked that it may write, and wrote nothing doing so.
    let scanned = client.scan().table_name(&table).send().await.unwrap();
    assert_eq!(scanned.count(), 0, "{scanned:?}");
    let description = client
        .describe_table()
        .table_name(&table)
        .send()
        .await
        .unwrap();
    let table_description = description.table().unwrap();
    let keys: Vec<_> = table_description
        .key_schema()
        .iter()
        .map(|element| (element.attribute_name(), element.key_type().clone()))
        .collect();
    assert_eq!(
        keys,
        [
            ("pk", aws_sdk_dynamodb::types::KeyType::Hash),
            ("sk", aws_sdk_dynamodb::types::KeyType::Range)
        ]
    );
    assert_ttl_is_on(&client, &table).await;

    // It works, and connecting again to the existing table is fine.
    let _ = store.register(mocker("mocker", "m1")).await.unwrap();
    let again = DynamoDbPoolStore::connect(lease(), ensure).await.unwrap();
    assert_eq!(again.list().await.unwrap().items.len(), 1);
}

/// A run that created the table but stopped before turning on its TTL left
/// TTL off. The next run with `ensure_table` turns it on.
#[tokio::test]
async fn ensure_table_turns_on_a_ttl_left_off() {
    let endpoint = skip_without_dynamodb!();
    if without_credentials() {
        return;
    }
    let client = dynamodb_support::test_client(&endpoint);
    let table = dynamodb_support::fresh_table_name();
    dynamodb_support::create_table_without_ttl(&client, &table).await;

    let mut ensure = config(&table, &endpoint);
    ensure.ensure_table = true;
    let _store = DynamoDbPoolStore::connect(lease(), ensure).await.unwrap();
    assert_ttl_is_on(&client, &table).await;
}

/// Needs no dynamodb-local: nothing listens on port 1.
#[tokio::test]
async fn an_unreachable_dynamodb_is_unavailable() {
    let nowhere = Url::parse("http://127.0.0.1:1").unwrap();
    let store = DynamoDbPoolStore::with_client(
        lease(),
        config("gds-unreachable", &nowhere),
        test_client_without_retries(nowhere.as_str()),
        Arc::new(SystemClock),
    );
    assert!(matches!(
        store.list().await,
        Err(DiscoveryError::Unavailable(_))
    ));
    assert!(matches!(
        store.register(mocker("mocker", "m1")).await,
        Err(DiscoveryError::Unavailable(_))
    ));
}
