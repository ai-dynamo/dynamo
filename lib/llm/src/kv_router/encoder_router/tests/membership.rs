// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn committed_encoder_admission_survives_membership_changes_and_endpoint_reuse() {
    if isolated(concat!(
        module_path!(),
        "::committed_encoder_admission_survives_membership_changes_and_endpoint_reuse"
    ))
    .await
    {
        return;
    }
    let mut fixture = Fixture::new().await;
    let router = fixture.router.clone();
    let endpoint = fixture.endpoint.clone();
    let ids = fixture.ids.clone();
    let group_id = endpoint.id().to_string();
    assert_eq!(
        router.available_worker_ids_for(&group_id, &endpoint.id()),
        Some(HashSet::from([ids[0]]))
    );
    assert!(
        router
            .available_worker_ids_for("other-group", &endpoint.id())
            .is_none()
    );
    let mut original = multimodal_request();
    let consumer_routing = json!({
        "backend_instance_id": 11,
        "prefill_worker_id": 12,
        "decode_worker_id": 13,
        "dp_rank": 1,
        "prefill_dp_rank": 2,
        "lora_name": "adapter",
        "cache_salt": "tenant",
        "priority": 3,
    });
    original.routing = Some(serde_json::from_value(consumer_routing.clone()).unwrap());
    let downstream = Arc::new(CaptureEngine::default());
    let response = router
        .generate(SingleIn::new(original), downstream.clone())
        .await
        .unwrap();
    let _: Vec<_> = response.collect().await;
    {
        let captured = downstream.request.lock().unwrap();
        let captured = captured.as_ref().unwrap();
        assert_eq!(
            serde_json::to_value(&captured.routing).unwrap(),
            consumer_routing
        );
        assert_eq!(
            captured.encoder_result.as_ref().unwrap()["routing"],
            json!({"lora_name": "adapter", "cache_salt": "tenant", "priority": 3})
        );
    }
    for _ in 0..6 {
        assert_eq!(encoded_worker(&router).await, Some(ids[0]));
    }

    fixture.admissions.send_replace(vec![ids[0], ids[1]]);
    tokio::time::timeout(Duration::from_secs(5), async {
        let mut seen = HashSet::new();
        while seen.len() < 2 {
            let worker = encoded_worker(&router).await.unwrap();
            assert!(
                ids[..2].contains(&worker),
                "rejected encoder received a request"
            );
            seen.insert(worker);
        }
    })
    .await
    .expect("compatible members must become reachable through the admission channel");

    let retired = router.binding.load_full().unwrap();
    fixture.admissions.send_replace(Vec::new());
    router.set_target(None);
    assert!(
        router
            .available_worker_ids_for(&group_id, &endpoint.id())
            .is_none()
    );
    let (successor_admissions, successor_ids) = watch::channel(vec![ids[2]]);
    let previous = std::mem::replace(&mut fixture.admissions, successor_admissions);
    drop(previous);
    router.set_target(Some(fixture.target(2, successor_ids)));
    assert_eq!(
        router.available_worker_ids_for(&group_id, &endpoint.id()),
        Some(HashSet::new())
    );
    assert!(
        router
            .available_worker_ids_for("other-group", &endpoint.id())
            .is_none()
    );
    fixture.wait_ready(2).await;
    assert_eq!(
        router.available_worker_ids_for(&group_id, &endpoint.id()),
        Some(HashSet::from([ids[2]]))
    );
    for _ in 0..6 {
        assert_eq!(encoded_worker(&router).await, Some(ids[2]));
    }
    assert!(
        tokio::time::timeout(
            Duration::from_secs(5),
            retired
                .router
                .generate(SingleIn::new(multimodal_request()),)
        )
        .await
        .expect("retired admission must fail promptly")
        .is_err()
    );

    drop(router);
    fixture.shutdown().await;
}
