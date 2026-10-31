// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_global_view_receiver::PoolViews;

#[test]
fn pool_views_iterator_does_not_borrow_the_lookup_key() {
    let mut receiver = PoolViews::<&str, String, u64>::new();
    let relay = receiver.open("relay-a");
    receiver
        .replace_snapshot(&relay, [(String::from("blue-dgd"), 10)])
        .unwrap();

    let views;
    {
        let key = String::from("blue-dgd");
        views = receiver.pool_views(&key);
    }
    assert_eq!(views.map(|(_, value)| *value).collect::<Vec<_>>(), vec![10]);
}
