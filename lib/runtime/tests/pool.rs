// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::utils::pool::{PoolExt, PoolItem, PoolValue, ReturnHandle, Returnable};
use std::sync::{Arc, Mutex};
use tokio::sync::Notify;

pub struct IndexedPoolState<T: Returnable + Ord + Eq + PartialEq> {
    pool: Arc<Mutex<Vec<PoolValue<T>>>>,
    available: Arc<Notify>,
}

impl<T: Returnable + Ord + Eq + PartialEq> ReturnHandle<T> for IndexedPoolState<T> {
    fn return_to_pool(&self, value: PoolValue<T>) {
        let mut pool = self.pool.lock().unwrap();
        pool.push(value);
        pool.sort_by(|a, b| a.get().cmp(b.get()));
        self.available.notify_one();
    }
}

/// A pool that maintains items in sorted order
pub struct IndexedPool<T: Returnable + Ord + Eq + PartialEq> {
    state: Arc<IndexedPoolState<T>>,
    capacity: usize,
}

impl<T: Returnable + Ord + Eq + PartialEq> IndexedPool<T> {
    /// Create a new indexed pool with the given initial elements
    pub fn new(mut initial_elements: Vec<PoolValue<T>>) -> Self {
        let capacity = initial_elements.len();

        // Sort the initial elements
        initial_elements.sort_by(|a, b| a.get().cmp(b.get()));

        let state = Arc::new(IndexedPoolState {
            pool: Arc::new(Mutex::new(initial_elements)),
            available: Arc::new(Notify::new()),
        });

        Self { state, capacity }
    }

    /// Create a new pool with initial boxed elements
    pub fn new_boxed(initial_elements: Vec<Box<T>>) -> Self {
        let initial_values = initial_elements
            .into_iter()
            .map(PoolValue::from_boxed)
            .collect();
        Self::new(initial_values)
    }

    /// Create a new pool with initial direct elements
    pub fn new_direct(initial_elements: Vec<T>) -> Self {
        let initial_values = initial_elements
            .into_iter()
            .map(PoolValue::from_direct)
            .collect();
        Self::new(initial_values)
    }

    /// Get a snapshot of the current pool contents for testing
    pub async fn get_contents(&self) -> Vec<T>
    where
        T: Clone,
    {
        let pool = self.state.pool.lock().unwrap();
        pool.iter().map(|v| v.get().clone()).collect()
    }

    async fn try_acquire(&self) -> Option<PoolItem<T>> {
        let mut pool = self.state.pool.lock().unwrap();
        if pool.is_empty() {
            return None;
        }

        // Take the first (smallest) element
        let value = pool.remove(0);
        // Use the factory method instead of direct construction
        Some(self.create_pool_item(value, self.state.clone()))
    }

    async fn acquire(&self) -> PoolItem<T> {
        loop {
            if let Some(guard) = self.try_acquire().await {
                return guard;
            }
            self.state.available.notified().await;
        }
    }
}

impl<T: Returnable + Ord + Eq + PartialEq + Send + Sync + 'static> PoolExt<T> for IndexedPool<T> {}

impl<T: Returnable + Ord + Eq + PartialEq> Clone for IndexedPool<T> {
    fn clone(&self) -> Self {
        IndexedPool {
            state: self.state.clone(),
            capacity: self.capacity,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct NonResettableInt(i32);

impl Returnable for NonResettableInt {
    fn on_return(&mut self) {}
}

impl From<i32> for NonResettableInt {
    fn from(value: i32) -> Self {
        NonResettableInt(value)
    }
}

#[tokio::test]
async fn test_indexed_pool_sorting() {
    // Create an indexed pool with unsorted elements
    let initial_elements = vec![
        PoolValue::Direct(NonResettableInt::from(5)),
        PoolValue::Direct(NonResettableInt::from(3)),
        PoolValue::Direct(NonResettableInt::from(1)),
        PoolValue::Direct(NonResettableInt::from(4)),
        PoolValue::Direct(NonResettableInt::from(2)),
    ];
    let pool = IndexedPool::new(initial_elements);

    // Verify initial sorting
    let contents = pool.get_contents().await;
    assert_eq!(
        contents,
        vec![
            NonResettableInt(1),
            NonResettableInt(2),
            NonResettableInt(3),
            NonResettableInt(4),
            NonResettableInt(5)
        ]
    );

    // Acquire an item (should be the smallest)
    let mut item1 = pool.acquire().await;
    assert_eq!(*item1, NonResettableInt(1));

    // Acquire another item
    let mut item2 = pool.acquire().await;
    assert_eq!(*item2, NonResettableInt(2));

    // Modify item1 to be larger than all remaining items
    *item1 = NonResettableInt(10);

    // Return item1 to the pool - should go at the end
    drop(item1);

    // Check the order after returning
    let contents = pool.get_contents().await;
    assert_eq!(
        contents,
        vec![
            NonResettableInt(3),
            NonResettableInt(4),
            NonResettableInt(5),
            NonResettableInt(10)
        ]
    );

    // Modify item2 to be in the middle
    *item2 = NonResettableInt(4);

    // Return item2 to the pool - should be inserted in the middle
    drop(item2);

    // Check the final order
    let contents = pool.get_contents().await;
    assert_eq!(
        contents,
        vec![
            NonResettableInt(3),
            NonResettableInt(4),
            NonResettableInt(4),
            NonResettableInt(5),
            NonResettableInt(10)
        ]
    );
    // // Test returning to a different pool
    // let pool2 = IndexedPool::new(vec![PoolValue::Direct(NonResettableInt(42))]);

    // // Acquire from first pool
    // let mut item = pool.acquire().await;
    // assert_eq!(*item, NonResettableInt(3));

    // // Modify and return to second pool
    // *item = NonResettableInt(8);
    // item.return_to_different_pool(&pool2).await;

    // // Verify item is in the second pool in sorted order
    // let contents = pool2.get_contents().await;
    // assert_eq!(contents, vec![NonResettableInt(8), NonResettableInt(42)]);
}
