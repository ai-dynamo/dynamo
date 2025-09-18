// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::Mutex;
use tokio::sync::Notify;

/// Trait for items that can be returned to a pool
pub trait Returnable: Send + Sync + 'static {
    /// Called when an item is returned to the pool
    fn on_return(&mut self) {}
}

pub trait ReturnHandle<T: Returnable>: Send + Sync + 'static {
    fn return_to_pool(&self, value: PoolValue<T>);
}

/// Enum to hold either a `Box<T>` or `T` directly
pub enum PoolValue<T: Returnable> {
    Boxed(Box<T>),
    Direct(T),
}

impl<T: Returnable> PoolValue<T> {
    /// Create a new PoolValue from a boxed item
    pub fn from_boxed(value: Box<T>) -> Self {
        PoolValue::Boxed(value)
    }

    /// Create a new PoolValue from a direct item
    pub fn from_direct(value: T) -> Self {
        PoolValue::Direct(value)
    }

    /// Get a reference to the underlying item
    pub fn get(&self) -> &T {
        match self {
            PoolValue::Boxed(boxed) => boxed.as_ref(),
            PoolValue::Direct(direct) => direct,
        }
    }

    /// Get a mutable reference to the underlying item
    pub fn get_mut(&mut self) -> &mut T {
        match self {
            PoolValue::Boxed(boxed) => boxed.as_mut(),
            PoolValue::Direct(direct) => direct,
        }
    }

    /// Call on_return on the underlying item
    pub fn on_return(&mut self) {
        self.get_mut().on_return();
    }
}

impl<T: Returnable> Deref for PoolValue<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.get()
    }
}

impl<T: Returnable> DerefMut for PoolValue<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.get_mut()
    }
}

// Private module to restrict access to PoolItem constructor
mod private {
    // This type can only be constructed within this module
    #[derive(Clone, Copy)]
    pub struct PoolItemToken(());

    impl PoolItemToken {
        pub(super) fn new() -> Self {
            PoolItemToken(())
        }
    }
}

/// Core trait defining pool operations
pub trait PoolExt<T: Returnable>: Send + Sync + 'static {
    /// Create a new PoolItem (only available to implementors)
    fn create_pool_item(
        &self,
        value: PoolValue<T>,
        handle: Arc<dyn ReturnHandle<T>>,
    ) -> PoolItem<T> {
        PoolItem::new(value, handle)
    }
}

/// An item borrowed from a pool
pub struct PoolItem<T: Returnable> {
    value: Option<PoolValue<T>>,
    handle: Arc<dyn ReturnHandle<T>>,
    _token: private::PoolItemToken,
}

impl<T: Returnable> PoolItem<T> {
    /// Create a new PoolItem (only available within this module)
    fn new(value: PoolValue<T>, handle: Arc<dyn ReturnHandle<T>>) -> Self {
        Self {
            value: Some(value),
            handle,
            _token: private::PoolItemToken::new(),
        }
    }

    /// Convert this unique PoolItem into a shared reference
    pub fn into_shared(self) -> SharedPoolItem<T> {
        SharedPoolItem {
            inner: Arc::new(self),
        }
    }

    /// Check if this item still contains a value
    pub fn has_value(&self) -> bool {
        self.value.is_some()
    }
}

impl<T: Returnable> Deref for PoolItem<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().unwrap().get()
    }
}

impl<T: Returnable> DerefMut for PoolItem<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().unwrap().get_mut()
    }
}

impl<T: Returnable> Drop for PoolItem<T> {
    fn drop(&mut self) {
        if let Some(mut value) = self.value.take() {
            value.on_return();
            // Use blocking version for drop
            self.handle.return_to_pool(value);
        }
    }
}

/// A shared reference to a pooled item
pub struct SharedPoolItem<T: Returnable> {
    inner: Arc<PoolItem<T>>,
}

impl<T: Returnable> Clone for SharedPoolItem<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<T: Returnable> SharedPoolItem<T> {
    /// Get a reference to the underlying item
    pub fn get(&self) -> &T {
        self.inner.value.as_ref().unwrap().get()
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.inner)
    }
}

impl<T: Returnable> Deref for SharedPoolItem<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.inner.value.as_ref().unwrap().get()
    }
}

/// Standard pool implementation
pub struct Pool<T: Returnable> {
    state: Arc<PoolState<T>>,
    capacity: usize,
}

struct PoolState<T: Returnable> {
    pool: Arc<Mutex<VecDeque<PoolValue<T>>>>,
    available: Arc<Notify>,
}

impl<T: Returnable> ReturnHandle<T> for PoolState<T> {
    fn return_to_pool(&self, value: PoolValue<T>) {
        let mut pool = self.pool.lock().unwrap();
        pool.push_back(value);
        self.available.notify_one();
    }
}

impl<T: Returnable> Pool<T> {
    /// Create a new pool with the given initial elements
    pub fn new(initial_elements: Vec<PoolValue<T>>) -> Self {
        let capacity = initial_elements.len();
        let pool = initial_elements
            .into_iter()
            .collect::<VecDeque<PoolValue<T>>>();

        let state = Arc::new(PoolState {
            pool: Arc::new(Mutex::new(pool)),
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

    async fn try_acquire(&self) -> Option<PoolItem<T>> {
        let mut pool = self.state.pool.lock().unwrap();
        pool.pop_front()
            .map(|value| PoolItem::new(value, self.state.clone()))
    }

    async fn acquire(&self) -> PoolItem<T> {
        loop {
            if let Some(guard) = self.try_acquire().await {
                return guard;
            }
            self.state.available.notified().await;
        }
    }

    fn notify_return(&self) {
        self.state.available.notify_one();
    }

    fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T: Returnable> PoolExt<T> for Pool<T> {}

impl<T: Returnable> Clone for Pool<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            capacity: self.capacity,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::time::{Duration, timeout};

    // Implement Returnable for u32 just for testing
    impl Returnable for u32 {
        fn on_return(&mut self) {
            *self = 0;
            tracing::debug!("Resetting u32 to 0");
        }
    }

    #[tokio::test]
    async fn test_acquire_release() {
        let initial_elements = vec![
            PoolValue::Direct(1),
            PoolValue::Direct(2),
            PoolValue::Direct(3),
            PoolValue::Direct(4),
            PoolValue::Direct(5),
        ];
        let pool = Pool::new(initial_elements);

        // Acquire an element from the pool
        if let Some(mut item) = pool.try_acquire().await {
            assert_eq!(*item, 1); // It should be the first element we put in

            // Modify the value
            *item += 10;
            assert_eq!(*item, 11);

            // The item will be dropped at the end of this scope,
            // and the value will be returned to the pool
        }

        // Acquire all remaining elements and the one we returned
        let mut values = Vec::new();
        let mut items = Vec::new();
        while let Some(item) = pool.try_acquire().await {
            values.push(*item);
            items.push(item);
        }

        // The last element in `values` should be the one we returned, and it should be on_return to 0
        assert_eq!(values, vec![2, 3, 4, 5, 0]);

        // Test the awaitable acquire
        let pool_clone = pool.clone();
        let task = tokio::spawn(async move {
            let first_acquired = pool_clone.acquire().await;
            assert_eq!(*first_acquired, 0);
        });

        timeout(Duration::from_secs(1), task)
            .await
            .expect_err("Expected timeout");

        // Drop the guards to return the PoolItems to the pool.
        items.clear();

        let pool_clone = pool.clone();
        let task = tokio::spawn(async move {
            let first_acquired = pool_clone.acquire().await;
            assert_eq!(*first_acquired, 0);
        });

        // Now the task should be able to finish.
        timeout(Duration::from_secs(1), task)
            .await
            .expect("Task did not complete in time")
            .unwrap();
    }

    #[tokio::test]
    async fn test_shared_items() {
        let initial_elements = vec![
            PoolValue::Direct(1),
            // PoolValue::Direct(2),
            // PoolValue::Direct(3),
        ];
        let pool = Pool::new(initial_elements);

        // Acquire and convert to shared
        let mut item = pool.acquire().await;
        *item += 10; // Modify before sharing
        let shared = item.into_shared();
        assert_eq!(*shared, 11);

        // Create a clone of the shared item
        let shared_clone = shared.clone();
        assert_eq!(*shared_clone, 11);

        // Drop the original shared item
        drop(shared);

        // Clone should still be valid
        assert_eq!(*shared_clone, 11);

        // Drop the clone
        drop(shared_clone);

        // Now we should be able to acquire the item again
        let item = pool.acquire().await;
        assert_eq!(*item, 0); // Value should be on_return
    }

    #[tokio::test]
    async fn test_boxed_values() {
        let initial_elements = vec![
            PoolValue::Boxed(Box::new(1)),
            // PoolValue::Boxed(Box::new(2)),
            // PoolValue::Boxed(Box::new(3)),
        ];
        let pool = Pool::new(initial_elements);

        // Acquire an element from the pool
        let mut item = pool.acquire().await;
        assert_eq!(*item, 1);

        // Modify and return to pool
        *item += 10;
        drop(item);

        // Should get on_return value when acquired again
        let item = pool.acquire().await;
        assert_eq!(*item, 0);
    }

    #[tokio::test]
    async fn test_pool_item_creation() {
        let pool = Pool::new(vec![PoolValue::Direct(1)]);

        // This works - acquiring from the pool
        let item = pool.acquire().await;
        assert_eq!(*item, 1);

        // This would not compile - can't create PoolItem directly
        // let invalid_item = PoolItem {
        //     value: Some(PoolValue::Direct(2)),
        //     pool: pool.clone(),
        //     _token: /* can't create this */
        // };
    }
}
