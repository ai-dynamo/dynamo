// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::{Condvar, Mutex};
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

pub struct SyncPool<T: Returnable> {
    state: Arc<SyncPoolState<T>>,
    capacity: usize,
}

struct SyncPoolState<T: Returnable> {
    pool: Mutex<VecDeque<PoolValue<T>>>,
    available: Condvar,
}

impl<T: Returnable> SyncPool<T> {
    pub fn new(initial_elements: Vec<PoolValue<T>>) -> Self {
        let capacity = initial_elements.len();
        let pool = initial_elements
            .into_iter()
            .collect::<VecDeque<PoolValue<T>>>();

        let state = Arc::new(SyncPoolState {
            pool: Mutex::new(pool),
            available: Condvar::new(),
        });

        Self { state, capacity }
    }

    pub fn new_direct(initial_elements: Vec<T>) -> Self {
        let initial_values = initial_elements
            .into_iter()
            .map(PoolValue::from_direct)
            .collect();
        Self::new(initial_values)
    }

    pub fn try_acquire(&self) -> Option<SyncPoolItem<T>> {
        let mut pool = self.state.pool.lock().unwrap();
        pool.pop_front()
            .map(|value| SyncPoolItem::new(value, self.state.clone()))
    }

    pub fn acquire_blocking(&self) -> SyncPoolItem<T> {
        let mut pool = self.state.pool.lock().unwrap();

        while pool.is_empty() {
            tracing::debug!("SyncPool: waiting for available resource (pool empty)");
            pool = self.state.available.wait(pool).unwrap();
            tracing::debug!(
                "SyncPool: woke up, checking pool again (size: {})",
                pool.len()
            );
        }

        let value = pool.pop_front().unwrap();
        tracing::debug!("SyncPool: acquired resource, pool size now: {}", pool.len());
        SyncPoolItem::new(value, self.state.clone())
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

impl<T: Returnable> Clone for SyncPool<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
            capacity: self.capacity,
        }
    }
}

pub struct SyncPoolItem<T: Returnable> {
    value: Option<PoolValue<T>>,
    state: Arc<SyncPoolState<T>>,
}

impl<T: Returnable> SyncPoolItem<T> {
    fn new(value: PoolValue<T>, state: Arc<SyncPoolState<T>>) -> Self {
        Self {
            value: Some(value),
            state,
        }
    }
}

impl<T: Returnable> Deref for SyncPoolItem<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value.as_ref().unwrap().get()
    }
}

impl<T: Returnable> DerefMut for SyncPoolItem<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.value.as_mut().unwrap().get_mut()
    }
}

impl<T: Returnable> Drop for SyncPoolItem<T> {
    fn drop(&mut self) {
        if let Some(mut value) = self.value.take() {
            value.on_return();

            let mut pool = self.state.pool.lock().unwrap();
            pool.push_back(value);
            tracing::debug!(
                "SyncPool: returned resource, pool size now: {}, notifying waiters",
                pool.len()
            );

            self.state.available.notify_one();
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::thread;
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

    #[test]
    fn test_sync_pool_basic_acquire_release() {
        let initial_elements = vec![1u32, 2, 3];
        let pool = SyncPool::new_direct(initial_elements);

        // Try acquire (non-blocking)
        let item1 = pool.try_acquire().unwrap();
        assert_eq!(*item1, 1);

        let item2 = pool.try_acquire().unwrap();
        assert_eq!(*item2, 2);

        // Pool should have one item left
        let item3 = pool.try_acquire().unwrap();
        assert_eq!(*item3, 3);

        // Pool should be empty now
        assert!(pool.try_acquire().is_none());

        // Drop items to return to pool
        drop(item1); // Returns 0 (after on_return)
        drop(item2); // Returns 0 (after on_return)
        drop(item3); // Returns 0 (after on_return)

        // Should be able to acquire again
        let item = pool.try_acquire().unwrap();
        assert_eq!(*item, 0); // Value was reset by on_return
    }

    #[test]
    fn test_sync_pool_blocking_acquire() {
        let pool = SyncPool::new_direct(vec![42u32]);

        // Acquire the only item
        let item = pool.acquire_blocking();
        assert_eq!(*item, 42);

        let pool_clone = pool.clone();
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        // Spawn a thread that will wait for the item
        let handle = thread::spawn(move || {
            counter_clone.store(1, Ordering::SeqCst); // Mark that we're waiting
            let waiting_item = pool_clone.acquire_blocking(); // This will block
            counter_clone.store(2, Ordering::SeqCst); // Mark that we got it
            assert_eq!(*waiting_item, 0); // Should be reset value
        });

        // Give the thread time to start waiting
        thread::sleep(Duration::from_millis(10));
        assert_eq!(counter.load(Ordering::SeqCst), 1); // Should be waiting

        // Drop the item to trigger condvar notification
        drop(item);

        // Wait for the other thread to complete
        handle.join().unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2); // Should have completed
    }

    #[test]
    fn test_sync_pool_multiple_waiters() {
        let pool = SyncPool::new_direct(vec![1u32]);

        // Acquire the only item
        let item = pool.acquire_blocking();

        let pool_clone1 = pool.clone();
        let pool_clone2 = pool.clone();
        let completed = Arc::new(AtomicUsize::new(0));
        let completed1 = completed.clone();
        let completed2 = completed.clone();

        // Spawn two threads that will wait
        let handle1 = thread::spawn(move || {
            let _item = pool_clone1.acquire_blocking(); // Will block
            completed1.fetch_add(1, Ordering::SeqCst); // Mark completion
            // Item drops here, potentially waking thread 2
        });

        let handle2 = thread::spawn(move || {
            let _item = pool_clone2.acquire_blocking(); // Will block
            completed2.fetch_add(1, Ordering::SeqCst); // Mark completion
            // Item drops here
        });

        // Give threads time to start waiting
        thread::sleep(Duration::from_millis(50));
        assert_eq!(completed.load(Ordering::SeqCst), 0); // Both should be waiting

        // Drop the item - should wake exactly one thread
        drop(item);

        // Wait for both threads to complete
        handle1.join().unwrap();
        handle2.join().unwrap();

        // Both threads should have completed eventually
        assert_eq!(completed.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_sync_vs_async_pool_compatibility() {
        // Test that both pool types work with the same Returnable type
        let async_pool = Pool::new_direct(vec![1u32, 2u32]);
        let sync_pool = SyncPool::new_direct(vec![3u32, 4u32]);

        // Both should work
        let async_rt = tokio::runtime::Runtime::new().unwrap();
        let async_item = async_rt.block_on(async { async_pool.acquire().await });
        assert_eq!(*async_item, 1);

        let sync_item = sync_pool.acquire_blocking();
        assert_eq!(*sync_item, 3);

        // Both use the same Returnable trait
        drop(async_item); // Should reset to 0
        drop(sync_item); // Should reset to 0
    }

    #[test]
    fn test_sync_pool_condvar_performance() {
        let pool = SyncPool::new_direct((0..10).collect::<Vec<u32>>());
        let start = std::time::Instant::now();

        // Rapid acquire/release cycles
        for _ in 0..1000 {
            let item = pool.acquire_blocking();
            // Simulate minimal work
            let _ = *item + 1;
            drop(item); // Return to pool
        }

        let duration = start.elapsed();
        println!("1000 sync pool operations took {:?}", duration);

        // Should be fast (< 10ms on most systems)
        assert!(duration < Duration::from_millis(50));
    }
}
