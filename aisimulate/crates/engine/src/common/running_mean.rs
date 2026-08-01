// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::ops::{Add, Div, Sub};

/// A generic running mean calculator with a fixed-size sliding window.
/// Maintains a running sum and count to compute the mean in O(1) time.
#[derive(Debug, Clone)]
pub struct RunningMean<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Default + From<u16>,
{
    max_size: u16,
    sum: T,
    values: VecDeque<T>,
}

impl<T> RunningMean<T>
where
    T: Copy + Add<Output = T> + Sub<Output = T> + Div<Output = T> + Default + From<u16>,
{
    pub fn new(max_size: u16) -> Self {
        Self {
            max_size,
            sum: T::default(),
            values: VecDeque::with_capacity(max_size as usize),
        }
    }

    pub fn push(&mut self, value: T) {
        // If at capacity, remove the oldest value from sum
        if self.values.len() >= self.max_size as usize
            && let Some(old_value) = self.values.pop_front()
        {
            self.sum = self.sum - old_value;
        }

        // Add new value
        self.sum = self.sum + value;
        self.values.push_back(value);
    }

    pub fn mean(&self) -> T {
        if self.values.is_empty() {
            T::default()
        } else {
            self.sum / T::from(self.values.len() as u16)
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Clear all values from the window.
    pub fn clear(&mut self) {
        self.sum = T::default();
        self.values.clear();
    }
}
