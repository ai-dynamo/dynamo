// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod external;
mod logical;

#[cfg(test)]
mod tests;

pub use external::{ExternalBlockAssignments, zip_assigned, zip_assigned_pending};
pub use logical::{LogicalBlockAssignmentError, LogicalBlockAssignments};
