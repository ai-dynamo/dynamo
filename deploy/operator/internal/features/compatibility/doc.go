/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package compatibility defines per-resource gates that preserve behavior
// across operator upgrades.
//
// Compatibility gates are evaluated per resource from durable metadata or the
// resolved Dynamo runtime version. Unlike the operator-wide gates in the parent
// features package, they must not be included in operator gate snapshots or
// namespace ownership Leases.
package compatibility
