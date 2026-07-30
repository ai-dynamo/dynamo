/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package compatibility defines per-resource gates that preserve behavior
// across operator upgrades.
//
// A compatibility gate combines a hard runtime capability requirement with an
// origin-version constraint that controls whether a feature becomes the default
// for a resource. An explicit opt-in may bypass the origin constraint, but never
// the runtime requirement.
//
// Unlike the operator-wide gates in the parent features package, compatibility
// gates must not be included in operator gate snapshots or namespace ownership
// Leases.
package compatibility
