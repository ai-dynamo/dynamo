/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package runtime defines explicit feature gates controlled by the Dynamo
// runtime compatibility version.
//
// Runtime gates make rendered defaults stable across operator upgrades. A
// runtime version change is the trigger for adopting new defaults, while
// explicit user configuration remains authoritative.
package runtime
