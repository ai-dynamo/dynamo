// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package validation implements structural admission validation after source
// schema validation and typed conversion to the storage API version.
//
// Storage-version validators own every rule whose inputs survive conversion,
// regardless of the submitted API version. Source-specific field paths affect
// diagnostics, not rule ownership. Shared rules must not be disabled for Alpha
// requests or copied into Alpha validators.
//
// Compatibility validators supplement that traversal only for source-only
// information, lossy conversion, and source-only warnings. Their Go docs must
// explain the information conversion cannot preserve. They must not compare
// whole compatibility objects or receive full storage-version objects to
// repeat shared rules.
//
// Conversion does not execute the destination API version's schema or CEL.
// Missing source-schema coverage is still owned by shared storage-version Go
// validation when converted values contain the inputs required by the rule.
package validation
