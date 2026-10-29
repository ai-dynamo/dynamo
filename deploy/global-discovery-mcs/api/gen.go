// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package api holds the registry API contract: the protobuf sources under
// proto/, the generated Go and OpenAPI code under gen/, the well-known
// Kubernetes label and annotation keys, and the derived Service name of an
// import.
package api

// make proto runs buf from the module root, where buf.yaml and buf.gen.yaml
// live, with the pinned plugins, and adds the license headers.
//go:generate make -C .. proto
