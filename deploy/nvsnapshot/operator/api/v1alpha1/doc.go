// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Package v1alpha1 contains the v1alpha1 API surface for NVSnapshot:
// the Snapshot, SnapshotContent, and SnapshotJob CRDs and their
// shared types. See docs/proposals/0001-nvsnapshot-api/ for the
// design rationale.

// +k8s:deepcopy-gen=package
// +kubebuilder:object:generate=true
// +groupName=nvsnapshot.io

package v1alpha1
