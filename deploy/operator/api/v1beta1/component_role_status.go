// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1beta1

import corev1 "k8s.io/api/core/v1"

// ComponentRoleStatus describes the lowered target configuration of one logical
// role. It does not describe the configuration of individual running Pods.
type ComponentRoleStatus struct {
	// name identifies the logical role, such as main, leader, or worker.
	// +required
	Name string `json:"name"`

	// podSpec is a sparse projection of the lowered target PodSpec, not a complete
	// or deployable Pod specification. It currently includes containers and
	// initContainers with their name, image, command, args, and env declarations.
	// Environment names have their final binding, in last-definition order. Known
	// values are evaluated once; unavailable inputs and oversized expansions retain
	// their original value or valueFrom. Consumers must not replay these values.
	// Remaining $(...) strings require conservative handling: they may be
	// unavailable references or already evaluated escaped literals.
	// ConfigMaps, Secrets, images, and provider environments are not read. Omitted
	// fields are unavailable, not Kubernetes defaults or proof of absence.
	// The projection is preserved without Pod admission validation or defaulting.
	// +required
	// +kubebuilder:validation:Schemaless
	// +kubebuilder:validation:Type=object
	// +kubebuilder:pruning:PreserveUnknownFields
	PodSpec corev1.PodSpec `json:"podSpec"`
}
