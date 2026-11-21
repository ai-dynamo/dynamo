// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

// GlobalDiscoveryConfiguration is where the DynamoPoolRelays with
// spec.globalDiscovery register their pools. Set both fields or neither.
type GlobalDiscoveryConfiguration struct {
	// URL is the Global Discovery Service's root, such as
	// `http://global-discovery.global-router.svc:8080/`.
	// +optional
	URL string `json:"url,omitempty"`

	// Region is this cluster's region, which the pools register in.
	// +optional
	Region string `json:"region,omitempty"`
}
