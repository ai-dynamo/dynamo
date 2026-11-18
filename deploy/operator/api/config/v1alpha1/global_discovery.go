// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package v1alpha1

// GlobalDiscoveryConfiguration holds the defaults for the DynamoPoolRelays that
// register their pools with a Global Discovery Service.
type GlobalDiscoveryConfiguration struct {
	// URL is the Global Discovery Service's root, for relays that set none.
	// +optional
	URL string `json:"url,omitempty"`

	// Region is this cluster's region, for relays that set none.
	// +optional
	Region string `json:"region,omitempty"`
}
