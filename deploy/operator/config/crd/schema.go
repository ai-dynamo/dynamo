/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package crd exposes the checked-in CRD definitions to Go code that needs
// the same structural schema used by the Kubernetes API server.
package crd

import _ "embed"

// dynamoGraphDeploymentDefinition is generated from the API types by the
// operator's `make manifests` target.
//
//go:embed bases/nvidia.com_dynamographdeployments.yaml
var dynamoGraphDeploymentDefinition string

// DynamoGraphDeploymentDefinition returns the generated DGD CRD definition.
func DynamoGraphDeploymentDefinition() string {
	return dynamoGraphDeploymentDefinition
}
