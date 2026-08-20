/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Package v1beta2 contains API Schema definitions for the nvidia.com v1beta2 API group.
// +kubebuilder:object:generate=true
// +groupName=nvidia.com
package v1beta2

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/scheme"
)

var (
	// GroupVersion is the group version used to register these objects.
	GroupVersion = schema.GroupVersion{Group: "nvidia.com", Version: "v1beta2"}

	// DynamoGraphDeploymentRequestGVK is the v1beta2 DynamoGraphDeploymentRequest kind.
	DynamoGraphDeploymentRequestGVK = GroupVersion.WithKind("DynamoGraphDeploymentRequest")

	// DynamoGraphDeploymentRunGVK is the v1beta2 DynamoGraphDeploymentRun kind.
	DynamoGraphDeploymentRunGVK = GroupVersion.WithKind("DynamoGraphDeploymentRun")

	// DynamoGraphDeploymentCandidateGVK is the v1beta2 DynamoGraphDeploymentCandidate kind.
	DynamoGraphDeploymentCandidateGVK = GroupVersion.WithKind("DynamoGraphDeploymentCandidate")

	// SchemeBuilder is used to add Go types to the GroupVersionKind scheme.
	SchemeBuilder = &scheme.Builder{GroupVersion: GroupVersion}

	// AddToScheme adds the types in this group-version to the given scheme.
	AddToScheme = SchemeBuilder.AddToScheme
)
