/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package models

type ClusterAssociate struct {
	ClusterId              uint     `json:"cluster_id"`
	AssociatedClusterCache *Cluster `gorm:"foreignkey:ClusterId"`
}

type DeploymentAssociate struct {
	DeploymentId              uint        `json:"deployment_id"`
	AssociatedDeploymentCache *Deployment `gorm:"foreignkey:DeploymentId;constraint:OnDelete:CASCADE;"`
}

type DeploymentRevisionAssociate struct {
	DeploymentRevisionId              uint                `json:"deployment_revision_id"`
	AssociatedDeploymentRevisionCache *DeploymentRevision `gorm:"foreignkey:DeploymentRevisionId;constraint:OnDelete:CASCADE;"`
}

type DynamoNimVersionAssociate struct {
	DynamoNimVersionId  string `json:"dynamo_nim_version_id"`
	DynamoNimVersionTag string `json:"dynamo_nim_version_tag"`
}

type DmsAssociate struct {
	KubeRequestId    string
	KubeDeploymentId string
}

type OrganizationAssociate struct {
	OrganizationId string `json:"organization_id"` // Set via http headers
}

type CreatorAssociate struct {
	UserId string `json:"user_id"` // Set via http headers
}
