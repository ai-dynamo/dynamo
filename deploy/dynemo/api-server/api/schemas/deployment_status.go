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

package schemas

type DeploymentStatus string

const (
	DeploymentStatusUnknown             DeploymentStatus = "unknown"
	DeploymentStatusNonDeployed         DeploymentStatus = "non-deployed"
	DeploymentStatusRunning             DeploymentStatus = "running"
	DeploymentStatusUnhealthy           DeploymentStatus = "unhealthy"
	DeploymentStatusFailed              DeploymentStatus = "failed"
	DeploymentStatusDeploying           DeploymentStatus = "deploying"
	DeploymentStatusTerminating         DeploymentStatus = "terminating"
	DeploymentStatusTerminated          DeploymentStatus = "terminated"
	DeploymentStatusImageBuilding       DeploymentStatus = "image-building"
	DeploymentStatusImageBuildFailed    DeploymentStatus = "image-build-failed"
	DeploymentStatusImageBuildSucceeded DeploymentStatus = "image-build-succeeded"
)

func (d DeploymentStatus) Ptr() *DeploymentStatus {
	return &d
}
