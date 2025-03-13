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

package conversion

import (
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/common"
	"github.com/ai-dynamo/dynamo/deploy/dynamo/operator/api/dynamo/modelschemas"
)

func ConvertToDeploymentTargetResourceItem(src *common.ResourceItem) (dest *modelschemas.DeploymentTargetResourceItem) {
	if src == nil {
		return
	}
	dest = &modelschemas.DeploymentTargetResourceItem{
		CPU:    src.CPU,
		Memory: src.Memory,
		GPU:    src.GPU,
		Custom: src.Custom,
	}
	return
}

func ConvertToDeploymentTargetResources(src *common.Resources) (dest *modelschemas.DeploymentTargetResources) {
	if src == nil {
		return
	}
	dest = &modelschemas.DeploymentTargetResources{
		Requests: ConvertToDeploymentTargetResourceItem(src.Requests),
		Limits:   ConvertToDeploymentTargetResourceItem(src.Limits),
	}
	return
}

func ConvertFromDeploymentTargetResourceItem(src *modelschemas.DeploymentTargetResourceItem) (dest *common.ResourceItem) {
	if src == nil {
		return
	}
	dest = &common.ResourceItem{
		CPU:    src.CPU,
		Memory: src.Memory,
		GPU:    src.GPU,
		Custom: src.Custom,
	}
	return
}

func ConvertFromDeploymentTargetResources(src *modelschemas.DeploymentTargetResources) (dest *common.Resources) {
	if src == nil {
		return
	}
	dest = &common.Resources{
		Requests: ConvertFromDeploymentTargetResourceItem(src.Requests),
		Limits:   ConvertFromDeploymentTargetResourceItem(src.Limits),
	}
	return
}
