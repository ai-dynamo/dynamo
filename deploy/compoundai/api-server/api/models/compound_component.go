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

import (
	"time"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
)

type CompoundComponent struct {
	Resource
	ClusterAssociate

	Version           string                                   `json:"version"`
	KubeNamespace     string                                   `json:"kube_namespace"`
	Description       string                                   `json:"description"`
	Manifest          *schemas.CompoundComponentManifestSchema `json:"manifest" type:"jsonb"`
	LatestInstalledAt *time.Time                               `json:"latest_installed_at"`
	LatestHeartbeatAt *time.Time                               `json:"latest_heartbeat_at"`
}

func (d *CompoundComponent) GetResourceType() schemas.ResourceType {
	return schemas.ResourceTypeCompoundAIComponent
}
