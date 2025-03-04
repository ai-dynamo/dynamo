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

package schemasv1

import (
	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"
)

type KubePodStatusSchema struct {
	Phase     apiv1.PodPhase `json:"phase"`
	Ready     bool           `json:"ready"`
	StartTime *metav1.Time   `json:"start_time"`
	IsOld     bool           `json:"is_old"`
	IsCanary  bool           `json:"is_canary"`
	HostIp    string         `json:"host_ip"`
}

type KubePodSchema struct {
	Name             string                     `json:"name"`
	Namespace        string                     `json:"namespace"`
	Annotations      map[string]string          `json:"annotations"`
	Labels           map[string]string          `json:"labels"`
	NodeName         string                     `json:"node_name"`
	RunnerName       *string                    `json:"runner_name"`
	DeploymentTarget *DeploymentTargetSchema    `json:"deployment_target"`
	CommitId         string                     `json:"commit_id"`
	Status           KubePodStatusSchema        `json:"status"`
	RawStatus        apiv1.PodStatus            `json:"raw_status"`
	PodStatus        modelschemas.KubePodStatus `json:"pod_status"`
	Warnings         []apiv1.Event              `json:"warnings"`
}
