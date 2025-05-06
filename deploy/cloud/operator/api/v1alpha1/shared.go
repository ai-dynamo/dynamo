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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type BaseStatus struct {
	Version    string             `json:"version,omitempty"`
	State      string             `json:"state,omitempty"`
	Conditions []metav1.Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,1,rep,name=conditions"`
}

func (b *BaseStatus) GetConditions() []metav1.Condition {
	return b.Conditions
}

func (b *BaseStatus) SetConditions(conditions []metav1.Condition) {
	b.Conditions = conditions
}

type BaseCRD struct {
	Status *BaseStatus `json:"status,omitempty"`
}

func (b *BaseCRD) GetStatusConditions() []metav1.Condition {
	if b.Status != nil {
		return b.Status.GetConditions()
	}
	return nil
}

func (b *BaseCRD) SetStatusConditions(conditions []metav1.Condition) {
	status := b.Status
	if status == nil {
		status = &BaseStatus{}
		b.Status = status
	}
	status.Conditions = conditions
}

func (b *BaseCRD) GetVersion() string {
	if b.Status != nil {
		return b.Status.Version
	}
	return ""
}

func (b *BaseCRD) SetVersion(version string) {
	status := b.Status
	if status == nil {
		status = &BaseStatus{}
		b.Status = status
	}
	status.Version = version
}

func (b *BaseCRD) SetState(state string) {
	status := b.Status
	if status == nil {
		status = &BaseStatus{}
		b.Status = status
	}
	status.State = state
}

func (n *BaseCRD) GetHelmVersionMatrix() map[string]string {
	return nil
}
