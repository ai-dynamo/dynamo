/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestDCD_DivergedLegacyCarrierDoesNotOverrideSparseSpokeSave(t *testing.T) {
	savedNamespace := "saved"
	data, err := marshalDCDSpokeSpec(&DynamoComponentDeploymentSpec{
		DynamoComponentDeploymentSharedSpec: DynamoComponentDeploymentSharedSpec{
			DynamoNamespace: &savedNamespace,
		},
	}, false)
	if err != nil {
		t.Fatalf("marshal DCD spoke spec: %v", err)
	}

	hub := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "dcd",
			Namespace: "ns",
			Annotations: map[string]string{
				annDCDSpec:                        string(data),
				"nvidia.com/dcd-dynamo-namespace": "stale",
			},
		},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "dcd",
			},
		},
	}

	spoke := &DynamoComponentDeployment{}
	if err := spoke.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom: %v", err)
	}
	if spoke.Spec.DynamoNamespace == nil || *spoke.Spec.DynamoNamespace != savedNamespace {
		t.Fatalf("DynamoNamespace = %v, want %q from sparse save", spoke.Spec.DynamoNamespace, savedNamespace)
	}
}
