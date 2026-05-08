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

package v1alpha1

import (
	"encoding/json"
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestBugDGDRStaleHubDeployedPhaseRequiresDGDNameMatch(t *testing.T) {
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annDGDRStatus: mustDGDRHubStatusAnnotation(t, v1beta1.DynamoGraphDeploymentRequestStatus{
					Phase:   v1beta1.DGDRPhaseDeployed,
					DGDName: "old-dgd",
				}),
			},
		},
		Status: DynamoGraphDeploymentRequestStatus{
			State: DGDRStateReady,
			Deployment: &DeploymentStatus{
				Name: "new-dgd",
			},
		},
	}

	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	if dst.Status.Phase != v1beta1.DGDRPhaseReady {
		t.Fatalf("phase = %q, want %q", dst.Status.Phase, v1beta1.DGDRPhaseReady)
	}
	if dst.Status.DGDName != "new-dgd" {
		t.Fatalf("dgdName = %q, want %q", dst.Status.DGDName, "new-dgd")
	}
}

func TestBugDGDRStaleHubProfilingSubstatusRequiresProfilingPhase(t *testing.T) {
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annDGDRStatus: mustDGDRHubStatusAnnotation(t, v1beta1.DynamoGraphDeploymentRequestStatus{
					ProfilingPhase:   v1beta1.ProfilingPhaseSweepingDecode,
					ProfilingJobName: "old-profiling-job",
				}),
			},
		},
		Status: DynamoGraphDeploymentRequestStatus{
			State: DGDRStateReady,
		},
	}

	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	if dst.Status.Phase != v1beta1.DGDRPhaseReady {
		t.Fatalf("phase = %q, want %q", dst.Status.Phase, v1beta1.DGDRPhaseReady)
	}
	if dst.Status.ProfilingPhase != "" {
		t.Fatalf("profilingPhase = %q, want empty", dst.Status.ProfilingPhase)
	}
	if dst.Status.ProfilingJobName != "" {
		t.Fatalf("profilingJobName = %q, want empty", dst.Status.ProfilingJobName)
	}
}

func TestBugDGDRStaleHubDeploymentInfoRequiresDGDNameMatch(t *testing.T) {
	replicas := int32(3)
	availableReplicas := int32(2)
	src := &DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annDGDRStatus: mustDGDRHubStatusAnnotation(t, v1beta1.DynamoGraphDeploymentRequestStatus{
					DGDName: "old-dgd",
					DeploymentInfo: &v1beta1.DeploymentInfoStatus{
						Replicas:          &replicas,
						AvailableReplicas: &availableReplicas,
					},
				}),
			},
		},
		Status: DynamoGraphDeploymentRequestStatus{
			State: DGDRStateReady,
			Deployment: &DeploymentStatus{
				Name: "new-dgd",
			},
		},
	}

	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	if dst.Status.DGDName != "new-dgd" {
		t.Fatalf("dgdName = %q, want %q", dst.Status.DGDName, "new-dgd")
	}
	if dst.Status.DeploymentInfo != nil {
		t.Fatalf("deploymentInfo = %#v, want nil", dst.Status.DeploymentInfo)
	}
}

func mustDGDRHubStatusAnnotation(t *testing.T, status v1beta1.DynamoGraphDeploymentRequestStatus) string {
	t.Helper()
	data, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("marshal DGDR hub status annotation: %v", err)
	}
	return string(data)
}
