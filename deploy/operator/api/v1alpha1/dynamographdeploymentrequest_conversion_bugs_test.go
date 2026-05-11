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
	"reflect"
	"testing"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestBugDGDRStaleHubDeployedPhaseRequiresDGDNameMatch(t *testing.T) {
	const newDGDName = "new-deployed-dgd"

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
				Name: newDGDName,
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
	if dst.Status.DGDName != newDGDName {
		t.Fatalf("dgdName = %q, want %q", dst.Status.DGDName, newDGDName)
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
	const newDGDName = "new-info-dgd"

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
				Name: newDGDName,
			},
		},
	}

	dst := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(dst); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	if dst.Status.DGDName != newDGDName {
		t.Fatalf("dgdName = %q, want %q", dst.Status.DGDName, newDGDName)
	}
	if dst.Status.DeploymentInfo != nil {
		t.Fatalf("deploymentInfo = %#v, want nil", dst.Status.DeploymentInfo)
	}
}

func TestBugDGDRStaleAlphaDeploymentDeletedRequiresDGDNameMatch(t *testing.T) {
	const newDGDName = "new-deleted-dgd"

	src := &v1beta1.DynamoGraphDeploymentRequest{
		ObjectMeta: metav1.ObjectMeta{
			Annotations: map[string]string{
				annDGDRStatus: mustDGDRAlphaStatusAnnotation(t, DynamoGraphDeploymentRequestStatus{
					State: DGDRStateDeploymentDeleted,
					Deployment: &DeploymentStatus{
						Name:    "old-dgd",
						Created: true,
					},
				}),
			},
		},
		Status: v1beta1.DynamoGraphDeploymentRequestStatus{
			Phase:   v1beta1.DGDRPhaseReady,
			DGDName: newDGDName,
		},
	}

	dst := &DynamoGraphDeploymentRequest{}
	if err := dst.ConvertFrom(src); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	if dst.Status.State != DGDRStateReady {
		t.Fatalf("state = %q, want %q", dst.Status.State, DGDRStateReady)
	}
	if dst.Status.Deployment == nil {
		t.Fatal("deployment = nil, want minimal live deployment")
	}
	if dst.Status.Deployment.Name != newDGDName {
		t.Fatalf("deployment.name = %q, want %q", dst.Status.Deployment.Name, newDGDName)
	}
	if dst.Status.Deployment.Created {
		t.Fatal("deployment.created = true, want false")
	}
}

func TestBugDGDRProfilingConfigPreservesUnprojectableTypedKeys(t *testing.T) {
	config := []byte(`{
		"sla": {
			"ttft": "not-a-number",
			"itl": false,
			"optimizationType": "priority",
			"isl": "1024",
			"osl": null
		},
		"deployment": {
			"modelCache": {
				"pvcName": 123,
				"modelPathInPvc": "",
				"pvcMountPath": false
			}
		},
		"planner": []
	}`)
	src := &DynamoGraphDeploymentRequest{
		Spec: DynamoGraphDeploymentRequestSpec{
			ProfilingConfig: ProfilingConfigSpec{
				Config: &apiextensionsv1.JSON{Raw: config},
			},
		},
	}

	hub := &v1beta1.DynamoGraphDeploymentRequest{}
	if err := src.ConvertTo(hub); err != nil {
		t.Fatalf("ConvertTo() error = %v", err)
	}

	restored := &DynamoGraphDeploymentRequest{}
	if err := restored.ConvertFrom(hub); err != nil {
		t.Fatalf("ConvertFrom() error = %v", err)
	}

	if restored.Spec.ProfilingConfig.Config == nil {
		t.Fatal("profilingConfig.config = nil, want preserved opaque JSON")
	}
	assertDGDRJSONEqual(t, config, restored.Spec.ProfilingConfig.Config.Raw)
}

func mustDGDRHubStatusAnnotation(t *testing.T, status v1beta1.DynamoGraphDeploymentRequestStatus) string {
	t.Helper()
	data, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("marshal DGDR hub status annotation: %v", err)
	}
	return string(data)
}

func assertDGDRJSONEqual(t *testing.T, wantRaw, gotRaw []byte) {
	t.Helper()
	var want, got any
	if err := json.Unmarshal(wantRaw, &want); err != nil {
		t.Fatalf("unmarshal wanted JSON: %v", err)
	}
	if err := json.Unmarshal(gotRaw, &got); err != nil {
		t.Fatalf("unmarshal got JSON: %v", err)
	}
	if reflect.DeepEqual(want, got) {
		return
	}
	wantJSON, _ := json.Marshal(want)
	gotJSON, _ := json.Marshal(got)
	t.Fatalf("JSON mismatch:\nwant: %s\n got: %s", wantJSON, gotJSON)
}

func mustDGDRAlphaStatusAnnotation(t *testing.T, status DynamoGraphDeploymentRequestStatus) string {
	t.Helper()
	data, err := json.Marshal(status)
	if err != nil {
		t.Fatalf("marshal DGDR alpha status annotation: %v", err)
	}
	return string(data)
}
