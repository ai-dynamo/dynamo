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

package controller

import (
	"context"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dra"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	resourcev1 "k8s.io/api/resource/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDGDGMSResourceClaimsReconciler_DRAValidation(t *testing.T) {
	t.Log("Define GMS configurations with and without DRA requirements")
	tests := []struct {
		name    string
		spec    v1beta1.DynamoComponentDeploymentSharedSpec
		wantErr bool
	}{
		{
			name: "intra-pod failover does not require DRA",
			spec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				Experimental: &v1beta1.ExperimentalSpec{
					Failover: &v1beta1.FailoverSpec{Mode: v1beta1.GMSModeIntraPod},
				},
			},
		},
		{
			name: "inter-pod failover requires DRA",
			spec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				Experimental: &v1beta1.ExperimentalSpec{
					Failover: &v1beta1.FailoverSpec{Mode: v1beta1.GMSModeInterPod},
				},
			},
			wantErr: true,
		},
		{
			name: "gpu memory service requires DRA",
			spec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				Experimental: &v1beta1.ExperimentalSpec{
					GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{},
				},
			},
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Build the DGD and GMS resource-claims reconciler")
			g := gomega.NewGomegaWithT(t)
			r := &DynamoGraphDeploymentReconciler{
				RuntimeConfig: &controller_common.RuntimeConfig{},
			}
			dgd := &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{tt.spec},
				},
			}

			t.Log("Reconcile GMS ResourceClaimTemplates")
			err := newDGDGMSResourceClaimsReconciler(
				newTestDGDResourceSyncer(r),
				r.RuntimeConfig.Gate,
			).Reconcile(context.Background(), dgd)
			t.Log("Verify validation matches the DRA requirement")
			if tt.wantErr {
				g.Expect(err).To(gomega.HaveOccurred())
				g.Expect(err.Error()).To(gomega.ContainSubstring("requires DRA"))
				return
			}
			g.Expect(err).NotTo(gomega.HaveOccurred())
		})
	}
}

func TestDGDGMSResourceClaimsReconciler_ToleratesNonGMSComponents(t *testing.T) {
	t.Log("Build a DGD containing only components without GMS")
	ctx := context.Background()
	s := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "frontend",
					ComponentType: v1beta1.ComponentTypeFrontend,
				},
				{
					ComponentName: "decode",
					ComponentType: v1beta1.ComponentTypeDecode,
				},
			},
		},
	}
	r := &DynamoGraphDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(s).
			WithObjects(dgd).
			Build(),
		Recorder:      events.NewFakeRecorder(100),
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{DRA: true}},
	}

	t.Log("Reconcile GMS ResourceClaimTemplates")
	if err := newDGDGMSResourceClaimsReconciler(newTestDGDResourceSyncer(r), r.RuntimeConfig.Gate).Reconcile(ctx, dgd); err != nil {
		t.Fatalf("dgdGMSResourceClaimsReconciler.Reconcile() returned error for non-GMS components: %v", err)
	}
}

func TestDGDGMSResourceClaimsReconciler_CleansStaleNonGMSResourceClaimTemplate(t *testing.T) {
	t.Log("Build a non-GMS DGD with a stale ResourceClaimTemplate")
	ctx := context.Background()
	s := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "decode",
					ComponentType: v1beta1.ComponentTypeDecode,
				},
			},
		},
	}
	templateName := "test-dgd-decode-gpu"
	rct := &resourcev1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{Name: templateName, Namespace: "default"},
	}
	cl := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(dgd, rct).
		Build()
	r := &DynamoGraphDeploymentReconciler{
		Client:        cl,
		Recorder:      events.NewFakeRecorder(100),
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{DRA: true}},
	}

	t.Log("Reconcile GMS ResourceClaimTemplates")
	if err := newDGDGMSResourceClaimsReconciler(newTestDGDResourceSyncer(r), r.RuntimeConfig.Gate).Reconcile(ctx, dgd); err != nil {
		t.Fatalf("dgdGMSResourceClaimsReconciler.Reconcile() returned error: %v", err)
	}

	t.Log("Verify the stale template was deleted")
	got := &resourcev1.ResourceClaimTemplate{}
	err := cl.Get(ctx, client.ObjectKey{Name: templateName, Namespace: "default"}, got)
	if !apierrors.IsNotFound(err) {
		t.Fatalf("expected stale ResourceClaimTemplate to be deleted, got %v", err)
	}
}

func TestDGDGMSResourceClaimsReconciler_DoesNotDeleteCheckpointTemplate(t *testing.T) {
	t.Log("Build a DGD with a checkpoint-owned GMS ResourceClaimTemplate")
	ctx := context.Background()
	s := newDynamoGraphDeploymentControllerTestScheme(t)
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	require.NoError(t, err)

	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
					ComponentType: v1beta1.ComponentTypeWorker,
					PodTemplate: &corev1.PodTemplateSpec{
						Spec: corev1.PodSpec{
							Containers: []corev1.Container{{
								Name:  commonconsts.MainContainerName,
								Image: "checkpoint-writer:latest",
								Resources: corev1.ResourceRequirements{
									Limits: corev1.ResourceList{
										corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
									},
								},
							}},
						},
					},
					Experimental: &v1beta1.ExperimentalSpec{
						GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{},
						Checkpoint: &v1beta1.ComponentCheckpointConfig{
							Enabled: true,
							Mode:    v1beta1.CheckpointModeAuto,
							Identity: &v1beta1.DynamoCheckpointIdentity{
								Model:            identity.Model,
								BackendFramework: identity.BackendFramework,
							},
						},
					},
				},
			},
		},
	}
	existingCheckpoint := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: "checkpoint-" + hash, Namespace: "default"},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: &identity,
			Job: v1alpha1.DynamoCheckpointJobConfig{
				TargetContainerName: commonconsts.MainContainerName,
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name: commonconsts.MainContainerName,
							Resources: corev1.ResourceRequirements{
								Claims: []corev1.ResourceClaim{{Name: dra.ClaimName}},
							},
						}},
					},
				},
			},
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			IdentityHash: hash,
		},
	}
	checkpointTemplate := &resourcev1.ResourceClaimTemplate{
		ObjectMeta: metav1.ObjectMeta{
			Name:      checkpointGMSResourceClaimTemplateName(hash),
			Namespace: "default",
			OwnerReferences: []metav1.OwnerReference{
				*metav1.NewControllerRef(existingCheckpoint, v1alpha1.GroupVersion.WithKind("DynamoCheckpoint")),
			},
		},
		Spec: resourcev1.ResourceClaimTemplateSpec{
			Spec: resourcev1.ResourceClaimSpec{
				Devices: resourcev1.DeviceClaim{
					Requests: []resourcev1.DeviceRequest{{
						Name: "gpus",
						Exactly: &resourcev1.ExactDeviceRequest{
							DeviceClassName: dra.DefaultDeviceClassName,
							AllocationMode:  resourcev1.DeviceAllocationModeExactCount,
							Count:           1,
						},
					}},
				},
			},
		},
	}
	deviceClass := &resourcev1.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: dra.DefaultDeviceClassName}}
	cl := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(dgd, existingCheckpoint, checkpointTemplate, deviceClass).
		Build()
	r := &DynamoGraphDeploymentReconciler{
		Client:        cl,
		Config:        &configv1alpha1.OperatorConfiguration{},
		Recorder:      events.NewFakeRecorder(100),
		RuntimeConfig: &controller_common.RuntimeConfig{Gate: features.Gates{DRA: true}},
	}

	t.Log("Reconcile GMS ResourceClaimTemplates")
	require.NoError(t, newDGDGMSResourceClaimsReconciler(newTestDGDResourceSyncer(r), r.RuntimeConfig.Gate).Reconcile(ctx, dgd))

	t.Log("Verify the checkpoint-owned template remains unchanged")
	template := &resourcev1.ResourceClaimTemplate{}
	require.NoError(t, cl.Get(ctx, client.ObjectKey{
		Name:      checkpointGMSResourceClaimTemplateName(hash),
		Namespace: "default",
	}, template))
	require.Len(t, template.Spec.Spec.Devices.Requests, 1)
	request := template.Spec.Spec.Devices.Requests[0]
	require.NotNil(t, request.Exactly)
	assert.Equal(t, int64(1), request.Exactly.Count)
	assert.Equal(t, dra.DefaultDeviceClassName, request.Exactly.DeviceClassName)
	controllerRef := metav1.GetControllerOf(template)
	require.NotNil(t, controllerRef)
	assert.Equal(t, "DynamoCheckpoint", controllerRef.Kind)
	assert.Equal(t, existingCheckpoint.Name, controllerRef.Name)
}
