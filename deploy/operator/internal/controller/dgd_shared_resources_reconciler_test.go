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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/onsi/gomega"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDGDSharedResourcesReconciler_ValidatesGMSResourceClaimTemplatesBeforePathway(t *testing.T) {
	t.Log("Build a DGD whose GMS configuration requires disabled DRA support")
	ctx := context.Background()
	g := gomega.NewGomegaWithT(t)
	s := newDynamoGraphDeploymentControllerTestScheme(t)
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default", UID: types.UID("dgd-uid")},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "decode",
					ComponentType: v1beta1.ComponentTypeDecode,
					Experimental: &v1beta1.ExperimentalSpec{
						GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{},
					},
				},
			},
		},
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(dgd).
		Build()
	recorder := events.NewFakeRecorder(100)
	config := &configv1alpha1.OperatorConfiguration{
		Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"},
	}
	runtimeConfig := &controller_common.RuntimeConfig{}

	t.Log("Reconcile shared resources through their explicit dependencies")
	sharedResources := newDGDSharedResourcesReconciler(
		kubeClient,
		recorder,
		config,
		runtimeConfig,
		nil,
		nil,
		nil,
		nil,
	)
	_, err := sharedResources.Reconcile(ctx, dgd)

	t.Log("Verify DRA validation fails before workload-path reconciliation")
	g.Expect(err).To(gomega.HaveOccurred())
	g.Expect(err.Error()).To(gomega.ContainSubstring("requires DRA"))
	g.Expect(err.Error()).To(gomega.ContainSubstring("explicitly disabled"))
}

func TestDGDSharedResourcesReconciler_PreservesCheckpointResultOnLaterFailure(t *testing.T) {
	t.Log("Build a ready checkpoint and a DGD that fails later EPP reconciliation")
	ctx := context.Background()
	identity := v1alpha1.DynamoCheckpointIdentity{
		Model:            "meta-llama/Llama-2-7b-hf",
		BackendFramework: "vllm",
	}
	hash, err := checkpoint.ComputeIdentityHash(identity)
	require.NoError(t, err)
	reference := "referenced-checkpoint"
	referenced := &v1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: reference, Namespace: "default"},
		Spec: v1alpha1.DynamoCheckpointSpec{
			Identity: identity,
			Job: v1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main"}}},
				},
			},
		},
		Status: v1alpha1.DynamoCheckpointStatus{
			Phase:        v1alpha1.DynamoCheckpointPhaseReady,
			IdentityHash: hash,
		},
	}
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-dgd", Namespace: "default", UID: types.UID("dgd-uid")},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{
					ComponentName: "worker",
					ComponentType: v1beta1.ComponentTypeWorker,
					Experimental: &v1beta1.ExperimentalSpec{
						Checkpoint: &v1beta1.ComponentCheckpointConfig{
							Enabled:       true,
							CheckpointRef: &reference,
						},
					},
				},
				{
					ComponentName: "epp",
					ComponentType: v1beta1.ComponentTypeEPP,
					// Invalid legacy Go-EPP config deliberately fails after checkpoint reconciliation.
					EPPConfig: &v1beta1.EPPConfig{},
				},
			},
		},
	}
	s := newDynamoGraphDeploymentControllerTestScheme(t)
	kubeClient := fake.NewClientBuilder().
		WithScheme(s).
		WithObjects(referenced).
		WithStatusSubresource(referenced).
		Build()
	config := &configv1alpha1.OperatorConfiguration{
		Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"},
	}
	runtimeConfig := &controller_common.RuntimeConfig{
		Gate: features.Gates{Checkpoint: true},
	}
	reconciler := newDGDSharedResourcesReconciler(
		kubeClient,
		events.NewFakeRecorder(10),
		config,
		runtimeConfig,
		nil,
		nil,
		nil,
		nil,
	)

	t.Log("Reconcile the ordered shared-resource sequence")
	result, err := reconciler.Reconcile(ctx, dgd)

	t.Log("Verify the later error preserves checkpoint progress")
	require.ErrorContains(t, err, "EPP configuration is required")
	require.Contains(t, result.Infos, "worker")
	assert.True(t, result.Infos["worker"].Ready)
	assert.Equal(t, reference, result.Statuses["worker"].CheckpointName)
	assert.Equal(t, hash, result.Statuses["worker"].CheckpointID)
}
