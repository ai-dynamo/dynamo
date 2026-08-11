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
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGroveWorkloadsReconciler_EvaluatesReadinessOnce(t *testing.T) {
	dgd := betaDGD(t, &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ComponentType: consts.ComponentTypeFrontend,
					Replicas:      ptr.To(int32(1)),
				},
			},
		},
	})
	podClique := &grovev1alpha1.PodClique{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "graph-0-frontend",
			Namespace:  "default",
			Generation: 1,
		},
		Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1},
		Status: grovev1alpha1.PodCliqueStatus{
			Replicas:           1,
			ReadyReplicas:      1,
			UpdatedReplicas:    1,
			ScheduledReplicas:  1,
			ObservedGeneration: ptr.To(int64(1)),
		},
	}

	podCliqueReads := 0
	scaleClient := &recordingGroveScaleClient{}
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd, podClique).
		WithStatusSubresource(dgd, podClique).
		WithInterceptorFuncs(interceptor.Funcs{
			Get: func(
				ctx context.Context,
				reader client.WithWatch,
				key client.ObjectKey,
				object client.Object,
				options ...client.GetOption,
			) error {
				if _, ok := object.(*grovev1alpha1.PodClique); ok {
					require.Len(t, scaleClient.updates, 1, "readiness must be observed after scaling")
					podCliqueReads++
				}
				return reader.Get(ctx, key, object, options...)
			},
		}).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        kubeClient,
		Config:        &configv1alpha1.OperatorConfiguration{},
		Recorder:      events.NewFakeRecorder(10),
		RuntimeConfig: &commoncontroller.RuntimeConfig{},
		ScaleClient:   scaleClient,
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(string, string) ([]string, error) {
				return nil, nil
			},
		},
	}

	result, err := reconciler.newGroveProgram().workloads.Reconcile(
		context.Background(),
		dgd,
		nil,
		nil,
	)
	require.NoError(t, err)
	assert.Equal(t, nvidiacomv1beta1.DGDStateSuccessful, result.State)
	assert.Equal(t, 1, podCliqueReads)
}

func TestGroveServiceWaitsForReadyWorkloadBeforeDisaggregatedSetCutover(t *testing.T) {
	dgd := betaDGD(t, &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"frontend": {ComponentType: consts.ComponentTypeFrontend},
			},
		},
	})
	component := &dgd.Spec.Components[0]
	service := &corev1.Service{ObjectMeta: metav1.ObjectMeta{
		Name:      dynamo.GetDCDResourceName(dgd, component.ComponentName, ""),
		Namespace: dgd.Namespace,
	}}
	setDisaggregatedSetServiceSelector(service, "graph", "frontend", "old12345")
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(dgd, service).
		Build()
	reconciler := newGroveStableResourcesReconciler(
		kubeClient,
		record.NewFakeRecorder(10),
		&configv1alpha1.OperatorConfiguration{},
	)

	_, err := reconciler.reconcileComponentService(t.Context(), dgd, dgd, component, false, false)
	require.NoError(t, err)
	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(service), service))
	require.True(t, isDisaggregatedSetServiceSelector(service))

	_, err = reconciler.reconcileComponentService(t.Context(), dgd, dgd, component, false, true)
	require.NoError(t, err)
	require.NoError(t, kubeClient.Get(t.Context(), client.ObjectKeyFromObject(service), service))
	require.False(t, isDisaggregatedSetServiceSelector(service))
}
