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
	"errors"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
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
		false,
		nil,
	)
	require.NoError(t, err)
	assert.Equal(t, nvidiacomv1beta1.DGDStateSuccessful, result.State)
	assert.Equal(t, 1, podCliqueReads)
}

func TestGroveWorkloadsReconciler_ReconcilePodCliqueSetRejectsStaleObservation(t *testing.T) {
	dgd := betaDGD(t, &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
	})
	existing := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{
			Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{{Name: "old"}},
		}},
	}
	updateCalls := 0
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithObjects(existing).
		WithInterceptorFuncs(interceptor.Funcs{
			Update: func(
				_ context.Context,
				_ client.WithWatch,
				object client.Object,
				_ ...client.UpdateOption,
			) error {
				updateCalls++
				assert.Equal(t, existing.ResourceVersion, object.GetResourceVersion())
				return apierrors.NewConflict(
					schema.GroupResource{Group: "grove.io", Resource: "podcliquesets"},
					object.GetName(),
					errors.New("stale PodCliqueSet"),
				)
			},
		}).
		Build()
	observed := &grovev1alpha1.PodCliqueSet{}
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(existing), observed))
	desired := observed.DeepCopy()
	desired.Spec.Template.Cliques[0].Name = "new"
	reconciler := &groveWorkloadsReconciler{syncer: newDGDResourceSyncer(kubeClient, nil)}

	_, err := reconciler.reconcilePodCliqueSet(context.Background(), dgd, &grovePodCliqueSetRender{
		existing: observed,
		desired:  desired,
	})

	require.Error(t, err)
	assert.True(t, apierrors.IsConflict(err))
	assert.Equal(t, 1, updateCalls)
}

func TestGroveWorkloadsReconciler_ReconcilePodCliqueSetReturnsCreateConflict(t *testing.T) {
	dgd := betaDGD(t, &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", UID: "dgd-uid"},
	})
	desired := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
	}
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithInterceptorFuncs(interceptor.Funcs{
			Create: func(
				_ context.Context,
				_ client.WithWatch,
				object client.Object,
				_ ...client.CreateOption,
			) error {
				return apierrors.NewAlreadyExists(
					schema.GroupResource{Group: "grove.io", Resource: "podcliquesets"},
					object.GetName(),
				)
			},
		}).
		Build()
	reconciler := &groveWorkloadsReconciler{syncer: newDGDResourceSyncer(kubeClient, nil)}

	_, err := reconciler.reconcilePodCliqueSet(context.Background(), dgd, &grovePodCliqueSetRender{desired: desired})

	require.Error(t, err)
	assert.True(t, apierrors.IsAlreadyExists(err))
}
