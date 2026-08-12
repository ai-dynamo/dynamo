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

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/checkpoint"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGroveScaler_ReconcileTargetsExpectedGroveChildren(t *testing.T) {
	dgd := betaDGD(t, &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"frontend": {
					ComponentType: consts.ComponentTypeFrontend,
					Replicas:      ptr.To(int32(2)),
				},
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(3)),
					Multinode:     &nvidiacomv1alpha1.MultinodeSpec{NodeCount: 2},
				},
				"gated": {
					ComponentType: consts.ComponentTypeWorker,
				},
				"defaulted": {
					ComponentType: consts.ComponentTypeWorker,
				},
			},
		},
	})
	frontend := &grovev1alpha1.PodClique{ObjectMeta: metav1.ObjectMeta{Name: "graph-0-frontend", Namespace: "default"}, Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1}}
	worker := &grovev1alpha1.PodCliqueScalingGroup{ObjectMeta: metav1.ObjectMeta{Name: "graph-0-worker", Namespace: "default"}, Spec: grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: 1}}
	gated := &grovev1alpha1.PodClique{ObjectMeta: metav1.ObjectMeta{Name: "graph-0-gated", Namespace: "default"}, Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1}}
	kubeClient := fake.NewClientBuilder().
		WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
		WithRESTMapper(groveScaleRESTMapper()).
		WithObjects(frontend, worker, gated).
		WithInterceptorFuncs(groveScaleInterceptor(interceptor.Funcs{}, nil)).
		Build()

	err := newGroveScaler(kubeClient).Reconcile(
		context.Background(),
		dgd,
		map[string]*checkpoint.CheckpointInfo{
			"gated": {
				Enabled:       true,
				StartupPolicy: nvidiacomv1alpha1.CheckpointStartupPolicyWaitForCheckpoint,
			},
		},
	)
	require.NoError(t, err)

	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(frontend), frontend))
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(worker), worker))
	require.NoError(t, kubeClient.Get(context.Background(), client.ObjectKeyFromObject(gated), gated))
	assert.Equal(t, int32(2), frontend.Spec.Replicas)
	assert.Equal(t, int32(3), worker.Spec.Replicas)
	assert.Equal(t, int32(0), gated.Spec.Replicas)
}

func groveScaleInterceptor(funcs interceptor.Funcs, onUpdate func()) interceptor.Funcs {
	funcs.SubResourceGet = func(
		ctx context.Context,
		reader client.Client,
		subResourceName string,
		object client.Object,
		subResource client.Object,
		_ ...client.SubResourceGetOption,
	) error {
		if subResourceName != "scale" {
			return errors.New("unexpected subresource get")
		}
		if err := reader.Get(ctx, client.ObjectKeyFromObject(object), object); err != nil {
			return err
		}
		scaleObject := subResource.(*autoscalingv1.Scale)
		scaleObject.ObjectMeta = metav1.ObjectMeta{
			Name:            object.GetName(),
			Namespace:       object.GetNamespace(),
			ResourceVersion: object.GetResourceVersion(),
		}
		switch resource := object.(type) {
		case *unstructured.Unstructured:
			replicas, found, err := unstructured.NestedInt64(resource.Object, "spec", "replicas")
			if err != nil || !found {
				return errors.New("scale resource has no replicas")
			}
			scaleObject.Spec.Replicas = int32(replicas)
		default:
			return errors.New("unexpected scale resource")
		}
		return nil
	}
	funcs.SubResourceUpdate = func(
		ctx context.Context,
		writer client.Client,
		subResourceName string,
		object client.Object,
		options ...client.SubResourceUpdateOption,
	) error {
		if subResourceName != "scale" {
			return errors.New("unexpected subresource update")
		}
		updateOptions := &client.SubResourceUpdateOptions{}
		updateOptions.ApplyOptions(options)
		scaleObject := updateOptions.SubResourceBody.(*autoscalingv1.Scale)
		if err := writer.Get(ctx, client.ObjectKeyFromObject(object), object); err != nil {
			return err
		}
		switch resource := object.(type) {
		case *unstructured.Unstructured:
			if err := unstructured.SetNestedField(resource.Object, int64(scaleObject.Spec.Replicas), "spec", "replicas"); err != nil {
				return err
			}
		default:
			return errors.New("unexpected scale resource")
		}
		if onUpdate != nil {
			onUpdate()
		}
		return writer.Update(ctx, object)
	}
	return funcs
}

func TestGroveScaler_ReconcileHandlesScaleReadErrors(t *testing.T) {
	dgd := betaDGD(t, &nvidiacomv1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"},
		Spec: nvidiacomv1alpha1.DynamoGraphDeploymentSpec{
			Services: map[string]*nvidiacomv1alpha1.DynamoComponentDeploymentSharedSpec{
				"worker": {
					ComponentType: consts.ComponentTypeWorker,
					Replicas:      ptr.To(int32(2)),
				},
			},
		},
	})

	t.Run("not found is retried by a later reconciliation", func(t *testing.T) {
		kubeClient := fake.NewClientBuilder().
			WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
			WithRESTMapper(groveScaleRESTMapper()).
			Build()
		require.NoError(t, newGroveScaler(kubeClient).Reconcile(context.Background(), dgd, nil))
	})

	t.Run("other errors are propagated", func(t *testing.T) {
		kubeClient := fake.NewClientBuilder().
			WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).
			WithRESTMapper(groveScaleRESTMapper()).
			WithInterceptorFuncs(interceptor.Funcs{
				SubResourceGet: func(context.Context, client.Client, string, client.Object, client.Object, ...client.SubResourceGetOption) error {
					return errors.New("scale read failed")
				},
			}).
			Build()
		err := newGroveScaler(kubeClient).Reconcile(context.Background(), dgd, nil)
		require.ErrorContains(t, err, "scale read failed")
	})
}

func groveScaleRESTMapper() meta.RESTMapper {
	groupVersion := schema.GroupVersion{Group: consts.PodCliqueGVR.Group, Version: consts.PodCliqueGVR.Version}
	mapper := meta.NewDefaultRESTMapper([]schema.GroupVersion{groupVersion})
	mapper.AddSpecific(
		groupVersion.WithKind("PodClique"),
		consts.PodCliqueGVR,
		consts.PodCliqueGVR,
		meta.RESTScopeNamespace,
	)
	mapper.AddSpecific(
		groupVersion.WithKind("PodCliqueScalingGroup"),
		consts.PodCliqueScalingGroupGVR,
		consts.PodCliqueScalingGroupGVR,
		meta.RESTScopeNamespace,
	)
	return mapper
}
