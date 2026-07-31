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
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	autoscalingv1 "k8s.io/api/autoscaling/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/scale"
	"k8s.io/utils/ptr"
)

type groveScaleUpdate struct {
	namespace string
	resource  schema.GroupResource
	name      string
	replicas  int32
}

type recordingGroveScaleClient struct {
	currentReplicas int32
	getErrors       map[string]error
	updates         []groveScaleUpdate
}

func (c *recordingGroveScaleClient) Scales(namespace string) scale.ScaleInterface {
	return &recordingGroveScaleInterface{client: c, namespace: namespace}
}

type recordingGroveScaleInterface struct {
	client    *recordingGroveScaleClient
	namespace string
}

func (s *recordingGroveScaleInterface) Get(
	_ context.Context,
	resource schema.GroupResource,
	name string,
	_ metav1.GetOptions,
) (*autoscalingv1.Scale, error) {
	if err := s.client.getErrors[name]; err != nil {
		return nil, err
	}
	return &autoscalingv1.Scale{
		ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"},
		Spec:       autoscalingv1.ScaleSpec{Replicas: s.client.currentReplicas},
	}, nil
}

func (s *recordingGroveScaleInterface) Update(
	_ context.Context,
	resource schema.GroupResource,
	scaleObject *autoscalingv1.Scale,
	_ metav1.UpdateOptions,
) (*autoscalingv1.Scale, error) {
	s.client.updates = append(s.client.updates, groveScaleUpdate{
		namespace: s.namespace,
		resource:  resource,
		name:      scaleObject.Name,
		replicas:  scaleObject.Spec.Replicas,
	})
	return scaleObject, nil
}

func (*recordingGroveScaleInterface) Patch(
	context.Context,
	schema.GroupVersionResource,
	string,
	types.PatchType,
	[]byte,
	metav1.PatchOptions,
) (*autoscalingv1.Scale, error) {
	return nil, errors.New("unexpected scale patch")
}

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
	scaleClient := &recordingGroveScaleClient{currentReplicas: 1}

	err := newGroveScaler(scaleClient).Reconcile(
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

	assert.ElementsMatch(t, []groveScaleUpdate{
		{
			namespace: "default",
			resource:  consts.PodCliqueGVR.GroupResource(),
			name:      "graph-0-frontend",
			replicas:  2,
		},
		{
			namespace: "default",
			resource:  consts.PodCliqueScalingGroupGVR.GroupResource(),
			name:      "graph-0-worker",
			replicas:  3,
		},
		{
			namespace: "default",
			resource:  consts.PodCliqueGVR.GroupResource(),
			name:      "graph-0-gated",
			replicas:  0,
		},
	}, scaleClient.updates)
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
		scaleClient := &recordingGroveScaleClient{
			getErrors: map[string]error{
				"graph-0-worker": apierrors.NewNotFound(
					consts.PodCliqueGVR.GroupResource(),
					"graph-0-worker",
				),
			},
		}
		require.NoError(t, newGroveScaler(scaleClient).Reconcile(context.Background(), dgd, nil))
		assert.Empty(t, scaleClient.updates)
	})

	t.Run("other errors are propagated", func(t *testing.T) {
		scaleClient := &recordingGroveScaleClient{
			getErrors: map[string]error{"graph-0-worker": errors.New("scale read failed")},
		}
		err := newGroveScaler(scaleClient).Reconcile(context.Background(), dgd, nil)
		require.ErrorContains(t, err, "scale read failed")
	})
}
