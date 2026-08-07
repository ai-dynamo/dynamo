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
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/record"
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
		Recorder:      record.NewFakeRecorder(10),
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

func TestStampGroveCheckpointBindings(t *testing.T) {
	t.Log("Build engine and unrelated cliques with stale whole-map copies")
	desired := groveBindingPCS(
		`stale`,
		groveBindingClique("engine-a", "checkpoint-a", "uid-a/1"),
		&grovev1alpha1.PodCliqueTemplateSpec{
			Name: "gms",
			Annotations: map[string]string{
				consts.GroveCheckpointBindingsAnnotation: "stale",
			},
		},
	)

	t.Log("Stamp only the canonical top-level checkpoint map")
	require.NoError(t, stampGroveCheckpointBindings(desired, map[string]string{
		"checkpoint-b": "uid-b/2",
		"checkpoint-a": "uid-a/1",
	}))

	assert.Equal(t,
		`{"checkpoint-a":"uid-a/1","checkpoint-b":"uid-b/2"}`,
		desired.Annotations[consts.GroveCheckpointBindingsAnnotation],
	)
	assert.Equal(t, "checkpoint-a",
		desired.Spec.Template.Cliques[0].Annotations[consts.CheckpointNameAnnotation],
	)
	assert.Equal(t, "uid-a/1",
		desired.Spec.Template.Cliques[0].Annotations[consts.CheckpointBindingAnnotation],
	)
	for _, clique := range desired.Spec.Template.Cliques {
		assert.NotContains(t, clique.Annotations, consts.GroveCheckpointBindingsAnnotation)
	}
	assert.NotContains(t, desired.Spec.Template.Cliques[1].Annotations, consts.CheckpointNameAnnotation)
	assert.NotContains(t, desired.Spec.Template.Cliques[1].Annotations, consts.CheckpointBindingAnnotation)
}

func TestValidateGroveCheckpointBindings(t *testing.T) {
	matching := groveBindingPCS(
		`{"checkpoint-a":"uid-a/1"}`,
		groveBindingClique("engine", "checkpoint-a", "uid-a/1"),
	)
	newClique := groveBindingPCS(
		`{"checkpoint-a":"uid-a/1","checkpoint-b":"uid-b/1"}`,
		groveBindingClique("engine", "checkpoint-a", "uid-a/1"),
		groveBindingClique("engine-new", "checkpoint-b", "uid-b/1"),
	)

	tests := []struct {
		name     string
		existing *grovev1alpha1.PodCliqueSet
		desired  *grovev1alpha1.PodCliqueSet
		wantErr  string
	}{
		{name: "new workload", desired: matching},
		{name: "steady reconcile", existing: matching, desired: matching},
		{
			name: "existing clique without binding copies fails closed",
			existing: groveBindingPCS(
				`{"checkpoint-a":"uid-a/1"}`,
				&grovev1alpha1.PodCliqueTemplateSpec{Name: "engine"},
			),
			desired: matching,
			wantErr: "copies are required",
		},
		{
			name: "missing PCS map fails closed",
			existing: groveBindingPCS(
				"",
				groveBindingClique("engine", "checkpoint-a", "uid-a/1"),
			),
			desired: matching,
			wantErr: "copies disagree",
		},
		{
			name: "divergent clique and PCS copies fail",
			existing: groveBindingPCS(
				`{"checkpoint-a":"uid-a/1"}`,
				groveBindingClique("engine", "checkpoint-a", "other/1"),
			),
			desired: matching,
			wantErr: "copies disagree",
		},
		{
			name:     "same-name replacement fails",
			existing: matching,
			desired: groveBindingPCS(
				`{"checkpoint-a":"replacement/1"}`,
				groveBindingClique("engine", "checkpoint-a", "replacement/1"),
			),
			wantErr: "is immutable",
		},
		{
			name:     "missing map entry does not make checkpoint new",
			existing: matching,
			desired: groveBindingPCS(
				`{"checkpoint-b":"uid-b/1"}`,
				groveBindingClique("engine", "checkpoint-b", "uid-b/1"),
			),
			wantErr: "is immutable",
		},
		{
			name:     "new checkpoint on new rollout clique is allowed",
			existing: matching,
			desired:  newClique,
		},
		{
			name:     "explicit checkpoints remain unbound",
			existing: matching,
			desired:  groveBindingPCS("", &grovev1alpha1.PodCliqueTemplateSpec{Name: "engine"}),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Validate PCS and same-named clique binding copies")
			err := validateGroveCheckpointBindings(tt.existing, tt.desired)
			if tt.wantErr == "" {
				require.NoError(t, err)
			} else {
				require.ErrorContains(t, err, tt.wantErr)
			}
		})
	}
}

func groveBindingPCS(
	bindings string,
	cliques ...*grovev1alpha1.PodCliqueTemplateSpec,
) *grovev1alpha1.PodCliqueSet {
	annotations := map[string]string{}
	if bindings != "" {
		annotations[consts.GroveCheckpointBindingsAnnotation] = bindings
	}
	return &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:        "graph",
			Namespace:   "default",
			Annotations: annotations,
		},
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: cliques,
			},
		},
	}
}

func groveBindingClique(name, checkpointName, binding string) *grovev1alpha1.PodCliqueTemplateSpec {
	return &grovev1alpha1.PodCliqueTemplateSpec{
		Name: name,
		Annotations: map[string]string{
			consts.CheckpointNameAnnotation:    checkpointName,
			consts.CheckpointBindingAnnotation: binding,
		},
	}
}
