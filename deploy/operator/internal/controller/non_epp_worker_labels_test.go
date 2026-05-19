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
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

type nonEPPWorkerIdentityLabels struct {
	name             string
	labels           map[string]string
	subComponentType string
}

type nonEPPWorkerIdentityRender struct {
	labelSets       []nonEPPWorkerIdentityLabels
	serviceSelector map[string]string
	existingHash    string
	desiredHash     string
}

func TestNonEPPWorkerIdentityLabelsDoNotTriggerRollout(t *testing.T) {
	ctx := context.Background()
	tests := []struct {
		name   string
		render func(context.Context, *testing.T) nonEPPWorkerIdentityRender
	}{
		{name: "Deployment", render: renderNonEPPDeploymentWorkerIdentity},
		{name: "LeaderWorkerSet", render: renderNonEPPLeaderWorkerSetWorkerIdentity},
		{name: "Grove PodCliqueSet", render: renderNonEPPGroveWorkerIdentity},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.render(ctx, t)
			for _, labelSet := range got.labelSets {
				require.Equal(t, commonconsts.ComponentTypeWorker, labelSet.labels[commonconsts.KubeLabelDynamoComponentType], labelSet.name)
				require.Equal(t, labelSet.subComponentType, labelSet.labels[commonconsts.KubeLabelDynamoSubComponentType], labelSet.name)
				require.NotContains(t, labelSet.labels, commonconsts.KubeLabelDynamoComponentClass, labelSet.name)
			}

			if got.serviceSelector != nil {
				require.Equal(t, commonconsts.ComponentTypeWorker, got.serviceSelector[commonconsts.KubeLabelDynamoComponentType])
			}
			if got.existingHash != "" || got.desiredHash != "" {
				require.Equal(t, got.existingHash, got.desiredHash)
			}
		})
	}
}

func renderNonEPPDeploymentWorkerIdentity(ctx context.Context, t *testing.T) nonEPPWorkerIdentityRender {
	t.Helper()
	s := newNonEPPWorkerIdentityScheme(t)
	dcd := newNonEPPDecodeDCD(false)
	seed := &appsv1.Deployment{
		ObjectMeta: metav1.ObjectMeta{Name: dcd.Name, Namespace: dcd.Namespace},
		Spec: appsv1.DeploymentSpec{
			Template: corev1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{Labels: newLegacyWorkerLabels(commonconsts.ComponentTypeDecode)},
			},
		},
	}

	existing, toDelete, err := newNonEPPDCDReconciler(t, s, dcd, seed).generateDeployment(ctx, generateResourceOption{dynamoComponentDeployment: dcd})
	require.NoError(t, err)
	require.False(t, toDelete)
	desired, toDelete, err := newNonEPPDCDReconciler(t, s, dcd, existing).generateDeployment(ctx, generateResourceOption{dynamoComponentDeployment: dcd})
	require.NoError(t, err)
	require.False(t, toDelete)
	return nonEPPWorkerIdentityRender{
		labelSets: []nonEPPWorkerIdentityLabels{{
			name:             "deployment pod template",
			labels:           desired.Spec.Template.Labels,
			subComponentType: commonconsts.ComponentTypeDecode,
		}},
		existingHash: specHash(t, existing),
		desiredHash:  specHash(t, desired),
	}
}

func renderNonEPPLeaderWorkerSetWorkerIdentity(ctx context.Context, t *testing.T) nonEPPWorkerIdentityRender {
	t.Helper()
	s := newNonEPPWorkerIdentityScheme(t)
	dcd := newNonEPPDecodeDCD(true)
	seed := &leaderworkersetv1.LeaderWorkerSet{
		ObjectMeta: metav1.ObjectMeta{Name: leaderWorkerSetName(dcd), Namespace: dcd.Namespace},
		Spec: leaderworkersetv1.LeaderWorkerSetSpec{
			LeaderWorkerTemplate: leaderworkersetv1.LeaderWorkerTemplate{
				LeaderTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Labels: newLegacyWorkerLabels(commonconsts.ComponentTypeDecode)},
				},
				WorkerTemplate: corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Labels: newLegacyWorkerLabels(commonconsts.ComponentTypeDecode)},
				},
			},
		},
	}

	existing, toDelete, err := newNonEPPDCDReconciler(t, s, dcd, seed).generateLeaderWorkerSet(ctx, generateResourceOption{dynamoComponentDeployment: dcd})
	require.NoError(t, err)
	require.False(t, toDelete)
	desired, toDelete, err := newNonEPPDCDReconciler(t, s, dcd, existing).generateLeaderWorkerSet(ctx, generateResourceOption{dynamoComponentDeployment: dcd})
	require.NoError(t, err)
	require.False(t, toDelete)
	return nonEPPWorkerIdentityRender{
		labelSets: []nonEPPWorkerIdentityLabels{
			{
				name:             "leaderworkerset leader pod template",
				labels:           desired.Spec.LeaderWorkerTemplate.LeaderTemplate.Labels,
				subComponentType: commonconsts.ComponentTypeDecode,
			},
			{
				name:             "leaderworkerset worker pod template",
				labels:           desired.Spec.LeaderWorkerTemplate.WorkerTemplate.Labels,
				subComponentType: commonconsts.ComponentTypeDecode,
			},
		},
		existingHash: specHash(t, existing),
		desiredHash:  specHash(t, desired),
	}
}

func renderNonEPPGroveWorkerIdentity(ctx context.Context, t *testing.T) nonEPPWorkerIdentityRender {
	t.Helper()
	dgd := newNonEPPGroveDGD()
	seed := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: dgd.Name, Namespace: dgd.Namespace},
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{
						Name: "vllmprefillworker",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent: "VllmPrefillWorker",
						},
						Annotations: map[string]string{
							commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.1.0",
						},
					},
					{Name: "frontend"},
					{
						Name: "vllmdecodeworker",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent: "VllmDecodeWorker",
						},
					},
				},
			},
		},
	}
	seed.Spec.Template.Cliques[0].Labels = mergeStringMaps(seed.Spec.Template.Cliques[0].Labels, newLegacyWorkerLabels(commonconsts.ComponentTypePrefill))
	seed.Spec.Template.Cliques[2].Labels = mergeStringMaps(seed.Spec.Template.Cliques[2].Labels, newLegacyWorkerLabels(commonconsts.ComponentTypeDecode))

	renderDGD, existing := renderGrovePodCliqueSetFromExisting(ctx, t, dgd, seed)
	renderDGD, desired := renderGrovePodCliqueSetFromExisting(ctx, t, dgd, existing.DeepCopy())
	prefill := requireGroveClique(t, desired, "vllmprefillworker")
	decode := requireGroveClique(t, desired, "vllmdecodeworker")
	require.Equal(t, "1.1.0", prefill.Annotations[commonconsts.KubeAnnotationDynamoOperatorOriginVersion])

	decodeComponent := renderDGD.GetComponentByName("VllmDecodeWorker")
	require.NotNil(t, decodeComponent)
	service, err := dynamo.GenerateComponentService(dynamo.ComponentServiceParams{
		ServiceName:     dynamo.GetDCDResourceName(renderDGD, "VllmDecodeWorker", ""),
		Namespace:       renderDGD.Namespace,
		ComponentType:   string(decodeComponent.ComponentType),
		DynamoNamespace: renderDGD.GetDynamoNamespaceForComponent(decodeComponent),
		ComponentName:   "VllmDecodeWorker",
		Labels:          dynamo.GetDGDComponentResourceLabels(renderDGD, "VllmDecodeWorker", decodeComponent),
		IsK8sDiscovery:  true,
	})
	require.NoError(t, err)

	return nonEPPWorkerIdentityRender{
		labelSets: []nonEPPWorkerIdentityLabels{
			{
				name:             "grove prefill clique",
				labels:           prefill.Labels,
				subComponentType: commonconsts.ComponentTypePrefill,
			},
			{
				name:             "grove decode clique",
				labels:           decode.Labels,
				subComponentType: commonconsts.ComponentTypeDecode,
			},
		},
		serviceSelector: service.Spec.Selector,
		existingHash:    specHash(t, existing),
		desiredHash:     specHash(t, desired),
	}
}

func TestGroveNativeWorkerIdentityLabelsStayNative(t *testing.T) {
	ctx := context.Background()
	dgd := &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "native-dgd", Namespace: "jsm"},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "prefill", ComponentType: v1beta1.ComponentTypePrefill, Replicas: ptr.To(int32(1))},
			},
		},
	}
	existingPCS := &grovev1alpha1.PodCliqueSet{
		ObjectMeta: metav1.ObjectMeta{Name: "native-dgd", Namespace: "jsm"},
		Spec: grovev1alpha1.PodCliqueSetSpec{
			Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
					{
						Name: "prefill",
						Labels: map[string]string{
							commonconsts.KubeLabelDynamoComponent:     "prefill",
							commonconsts.KubeLabelDynamoComponentType: commonconsts.ComponentTypePrefill,
						},
					},
				},
			},
		},
	}

	renderDGD, desired := renderGrovePodCliqueSetFromExisting(ctx, t, dgd, existingPCS)
	prefillComponent := renderDGD.GetComponentByName("prefill")
	require.NotNil(t, prefillComponent)
	require.Equal(t, v1beta1.ComponentTypePrefill, prefillComponent.ComponentType)
	prefillClique := requireGroveClique(t, desired, "prefill")
	require.Equal(t, commonconsts.ComponentTypePrefill, prefillClique.Labels[commonconsts.KubeLabelDynamoComponentType])
}

func renderGrovePodCliqueSetFromExisting(
	ctx context.Context,
	t *testing.T,
	dgd *v1beta1.DynamoGraphDeployment,
	existingPCS *grovev1alpha1.PodCliqueSet,
) (*v1beta1.DynamoGraphDeployment, *grovev1alpha1.PodCliqueSet) {
	t.Helper()
	kubeClient := fake.NewClientBuilder().
		WithScheme(newNonEPPWorkerIdentityScheme(t)).
		WithObjects(dgd, existingPCS).
		Build()
	reconciler := &DynamoGraphDeploymentReconciler{Client: kubeClient}
	renderDGD, existing, err := reconciler.prepareGroveRenderDeployment(ctx, dgd)
	require.NoError(t, err)
	require.NotNil(t, existing)
	generated, err := dynamo.GenerateGrovePodCliqueSet(ctx, renderDGD, &configv1alpha1.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, kubeClient, nil, nil, nil, nil)
	require.NoError(t, err)
	preserveGrovePodCliqueSetOrder(generated, existing)
	return renderDGD, generated
}

func newNonEPPDCDReconciler(
	t *testing.T,
	s *runtime.Scheme,
	dcd *v1beta1.DynamoComponentDeployment,
	objects ...client.Object,
) *DynamoComponentDeploymentReconciler {
	t.Helper()
	objects = append([]client.Object{dcd}, objects...)
	return &DynamoComponentDeploymentReconciler{
		Client: fake.NewClientBuilder().
			WithScheme(s).
			WithObjects(objects...).
			Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &controller_common.RuntimeConfig{LWSEnabled: true},
		DockerSecretRetriever: &mockDockerSecretRetriever{
			GetSecretsFunc: func(namespace, imageName string) ([]string, error) {
				return nil, nil
			},
		},
	}
}

func newNonEPPDecodeDCD(multinode bool) *v1beta1.DynamoComponentDeployment {
	dcd := &v1beta1.DynamoComponentDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "qwen-decode-db6b6891",
			Namespace: "default",
			Labels: map[string]string{
				commonconsts.KubeLabelDynamoComponent:           "decode",
				commonconsts.KubeLabelDynamoGraphDeploymentName: "qwen",
				commonconsts.KubeLabelDynamoWorkerHash:          "db6b6891",
				commonconsts.KubeLabelDynamoComponentType:       commonconsts.ComponentTypeDecode,
			},
		},
		Spec: v1beta1.DynamoComponentDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
			DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
				ComponentName: "decode",
				ComponentType: v1beta1.ComponentTypeDecode,
				PodTemplate: &corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{
							Name:    commonconsts.MainContainerName,
							Image:   "test-image:latest",
							Command: []string{"python3"},
							Args:    []string{"-m", "dynamo.vllm"},
							Resources: corev1.ResourceRequirements{
								Limits: corev1.ResourceList{
									corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
								},
							},
						}},
					},
				},
			},
		},
	}
	if multinode {
		dcd.Spec.Multinode = &v1beta1.MultinodeSpec{NodeCount: 2}
	}
	return dcd
}

func newNonEPPGroveDGD() *v1beta1.DynamoGraphDeployment {
	return &v1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "vllm-disagg-planner",
			Namespace: "jsm",
			Annotations: map[string]string{
				commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.1.0",
			},
		},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
				{ComponentName: "Frontend", ComponentType: v1beta1.ComponentTypeFrontend, Replicas: ptr.To(int32(1))},
				{ComponentName: "Planner", ComponentType: v1beta1.ComponentTypePlanner, Replicas: ptr.To(int32(1))},
				{ComponentName: "VllmDecodeWorker", ComponentType: v1beta1.ComponentTypeDecode, Replicas: ptr.To(int32(1))},
				{ComponentName: "VllmPrefillWorker", ComponentType: v1beta1.ComponentTypePrefill, Replicas: ptr.To(int32(1))},
			},
		},
	}
}

func newNonEPPWorkerIdentityScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	s := runtime.NewScheme()
	for _, addToScheme := range []func(*runtime.Scheme) error{
		appsv1.AddToScheme,
		corev1.AddToScheme,
		v1beta1.AddToScheme,
		grovev1alpha1.AddToScheme,
		leaderworkersetv1.AddToScheme,
	} {
		require.NoError(t, addToScheme(s))
	}
	return s
}

func newLegacyWorkerLabels(subComponentType string) map[string]string {
	return map[string]string{
		commonconsts.KubeLabelDynamoComponentType:    commonconsts.ComponentTypeWorker,
		commonconsts.KubeLabelDynamoSubComponentType: subComponentType,
	}
}

func mergeStringMaps(dst map[string]string, src map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range dst {
		out[k] = v
	}
	for k, v := range src {
		out[k] = v
	}
	return out
}

func requireGroveClique(t *testing.T, pcs *grovev1alpha1.PodCliqueSet, name string) *grovev1alpha1.PodCliqueTemplateSpec {
	t.Helper()
	for _, clique := range pcs.Spec.Template.Cliques {
		if clique.Name == name {
			return clique
		}
	}
	t.Fatalf("expected rendered grove clique %q", name)
	return nil
}

func specHash(t *testing.T, obj client.Object) string {
	t.Helper()
	hash, err := controller_common.GetSpecHash(obj)
	require.NoError(t, err)
	return hash
}
