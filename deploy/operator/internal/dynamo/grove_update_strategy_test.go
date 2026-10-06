// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"context"
	"errors"
	"fmt"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/utils/ptr"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestGenerateGrovePodCliqueSet_CoherentStrategyPreservesTemplates(t *testing.T) {
	for _, test := range []struct{ multinode, workerHashSuffix bool }{{false, false}, {false, true}, {true, false}, {true, true}} {
		multinode, workerHashSuffix := test.multinode, test.workerHashSuffix
		t.Run(fmt.Sprintf("multinode=%t/hashSuffix=%t", multinode, workerHashSuffix), func(t *testing.T) {
			t.Log("Render an existing disaggregated graph with RollingRecreate")
			dgd := &v1beta1.DynamoGraphDeployment{
				ObjectMeta: metav1.ObjectMeta{
					Name: "graph", Namespace: "default",
					Annotations: map[string]string{commonconsts.KubeAnnotationGroveUpdateStrategy: "RollingRecreate", commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.6.0"},
				},
				Spec: v1beta1.DynamoGraphDeploymentSpec{
					BackendFramework: "vllm",
					Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
						{ComponentName: "Prefill", ComponentType: commonconsts.ComponentTypePrefill, Replicas: ptr.To(int32(1))},
						{ComponentName: "Decode", ComponentType: commonconsts.ComponentTypeDecode, Replicas: ptr.To(int32(2))},
					},
				},
			}
			if multinode {
				for i := range dgd.Spec.Components {
					dgd.Spec.Components[i].Multinode = &v1beta1.MultinodeSpec{NodeCount: 2}
				}
			}
			config := &configv1alpha1.OperatorConfiguration{}
			runtimeConfig := &controller_common.RuntimeConfig{}
			oldPCS, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, config, runtimeConfig, nil, &mockSecretsRetriever{}, nil, nil, workerHashSuffix, nil)
			require.NoError(t, err)
			oldHash, err := ComputeDGDWorkersSpecHash(dgd)
			require.NoError(t, err)

			t.Log("Adopt the coherent default without changing the workload or worker generation")
			delete(dgd.Annotations, commonconsts.KubeAnnotationGroveUpdateStrategy)
			newPCS, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, config, runtimeConfig, nil, &mockSecretsRetriever{}, nil, nil, workerHashSuffix, nil)
			require.NoError(t, err)
			require.NotNil(t, newPCS.Spec.UpdateStrategy)
			require.Equal(t, grovev1alpha1.CoherentStrategy, newPCS.Spec.UpdateStrategy.Type)
			assert.Equal(t, oldPCS.Spec.Template, newPCS.Spec.Template)
			newHash, err := ComputeDGDWorkersSpecHash(dgd)
			require.NoError(t, err)
			assert.Equal(t, oldHash, newHash)

			t.Log("Validate the strategy-only update against the pinned Grove CRD")
			newGrovePodCliqueSetRequestValidator(t).validate(t, newPCS, oldPCS)
		})
	}
}

func TestGroveUpdateStrategyPolicy(t *testing.T) {
	for _, origin := range []string{"", "1.1.0", "1.6.0"} {
		for _, annotation := range []string{"", string(grovev1alpha1.CoherentStrategy), "RollingRecreate", "OnDelete"} {
			for _, disagg := range []bool{false, true} {
				t.Run(fmt.Sprintf("origin=%s/annotation=%s/disagg=%t", origin, annotation, disagg), func(t *testing.T) {
					t.Log("Author an old or new graph with optional explicit strategy")
					dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", Annotations: map[string]string{}}, Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
						{ComponentName: "Worker", ComponentType: commonconsts.ComponentTypeWorker, Replicas: ptr.To(int32(2))},
					}}}
					if origin != "" {
						dgd.Annotations[commonconsts.KubeAnnotationDynamoOperatorOriginVersion] = origin
					}
					if annotation != "" {
						dgd.Annotations[commonconsts.KubeAnnotationGroveUpdateStrategy] = annotation
					}
					if disagg {
						dgd.Spec.Components = []v1beta1.DynamoComponentDeploymentSharedSpec{
							{ComponentName: "Prefill", ComponentType: commonconsts.ComponentTypePrefill, Replicas: ptr.To(int32(2))},
							{ComponentName: "Decode", ComponentType: commonconsts.ComponentTypeDecode, Replicas: ptr.To(int32(3))},
						}
					}
					original := dgd.DeepCopy()

					t.Log("Both ordinary and LPX envelopes follow origin version rather than topology")
					ordinary, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, &configv1alpha1.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, nil, &mockSecretsRetriever{}, nil, nil, true, nil)
					require.NoError(t, err)
					lpx, err := RenderLPXPodCliqueSet(t.Context(), dgd, &configv1alpha1.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, "lpx-graph", nil)
					require.NoError(t, err)
					want := annotation
					if want == "" && origin == "1.6.0" {
						want = string(grovev1alpha1.CoherentStrategy)
					}
					if want == "" {
						require.Nil(t, ordinary.Spec.UpdateStrategy)
					} else {
						require.NotNil(t, ordinary.Spec.UpdateStrategy)
						require.Equal(t, grovev1alpha1.UpdateStrategyType(want), ordinary.Spec.UpdateStrategy.Type)
					}
					require.Equal(t, ordinary.Spec.UpdateStrategy, lpx.Spec.UpdateStrategy)
					require.Equal(t, original, dgd)
				})
			}
		}
	}
}

func TestGroveUpdateStrategyTransitionsWait(t *testing.T) {
	for _, multinode := range []bool{false, true} {
		for _, observed := range []string{"", "RollingRecreate", string(grovev1alpha1.CoherentStrategy)} {
			for _, annotation := range []string{"", string(grovev1alpha1.CoherentStrategy), "RollingRecreate", "OnDelete"} {
				t.Run(fmt.Sprintf("multinode=%t/observed=%s/annotation=%s", multinode, observed, annotation), func(t *testing.T) {
					t.Log("Render a PCS and mark a standalone or scaling-group rollout in progress")
					dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default", Annotations: map[string]string{commonconsts.KubeAnnotationDynamoOperatorOriginVersion: "1.6.0"}}, Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "Worker", ComponentType: commonconsts.ComponentTypeWorker, Replicas: ptr.To(int32(3))}}}}
					if multinode {
						dgd.Spec.Components[0].Multinode = &v1beta1.MultinodeSpec{NodeCount: 2}
					}
					config, runtime := &configv1alpha1.OperatorConfiguration{}, &controller_common.RuntimeConfig{}
					existing, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, config, runtime, nil, &mockSecretsRetriever{}, nil, nil, true, nil)
					require.NoError(t, err)
					existing.Spec.UpdateStrategy = nil
					if observed != "" {
						existing.Spec.UpdateStrategy = &grovev1alpha1.PodCliqueSetUpdateStrategy{Type: grovev1alpha1.UpdateStrategyType(observed)}
					}
					existing.Status.UpdateProgress = &grovev1alpha1.PodCliqueSetUpdateProgress{UpdateStartedAt: metav1.Now()}
					if annotation != "" {
						dgd.Annotations[commonconsts.KubeAnnotationGroveUpdateStrategy] = annotation
					}
					before := existing.DeepCopy()

					t.Log("Preserve the active strategy for both implicit and explicit transitions")
					desired, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, config, runtime, nil, &mockSecretsRetriever{}, nil, existing, true, nil)
					require.NoError(t, err)
					require.Equal(t, existing.Spec, desired.Spec)
					lpx, err := RenderLPXPodCliqueSet(t.Context(), dgd, config, runtime, "lpx-graph", existing)
					require.NoError(t, err)
					require.Equal(t, existing.Spec.UpdateStrategy, lpx.Spec.UpdateStrategy)
					require.Equal(t, before, existing)

					t.Log("Apply the pending intent after Grove finishes the current rollout")
					existing.Status.UpdateProgress.UpdateEndedAt = ptr.To(metav1.Now())
					desired, err = GenerateGrovePodCliqueSet(t.Context(), dgd, nil, config, runtime, nil, &mockSecretsRetriever{}, nil, existing, true, nil)
					require.NoError(t, err)
					want := annotation
					if want == "" {
						want = string(grovev1alpha1.CoherentStrategy)
					}
					require.Equal(t, grovev1alpha1.UpdateStrategyType(want), desired.Spec.UpdateStrategy.Type)
					require.Equal(t, existing.Spec.Template, desired.Spec.Template)
				})
			}
		}
	}
}

func TestParseGroveUpdateStrategy(t *testing.T) {
	for _, value := range []string{string(grovev1alpha1.CoherentStrategy), "RollingRecreate", "OnDelete", "ondelete", " OnDelete ", "BlueGreen", ""} {
		t.Run(value, func(t *testing.T) {
			t.Log("Parse exact values without normalization")
			strategy, err := ParseGroveUpdateStrategy(value)
			if value == string(grovev1alpha1.CoherentStrategy) || value == "RollingRecreate" || value == "OnDelete" {
				require.NoError(t, err)
				require.Equal(t, value, string(strategy))
			} else {
				require.Error(t, err)
			}
		})
	}
}

func TestCheckGroveUpdateStrategySupport(t *testing.T) {
	for _, test := range []struct {
		name, version                         string
		served, coherent, denied, wantSupport bool
	}{
		{name: "old CRD", version: "v1alpha1", served: true},
		{name: "supported CRD", version: "v1alpha1", served: true, coherent: true, wantSupport: true},
		{name: "unserved version", version: "v1alpha1", coherent: true},
		{name: "different version", version: "v1beta1", served: true, coherent: true},
		{name: "read forbidden", version: "v1alpha1", served: true, coherent: true, denied: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Install a PCS schema with the selected strategy capability")
			scheme := runtime.NewScheme()
			require.NoError(t, apiextensionsv1.AddToScheme(scheme))
			values := []apiextensionsv1.JSON{{Raw: []byte(`"RollingRecreate"`)}, {Raw: []byte(`"OnDelete"`)}}
			if test.coherent {
				values = append(values, apiextensionsv1.JSON{Raw: []byte(`"Coherent"`)})
			}
			crd := &apiextensionsv1.CustomResourceDefinition{ObjectMeta: metav1.ObjectMeta{Name: "podcliquesets.grove.io"}, Spec: apiextensionsv1.CustomResourceDefinitionSpec{Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{Name: test.version, Served: test.served, Schema: &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{Properties: map[string]apiextensionsv1.JSONSchemaProps{
				"spec": {Properties: map[string]apiextensionsv1.JSONSchemaProps{"updateStrategy": {Properties: map[string]apiextensionsv1.JSONSchemaProps{"type": {Enum: values}}}}},
			}}}}}}}
			builder := fake.NewClientBuilder().WithScheme(scheme).WithObjects(crd)
			if test.denied {
				builder.WithInterceptorFuncs(interceptor.Funcs{Get: func(context.Context, client.WithWatch, client.ObjectKey, client.Object, ...client.GetOption) error {
					return apierrors.NewForbidden(schema.GroupResource{Group: "apiextensions.k8s.io", Resource: "customresourcedefinitions"}, crd.Name, errors.New("RBAC denied"))
				}})
			}
			reader := builder.Build()
			pcs := &grovev1alpha1.PodCliqueSet{Spec: grovev1alpha1.PodCliqueSetSpec{UpdateStrategy: &grovev1alpha1.PodCliqueSetUpdateStrategy{Type: grovev1alpha1.CoherentStrategy}}}

			t.Log("Reject unsupported coherent intent without hiding permission errors")
			err := CheckGroveUpdateStrategySupport(t.Context(), reader, pcs)
			if test.denied {
				require.True(t, apierrors.IsForbidden(err))
				require.NotErrorIs(t, err, ErrGroveCoherentUnsupported)
			} else if test.wantSupport {
				require.NoError(t, err)
			} else {
				require.ErrorIs(t, err, ErrGroveCoherentUnsupported)
			}

			t.Log("Legacy strategies do not require capability reads")
			pcs.Spec.UpdateStrategy = nil
			require.NoError(t, CheckGroveUpdateStrategySupport(t.Context(), reader, pcs))
		})
	}
}
