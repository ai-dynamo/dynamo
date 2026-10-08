// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"encoding/json"
	"fmt"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/provideroverride"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/utils/ptr"
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

			t.Log("Explicitly opt into Coherent without changing the workload or worker generation")
			dgd.Annotations[commonconsts.KubeAnnotationGroveUpdateStrategy] = "Coherent"
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
	const nativeMinimumForm = "provider minimum"
	for _, origin := range []string{"", "1.1.0", "1.6.0"} {
		for _, annotation := range []string{"", string(grovev1alpha1.CoherentStrategy), "RollingRecreate", "OnDelete"} {
			for _, disagg := range []bool{false, true} {
				for _, form := range []string{"omitted", "legacy", nativeMinimumForm} {
					t.Run(fmt.Sprintf("origin=%s/annotation=%s/disagg=%t/form=%s", origin, annotation, disagg, form), func(t *testing.T) {
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
						if form == "legacy" {
							dgd.Spec.Components[0].MinAvailable = ptr.To(int32(1))
						}
						if form == nativeMinimumForm {
							dgd.Spec.Components[0].ProviderOverride = &v1beta1.ProviderOverride{APIVersion: provideroverride.GroveAPIVersion, Target: provideroverride.TargetPodCliqueTemplateSpec, Value: apiextensionsv1.JSON{Raw: []byte(`{"spec":{"minAvailable":1}}`)}}
						}
						original := dgd.DeepCopy()

						t.Log("Only the strategy annotation selects Coherent for ordinary and LPX envelopes")
						ordinary, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, &configv1alpha1.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, nil, &mockSecretsRetriever{}, nil, nil, true, nil)
						require.NoError(t, err)
						lpx, err := RenderLPXPodCliqueSet(t.Context(), dgd, &configv1alpha1.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, "lpx-graph", nil)
						require.NoError(t, err)
						want := annotation
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
						require.Nil(t, desired.Spec.UpdateStrategy)
					} else {
						require.Equal(t, grovev1alpha1.UpdateStrategyType(want), desired.Spec.UpdateStrategy.Type)
					}
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

func TestGroveMinAvailableMigrationPreservesWorkload(t *testing.T) {
	for _, multinode := range []bool{false, true} {
		t.Run(fmt.Sprintf("multinode=%t", multinode), func(t *testing.T) {
			t.Log("Render a legacy graph and record its complete workload identity")
			dgd := &v1beta1.DynamoGraphDeployment{ObjectMeta: metav1.ObjectMeta{Name: "graph", Namespace: "default"}, Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{ComponentName: "Worker", ComponentType: commonconsts.ComponentTypeWorker, Replicas: ptr.To(int32(4)), MinAvailable: ptr.To(int32(2))}}}}
			component := &dgd.Spec.Components[0]
			if multinode {
				component.Multinode = &v1beta1.MultinodeSpec{NodeCount: 2}
			}
			oldHash, err := ComputeDGDWorkersSpecHash(dgd)
			require.NoError(t, err)
			oldPCS, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, &configv1alpha1.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, nil, &mockSecretsRetriever{}, nil, nil, true, nil)
			require.NoError(t, err)

			t.Log("Move the same minimum to its native owner without rolling worker templates")
			component.MinAvailable = nil
			fragment, target := `{"spec":{"minAvailable":2}}`, provideroverride.TargetPodCliqueTemplateSpec
			if multinode {
				fragment, target = `{"minAvailable":2}`, provideroverride.TargetPodCliqueScalingGroupConfig
			}
			component.ProviderOverride = &v1beta1.ProviderOverride{APIVersion: provideroverride.GroveAPIVersion, Target: target, Value: apiextensionsv1.JSON{Raw: []byte(fragment)}}
			newHash, err := ComputeDGDWorkersSpecHash(dgd)
			require.NoError(t, err)
			require.Equal(t, oldHash, newHash)
			newPCS, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, &configv1alpha1.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, nil, &mockSecretsRetriever{}, nil, nil, true, nil)
			require.NoError(t, err)
			require.Equal(t, oldPCS.Spec.Template, newPCS.Spec.Template)
			require.Nil(t, newPCS.Spec.UpdateStrategy)

			t.Log("Compose the native fragment without erasing generated container templates")
			composed, err := provideroverride.ComposeGroveOverrides(dgd, newPCS)
			require.NoError(t, err)
			var typed grovev1alpha1.PodCliqueSet
			require.NoError(t, runtime.DefaultUnstructuredConverter.FromUnstructured(composed.Object, &typed))
			require.Equal(t, mustJSON(t, oldPCS.Spec.Template), mustJSON(t, typed.Spec.Template))
			newGrovePodCliqueSetRequestValidator(t).validate(t, &typed, oldPCS)
		})
	}
}

// mustJSON compares wire templates, where empty annotations and omission are equivalent.
func mustJSON(t *testing.T, value any) string {
	t.Helper()
	raw, err := json.Marshal(value)
	require.NoError(t, err)
	return string(raw)
}

func TestGroveScalingBlocked(t *testing.T) {
	for _, test := range []struct {
		name                           string
		strategy                       grovev1alpha1.UpdateStrategyType
		missingPCS, noProgress, active bool
		observed                       *int64
		blocked                        bool
	}{
		{name: "before creation", missingPCS: true},
		{name: "initial configuration without observed generation", noProgress: true},
		{name: "initial coherent configuration without observed generation", strategy: grovev1alpha1.CoherentStrategy, noProgress: true, blocked: true},
		{name: "coherent configuration before an update starts", strategy: grovev1alpha1.CoherentStrategy, observed: ptr.To(int64(1)), noProgress: true, blocked: true},
		{name: "acknowledged coherent configuration without a rollout", strategy: grovev1alpha1.CoherentStrategy, observed: ptr.To(int64(2)), noProgress: true},
		{name: "active coherent update with lagging observed generation", strategy: grovev1alpha1.CoherentStrategy, observed: ptr.To(int64(1)), active: true, blocked: true},
		{name: "active coherent update without observed generation", strategy: grovev1alpha1.CoherentStrategy, active: true, blocked: true},
		{name: "active coherent update after generation acknowledgement", strategy: grovev1alpha1.CoherentStrategy, observed: ptr.To(int64(2)), active: true, blocked: true},
		{name: "completed coherent update with lagging observed generation", strategy: grovev1alpha1.CoherentStrategy, observed: ptr.To(int64(1)), blocked: true},
		{name: "completed acknowledged coherent update", strategy: grovev1alpha1.CoherentStrategy, observed: ptr.To(int64(2))},
		{name: "active rolling recreate with lagging observed generation", strategy: grovev1alpha1.RollingRecreateStrategy, observed: ptr.To(int64(1)), active: true},
		{name: "on delete with lagging observed generation", strategy: grovev1alpha1.OnDeleteStrategy, observed: ptr.To(int64(1))},
		{name: "implicit rolling recreate with lagging observed generation", observed: ptr.To(int64(1)), active: true},
	} {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Observe the PCS strategy, generation acknowledgement, and update progress")
			pcs := &grovev1alpha1.PodCliqueSet{
				ObjectMeta: metav1.ObjectMeta{Generation: 2},
				Status: grovev1alpha1.PodCliqueSetStatus{
					ObservedGeneration: test.observed,
					UpdateProgress:     &grovev1alpha1.PodCliqueSetUpdateProgress{UpdateStartedAt: metav1.Now()},
				},
			}
			if test.strategy != "" {
				pcs.Spec.UpdateStrategy = &grovev1alpha1.PodCliqueSetUpdateStrategy{Type: test.strategy}
			}
			if !test.active {
				pcs.Status.UpdateProgress.UpdateEndedAt = ptr.To(metav1.Now())
			}
			if test.noProgress {
				pcs.Status.UpdateProgress = nil
			}
			if test.missingPCS {
				pcs = nil
			}
			before := pcs.DeepCopy()

			t.Log("Coherent scaling waits for the current generation and rollout completion")
			require.Equal(t, test.blocked, GroveScalingBlocked(pcs))
			require.Equal(t, before, pcs)
		})
	}
}
