// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"fmt"
	"slices"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/events"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	leaderworkersetv1 "sigs.k8s.io/lws/api/leaderworkerset/v1"
)

// environmentSyncGraph constructs an independently owned frontend-only graph so
// worker-generation rotation cannot hide a missed order-only resource update.
func environmentSyncGraph(origin string) *v1beta1.DynamoGraphDeployment {
	return &v1beta1.DynamoGraphDeployment{Name: "graph", Namespace: "ns", UID: "graph-uid",
		Annotations: map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: origin},
		Spec: v1beta1.DynamoGraphDeploymentSpec{BackendFramework: "vllm", Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
			ComponentName: "frontend", ComponentType: v1beta1.ComponentTypeFrontend, Replicas: new(int32(1)),
			PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{
				Name: "main", Image: "runtime:1.7.0", Env: []corev1.EnvVar{{Name: "B", Value: "$(A)"}, {Name: "A", Value: "x"}},
			}}}},
		}}},
	}
}

func requireEnvironmentOrder(t *testing.T, env []corev1.EnvVar, forward bool) {
	t.Helper()
	var actual []corev1.EnvVar
	for _, variable := range env {
		if variable.Name == "A" || variable.Name == "B" {
			actual = append(actual, variable)
		}
	}
	want := []corev1.EnvVar{{Name: "A", Value: "x"}, {Name: "B", Value: "$(A)"}}
	if forward {
		slices.Reverse(want)
	}
	require.Equal(t, want, actual)
}

func TestEnvironmentOrderThroughDCDSync(t *testing.T) {
	for _, origin := range []string{"1.8.0", "1.7.0", ""} {
		for _, auxiliary := range []bool{false, true} {
			t.Run(fmt.Sprintf("origin=%s/DS-auxiliary=%t", origin, auxiliary), func(t *testing.T) {
				t.Log("Seed the predecessor's exact DCD and legacy hash bookkeeping")
				dgd := environmentSyncGraph(origin)
				children, err := dynamo.GenerateDynamoComponentsDeployments(dgd, nil, nil, dynamo.RollingUpdateContext{})
				require.NoError(t, err)
				existing := children["frontend"]
				existing.UID, existing.Generation = "dcd-uid", 7
				existing.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}
				legacyHash, err := common.GetSpecHash(existing)
				require.NoError(t, err)
				existing.Annotations[common.NvidiaAnnotationHashKey] = legacyHash
				existing.Annotations[common.NvidiaAnnotationGenerationKey] = "7"
				kubeClient := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(dgd, existing).Build()
				require.NoError(t, kubeClient.Get(t.Context(), types.NamespacedName{Name: existing.Name, Namespace: existing.Namespace}, existing))
				original := existing.DeepCopy()
				recorder := events.NewFakeRecorder(20)
				workloads := newComponentWorkloadsReconciler(kubeClient, recorder, newDGDWorkerRolloutReconciler(kubeClient, recorder))
				auxiliaryDCDs := newDisaggregatedSetAuxiliaryDCDReconciler(kubeClient, recorder)
				ordered := origin == "1.8.0"
				previousVersion := ""

				for _, phase := range []string{"operator upgrade", "unchanged reconcile", "order-only input edit"} {
					t.Log(phase)
					edit := phase == "order-only input edit"
					if edit {
						slices.Reverse(dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Env)
					}
					if auxiliary {
						children, err = dynamo.GenerateDynamoComponentsDeployments(dgd, nil, nil, dynamo.RollingUpdateContext{})
						require.NoError(t, err)
						_, _, err = auxiliaryDCDs.Reconcile(t.Context(), dgd, children, disaggregatedSetSelection{})
					} else {
						_, err = workloads.Reconcile(t.Context(), dgd, nil, nil)
					}
					require.NoError(t, err)
					actual := &v1beta1.DynamoComponentDeployment{}
					require.NoError(t, kubeClient.Get(t.Context(), types.NamespacedName{Name: existing.Name, Namespace: existing.Namespace}, actual))
					require.Equal(t, original.UID, actual.UID)
					requireEnvironmentOrder(t, actual.Spec.PodTemplate.Spec.Containers[0].Env, !edit || !ordered)
					if !edit || !ordered {
						require.Equal(t, original.Spec, actual.Spec, "an operator-only upgrade must not alter a workload")
						require.Equal(t, original.Generation, actual.Generation)
					}
					if phase == "unchanged reconcile" || (edit && !ordered) {
						require.Equal(t, previousVersion, actual.ResourceVersion, "unchanged legacy/steady-state content must not be rewritten")
					}
					hash, err := common.GetSpecHash(actual, common.WithPreservedListOrder(ordered))
					require.NoError(t, err)
					require.Equal(t, hash, actual.Annotations[common.NvidiaAnnotationHashKey])
					previousVersion = actual.ResourceVersion
				}
			})
		}
	}
}

func TestEnvironmentOrderThroughGroveSync(t *testing.T) {
	for _, origin := range []string{"1.8.0", "1.7.0", ""} {
		t.Run("origin="+origin, func(t *testing.T) {
			t.Log("Seed a frontend-only PCS with the predecessor's list-insensitive hash")
			dgd := environmentSyncGraph(origin)
			config := &configv1alpha1.OperatorConfiguration{}
			runtimeConfig := &common.RuntimeConfig{Gate: features.Gates{Grove: true}}
			secrets := &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}
			existing, err := dynamo.GenerateGrovePodCliqueSet(t.Context(), dgd, nil, config, runtimeConfig, nil, secrets, nil, nil, false, nil)
			require.NoError(t, err)
			existing.UID, existing.Generation = "pcs-uid", 7
			existing.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}
			legacyHash, err := common.GetSpecHash(existing)
			require.NoError(t, err)
			if existing.Annotations == nil {
				existing.Annotations = map[string]string{}
			}
			existing.Annotations[common.NvidiaAnnotationHashKey] = legacyHash
			existing.Annotations[common.NvidiaAnnotationGenerationKey] = "7"
			kubeClient := fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).WithObjects(existing).Build()
			require.NoError(t, kubeClient.Get(t.Context(), types.NamespacedName{Name: existing.Name, Namespace: existing.Namespace}, existing))
			original := existing.DeepCopy()
			renderer := groveWorkloadRenderer{reader: kubeClient, config: config, runtimeConfig: runtimeConfig, dockerSecretRetriever: secrets}
			reconciler := &groveWorkloadsReconciler{syncer: newDGDResourceSyncer(kubeClient, events.NewFakeRecorder(20))}
			ordered := origin == "1.8.0"

			for _, phase := range []string{"operator upgrade", "unchanged reconcile", "order-only input edit"} {
				t.Log(phase)
				edit := phase == "order-only input edit"
				if edit {
					slices.Reverse(dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Env)
				}
				rendered, err := renderer.Render(t.Context(), groveReconcileRequest{DGD: dgd}, nil, nil, false)
				require.NoError(t, err)
				actual, written, err := reconciler.reconcilePodCliqueSet(t.Context(), dgd, rendered)
				require.NoError(t, err)
				require.Equal(t, original.UID, actual.UID)
				requireEnvironmentOrder(t, actual.Spec.Template.Cliques[0].Spec.PodSpec.Containers[0].Env, ordered && !edit)
				if !edit || !ordered {
					require.Equal(t, original.Spec, actual.Spec, "hash migration must only repair top-level bookkeeping")
					require.Equal(t, original.Generation, actual.Generation)
				}
				if phase == "unchanged reconcile" || !ordered {
					require.False(t, written)
				} else if edit {
					require.True(t, written)
				}
			}
		})
	}
}

func TestEnvironmentOrderThroughDCDWorkloadSync(t *testing.T) {
	type originCase struct {
		name, origin, templateOrigin string
		child, mixedRoles, ordered   bool
	}
	for _, multinode := range []bool{false, true} {
		cases := []originCase{
			{name: "standalone semantic origin is authoritative", origin: "1.8.0", templateOrigin: "1.7.0", ordered: true},
			{name: "standalone legacy origin is authoritative", origin: "1.7.0", templateOrigin: "1.8.0"},
			{name: "standalone missing origin is authoritative", templateOrigin: "1.8.0"},
			{name: "graph child inherits semantic template origin", origin: "1.7.0", templateOrigin: "1.8.0", child: true, ordered: true},
			{name: "graph child inherits legacy template origin", origin: "1.8.0", templateOrigin: "1.7.0", child: true},
		}
		if multinode {
			cases = append(cases, originCase{name: "any semantic role makes LWS order-sensitive", origin: "1.7.0", templateOrigin: "1.8.0", child: true, mixedRoles: true, ordered: true})
		}
		for _, test := range cases {
			t.Run(fmt.Sprintf("multinode=%t/%s", multinode, test.name), func(t *testing.T) {
				t.Log("Render the predecessor workload using the authoritative standalone or inherited origin")
				template := environmentSyncGraph(test.templateOrigin).Spec.Components[0].PodTemplate
				template.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: test.templateOrigin}
				template.Spec.Containers[0].Command = []string{"python3", "-m", "dynamo.vllm"}
				template.Spec.Containers[0].Args = []string{"--model", "test-model"}
				dcd := &v1beta1.DynamoComponentDeployment{Name: "worker", Namespace: "ns", UID: "dcd-uid",
					Annotations: map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: test.origin},
					Spec: v1beta1.DynamoComponentDeploymentSpec{BackendFramework: "vllm", DynamoComponentDeploymentSharedSpec: v1beta1.DynamoComponentDeploymentSharedSpec{
						ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker, Replicas: new(int32(1)), PodTemplate: template,
					}},
				}
				if test.child {
					dcd.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(environmentSyncGraph(test.origin), v1beta1.DynamoGraphDeploymentGVK)}
				}
				if multinode {
					dcd.Spec.Multinode = &v1beta1.MultinodeSpec{NodeCount: 2}
				}
				if test.mixedRoles {
					leader := template.DeepCopy()
					leader.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = "1.7.0"
					dcd.Spec.PodTemplate = nil
					dcd.Spec.Roles = []v1beta1.ComponentRoleSpec{{Name: "leader", PodTemplate: leader}, {Name: "worker", PodTemplate: template}}
				}
				require.Equal(t, test.ordered, dynamo.DCDUsesOrderedEnvironment(dcd))
				scheme := newDynamoGraphDeploymentControllerTestScheme(t)
				require.NoError(t, appsv1.AddToScheme(scheme))
				kubeClient := fake.NewClientBuilder().WithScheme(scheme).WithObjects(dcd).Build()
				reconciler := &DynamoComponentDeploymentReconciler{Client: kubeClient, Recorder: events.NewFakeRecorder(20), Config: &configv1alpha1.OperatorConfiguration{}, RuntimeConfig: &common.RuntimeConfig{}}
				option := generateResourceOption{dynamoComponentDeployment: dcd}
				var originalDeployment *appsv1.Deployment
				var originalLWS *leaderworkersetv1.LeaderWorkerSet
				if multinode {
					workload, _, err := reconciler.generateLeaderWorkerSet(t.Context(), option, leaderWorkerSetName(dcd))
					require.NoError(t, err)
					workload.UID, workload.Generation = "lws-uid", 7
					workload.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dcd, v1beta1.DynamoComponentDeploymentGVK)}
					hash, err := common.GetSpecHash(workload)
					require.NoError(t, err)
					workload.Annotations = map[string]string{common.NvidiaAnnotationHashKey: hash, common.NvidiaAnnotationGenerationKey: "7"}
					require.NoError(t, kubeClient.Create(t.Context(), workload))
					require.NoError(t, kubeClient.Get(t.Context(), types.NamespacedName{Name: workload.Name, Namespace: workload.Namespace}, workload))
					originalLWS = workload.DeepCopy()
				} else {
					workload, _, err := reconciler.generateDeployment(t.Context(), option)
					require.NoError(t, err)
					workload.UID, workload.Generation = "deployment-uid", 7
					workload.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dcd, v1beta1.DynamoComponentDeploymentGVK)}
					hash, err := common.GetSpecHash(workload)
					require.NoError(t, err)
					workload.Annotations[common.NvidiaAnnotationHashKey] = hash
					workload.Annotations[common.NvidiaAnnotationGenerationKey] = "7"
					require.NoError(t, kubeClient.Create(t.Context(), workload))
					require.NoError(t, kubeClient.Get(t.Context(), types.NamespacedName{Name: workload.Name, Namespace: workload.Namespace}, workload))
					originalDeployment = workload.DeepCopy()
				}

				for _, phase := range []string{"operator upgrade", "unchanged reconcile", "order-only input edit"} {
					t.Log(phase)
					edit := phase == "order-only input edit"
					if edit {
						for _, template := range dynamo.ComponentPodTemplates(&dcd.Spec.DynamoComponentDeploymentSharedSpec) {
							slices.Reverse(template.Spec.Containers[0].Env)
						}
					}
					if multinode {
						result, err := reconciler.reconcileLeaderWorkerSetResources(t.Context(), dcd)
						require.NoError(t, err)
						actual := &leaderworkersetv1.LeaderWorkerSet{}
						require.NoError(t, kubeClient.Get(t.Context(), types.NamespacedName{Name: originalLWS.Name, Namespace: originalLWS.Namespace}, actual))
						require.Equal(t, originalLWS.UID, actual.UID)
						requireEnvironmentOrder(t, actual.Spec.LeaderWorkerTemplate.WorkerTemplate.Spec.Containers[0].Env, test.ordered && !edit)
						if !edit || !test.ordered {
							require.Equal(t, originalLWS.Spec, actual.Spec)
							require.Equal(t, int64(7), actual.Generation)
						}
						if phase == "unchanged reconcile" || !test.ordered {
							require.False(t, result.modified)
						} else if edit {
							require.True(t, result.modified)
						}
					} else {
						modified, actual, err := reconciler.createOrUpdateOrDeleteDeployments(t.Context(), option)
						require.NoError(t, err)
						require.Equal(t, originalDeployment.UID, actual.UID)
						requireEnvironmentOrder(t, actual.Spec.Template.Spec.Containers[0].Env, test.ordered && !edit)
						if !edit || !test.ordered {
							require.Equal(t, originalDeployment.Spec, actual.Spec)
							require.Equal(t, originalDeployment.Generation, actual.Generation)
						}
						if phase == "unchanged reconcile" || !test.ordered {
							require.False(t, modified)
						} else if edit {
							require.True(t, modified)
						}
					}
				}
			})
		}
	}
}
