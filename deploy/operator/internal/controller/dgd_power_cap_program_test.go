// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	podcontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/pods"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/power"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	snapshotv1alpha1 "github.com/ai-dynamo/snapshot/api/v1alpha1"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
)

func powerCapTestDGD(t *testing.T, provider string) *v1beta1.DynamoGraphDeployment {
	t.Helper()
	dgd := createTestDGD("power", map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
		"worker": {ComponentType: consts.ComponentTypeWorker, Envs: []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "v1"}}},
	})
	dgd.UID, dgd.Generation = "graph-uid", 2
	dgd.Annotations = map[string]string{consts.KubeAnnotationWorkloadProvider: provider}
	for i := range dgd.Spec.Components {
		dgd.Spec.Components[i].PodTemplate.Annotations = map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}
	}
	return dgd
}

func powerCapTestScheme(t *testing.T) *runtime.Scheme {
	t.Helper()
	scheme := newDynamoGraphDeploymentControllerTestScheme(t)
	require.NoError(t, appsv1.AddToScheme(scheme))
	return scheme
}

// powerCapTestClient registers the production per-component Pod index that rollout queries.
func powerCapTestClient(scheme *runtime.Scheme) *fake.ClientBuilder {
	return fake.NewClientBuilder().WithScheme(scheme).WithIndex(&corev1.Pod{}, podcontroller.ComponentIndex, podcontroller.ComponentIndexValues)
}

// powerCapTestOwners constructs fresh complete owner chains for each test.
func powerCapTestOwners(dgd *v1beta1.DynamoGraphDeployment, provider string) ([]client.Object, *corev1.Pod) {
	pod := &corev1.Pod{Name: "worker-pod", Namespace: dgd.Namespace, UID: "pod-uid", Labels: map[string]string{
		consts.KubeLabelDynamoGraphDeploymentName: dgd.Name, consts.KubeLabelDynamoComponent: "worker", consts.KubeLabelDynamoComponentType: "worker", consts.KubeLabelDynamoSelector: "worker",
	}, Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: "300", "other": "retained"}}
	if provider == "grove" {
		pcs := &grovev1alpha1.PodCliqueSet{Name: "pcs", Namespace: dgd.Namespace, UID: "pcs-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}}
		group := &grovev1alpha1.PodCliqueScalingGroup{Name: "group", Namespace: dgd.Namespace, UID: "group-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}}
		clique := &grovev1alpha1.PodClique{Name: "clique", Namespace: dgd.Namespace, UID: "clique-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(group, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))}}
		pod.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(clique, grovev1alpha1.SchemeGroupVersion.WithKind("PodClique"))}
		return []client.Object{pcs, group, clique}, pod
	}
	dcd := &v1beta1.DynamoComponentDeployment{Name: "dcd", Namespace: dgd.Namespace, UID: "dcd-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}}
	deployment := &appsv1.Deployment{Name: "deployment", Namespace: dgd.Namespace, UID: "deployment-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dcd, v1beta1.DynamoComponentDeploymentGVK)}}
	rs := &appsv1.ReplicaSet{Name: "rs", Namespace: dgd.Namespace, UID: "rs-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(deployment, appsv1.SchemeGroupVersion.WithKind("Deployment"))}}
	pod.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(rs, appsv1.SchemeGroupVersion.WithKind("ReplicaSet"))}
	return []client.Object{dcd, deployment, rs}, pod
}

func TestPowerCapComponentRolloutSupersedesRestart(t *testing.T) {
	for _, active := range []bool{false, true} {
		t.Run(fmt.Sprintf("active=%t", active), func(t *testing.T) {
			t.Log("An ordinary image change with an unchanged cap starts rollout before committing restart state")
			dgd := powerCapTestDGD(t, "component")
			hash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
			require.NoError(t, err)
			dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = hash
			dcds, err := dynamo.GenerateDynamoComponentsDeployments(dgd, nil, nil, dynamo.RollingUpdateContext{NewWorkerHash: hash})
			require.NoError(t, err)
			dcd := dcds["worker"]
			dcd.UID = "old-worker-uid"
			dgd.Spec.Restart = &v1beta1.Restart{ID: "restart-1"}
			if active {
				dgd.Status.Restart = &v1beta1.RestartStatus{ObservedID: "restart-1", Phase: v1beta1.RestartPhaseRestarting, InProgress: []string{"worker"}}
			}
			dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Image = "different:v2"
			scheme := powerCapTestScheme(t)
			require.NoError(t, rbacv1.AddToScheme(scheme))
			c := powerCapTestClient(scheme).WithObjects(dgd, dcd).WithStatusSubresource(dgd).Build()
			config := &configv1alpha1.OperatorConfiguration{}
			config.Namespace.Restricted = "default"
			root := &DynamoGraphDeploymentReconciler{Client: c, Config: config, RuntimeConfig: &commoncontroller.RuntimeConfig{}, DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}}
			result, err := root.newComponentProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			require.True(t, rollingUpdateInProgress(result.Status.RollingUpdate))
			require.Equal(t, v1beta1.RestartPhaseSuperseded, result.Status.Restart.Phase)
			started := 0
			for _, event := range result.Events {
				if event.Reason == "RollingUpdateStarted" {
					started++
				}
			}
			require.Equal(t, 1, started)
		})
	}
}

func TestPowerCapProgramsPassThroughPropagationCondition(t *testing.T) {
	for _, provider := range []string{"component", "grove"} {
		for _, scenario := range []string{"success", "early-error", "terminal", "absent"} {
			t.Run(provider+"/"+scenario, func(t *testing.T) {
				t.Log("Reconcile a capped worker needing repair while the power controller's condition is stored")
				dgd := powerCapTestDGD(t, provider)
				dgd.Status.ObservedGeneration = 1
				dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
				stored := metav1.Condition{Type: power.ConditionType, Status: metav1.ConditionFalse, Reason: "PodAnnotationMismatch", Message: "owned by the power controller", ObservedGeneration: 1, LastTransitionTime: metav1.Unix(1, 0)}
				if scenario != "absent" {
					meta.SetStatusCondition(&dgd.Status.Conditions, stored)
				}
				if scenario == "terminal" && provider == "component" {
					unsupported := *dgd.Spec.Components[0].DeepCopy()
					unsupported.ComponentName, unsupported.ComponentType = "external", v1beta1.ComponentTypeLPX
					dgd.Spec.Components = append(dgd.Spec.Components, unsupported)
				}
				owners, pod := powerCapTestOwners(dgd, provider)
				scheme := powerCapTestScheme(t)
				require.NoError(t, rbacv1.AddToScheme(scheme))
				podPatches := 0
				c := powerCapTestClient(scheme).WithObjects(append(owners, dgd, pod)...).WithStatusSubresource(dgd).WithInterceptorFuncs(interceptor.Funcs{
					Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
						if _, ok := obj.(*corev1.Pod); ok {
							podPatches++
						}
						return c.Patch(ctx, obj, patch, opts...)
					},
				}).Build()
				config := &configv1alpha1.OperatorConfiguration{Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendEtcd}}
				if scenario != "early-error" {
					config.Namespace.Restricted = "default"
				}
				gates := features.Gates{Grove: scenario != "terminal"}
				root := &DynamoGraphDeploymentReconciler{Client: c, Config: config, RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: gates}, DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}}
				var program workloadProgram = root.newComponentProgram()
				if provider == "grove" {
					program = root.newGroveProgram()
				}
				result, err := program.Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})

				t.Log("The program never repairs Pods and returns the condition unchanged")
				switch scenario {
				case "success", "absent":
					require.NoError(t, err)
				case "terminal":
					require.True(t, errors.Is(err, reconcile.TerminalError(nil)))
				default:
					require.Error(t, err)
				}
				require.Zero(t, podPatches)
				got := meta.FindStatusCondition(result.Status.Conditions, power.ConditionType)
				if scenario == "absent" {
					require.Nil(t, got)
					return
				}
				require.Equal(t, stored, *got)
			})
		}
	}
}

func TestPowerCapUpgradeUsesNormalWorkerRollout(t *testing.T) {
	for _, provider := range []string{consts.WorkloadProviderComponent, consts.WorkloadProviderGrove} {
		for _, source := range []string{"beta", "alpha"} {
			for _, legacy := range []bool{false, true} {
				t.Run(fmt.Sprintf("%s/%s/legacy=%t", provider, source, legacy), func(t *testing.T) {
					t.Log("Start from a DGD and worker resources rendered by the pre-change operator")
					data, err := os.ReadFile(filepath.Join("testdata", "power-cap-upgrade", provider+"-"+source+".json"))
					require.NoError(t, err)
					var old struct {
						DGD *v1beta1.DynamoGraphDeployment
						DCD *v1beta1.DynamoComponentDeployment
						PCS *grovev1alpha1.PodCliqueSet
					}
					require.NoError(t, json.Unmarshal(data, &old))
					dgd := old.DGD
					dgd.UID, dgd.Generation = "upgraded-graph", 1
					const capKey = consts.KubeAnnotationGPUPowerLimit
					oldHash := dgd.Annotations[consts.AnnotationCurrentWorkerHashV2]
					require.NotEmpty(t, oldHash)
					if legacy {
						// The same stored generation can have an opaque v1 identity and no v2 record.
						dgd.Annotations[consts.AnnotationCurrentWorkerHash] = oldHash
						delete(dgd.Annotations, consts.AnnotationCurrentWorkerHashV2)
					}
					newHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
					require.NoError(t, err)
					require.NotEqual(t, oldHash, newHash)
					scheme := powerCapTestScheme(t)
					require.NoError(t, rbacv1.AddToScheme(scheme))
					builder := powerCapTestClient(scheme).WithStatusSubresource(dgd, &v1beta1.DynamoComponentDeployment{}).WithObjects(dgd)
					owner := *metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)
					if provider == consts.WorkloadProviderComponent {
						require.Equal(t, "300", old.DCD.Spec.PodTemplate.Annotations[capKey])
						old.DCD.OwnerReferences = []metav1.OwnerReference{owner}
						builder.WithObjects(old.DCD)
					} else {
						require.Equal(t, "300", old.PCS.Spec.Template.Cliques[0].Annotations[capKey])
						old.PCS.OwnerReferences = []metav1.OwnerReference{owner}
						funcs := groveScaleInterceptor(interceptor.Funcs{}, nil)
						scaleUpdate := funcs.SubResourceUpdate
						funcs.SubResourceUpdate = func(ctx context.Context, kube client.Client, subresource string, object client.Object, options ...client.SubResourceUpdateOption) error {
							if subresource == "status" {
								return kube.Status().Update(ctx, object, options...)
							}
							return scaleUpdate(ctx, kube, subresource, object, options...)
						}
						builder.WithObjects(old.PCS).WithRESTMapper(groveScaleRESTMapper()).WithInterceptorFuncs(funcs)
					}
					c := builder.Build()
					config := &configv1alpha1.OperatorConfiguration{}
					config.Namespace.Restricted = dgd.Namespace
					root := &DynamoGraphDeploymentReconciler{
						Client: c, Config: config, RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}},
						DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }},
					}
					var program workloadProgram = root.newComponentProgram()
					if provider == consts.WorkloadProviderGrove {
						program = root.newGroveProgram()
					}

					t.Log("Reconcile the upgrade and inspect the ordinary provider rollout inputs")
					result, err := program.Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
					require.NoError(t, err)
					var workloadKey client.ObjectKey
					var deploymentTemplate *corev1.PodTemplateSpec
					var groveTemplate *grovev1alpha1.PodCliqueSetTemplateSpec
					if old.DCD != nil {
						hash := newHash
						if legacy {
							// The existing v1 DCD keeps its name and its changed Deployment template rolls in place.
							hash = oldHash
						} else {
							require.True(t, rollingUpdateInProgress(result.Status.RollingUpdate))
						}
						dcd := &v1beta1.DynamoComponentDeployment{}
						workloadKey = client.ObjectKey{Namespace: dgd.Namespace, Name: dynamo.GetDCDResourceName(dgd, "worker", hash)}
						require.NoError(t, c.Get(t.Context(), workloadKey, dcd))
						require.NotContains(t, dcd.Spec.PodTemplate.Annotations, capKey)
						deploymentTemplate = dcd.Spec.PodTemplate.DeepCopy()
						dcd.Status.Component = &v1beta1.ComponentReplicaStatus{}
						require.NoError(t, c.Status().Update(t.Context(), dcd))
					} else {
						pcs := &grovev1alpha1.PodCliqueSet{}
						workloadKey = client.ObjectKeyFromObject(old.PCS)
						require.NoError(t, c.Get(t.Context(), workloadKey, pcs))
						require.NotContains(t, pcs.Spec.Template.Cliques[0].Annotations, capKey)
						require.Equal(t, newHash, pcs.Spec.Template.Cliques[0].Labels[consts.KubeLabelDynamoWorkerHash])
						groveTemplate = pcs.Spec.Template.DeepCopy()
					}

					t.Log("After upgrade lowering, a cap-only edit keeps the same worker identity")
					require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(dgd), dgd))
					dgd.Status = result.Status
					require.NoError(t, c.Status().Update(t.Context(), dgd))
					dgd.Spec.Components[0].PodTemplate.Annotations[capKey] = "350"
					require.NoError(t, c.Update(t.Context(), dgd))
					result, err = program.Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
					require.NoError(t, err)
					desiredWatts := result.Status.Components["worker"].GPUPowerLimitWatts
					require.NotNil(t, desiredWatts)
					require.Equal(t, int64(350), *desiredWatts)
					require.Equal(t, newHash, betaDGDWorkersSpecHash(t, dgd))
					if old.DCD != nil {
						dcd := &v1beta1.DynamoComponentDeployment{}
						require.NoError(t, c.Get(t.Context(), workloadKey, dcd))
						require.Equal(t, deploymentTemplate, dcd.Spec.PodTemplate)
					} else {
						pcs := &grovev1alpha1.PodCliqueSet{}
						require.NoError(t, c.Get(t.Context(), workloadKey, pcs))
						require.Equal(t, groveTemplate, &pcs.Spec.Template)
					}
				})
			}
		}
	}
}

func TestPowerCapEditDoesNotRecaptureCheckpoint(t *testing.T) {
	t.Log("Capture an automatic checkpoint for a capped worker")
	ctx := context.Background()
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        fake.NewClientBuilder().WithScheme(newDynamoGraphDeploymentControllerTestScheme(t)).Build(),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Checkpoint: true}},
	}
	const capKey = consts.KubeAnnotationGPUPowerLimit
	dgd := &v1beta1.DynamoGraphDeployment{
		Name:        "test-dgd",
		Namespace:   "default",
		UID:         types.UID("dgd-uid"),
		Annotations: map[string]string{consts.KubeAnnotationWorkloadProvider: consts.WorkloadProviderComponent},
		Spec: v1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: string(dynamo.BackendFrameworkVLLM),
			Components: []v1beta1.DynamoComponentDeploymentSharedSpec{{
				ComponentName: "worker",
				ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{Annotations: map[string]string{capKey: "300"}},
					Spec: corev1.PodSpec{
						Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "worker:latest"}},
					},
				},
				Experimental: &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{
					Enabled: true,
					Mode:    v1beta1.CheckpointModeAuto,
				}},
			}},
		},
	}
	workerHash := betaDGDWorkersSpecHash(t, dgd)
	dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = workerHash
	_, err := newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	require.NoError(t, err)
	jobs := &snapshotv1alpha1.SnapshotJobList{}
	require.NoError(t, reconciler.List(ctx, jobs, client.InNamespace("default")))
	require.Len(t, jobs.Items, 1)
	job := jobs.Items[0].DeepCopy()

	t.Log("A cap edit keeps the worker hash and does not start another capture")
	dgd.Spec.Components[0].PodTemplate.Annotations[capKey] = "350"
	require.Equal(t, workerHash, betaDGDWorkersSpecHash(t, dgd))
	_, err = newTestDGDCheckpointsReconciler(reconciler).Reconcile(ctx, dgd)
	require.NoError(t, err)
	require.NoError(t, reconciler.List(ctx, jobs, client.InNamespace("default")))
	require.Len(t, jobs.Items, 1)
	require.Equal(t, job.Name, jobs.Items[0].Name)
	require.Equal(t, job.Spec, jobs.Items[0].Spec)
}
