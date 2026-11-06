// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"errors"
	"fmt"
	"maps"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller/power"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
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
	return fake.NewClientBuilder().WithScheme(scheme).WithIndex(&corev1.Pod{}, dynamo.ComponentPodIndex, dynamo.ComponentPodIndexValues)
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

func TestPowerCapComponentPreservation(t *testing.T) {
	for _, key := range []string{consts.AnnotationCurrentWorkerHash, consts.AnnotationCurrentWorkerHashV2} {
		t.Run(key, func(t *testing.T) { testPowerCapComponentPreservation(t, key) })
	}
}

func testPowerCapComponentPreservation(t *testing.T, hashKey string) {
	t.Log("Render and store one active worker generation at 300 W")
	dgd := powerCapTestDGD(t, "component")
	hash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	dgd.Annotations[hashKey] = hash
	dcds, err := dynamo.GenerateDynamoComponentsDeployments(dgd, nil, nil, dynamo.RollingUpdateContext{NewWorkerHash: hash})
	require.NoError(t, err)
	dcd := dcds["worker"]
	dcd.UID = "dcd-uid"
	dcd.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}
	c := powerCapTestClient(powerCapTestScheme(t)).WithObjects(dgd, dcd).Build()
	renderer := newDCDWorkloadRenderer(c, &configv1alpha1.OperatorConfiguration{}, &commoncontroller.RuntimeConfig{}, &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }})
	template, err := renderer.generatePodTemplateSpec(context.Background(), dcd, dynamo.RoleMain, renderer.containerGPUCount(context.Background(), dcd))
	require.NoError(t, err)
	deployment := &appsv1.Deployment{Name: dcd.Name, Namespace: dcd.Namespace, UID: "deployment-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dcd, v1beta1.DynamoComponentDeploymentGVK)}, Spec: appsv1.DeploymentSpec{Template: *template}}
	require.NoError(t, c.Create(context.Background(), deployment))
	workloads := newComponentWorkloadsReconciler(c, nil, newDGDWorkerRolloutReconciler(c, nil))
	t.Log("An unchanged graph needs no exception on an operator-only upgrade")
	unchanged, err := workloads.resolvePowerCapPreservation(t.Context(), dgd, nil)
	require.NoError(t, err)
	require.Nil(t, unchanged)

	for _, cap := range []string{"350", "325"} {
		t.Log("Cap-only edits and replica scale-up preserve the observed identity and creation cap")
		dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = cap
		dgd.Spec.Components[0].Replicas = new(int32(3))
		before := dgd.DeepCopy()
		retained, err := workloads.resolvePowerCapPreservation(context.Background(), dgd, nil)
		require.NoError(t, err)
		require.NotNil(t, retained)
		require.Equal(t, hash, retained.hash)
		require.Equal(t, "300", retained.caps["worker"])
		require.Equal(t, cap, dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit])
		require.Equal(t, before, dgd)
	}

	t.Log("The production DCD writer scales the retained generation at its creation cap")
	retained, err := workloads.resolvePowerCapPreservation(t.Context(), dgd, nil)
	require.NoError(t, err)
	_, err = workloads.Reconcile(t.Context(), dgd, nil, nil, retained)
	require.NoError(t, err)
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(dcd), dcd))
	require.Equal(t, int32(3), *dcd.Spec.Replicas)
	require.Equal(t, "300", dcd.Spec.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit])
	require.Equal(t, hash, dgd.Annotations[hashKey])

	t.Log("An unrelated worker edit follows ordinary rollout instead")
	dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Image = "different:v2"
	retained, err = workloads.resolvePowerCapPreservation(context.Background(), dgd, nil)
	require.NoError(t, err)
	require.Nil(t, retained)
	if hashKey == consts.AnnotationCurrentWorkerHash {
		return // Legacy hash migration precedes the ordinary rollout writer in the full program.
	}
	_, err = workloads.Reconcile(t.Context(), dgd, nil, nil, nil)
	require.NoError(t, err)
	newHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	rolled := &v1beta1.DynamoComponentDeployment{}
	require.NoError(t, c.Get(t.Context(), client.ObjectKey{Name: dynamo.GetDCDResourceName(dgd, "worker", newHash), Namespace: dgd.Namespace}, rolled))
	require.Equal(t, "325", rolled.Spec.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit])
}

func TestPowerCapGrovePreservation(t *testing.T) {
	for _, suffixed := range []bool{false, true} {
		for _, strategy := range []string{"", "Coherent", "RollingRecreate", "OnDelete"} {
			t.Run(fmt.Sprintf("suffix=%t/strategy=%s", suffixed, strategy), func(t *testing.T) {
				t.Log("Store the actual Grove render at the active worker identity")
				dgd := powerCapTestDGD(t, "grove")
				if strategy != "" {
					dgd.Annotations[consts.KubeAnnotationGroveUpdateStrategy] = strategy
				}
				frontend := *dgd.Spec.Components[0].DeepCopy()
				frontend.ComponentName, frontend.ComponentType = "frontend", "frontend"
				delete(frontend.PodTemplate.Annotations, consts.KubeAnnotationGPUPowerLimit)
				dgd.Spec.Components = append(dgd.Spec.Components, frontend)
				secondWorker := *dgd.Spec.Components[0].DeepCopy()
				secondWorker.ComponentName = "second-worker"
				dgd.Spec.Components = append(dgd.Spec.Components, secondWorker)
				hash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
				require.NoError(t, err)
				dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = hash
				c := powerCapTestClient(powerCapTestScheme(t)).WithRESTMapper(groveScaleRESTMapper()).WithInterceptorFuncs(groveScaleInterceptor(interceptor.Funcs{}, nil)).WithObjects(dgd).Build()
				config := &configv1alpha1.OperatorConfiguration{}
				runtimeConfig := &commoncontroller.RuntimeConfig{}
				secrets := &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}
				pcs, err := dynamo.GenerateGrovePodCliqueSet(context.Background(), dgd, nil, config, runtimeConfig, c, secrets, nil, nil, suffixed, nil, nil)
				require.NoError(t, err)
				// Store the pinned real webhook defaults rather than the raw renderer result.
				selected := grovev1alpha1.UpdateStrategyType(strategy)
				if selected == "" {
					selected = grovev1alpha1.RollingRecreateStrategy
				}
				pcs.Spec.UpdateStrategy = &grovev1alpha1.PodCliqueSetUpdateStrategy{Type: selected}
				if pcs.Spec.Template.TerminationDelay == nil {
					pcs.Spec.Template.TerminationDelay = &metav1.Duration{Duration: 4 * time.Hour}
				}
				if pcs.Spec.Template.HeadlessServiceConfig == nil {
					pcs.Spec.Template.HeadlessServiceConfig = &grovev1alpha1.HeadlessServiceConfig{PublishNotReadyAddresses: true}
				}
				if selected != grovev1alpha1.OnDeleteStrategy {
					for _, clique := range pcs.Spec.Template.Cliques {
						maxUnavailable := int32(1)
						if selected == grovev1alpha1.CoherentStrategy && clique.Spec.MinAvailable != nil {
							maxUnavailable = *clique.Spec.MinAvailable
						}
						clique.RollingUpdate = &grovev1alpha1.RollingUpdateConfiguration{MaxUnavailable: &maxUnavailable}
					}
				}
				pcs.UID = "pcs-uid"
				pcs.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}
				require.NoError(t, c.Create(context.Background(), pcs))
				renderer := newGroveWorkloadRenderer(c, config, runtimeConfig, secrets)
				req := groveReconcileRequest{DGD: dgd}

				for _, cap := range []string{"350", "325"} {
					t.Log("Repeated edits preserve all names, runtime bindings and retained caps")
					dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = cap
					dgd.Spec.Components[0].Replicas = new(int32(3))
					beforeDGD, beforePCS := dgd.DeepCopy(), pcs.DeepCopy()
					rendered, err := renderer.Render(context.Background(), req, nil, nil, true)
					require.NoError(t, err)
					require.True(t, rendered.powerCapPreserved)
					require.Equal(t, pcs.Spec.Template.Cliques[0].Name, rendered.desired.Spec.Template.Cliques[0].Name)
					require.Equal(t, pcs.Spec.Template.Cliques[0].Labels, rendered.desired.Spec.Template.Cliques[0].Labels)
					require.Equal(t, pcs.Spec.Template.Cliques[0].Spec.PodSpec.Containers[0].Env, rendered.desired.Spec.Template.Cliques[0].Spec.PodSpec.Containers[0].Env)
					require.Equal(t, "300", rendered.desired.Spec.Template.Cliques[0].Annotations[consts.KubeAnnotationGPUPowerLimit])
					require.Equal(t, cap, dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit])
					require.Equal(t, beforeDGD, dgd)
					require.Equal(t, beforePCS, pcs)
				}

				t.Log("A rollout-policy edit keeps preservation and still reaches the written PCS")
				nextStrategy := grovev1alpha1.OnDeleteStrategy
				if strategy == string(grovev1alpha1.OnDeleteStrategy) {
					nextStrategy = grovev1alpha1.CoherentStrategy
				}
				previousAnnotations := maps.Clone(dgd.Annotations)
				dgd.Annotations[consts.KubeAnnotationGroveUpdateStrategy] = string(nextStrategy)
				rendered, err := renderer.Render(context.Background(), req, nil, nil, true)
				require.NoError(t, err)
				require.True(t, rendered.powerCapPreserved)
				require.Equal(t, nextStrategy, rendered.desired.Spec.UpdateStrategy.Type)
				require.Equal(t, "300", rendered.desired.Spec.Template.Cliques[0].Annotations[consts.KubeAnnotationGPUPowerLimit])
				dgd.Annotations = previousAnnotations

				t.Log("An independently reconciled frontend image edit keeps worker identity and retained cap")
				dgd.Spec.Components[1].PodTemplate.Spec.Containers[0].Image = "frontend:v2"
				rendered, err = renderer.Render(context.Background(), req, nil, nil, true)
				require.NoError(t, err)
				require.True(t, rendered.powerCapPreserved)
				require.Equal(t, pcs.Spec.Template.Cliques[0].Name, rendered.desired.Spec.Template.Cliques[0].Name)
				require.Equal(t, "frontend:v2", podCliqueSetCliqueForComponent(rendered.desired, "frontend").Spec.PodSpec.Containers[0].Image)
				if !suffixed && strategy == "" {
					t.Log("The real PCS writer and scale-subresource path retain creation caps while scaling")
					workloads := newGroveWorkloadsReconciler(c, nil, newDGDWorkerRolloutReconciler(c, nil), config, runtimeConfig, secrets)
					synced, _, err := workloads.reconcilePodCliqueSet(t.Context(), dgd, rendered)
					require.NoError(t, err)
					require.Equal(t, "300", podCliqueSetCliqueForComponent(synced, "worker").Annotations[consts.KubeAnnotationGPUPowerLimit])
					child := &grovev1alpha1.PodClique{Name: dynamo.GroveComponentResourceName(pcs.Name, "worker"), Namespace: dgd.Namespace, UID: "scale-clique-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}, Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1}}
					require.NoError(t, c.Create(t.Context(), child))
					_, err = newGroveScaler(c).Reconcile(t.Context(), req, nil, false)
					require.NoError(t, err)
					require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(child), child))
					require.Equal(t, int32(3), child.Spec.Replicas)
					require.Equal(t, hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
				}

				t.Log("A genuine worker change stops preservation")
				originalImage := dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Image
				dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Image = "different:v2"
				rendered, err = renderer.Render(context.Background(), req, nil, nil, true)
				require.NoError(t, err)
				require.False(t, rendered.powerCapPreserved)
				dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Image = originalImage
				t.Log("Already-updated creation caps and absent workloads use normal identity reconciliation")
				require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
				podCliqueSetCliqueForComponent(pcs, "worker").Annotations[consts.KubeAnnotationGPUPowerLimit] = "325"
				require.NoError(t, c.Update(t.Context(), pcs))
				rendered, err = renderer.Render(t.Context(), req, nil, nil, true)
				require.NoError(t, err)
				require.False(t, rendered.powerCapPreserved)
				require.NoError(t, c.Delete(t.Context(), pcs))
				rendered, err = renderer.Render(t.Context(), req, nil, nil, true)
				require.NoError(t, err)
				require.False(t, rendered.powerCapPreserved)
			})
		}
	}
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
				podReads, podPatches := 0, 0
				c := powerCapTestClient(scheme).WithObjects(append(owners, dgd, pod)...).WithStatusSubresource(dgd).WithInterceptorFuncs(interceptor.Funcs{
					List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
						if _, ok := list.(*corev1.PodList); ok {
							podReads++
						}
						return c.List(ctx, list, opts...)
					},
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

				t.Log("The program neither observes nor repairs Pods, and returns the condition unchanged")
				switch scenario {
				case "success", "absent":
					require.NoError(t, err)
				case "terminal":
					require.True(t, errors.Is(err, reconcile.TerminalError(nil)))
				default:
					require.Error(t, err)
				}
				require.Zero(t, podPatches)
				if provider == "grove" {
					require.Zero(t, podReads)
				}
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

func TestPowerCapGroveScalingGroupDefaults(t *testing.T) {
	for _, strategy := range []grovev1alpha1.UpdateStrategyType{grovev1alpha1.CoherentStrategy, grovev1alpha1.RollingRecreateStrategy, grovev1alpha1.OnDeleteStrategy} {
		t.Run(string(strategy), func(t *testing.T) {
			t.Log("Compare explicit webhook-defaulted scaling groups")
			source := &grovev1alpha1.PodCliqueSet{Spec: grovev1alpha1.PodCliqueSetSpec{UpdateStrategy: &grovev1alpha1.PodCliqueSetUpdateStrategy{Type: strategy}, Template: grovev1alpha1.PodCliqueSetTemplateSpec{
				Cliques:                      []*grovev1alpha1.PodCliqueTemplateSpec{{Name: "worker", Labels: map[string]string{consts.KubeLabelDynamoComponentType: "worker"}, Spec: grovev1alpha1.PodCliqueSpec{Replicas: 2}}},
				PodCliqueScalingGroupConfigs: []grovev1alpha1.PodCliqueScalingGroupConfig{{Name: "group", CliqueNames: []string{"worker"}, Replicas: new(int32(3)), MinAvailable: new(int32(2)), ScaleConfig: &grovev1alpha1.AutoScalingConfig{}}},
			}}}
			before := source.DeepCopy()
			observed := source.DeepCopy()
			observed.Spec.Template.PodCliqueScalingGroupConfigs[0].ScaleConfig.MinReplicas = new(int32(3))
			if strategy != grovev1alpha1.OnDeleteStrategy {
				value := int32(1)
				if strategy == grovev1alpha1.CoherentStrategy {
					value = 2
				}
				observed.Spec.Template.PodCliqueScalingGroupConfigs[0].RollingUpdate = &grovev1alpha1.RollingUpdateConfiguration{MaxUnavailable: &value}
			}
			require.Equal(t, grovePowerCapWorkerSpec(source), grovePowerCapWorkerSpec(observed))
			require.Equal(t, before, source)

			t.Log("Rollout policy edits are excluded like the worker hash excludes them")
			observed.Spec.UpdateStrategy = &grovev1alpha1.PodCliqueSetUpdateStrategy{Type: grovev1alpha1.OnDeleteStrategy}
			observed.Spec.Template.PodCliqueScalingGroupConfigs[0].RollingUpdate = &grovev1alpha1.RollingUpdateConfiguration{MaxUnavailable: new(int32(7))}
			observed.Spec.Template.Cliques[0].RollingUpdate = &grovev1alpha1.RollingUpdateConfiguration{MaxUnavailable: new(int32(5))}
			require.Equal(t, grovePowerCapWorkerSpec(source), grovePowerCapWorkerSpec(observed))
			require.NotNil(t, observed.Spec.UpdateStrategy, "only the comparison copy drops rollout policy")

			t.Log("MinAvailable is not treated as rollout policy and its drift stays visible")
			observed.Spec.Template.PodCliqueScalingGroupConfigs[0].MinAvailable = new(int32(1))
			require.NotEqual(t, grovePowerCapWorkerSpec(source), grovePowerCapWorkerSpec(observed))
		})
	}
}

func TestPowerCapComponentPreservationBoundaries(t *testing.T) {
	for _, scenario := range []string{"missing-dcd", "missing-deployment", "already-updated", "mixed-worker-drift"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("Observe a complete two-worker generation before testing preservation boundaries")
			dgd := powerCapTestDGD(t, "component")
			second := *dgd.Spec.Components[0].DeepCopy()
			second.ComponentName = "second"
			dgd.Spec.Components = append(dgd.Spec.Components, second)
			hash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
			require.NoError(t, err)
			dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = hash
			generated, err := dynamo.GenerateDynamoComponentsDeployments(dgd, nil, nil, dynamo.RollingUpdateContext{NewWorkerHash: hash})
			require.NoError(t, err)
			c := powerCapTestClient(powerCapTestScheme(t)).WithObjects(dgd).Build()
			renderer := newDCDWorkloadRenderer(c, &configv1alpha1.OperatorConfiguration{}, &commoncontroller.RuntimeConfig{}, &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }})
			deployments := map[string]*appsv1.Deployment{}
			for name, dcd := range generated {
				dcd.UID = types.UID(name + "-uid")
				dcd.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}
				require.NoError(t, c.Create(t.Context(), dcd))
				template, err := renderer.generatePodTemplateSpec(t.Context(), dcd, dynamo.RoleMain, renderer.containerGPUCount(t.Context(), dcd))
				require.NoError(t, err)
				dep := &appsv1.Deployment{Name: dcd.Name, Namespace: dcd.Namespace, UID: types.UID(name + "-deployment-uid"), OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dcd, v1beta1.DynamoComponentDeploymentGVK)}, Spec: appsv1.DeploymentSpec{Template: *template}}
				require.NoError(t, c.Create(t.Context(), dep))
				deployments[name] = dep
			}
			dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
			workloads := newComponentWorkloadsReconciler(c, nil, newDGDWorkerRolloutReconciler(c, nil))
			preserved, err := workloads.resolvePowerCapPreservation(t.Context(), dgd, nil)
			require.NoError(t, err)
			require.NotNil(t, preserved, "an unchanged second worker must not prevent cap-only preservation")
			switch scenario {
			case "missing-dcd":
				require.NoError(t, c.Delete(t.Context(), generated["worker"]))
			case "missing-deployment":
				require.NoError(t, c.Delete(t.Context(), deployments["worker"]))
			case "already-updated":
				generated["worker"].Spec.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
				deployments["worker"].Spec.Template.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
				require.NoError(t, c.Update(t.Context(), generated["worker"]))
				require.NoError(t, c.Update(t.Context(), deployments["worker"]))
			case "mixed-worker-drift":
				dgd.Spec.Components[1].PodTemplate.Spec.Containers[0].Image = "second:v2"
			}
			before := dgd.DeepCopy()
			preserved, err = workloads.resolvePowerCapPreservation(t.Context(), dgd, nil)
			require.NoError(t, err)
			require.Nil(t, preserved, "missing resources, already-current templates or any worker drift must use normal identity reconciliation")
			require.Equal(t, before, dgd)
		})
	}
}

func TestPowerCapComponentPreservationComparesHashedInputs(t *testing.T) {
	for _, scenario := range []string{"cap-only", "legacy-backend-framework", "subtype-change"} {
		t.Run(scenario, func(t *testing.T) {
			t.Log("Store one active component worker generation at 300 W with its owned Deployment")
			dgd := powerCapTestDGD(t, "component")
			hash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
			require.NoError(t, err)
			dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = hash
			generated, err := dynamo.GenerateDynamoComponentsDeployments(dgd, nil, nil, dynamo.RollingUpdateContext{NewWorkerHash: hash})
			require.NoError(t, err)
			dcd := generated["worker"]
			dcd.UID = "dcd-uid"
			dcd.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}
			if scenario == "legacy-backend-framework" {
				t.Log("An older operator stored a different backendFramework, which ordinary reconciliation keeps")
				dcd.Spec.BackendFramework = "sglang"
			}
			deployment := &appsv1.Deployment{Name: dcd.Name, Namespace: dcd.Namespace, UID: "deployment-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dcd, v1beta1.DynamoComponentDeploymentGVK)}}
			c := powerCapTestClient(powerCapTestScheme(t)).WithObjects(dgd, dcd, deployment).Build()
			workloads := newComponentWorkloadsReconciler(c, nil, newDGDWorkerRolloutReconciler(c, nil))

			t.Log("Edit the cap to 350 W, together with a v1alpha1 subtype in the subtype-change scenario")
			edited := dgd
			if scenario == "subtype-change" {
				edited = createTestDGD("power", map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
					"worker": {ComponentType: consts.ComponentTypeWorker, SubComponentType: "custom-prefill", Envs: []corev1.EnvVar{{Name: "WORKER_VERSION", Value: "v1"}}},
				})
				edited.UID, edited.Generation = dgd.UID, dgd.Generation
				maps.Copy(edited.Annotations, dgd.Annotations)
				edited.Spec.Components[0].PodTemplate.Annotations = map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}
				changed, err := dynamo.ComputeDGDWorkersSpecHash(edited)
				require.NoError(t, err)
				require.NotEqual(t, hash, changed, "the subtype must be a hashed worker input")
			}
			edited.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
			retained, err := workloads.resolvePowerCapPreservation(t.Context(), edited, nil)
			require.NoError(t, err)

			t.Log("Only a change confined to caps keeps the recorded generation")
			if scenario == "subtype-change" {
				require.Nil(t, retained)
				return
			}
			require.NotNil(t, retained)
			require.Equal(t, hash, retained.hash)
			require.Equal(t, "300", retained.caps["worker"])
		})
	}
}
