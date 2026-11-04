// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"errors"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
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
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

func TestPowerCapGrovePendingUpdate(t *testing.T) {
	for _, unacknowledged := range []string{"missing-observed-generation", "stale-observed-generation", "update-in-progress"} {
		t.Run(unacknowledged, func(t *testing.T) {
			t.Log("Let the production Grove program create the active PCS at 300 W")
			dgd := powerCapTestDGD(t, "grove")
			scheme := powerCapTestScheme(t)
			require.NoError(t, rbacv1.AddToScheme(scheme))
			funcs := groveScaleInterceptor(interceptor.Funcs{}, nil)
			scaleUpdate := funcs.SubResourceUpdate
			funcs.SubResourceUpdate = func(ctx context.Context, writer client.Client, subResource string, obj client.Object, opts ...client.SubResourceUpdateOption) error {
				if subResource == "status" {
					return writer.Status().Update(ctx, obj, opts...)
				}
				return scaleUpdate(ctx, writer, subResource, obj, opts...)
			}
			c := powerCapTestClient(scheme).WithRESTMapper(groveScaleRESTMapper()).WithInterceptorFuncs(funcs).WithObjects(dgd).WithStatusSubresource(dgd, &grovev1alpha1.PodCliqueSet{}).Build()
			config := &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"}, Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendEtcd}}
			root := &DynamoGraphDeploymentReconciler{Client: c, Config: config, RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}}, DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}}
			for range 2 {
				// The first pass writes the PCS; the next observes it and commits the worker hash.
				_, err := root.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
				require.NoError(t, err)
			}
			hash := dgd.Annotations[consts.AnnotationCurrentWorkerHashV2]
			require.NotEmpty(t, hash)
			sets := &grovev1alpha1.PodCliqueSetList{}
			require.NoError(t, c.List(t.Context(), sets))
			require.Len(t, sets.Items, 1)
			pcs := &sets.Items[0]
			require.Equal(t, "300", podCliqueSetCliqueForComponent(pcs, "worker").Annotations[consts.KubeAnnotationGPUPowerLimit])

			t.Log("Grove has not acknowledged or finished the stored PCS update")
			switch unacknowledged {
			case "stale-observed-generation":
				pcs.Status.ObservedGeneration = new(pcs.Generation - 1)
			case "update-in-progress":
				pcs.Status.ObservedGeneration = new(pcs.Generation)
				pcs.Status.UpdateProgress = &grovev1alpha1.PodCliqueSetUpdateProgress{UpdateStartedAt: metav1.Now()}
			}
			require.NoError(t, c.Status().Update(t.Context(), pcs))
			clique := &grovev1alpha1.PodClique{Name: "clique", Namespace: dgd.Namespace, UID: "clique-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}}
			require.NoError(t, c.Create(t.Context(), clique))
			_, pod := powerCapTestOwners(dgd, "grove")
			pod.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(clique, grovev1alpha1.SchemeGroupVersion.WithKind("PodClique"))}
			require.NoError(t, c.Create(t.Context(), pod))
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
			pendingVersion := pcs.ResourceVersion

			t.Log("A cap-only edit is pending: no template write, no hash commit and no Pod repair")
			dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
			result, err := root.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
			require.Equal(t, pendingVersion, pcs.ResourceVersion)
			require.Equal(t, "300", podCliqueSetCliqueForComponent(pcs, "worker").Annotations[consts.KubeAnnotationGPUPowerLimit])
			require.Equal(t, hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pod), pod))
			require.Equal(t, "300", pod.Annotations[consts.KubeAnnotationGPUPowerLimit])
			condition := meta.FindStatusCondition(result.Status.Conditions, powerCapPropagationCondition)
			require.Equal(t, metav1.ConditionFalse, condition.Status)
			require.Equal(t, "WaitingForWorkerUpdate", condition.Reason)
			require.LessOrEqual(t, result.RequeueAfter, powerCapRetry)
			require.Positive(t, result.RequeueAfter)

			t.Log("After Grove acknowledges and finishes, the retained identity converges by Pod repair")
			pcs.Status.ObservedGeneration = new(pcs.Generation)
			pcs.Status.UpdateProgress = nil
			require.NoError(t, c.Status().Update(t.Context(), pcs))
			acknowledgedVersion := pcs.ResourceVersion
			for _, wantReason := range []string{"PodAnnotationMismatch", "PodAnnotationsMatch"} {
				result, err = root.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
				require.NoError(t, err)
				require.Equal(t, wantReason, meta.FindStatusCondition(result.Status.Conditions, powerCapPropagationCondition).Reason)
			}
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pod), pod))
			require.Equal(t, "350", pod.Annotations[consts.KubeAnnotationGPUPowerLimit])
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
			require.Equal(t, acknowledgedVersion, pcs.ResourceVersion)
			require.Equal(t, "300", podCliqueSetCliqueForComponent(pcs, "worker").Annotations[consts.KubeAnnotationGPUPowerLimit])
			require.Equal(t, hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
		})
	}
}

func TestPowerCapGrovePendingSurvivesPodCliqueSetSyncFailure(t *testing.T) {
	t.Log("Let the production Grove program create the active PCS at 300 W")
	dgd := powerCapTestDGD(t, "grove")
	scheme := powerCapTestScheme(t)
	require.NoError(t, rbacv1.AddToScheme(scheme))
	failPCSWrites, podReads, podPatches := false, 0, 0
	funcs := groveScaleInterceptor(interceptor.Funcs{
		Update: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.UpdateOption) error {
			if failPCSWrites && obj.GetObjectKind().GroupVersionKind().Kind == "PodCliqueSet" {
				return errors.New("PCS write denied")
			}
			return c.Update(ctx, obj, opts...)
		},
		Patch: func(ctx context.Context, c client.WithWatch, obj client.Object, patch client.Patch, opts ...client.PatchOption) error {
			if failPCSWrites && obj.GetObjectKind().GroupVersionKind().Kind == "PodCliqueSet" {
				return errors.New("PCS write denied")
			}
			if _, ok := obj.(*corev1.Pod); ok {
				podPatches++
			}
			return c.Patch(ctx, obj, patch, opts...)
		},
		List: func(ctx context.Context, c client.WithWatch, list client.ObjectList, opts ...client.ListOption) error {
			if _, ok := list.(*corev1.PodList); ok && failPCSWrites {
				podReads++
			}
			return c.List(ctx, list, opts...)
		},
	}, nil)
	c := powerCapTestClient(scheme).WithRESTMapper(groveScaleRESTMapper()).WithInterceptorFuncs(funcs).WithObjects(dgd).WithStatusSubresource(dgd).Build()
	config := &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"}, Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendEtcd}}
	root := &DynamoGraphDeploymentReconciler{Client: c, Config: config, RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}}, DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}}
	for range 2 {
		// The first pass writes the PCS; the next observes it and commits the worker hash.
		_, err := root.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
		require.NoError(t, err)
	}
	sets := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, c.List(t.Context(), sets))
	require.Len(t, sets.Items, 1)
	pcs := &sets.Items[0]
	require.Nil(t, pcs.Status.ObservedGeneration, "Grove has not acknowledged the stored PCS")
	clique := &grovev1alpha1.PodClique{Name: "clique", Namespace: dgd.Namespace, UID: "clique-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))}}
	require.NoError(t, c.Create(t.Context(), clique))
	_, pod := powerCapTestOwners(dgd, "grove")
	pod.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(clique, grovev1alpha1.SchemeGroupVersion.WithKind("PodClique"))}
	require.NoError(t, c.Create(t.Context(), pod))

	t.Log("A cap edit with a rollout-policy change needs a PCS write, and that write fails")
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	dgd.Annotations[consts.KubeAnnotationGroveUpdateStrategy] = string(grovev1alpha1.OnDeleteStrategy)
	failPCSWrites = true
	result, err := root.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})

	t.Log("The workload error remains while propagation keeps waiting without Pod reads or patches")
	require.ErrorContains(t, err, "failed to reconcile the Grove PodCliqueSet")
	require.Zero(t, podReads)
	require.Zero(t, podPatches)
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pod), pod))
	require.Equal(t, "300", pod.Annotations[consts.KubeAnnotationGPUPowerLimit])
	condition := meta.FindStatusCondition(result.Status.Conditions, powerCapPropagationCondition)
	require.Equal(t, metav1.ConditionFalse, condition.Status)
	require.Equal(t, "WaitingForWorkerUpdate", condition.Reason)
}

func TestPowerCapComponentRestartAppliesToRetainedWorkers(t *testing.T) {
	t.Log("Store the active component generation at 300 W with its owned Deployment")
	dgd := powerCapTestDGD(t, "component")
	hash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = hash
	dcds, err := dynamo.GenerateDynamoComponentsDeployments(dgd, nil, nil, dynamo.RollingUpdateContext{NewWorkerHash: hash})
	require.NoError(t, err)
	dcd := dcds["worker"]
	dcd.UID = "worker-uid"
	dcd.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}
	deployment := &appsv1.Deployment{Name: dcd.Name, Namespace: dcd.Namespace, UID: "deployment-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dcd, v1beta1.DynamoComponentDeploymentGVK)}}
	scheme := powerCapTestScheme(t)
	require.NoError(t, rbacv1.AddToScheme(scheme))
	c := powerCapTestClient(scheme).WithObjects(dgd, dcd, deployment).WithStatusSubresource(dgd).Build()
	config := &configv1alpha1.OperatorConfiguration{}
	config.Namespace.Restricted = "default"
	root := &DynamoGraphDeploymentReconciler{Client: c, Config: config, RuntimeConfig: &commoncontroller.RuntimeConfig{}, DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}}

	t.Log("Request a restart together with a cap-only edit")
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	dgd.Spec.Restart = &v1beta1.Restart{ID: "restart-1"}
	result, err := root.newComponentProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
	require.NoError(t, err)

	t.Log("The restart is not superseded and stamps the retained worker in place")
	require.False(t, rollingUpdateInProgress(result.Status.RollingUpdate))
	require.NotNil(t, result.Status.Restart)
	require.NotEqual(t, v1beta1.RestartPhaseSuperseded, result.Status.Restart.Phase)
	require.Equal(t, hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
	require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(dcd), dcd))
	require.Equal(t, "300", dcd.Spec.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit])
	require.NotEmpty(t, dcd.Spec.PodTemplate.Annotations[consts.RestartAnnotation])
}

func TestPowerCapGroveRoleTemplateWorkerHasNoCap(t *testing.T) {
	t.Log("Observe a capped worker clique beside an uncapped role-template worker clique")
	dgd := powerCapTestDGD(t, "grove")
	roleWorker := *dgd.Spec.Components[0].DeepCopy()
	roleWorker.ComponentName, roleWorker.PodTemplate = "role-worker", nil
	dgd.Spec.Components = append(dgd.Spec.Components, roleWorker)
	dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = "active"
	workerLabels := map[string]string{consts.KubeLabelDynamoComponent: "worker", consts.KubeLabelDynamoComponentType: "worker", consts.KubeLabelDynamoWorkerHash: "active"}
	roleWorkerLabels := map[string]string{consts.KubeLabelDynamoComponent: "role-worker", consts.KubeLabelDynamoComponentType: "worker", consts.KubeLabelDynamoWorkerHash: "active"}
	pcs := &grovev1alpha1.PodCliqueSet{Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
		{Name: "worker", Labels: workerLabels, Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}},
		{Name: "role-worker", Labels: roleWorkerLabels},
	}}}}
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"

	t.Log("The template-less worker has no explicit cap and does not block the cap-only exception")
	retained, applicable := retainedGrovePowerCaps(groveReconcileRequest{DGD: dgd}, pcs)
	require.True(t, applicable)
	require.Equal(t, "active", retained.Hash)
	require.Equal(t, map[string]string{"worker": "300"}, retained.PowerCaps)

	t.Log("A cap on that worker's clique without a source cap is still drift")
	pcs.Spec.Template.Cliques[1].Annotations = map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}
	_, applicable = retainedGrovePowerCaps(groveReconcileRequest{DGD: dgd}, pcs)
	require.False(t, applicable)
}
