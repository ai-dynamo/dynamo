// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"context"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
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
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
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

			t.Log("A cap-only edit is pending: no template write and no hash commit")
			dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
			result, err := root.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
			require.Equal(t, pendingVersion, pcs.ResourceVersion)
			require.Equal(t, "300", podCliqueSetCliqueForComponent(pcs, "worker").Annotations[consts.KubeAnnotationGPUPowerLimit])
			require.Equal(t, hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
			require.Nil(t, meta.FindStatusCondition(result.Status.Conditions, power.ConditionType))

			t.Log("After Grove acknowledges and finishes, the retained identity stays without a template write")
			pcs.Status.ObservedGeneration = new(pcs.Generation)
			pcs.Status.UpdateProgress = nil
			require.NoError(t, c.Status().Update(t.Context(), pcs))
			acknowledgedVersion := pcs.ResourceVersion
			result, err = root.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
			require.NoError(t, err)
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
			require.Equal(t, acknowledgedVersion, pcs.ResourceVersion)
			require.Equal(t, "300", podCliqueSetCliqueForComponent(pcs, "worker").Annotations[consts.KubeAnnotationGPUPowerLimit])
			require.Equal(t, hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])

			t.Log("Component status reports the desired cap the power controller propagates, not the retained template's")
			require.Equal(t, new(int64(350)), result.Status.Components["worker"].GPUPowerLimitWatts)
		})
	}
}

func TestPowerCapComponentRestartAppliesToRetainedWorkers(t *testing.T) {
	for _, scenario := range []string{"restart-with-cap-edit", "cap-edit-after-restart"} {
		t.Run(scenario, func(t *testing.T) {
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
			dcd.Status.Conditions = []metav1.Condition{{Type: v1beta1.DynamoComponentDeploymentConditionTypeAvailable, Status: metav1.ConditionTrue, Reason: "Available", LastTransitionTime: metav1.Now()}}
			// The DCD controller reports the cap its retained template renders.
			dcd.Status.Component = &v1beta1.ComponentReplicaStatus{Replicas: 1, ReadyReplicas: new(int32(1)), GPUPowerLimitWatts: new(int64(300))}
			if scenario == "cap-edit-after-restart" {
				t.Log("An earlier restart completed and left its stamp on the worker DCD")
				dcd.Spec.PodTemplate.Annotations[consts.RestartAnnotation] = "restart-0"
				dgd.Spec.Restart = &v1beta1.Restart{ID: "restart-0"}
				dgd.Status.Restart = &v1beta1.RestartStatus{ObservedID: "restart-0", Phase: v1beta1.RestartPhaseCompleted}
			}
			deployment := &appsv1.Deployment{Name: dcd.Name, Namespace: dcd.Namespace, UID: "deployment-uid", OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dcd, v1beta1.DynamoComponentDeploymentGVK)}}
			scheme := powerCapTestScheme(t)
			require.NoError(t, rbacv1.AddToScheme(scheme))
			c := powerCapTestClient(scheme).WithObjects(dgd, dcd, deployment).WithStatusSubresource(dgd).Build()
			config := &configv1alpha1.OperatorConfiguration{}
			config.Namespace.Restricted = "default"
			root := &DynamoGraphDeploymentReconciler{Client: c, Config: config, RuntimeConfig: &commoncontroller.RuntimeConfig{}, DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}}
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(dcd), dcd))
			stamp := dcd.Spec.PodTemplate.Annotations[consts.RestartAnnotation]

			t.Log("Store an edit of only the cap, with a new restart in the restart-with-cap-edit scenario")
			key := client.ObjectKeyFromObject(dgd)
			require.NoError(t, c.Get(t.Context(), key, dgd))
			dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
			if scenario == "restart-with-cap-edit" {
				dgd.Spec.Restart = &v1beta1.Restart{ID: "restart-1"}
			}
			require.NoError(t, c.Update(t.Context(), dgd))

			t.Log("Reconcile through convergence, persisting the returned status and reloading the DGD between passes")
			for range 3 {
				require.NoError(t, c.Get(t.Context(), key, dgd))
				result, err := root.newComponentProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
				require.NoError(t, err)
				dgd.Status = result.Status
				require.NoError(t, c.Status().Update(t.Context(), dgd))
				require.False(t, rollingUpdateInProgress(dgd.Status.RollingUpdate))
				require.NotEqual(t, v1beta1.RestartPhaseSuperseded, dgd.Status.Restart.Phase)
			}
			require.NoError(t, c.Get(t.Context(), key, dgd))

			t.Log("The retained worker keeps its identity, creation cap and restart stamp, and the restart completes")
			require.Equal(t, hash, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
			dcdList := &v1beta1.DynamoComponentDeploymentList{}
			require.NoError(t, c.List(t.Context(), dcdList))
			require.Len(t, dcdList.Items, 1)
			require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(dcd), dcd))
			require.Equal(t, "300", dcd.Spec.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit])
			require.NotEmpty(t, dcd.Spec.PodTemplate.Annotations[consts.RestartAnnotation])
			if scenario == "cap-edit-after-restart" {
				require.Equal(t, stamp, dcd.Spec.PodTemplate.Annotations[consts.RestartAnnotation])
			}
			require.Equal(t, v1beta1.RestartPhaseCompleted, dgd.Status.Restart.Phase)

			t.Log("Component status reports the desired cap the power controller propagates, not the retained template's")
			require.Equal(t, new(int64(350)), dgd.Status.Components["worker"].GPUPowerLimitWatts)
		})
	}
}

func TestPowerCapGroveRoleTemplateWorkerHasNoCap(t *testing.T) {
	t.Log("Observe a capped worker clique beside an uncapped role-template worker clique")
	dgd := powerCapTestDGD(t, "grove")
	roleWorker := *dgd.Spec.Components[0].DeepCopy()
	roleWorker.ComponentName, roleWorker.PodTemplate = "role-worker", nil
	dgd.Spec.Components = append(dgd.Spec.Components, roleWorker)
	active, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = active
	workerLabels := map[string]string{consts.KubeLabelDynamoComponent: "worker", consts.KubeLabelDynamoComponentType: "worker", consts.KubeLabelDynamoWorkerHash: active}
	roleWorkerLabels := map[string]string{consts.KubeLabelDynamoComponent: "role-worker", consts.KubeLabelDynamoComponentType: "worker", consts.KubeLabelDynamoWorkerHash: active}
	pcs := &grovev1alpha1.PodCliqueSet{Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
		{Name: "worker", Labels: workerLabels, Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}},
		{Name: "role-worker", Labels: roleWorkerLabels},
	}}}}
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"

	t.Log("The template-less worker has no explicit cap and does not block the cap-only exception")
	retained, applicable, err := retainedGrovePowerCaps(groveReconcileRequest{DGD: dgd}, pcs)
	require.NoError(t, err)
	require.True(t, applicable)
	require.Equal(t, active, retained.Hash)
	require.Equal(t, map[string]string{"worker": "300"}, retained.PowerCaps)

	t.Log("A cap on that worker's clique without a source cap is still drift")
	pcs.Spec.Template.Cliques[1].Annotations = map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}
	_, applicable, err = retainedGrovePowerCaps(groveReconcileRequest{DGD: dgd}, pcs)
	require.NoError(t, err)
	require.False(t, applicable)
}

func TestPowerCapGroveUnsuffixedPodCliqueSetRequiresRecordedIdentity(t *testing.T) {
	for _, tc := range []struct {
		name           string
		recorded       bool
		changeRuntime  bool
		wantApplicable bool
	}{
		{name: "cap-only edit", recorded: true, wantApplicable: true},
		{name: "cap edit with a hash-only runtime version change", recorded: true, changeRuntime: true},
		{name: "no recorded hash"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Record a custom-image worker at 300 W whose legacy unsuffixed clique labels no worker hash")
			dgd := powerCapTestDGD(t, "grove")
			worker := &dgd.Spec.Components[0]
			worker.PodTemplate.Spec.Containers[0].Image = "registry.example/custom-runtime:dev"
			worker.RuntimeVersionOverride = "1.5.0"
			recorded, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
			require.NoError(t, err)
			if tc.recorded {
				dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = recorded
			}
			pcs := &grovev1alpha1.PodCliqueSet{Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{
				{Name: "worker", Labels: map[string]string{consts.KubeLabelDynamoComponent: "worker", consts.KubeLabelDynamoComponentType: "worker"}, Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}},
			}}}}

			t.Log("Edit the cap to 350 W, optionally with a change the worker hash covers")
			worker.PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
			if tc.changeRuntime {
				worker.RuntimeVersionOverride = "1.6.0"
			}

			t.Log("Only an edit whose capped identity equals the recorded hash keeps the old PodCliqueSet")
			retained, applicable, err := retainedGrovePowerCaps(groveReconcileRequest{DGD: dgd}, pcs)
			require.NoError(t, err)
			require.Equal(t, tc.wantApplicable, applicable)
			require.Empty(t, retained.Hash)
		})
	}
}

func TestPowerCapGroveRetainsGenerationAwaitingHashCommit(t *testing.T) {
	t.Log("Let the production Grove program commit worker hash A at 300 W")
	dgd := powerCapTestDGD(t, "grove")
	scheme := powerCapTestScheme(t)
	require.NoError(t, rbacv1.AddToScheme(scheme))
	c := powerCapTestClient(scheme).WithRESTMapper(groveScaleRESTMapper()).WithInterceptorFuncs(groveScaleInterceptor(interceptor.Funcs{}, nil)).WithObjects(dgd).WithStatusSubresource(dgd).Build()
	config := &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"}, Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendEtcd}}
	root := &DynamoGraphDeploymentReconciler{Client: c, Config: config, RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}}, DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}}
	for range 2 {
		// The first pass writes the PCS; the next observes it and commits the worker hash.
		_, err := root.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
		require.NoError(t, err)
	}
	hashA := dgd.Annotations[consts.AnnotationCurrentWorkerHashV2]
	require.NotEmpty(t, hashA)

	t.Log("An image update writes generation B; its hash commit waits for the next pass")
	dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Image = "different:v2"
	_, err := root.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
	require.NoError(t, err)
	sets := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, c.List(t.Context(), sets))
	require.Len(t, sets.Items, 1)
	pcs := &sets.Items[0]
	hashB := podCliqueSetCliqueForComponent(pcs, "worker").Labels[consts.KubeLabelDynamoWorkerHash]
	require.NotEqual(t, hashA, hashB)
	require.Equal(t, hashA, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])

	t.Log("A cap-only edit before that commit retains B at its 300 W creation cap instead of writing a third generation")
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	for range 2 {
		_, err = root.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
		require.NoError(t, err)
		require.NoError(t, c.Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
		require.Equal(t, hashB, podCliqueSetCliqueForComponent(pcs, "worker").Labels[consts.KubeLabelDynamoWorkerHash])
		require.Equal(t, "300", podCliqueSetCliqueForComponent(pcs, "worker").Annotations[consts.KubeAnnotationGPUPowerLimit])
	}

	t.Log("The DGD records the generation its workers run")
	require.Equal(t, hashB, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
}

func TestPowerCapGroveFirstGenerationCommitEmitsNoRolloutWarning(t *testing.T) {
	t.Log("Let the production Grove program write a new graph's first PCS at 300 W; its hash commit waits for the next pass")
	dgd := powerCapTestDGD(t, "grove")
	scheme := powerCapTestScheme(t)
	require.NoError(t, rbacv1.AddToScheme(scheme))
	c := powerCapTestClient(scheme).WithRESTMapper(groveScaleRESTMapper()).WithInterceptorFuncs(groveScaleInterceptor(interceptor.Funcs{}, nil)).WithObjects(dgd).WithStatusSubresource(dgd).Build()
	config := &configv1alpha1.OperatorConfiguration{Namespace: configv1alpha1.NamespaceConfiguration{Restricted: "default"}, Discovery: configv1alpha1.DiscoveryConfiguration{Backend: configv1alpha1.DiscoveryBackendEtcd}}
	recorder := events.NewFakeRecorder(20)
	root := &DynamoGraphDeploymentReconciler{Client: c, Recorder: recorder, Config: config, RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}}, DockerSecretRetriever: &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }}}
	_, err := root.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
	require.NoError(t, err)
	require.Empty(t, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
	sets := &grovev1alpha1.PodCliqueSetList{}
	require.NoError(t, c.List(t.Context(), sets))
	require.Len(t, sets.Items, 1)
	first := podCliqueSetCliqueForComponent(&sets.Items[0], "worker").Labels[consts.KubeLabelDynamoWorkerHash]
	require.NotEmpty(t, first)

	t.Log("A cap-only edit arrives before that commit and retains the first generation")
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	_, err = root.newGroveProgram().Reconcile(t.Context(), workloadProgramRequest{DGD: dgd})
	require.NoError(t, err)

	t.Log("The first generation is recorded as a first deploy is, with no rolling-update warning")
	require.Equal(t, first, dgd.Annotations[consts.AnnotationCurrentWorkerHashV2])
	close(recorder.Events)
	for event := range recorder.Events {
		require.NotContains(t, event, "RollingUpdateNotSupported")
	}
}

func TestGrovePowerCapWorkerSpecNormalizesOnlyGroveDefaults(t *testing.T) {
	for _, tc := range []struct {
		name      string
		set       func(*corev1.PodSpec)
		wantEqual bool
	}{
		{name: "restartPolicy defaulted by Grove", set: func(spec *corev1.PodSpec) { spec.RestartPolicy = corev1.RestartPolicyAlways }, wantEqual: true},
		{name: "terminationGracePeriodSeconds defaulted by Grove", set: func(spec *corev1.PodSpec) { spec.TerminationGracePeriodSeconds = new(int64(30)) }, wantEqual: true},
		{name: "explicit dnsPolicy", set: func(spec *corev1.PodSpec) { spec.DNSPolicy = corev1.DNSClusterFirst }},
		{name: "explicit schedulerName", set: func(spec *corev1.PodSpec) { spec.SchedulerName = corev1.DefaultSchedulerName }},
		{name: "explicit imagePullPolicy", set: func(spec *corev1.PodSpec) { spec.Containers[0].ImagePullPolicy = corev1.PullIfNotPresent }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			t.Log("Compare a worker clique with a copy that sets one Pod field explicitly")
			labels := map[string]string{consts.KubeLabelDynamoComponent: "worker", consts.KubeLabelDynamoComponentType: "worker"}
			source := &grovev1alpha1.PodCliqueSet{Spec: grovev1alpha1.PodCliqueSetSpec{Template: grovev1alpha1.PodCliqueSetTemplateSpec{Cliques: []*grovev1alpha1.PodCliqueTemplateSpec{{
				Name: "worker", Labels: labels, Spec: grovev1alpha1.PodCliqueSpec{Replicas: 1, PodSpec: corev1.PodSpec{Containers: []corev1.Container{{Name: consts.MainContainerName, Image: "example/engine:1.6.0"}}}},
			}}}}}
			edited := source.DeepCopy()
			tc.set(&edited.Spec.Template.Cliques[0].Spec.PodSpec)

			t.Log("Only fields Grove's webhook defaults compare equal; other explicit fields stay visible as drift")
			require.Equal(t, tc.wantEqual, apiequality.Semantic.DeepEqual(grovePowerCapWorkerSpec(source), grovePowerCapWorkerSpec(edited)))
		})
	}
}
