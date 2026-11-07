//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"net/url"
	"strconv"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	lpxv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo/lpx/scheduler/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	grovecommon "github.com/ai-dynamo/grove/operator/api/common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
)

func TestLPXPriorityPublicationAndDeadlineRetryThroughSetup(t *testing.T) {
	t.Log("Start production DGD and LPX controllers with a test-owned compiler build")
	buildDir := newLPXEnvtestBuild(t)
	config := &configv1alpha1.OperatorConfiguration{}
	config.LPX.Enabled = true
	config.MPI.SSHSecretName = "ssh-secret"
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LPX: true}}
	env := operatorenv.New(operatorenv.Options{
		Config: config, RuntimeConfig: runtimeConfig,
		Admission: operatorenv.AdmissionWebhooks{Mutating: true, Validating: true},
		SetupWebhooks: func(mgr ctrl.Manager, opts operatorenv.WebhookSetupOptions) error {
			if err := lpxv1alpha1.AddToScheme(mgr.GetScheme()); err != nil {
				return err
			}
			return setupProductionWebhooks(mgr, opts)
		},
	}).RunT(t)
	lprCRD := &apiextensionsv1.CustomResourceDefinition{
		Name: "lpupipelinerequests." + lpxv1alpha1.APIGroup,
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: lpxv1alpha1.APIGroup, Scope: apiextensionsv1.NamespaceScoped,
			Names: apiextensionsv1.CustomResourceDefinitionNames{Kind: "LpuPipelineRequest", ListKind: "LpuPipelineRequestList", Plural: "lpupipelinerequests"},
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{{
				Name: "v1alpha1", Served: true, Storage: true,
				Schema:       &apiextensionsv1.CustomResourceValidation{OpenAPIV3Schema: &apiextensionsv1.JSONSchemaProps{Type: "object", XPreserveUnknownFields: new(true)}},
				Subresources: &apiextensionsv1.CustomResourceSubresources{Status: &apiextensionsv1.CustomResourceSubresourceStatus{}},
			}},
		},
	}
	_, err := envtest.InstallCRDs(env.RESTConfig(), envtest.CRDInstallOptions{CRDs: []*apiextensionsv1.CustomResourceDefinition{lprCRD}})
	require.NoError(t, err)
	// The shared minimal Grove fixtures omit subresources. Enable status for
	// actor observations and scale for the controller's real capacity writes.
	for _, name := range []string{"podcliquesets.grove.io", "podcliquescalinggroups.grove.io", "podcliques.grove.io"} {
		crd := &apiextensionsv1.CustomResourceDefinition{}
		require.NoError(t, env.Client().Get(t.Context(), client.ObjectKey{Name: name}, crd))
		for i := range crd.Spec.Versions {
			crd.Spec.Versions[i].Subresources = &apiextensionsv1.CustomResourceSubresources{Status: &apiextensionsv1.CustomResourceSubresourceStatus{}}
			if name == "podcliquescalinggroups.grove.io" {
				crd.Spec.Versions[i].Subresources.Scale = &apiextensionsv1.CustomResourceSubresourceScale{
					SpecReplicasPath: ".spec.replicas", StatusReplicasPath: ".status.replicas",
				}
			}
		}
		require.NoError(t, env.Client().Update(t.Context(), crd))
	}
	config = env.OperatorConfig().DeepCopy()
	config.Namespace.Restricted = env.Namespace()
	env.StartManager(func(mgr ctrl.Manager) error {
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{Config: config, RuntimeConfig: runtimeConfig})
	})

	t.Log("Create authored DGD intent; only Dynamo may create the PCS and LPR payloads")
	source := newLPXHandoffSource(t, "node-local-v2-lpu-only")
	source.Namespace, source.UID, source.Generation = env.Namespace(), "", 0
	source.Spec.Components[0].Replicas = new(int32(2))
	source.Spec.Components[0].LPX.BuildID = (&url.URL{Scheme: "file", Path: buildDir}).String()
	source.Spec.Components[0].LPX.Scheduling = &v1beta1.SchedulingSpec{PriorityClassName: new("old"), AttemptDeadlineSeconds: new(int64(30))}
	require.NoError(t, env.Client().Create(t.Context(), source))
	key := client.ObjectKeyFromObject(source)
	child := &v1alpha1.LPXGraphDeployment{}
	pcs := &grovev1alpha1.PodCliqueSet{}
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		if !assert.NoError(c, env.Client().Get(t.Context(), key, child)) {
			return
		}
		assert.NoError(c, env.Client().Get(t.Context(), client.ObjectKey{Namespace: source.Namespace, Name: dynamo.PCSNameForLPX(child)}, pcs))
	}, 20*time.Second, 50*time.Millisecond)

	t.Log("Supply declared Grove observations; this test does not claim Grove actuation or LPX scheduling")
	pcs.Status.ObservedGeneration = new(pcs.Generation)
	pcs.Status.CurrentGenerationHash = new("fixture-generation")
	require.NoError(t, env.Client().Status().Update(t.Context(), pcs))
	groupTemplate := pcs.Spec.Template.PodCliqueScalingGroupConfigs[0]
	groupName := grovecommon.GeneratePodCliqueScalingGroupName(grovecommon.ResourceNameReplica{Name: pcs.Name, Replica: 0}, groupTemplate.Name)
	group := &grovev1alpha1.PodCliqueScalingGroup{
		Name: groupName, Namespace: pcs.Namespace,
		Labels:          grovecommon.GetDefaultLabelsForPodCliqueSetManagedResources(pcs.Name),
		Annotations:     groupTemplate.Annotations,
		OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(pcs, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueSet"))},
		Spec:            grovev1alpha1.PodCliqueScalingGroupSpec{Replicas: 2, MinAvailable: groupTemplate.MinAvailable, CliqueNames: groupTemplate.CliqueNames},
	}
	group.Labels[grovecommon.LabelPartOfKey] = pcs.Name
	group.Labels[grovecommon.LabelPodCliqueSetReplicaIndex] = "0"
	require.NoError(t, env.Client().Create(t.Context(), group))
	group.Status.ObservedGeneration = new(group.Generation)
	group.Status.CurrentPodCliqueSetGenerationHash = pcs.Status.CurrentGenerationHash
	require.NoError(t, env.Client().Status().Update(t.Context(), group))
	for replica := range 2 {
		parent := grovecommon.ResourceNameReplica{Name: group.Name, Replica: replica}
		for _, template := range pcs.Spec.Template.Cliques {
			clique := &grovev1alpha1.PodClique{
				Name: grovecommon.GeneratePodCliqueName(parent, template.Name), Namespace: pcs.Namespace,
				Labels: template.DeepCopy().Labels, Annotations: template.DeepCopy().Annotations,
				OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(group, grovev1alpha1.SchemeGroupVersion.WithKind("PodCliqueScalingGroup"))},
				Spec:            template.DeepCopy().Spec,
			}
			if clique.Labels == nil {
				clique.Labels = make(map[string]string)
			}
			clique.Labels[grovecommon.LabelPartOfKey] = pcs.Name
			clique.Labels[grovecommon.LabelPodCliqueSetReplicaIndex] = "0"
			clique.Labels[grovecommon.LabelPodCliqueScalingGroup] = group.Name
			clique.Labels[grovecommon.LabelPodCliqueScalingGroupReplicaIndex] = strconv.Itoa(replica)
			for i, dependency := range clique.Spec.StartsAfter {
				clique.Spec.StartsAfter[i] = grovecommon.GeneratePodCliqueName(parent, dependency)
			}
			require.NoError(t, env.Client().Create(t.Context(), clique))
			clique.Status.ObservedGeneration = new(clique.Generation)
			clique.Status.CurrentPodCliqueSetGenerationHash = pcs.Status.CurrentGenerationHash
			clique.Status.CurrentPodTemplateHash = new(template.Name + "-fixture-hash")
			require.NoError(t, env.Client().Status().Update(t.Context(), clique))
		}
	}
	requests := &lpxv1alpha1.LPUPipelineRequestList{}
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		assert.NoError(c, env.Client().List(t.Context(), requests, client.InNamespace(source.Namespace)))
		assert.Len(c, requests.Items, 2)
	}, 20*time.Second, 50*time.Millisecond)
	byReplica := make(map[int64]*lpxv1alpha1.LPUPipelineRequest)
	for i := range requests.Items {
		request := requests.Items[i].DeepCopy()
		require.Equal(t, new("old"), request.Spec.PriorityClassName)
		require.NotEmpty(t, request.UID)
		byReplica[request.Spec.MaterializationTarget.PodCliqueScalingGroupRef.ReplicaIndex] = request
	}
	initialTemplate, pcsUID := pcs.Spec.Template.DeepCopy(), pcs.UID

	t.Log("Persist scheduler observations for a healthy survivor and an expired scale-out")
	survivor, failed := byReplica[0], byReplica[1]
	survivor.Status = &lpxv1alpha1.LPUPipelineRequestStatus{Phase: lpxv1alpha1.RequestPhaseBound, ObservedGeneration: new(survivor.Generation)}
	require.NoError(t, env.Client().Status().Update(t.Context(), survivor))
	started := metav1.NewTime(time.Now().Add(-time.Hour))
	failed.Finalizers = []string{"test.example/scheduler-release"}
	require.NoError(t, env.Client().Update(t.Context(), failed))
	failed.Status = &lpxv1alpha1.LPUPipelineRequestStatus{Phase: lpxv1alpha1.RequestPhasePending, ObservedGeneration: new(failed.Generation), SchedulingStartedAt: &started}
	require.NoError(t, env.Client().Status().Update(t.Context(), failed))
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		assert.NoError(c, env.Client().Get(t.Context(), key, child))
		assert.True(c, meta.IsStatusConditionTrue(child.Status.Conditions, "SchedulingFailed"))
		assert.NoError(c, env.Client().Get(t.Context(), client.ObjectKeyFromObject(failed), failed))
		assert.False(c, failed.DeletionTimestamp.IsZero(), "child conditions: %+v", child.Status.Conditions)
		assert.NoError(c, env.Client().Get(t.Context(), client.ObjectKeyFromObject(group), group))
		assert.Equal(c, int32(1), group.Spec.Replicas)
	}, 20*time.Second, 50*time.Millisecond)
	failure := meta.FindStatusCondition(child.Status.Conditions, "SchedulingFailed").DeepCopy()
	failedUID := failed.UID
	require.NoError(t, env.Client().Get(t.Context(), client.ObjectKeyFromObject(survivor), survivor))
	beforeSurvivor := survivor.DeepCopy()

	t.Log("A priority-only API edit advances the real handoff but waits for the existing release barrier")
	require.NoError(t, retry.RetryOnConflict(retry.DefaultRetry, func() error {
		if err := env.Client().Get(t.Context(), key, source); err != nil {
			return err
		}
		source.Spec.Components[0].LPX.Scheduling.PriorityClassName = new("new")
		return env.Client().Update(t.Context(), source)
	}))
	revision, err := dynamo.LPXInputRevision(source, "")
	require.NoError(t, err)
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		assert.NoError(c, env.Client().Get(t.Context(), key, child))
		assert.Equal(c, revision, child.Spec.InputRevision)
		assert.Greater(c, child.Generation, failure.ObservedGeneration)
		ready := meta.FindStatusCondition(child.Status.Conditions, "Ready")
		if assert.NotNil(c, ready) {
			assert.Equal(c, child.Generation, ready.ObservedGeneration)
		}
	}, 20*time.Second, 50*time.Millisecond)
	retryCondition := meta.FindStatusCondition(child.Status.Conditions, "SchedulingFailed")
	require.NotNil(t, retryCondition)
	require.Equal(t, metav1.ConditionFalse, retryCondition.Status)
	require.Equal(t, "LPXSchedulingRetryAuthorized", retryCondition.Reason)
	require.Equal(t, child.Generation, retryCondition.ObservedGeneration)
	require.NoError(t, env.Client().Get(t.Context(), client.ObjectKeyFromObject(group), group))
	require.Equal(t, int32(1), group.Spec.Replicas)
	require.NoError(t, env.Client().Get(t.Context(), client.ObjectKeyFromObject(failed), failed))
	require.Equal(t, failedUID, failed.UID)
	require.Equal(t, new("old"), failed.Spec.PriorityClassName)
	require.False(t, failed.DeletionTimestamp.IsZero())

	t.Log("Ordinary finalization permits restored capacity and a new-class replacement only")
	failed.Finalizers = nil
	require.NoError(t, env.Client().Update(t.Context(), failed))
	replacement := &lpxv1alpha1.LPUPipelineRequest{}
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		if !assert.NoError(c, env.Client().Get(t.Context(), client.ObjectKeyFromObject(failed), replacement)) {
			return
		}
		assert.NotEqual(c, failedUID, replacement.UID)
		assert.Equal(c, new("new"), replacement.Spec.PriorityClassName)
		assert.True(c, replacement.DeletionTimestamp.IsZero())
		assert.NoError(c, env.Client().Get(t.Context(), client.ObjectKeyFromObject(group), group))
		assert.Equal(c, int32(2), group.Spec.Replicas)
	}, 20*time.Second, 50*time.Millisecond)
	require.NoError(t, env.Client().Get(t.Context(), client.ObjectKeyFromObject(survivor), survivor))
	require.Equal(t, beforeSurvivor, survivor)
	require.NoError(t, env.Client().Get(t.Context(), client.ObjectKeyFromObject(pcs), pcs))
	require.Equal(t, pcsUID, pcs.UID)
	require.Equal(t, initialTemplate, pcs.Spec.Template.DeepCopy())
}
