//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

func TestRetuneTargetStatusThroughGroveController(t *testing.T) {
	t.Log("Start production admission and the DGD controller without a Grove operator or running Pods")
	env := operatorenv.New(operatorenv.Options{
		RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}},
		Admission:     operatorenv.AdmissionWebhooks{Mutating: true, Validating: true},
		SetupWebhooks: setupProductionWebhooks,
	}).RunT(t)
	config := env.OperatorConfig().DeepCopy()
	config.Namespace.Restricted = env.Namespace()
	env.StartManager(func(mgr ctrl.Manager) error {
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{Config: config, RuntimeConfig: env.RuntimeConfig()})
	})

	t.Log("Request Retune with type-selected frontend and Planner names and an existing-claim reference")
	source := &v1beta1.DynamoGraphDeployment{Name: "retune-target", Namespace: env.Namespace(), Spec: v1beta1.DynamoGraphDeploymentSpec{
		BackendFramework: "vllm",
		Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "gateway", ComponentType: v1beta1.ComponentTypeFrontend,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/frontend:1.8.0"}}}}},
			{ComponentName: "optimizer", ComponentType: v1beta1.ComponentTypePlanner,
				Retune:      &v1beta1.RetuneSpec{Enabled: true, TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "traces"}, Router: &v1beta1.RetuneRouterSpec{}},
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/planner:1.8.0"}}}}},
			{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/worker:1.8.0"}}}}},
		},
	}}
	require.NoError(t, env.Client().Create(t.Context(), source))
	require.Equal(t, &v1beta1.RetuneRouterSpec{OverlapScoreCredit: new(float64(1)), PrefillLoadScale: new(float64(1)), Temperature: new(float64(0))}, source.GetComponentByName("optimizer").Retune.Router)
	key := client.ObjectKeyFromObject(source)
	pcs := &grovev1alpha1.PodCliqueSet{}
	pcsKey := client.ObjectKey{Name: dynamo.PCSNameForDGD(source, nil), Namespace: source.Namespace}

	t.Log("The real provider reconciliation persists target capability while readiness is still false")
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		if !assert.NoError(c, env.Client().Get(t.Context(), key, source)) || !assert.NoError(c, env.Client().Get(t.Context(), pcsKey, pcs)) {
			return
		}
		assert.Equal(c, source.Generation, source.Status.ObservedGeneration)
		assert.Equal(c, v1beta1.DGDStatePending, source.Status.State)
		ready := meta.FindStatusCondition(source.Status.Conditions, "Ready")
		if assert.NotNil(c, ready) {
			assert.Equal(c, metav1.ConditionFalse, ready.Status)
		}
		_, configuration, err := dynamo.RetuneConfiguration(source)
		if assert.NoError(c, err) {
			assert.Equal(c, &v1beta1.RetuneStatus{ConfigHash: configuration.ConfigHash}, source.Status.Components["optimizer"].Retune)
		}
	}, 20*time.Second, 50*time.Millisecond)

	t.Log("The persisted Grove target contains the owned wiring, not mutations to the authored DGD")
	pcsUID, pcsGeneration := pcs.UID, pcs.Generation
	_, configuration, err := dynamo.RetuneConfiguration(source)
	require.NoError(t, err)
	capabilityJSON, err := json.Marshal(configuration)
	require.NoError(t, err)
	require.Len(t, pcs.Spec.Template.Cliques, 3)
	for _, clique := range pcs.Spec.Template.Cliques {
		component := clique.Labels[consts.KubeLabelDynamoComponent]
		if component != "gateway" && component != "optimizer" {
			continue
		}
		main := dynamo.GetMainContainerFromPodSpec(&clique.Spec.PodSpec)
		require.NotNil(t, main)
		require.Contains(t, main.VolumeMounts, corev1.VolumeMount{Name: "dynamo-retune-traces", MountPath: "/trace"})
		if component == "gateway" {
			require.Contains(t, main.Env, corev1.EnvVar{Name: "DYN_ENABLE_FRONTEND_ADMIN_API", Value: "1"})
			require.Contains(t, main.Env, corev1.EnvVar{Name: "DYN_FRONTEND_RETUNE_CAPABILITY", Value: string(capabilityJSON)})
		} else {
			require.Contains(t, main.Env, corev1.EnvVar{Name: "DYN_PLANNER_RETUNE_CAPABILITY", Value: string(capabilityJSON)})
		}
	}
	require.Empty(t, source.GetComponentByName("gateway").PodTemplate.Spec.Containers[0].Env)
	pods := &corev1.PodList{}
	require.NoError(t, env.Client().List(t.Context(), pods, client.InNamespace(env.Namespace())))
	require.Empty(t, pods.Items)

	t.Log("Remove Retune intent and observe capability and generated bindings disappear in the next target")
	require.NoError(t, retry.RetryOnConflict(retry.DefaultRetry, func() error {
		if err := env.Client().Get(t.Context(), key, source); err != nil {
			return err
		}
		source.GetComponentByName("optimizer").Retune = nil
		return env.Client().Update(t.Context(), source)
	}))
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		if !assert.NoError(c, env.Client().Get(t.Context(), key, source)) || !assert.NoError(c, env.Client().Get(t.Context(), pcsKey, pcs)) {
			return
		}
		assert.Equal(c, source.Generation, source.Status.ObservedGeneration)
		assert.Nil(c, source.Status.Components["optimizer"].Retune)
		assert.Equal(c, pcsUID, pcs.UID)
		assert.Greater(c, pcs.Generation, pcsGeneration)
		for _, clique := range pcs.Spec.Template.Cliques {
			if clique.Labels[consts.KubeLabelDynamoComponent] == "gateway" || clique.Labels[consts.KubeLabelDynamoComponent] == "optimizer" {
				main := dynamo.GetMainContainerFromPodSpec(&clique.Spec.PodSpec)
				if !assert.NotNil(c, main) {
					continue
				}
				assert.NotContains(c, main.VolumeMounts, corev1.VolumeMount{Name: "dynamo-retune-traces", MountPath: "/trace"})
				assert.Nil(c, findEnv(main.Env, "DYN_ENABLE_FRONTEND_ADMIN_API"))
				assert.Nil(c, findEnv(main.Env, "DYN_REQUEST_TRACE_FILE_PATH"))
				assert.Nil(c, findEnv(main.Env, "DYN_PLANNER_RETUNE_CAPABILITY"))
				assert.Nil(c, findEnv(main.Env, "DYN_FRONTEND_RETUNE_CAPABILITY"))
			}
		}
	}, 20*time.Second, 50*time.Millisecond)
}

func TestRetuneTargetStatusWaitsForDCDLowering(t *testing.T) {
	t.Log("Run the graph controller without a child controller so target acknowledgments are independent of readiness")
	env := operatorenv.New(operatorenv.Options{
		RuntimeConfig: &commoncontroller.RuntimeConfig{},
		Admission:     operatorenv.AdmissionWebhooks{Mutating: true, Validating: true},
		SetupWebhooks: setupProductionWebhooks,
	}).RunT(t)
	config := env.OperatorConfig().DeepCopy()
	config.Namespace.Restricted = env.Namespace()
	env.StartManager(func(mgr ctrl.Manager) error {
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{Config: config, RuntimeConfig: env.RuntimeConfig()})
	})
	source := &v1beta1.DynamoGraphDeployment{Name: "retune-children", Namespace: env.Namespace(), Spec: v1beta1.DynamoGraphDeploymentSpec{
		BackendFramework: "vllm",
		Components: []v1beta1.DynamoComponentDeploymentSharedSpec{
			{ComponentName: "gateway", ComponentType: v1beta1.ComponentTypeFrontend,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/frontend:1.8.0"}}}}},
			{ComponentName: "optimizer", ComponentType: v1beta1.ComponentTypePlanner,
				Retune:      &v1beta1.RetuneSpec{Enabled: true, TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "traces"}},
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/planner:1.8.0"}}}}},
			{ComponentName: "worker", ComponentType: v1beta1.ComponentTypeWorker,
				PodTemplate: &corev1.PodTemplateSpec{Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "example/worker:1.8.0"}}}}},
		},
	}}
	require.NoError(t, env.Client().Create(t.Context(), source))
	key := client.ObjectKeyFromObject(source)
	children := &v1beta1.DynamoComponentDeploymentList{}

	t.Log("Applying child specs acknowledges the graph generation but not the Retune target")
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		if !assert.NoError(c, env.Client().Get(t.Context(), key, source)) || !assert.NoError(c, env.Client().List(t.Context(), children, client.InNamespace(env.Namespace()))) {
			return
		}
		assert.Equal(c, source.Generation, source.Status.ObservedGeneration)
		assert.Len(c, children.Items, 3)
		assert.Nil(c, source.Status.Components["optimizer"].Retune)
	}, 20*time.Second, 50*time.Millisecond)

	t.Log("Acknowledge the two relevant child lowerings while leaving all Pods and the worker unready")
	for i := range children.Items {
		child := &children.Items[i]
		if child.Spec.ComponentName != "gateway" && child.Spec.ComponentName != "optimizer" {
			continue
		}
		require.NoError(t, retry.RetryOnConflict(retry.DefaultRetry, func() error {
			if err := env.Client().Get(t.Context(), client.ObjectKeyFromObject(child), child); err != nil {
				return err
			}
			child.Status.ObservedGeneration = child.Generation
			child.Status.Component = &v1beta1.ComponentReplicaStatus{ComponentKind: v1beta1.ComponentKindDeployment}
			return env.Client().Status().Update(t.Context(), child)
		}))
	}
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		if !assert.NoError(c, env.Client().Get(t.Context(), key, source)) {
			return
		}
		_, configuration, err := dynamo.RetuneConfiguration(source)
		if assert.NoError(c, err) {
			assert.Equal(c, &v1beta1.RetuneStatus{ConfigHash: configuration.ConfigHash}, source.Status.Components["optimizer"].Retune)
		}
		assert.Equal(c, v1beta1.DGDStatePending, source.Status.State)
	}, 20*time.Second, 50*time.Millisecond)
}
