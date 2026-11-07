//go:build !clustertest

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/stretchr/testify/require"
	appsv1 "k8s.io/api/apps/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestPowerCapComponentPreservationIgnoresAPIServerDefaults(t *testing.T) {
	t.Log("Store one 300 W worker DCD through the API server and admission webhooks")
	env := sharedEnv.ForTest(t)
	c := env.Client()
	dgd := powerCapTestDGD(t, "component")
	dgd.Namespace, dgd.UID = env.Namespace(), "power-dgd-uid"
	dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Image = "example/engine:1.6.0"
	hash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
	require.NoError(t, err)
	dgd.Annotations[consts.AnnotationCurrentWorkerHashV2] = hash
	generated, err := dynamo.GenerateDynamoComponentsDeployments(dgd, nil, nil, dynamo.RollingUpdateContext{NewWorkerHash: hash})
	require.NoError(t, err)
	dcd := generated["worker"]
	dcd.OwnerReferences = []metav1.OwnerReference{*metav1.NewControllerRef(dgd, v1beta1.DynamoGraphDeploymentGVK)}
	require.NoError(t, c.Create(t.Context(), dcd))

	t.Log("Store its Deployment with a Kubernetes-discovery service account the API server mirrors")
	config := &configv1alpha1.OperatorConfiguration{}
	config.Discovery.Backend = configv1alpha1.DiscoveryBackendKubernetes
	renderer := newDCDWorkloadRenderer(c, config, &commoncontroller.RuntimeConfig{}, &mockDockerSecretRetriever{GetSecretsFunc: func(string, string) ([]string, error) { return nil, nil }})
	template, err := renderer.generatePodTemplateSpec(t.Context(), dcd, dynamo.RoleMain, renderer.containerGPUCount(t.Context(), dcd))
	require.NoError(t, err)
	require.NotEmpty(t, template.Spec.ServiceAccountName)
	require.Empty(t, template.Spec.DeprecatedServiceAccount)
	deployment := &appsv1.Deployment{Name: dcd.Name, Namespace: dcd.Namespace, OwnerReferences: []metav1.OwnerReference{*metav1.NewControllerRef(dcd, v1beta1.DynamoComponentDeploymentGVK)}, Spec: appsv1.DeploymentSpec{Selector: &metav1.LabelSelector{MatchLabels: template.Labels}, Template: *template}}
	require.NoError(t, c.Create(t.Context(), deployment))
	require.Equal(t, template.Spec.ServiceAccountName, deployment.Spec.Template.Spec.DeprecatedServiceAccount)

	t.Log("A cap-only edit keeps the observed generation and its creation cap")
	dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	workloads := newComponentWorkloadsReconciler(c, nil, newDGDWorkerRolloutReconciler(c, nil))
	retained, err := workloads.resolvePowerCapPreservation(t.Context(), dgd, nil)
	require.NoError(t, err)
	require.NotNil(t, retained)
	require.Equal(t, hash, retained.hash)
	require.Equal(t, "300", retained.caps["worker"])
}
