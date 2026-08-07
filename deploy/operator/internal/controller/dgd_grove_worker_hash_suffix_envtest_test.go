//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	dynamotesting "github.com/ai-dynamo/dynamo/deploy/operator/internal/testing"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	groveSuffixTestTimeout  = 10 * time.Second
	groveSuffixTestInterval = 100 * time.Millisecond
	legacyGroveDGDSeeder    = "operatorenv-grove-legacy-seeder"
)

func TestGroveWorkerHashSuffixDefaultedForNewDGD(t *testing.T) {
	ctx := context.Background()
	env := newGroveWorkerHashSuffixTestEnv(t)
	startGroveWorkerHashSuffixTestController(t, env)

	dgd := newGroveWorkerHashSuffixTestDGD(env.Namespace(), "new-without-pcs")
	require.NoError(t, env.Client().Create(ctx, dgd))

	createdDGD := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, env.Client().Get(ctx, types.NamespacedName{Name: dgd.Name, Namespace: env.Namespace()}, createdDGD))
	require.Equal(t, createdDGD.GetAnnotations()[consts.AnnotationGroveWorkerHashSuffixEnabled], "true")
	wantHash, err := dynamo.ComputeDGDWorkersSpecHash(createdDGD)
	require.NoError(t, err)
	waitForGroveWorkerHashSuffixes(t, ctx, env, createdDGD, wantHash)
}

func TestGroveWorkerHashSuffixForExistingDGD(t *testing.T) {
	ctx := context.Background()
	env := newGroveWorkerHashSuffixTestEnv(t)
	dgd := newGroveWorkerHashSuffixTestDGD(env.Namespace(), "existing-without-suffix")
	createLegacyGroveWorkerHashSuffixTestDGD(t, ctx, env, dgd)

	startGroveWorkerHashSuffixTestController(t, env)
	waitForGroveWorkerHashSuffixes(t, ctx, env, dgd, "")

	current := &nvidiacomv1beta1.DynamoGraphDeployment{}
	key := types.NamespacedName{Name: dgd.Name, Namespace: dgd.Namespace}
	require.NoError(t, env.Client().Get(ctx, key, current))
	require.Empty(t, current.GetAnnotations()[consts.AnnotationGroveWorkerHashSuffixEnabled])

	frontend := current.GetComponentByName("frontend")
	frontend.PodTemplate.Spec.Containers[0].Env = append(frontend.PodTemplate.Spec.Containers[0].Env,
		corev1.EnvVar{Name: "FRONTEND_CONFIG_REVISION", Value: "2"})
	require.NoError(t, env.Client().Update(ctx, current))

	require.NoError(t, env.Client().Get(ctx, key, current))
	require.Empty(t, current.GetAnnotations()[consts.AnnotationGroveWorkerHashSuffixEnabled])
	waitForGroveFrontendEnv(t, ctx, env, current, "FRONTEND_CONFIG_REVISION", "2")
	checkGroveWorkerHashSuffixes(t, ctx, env, current, "")

	require.NoError(t, env.Client().Get(ctx, key, current))
	prefill := current.GetComponentByName("prefill")
	prefill.PodTemplate.Spec.Containers[0].Env[0].Value = "8192"
	require.NoError(t, env.Client().Update(ctx, current))
	require.NoError(t, env.Client().Get(ctx, key, current))
	require.Equal(t, current.GetAnnotations()[consts.AnnotationGroveWorkerHashSuffixEnabled], "true")

	wantHash, err := dynamo.ComputeDGDWorkersSpecHash(current)
	require.NoError(t, err)
	waitForGroveWorkerHashSuffixes(t, ctx, env, current, wantHash)
}

func newGroveWorkerHashSuffixTestEnv(t *testing.T) *operatorenv.TestEnv {
	t.Helper()
	return operatorenv.New(operatorenv.Options{
		Admission: operatorenv.AdmissionWebhooks{
			Mutating:            true,
			Validating:          true,
			MutatingBypassUsers: []string{legacyGroveDGDSeeder},
		},
		SetupWebhooks: setupProductionWebhooks,
		RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}},
	}).RunT(t)
}

func createLegacyGroveWorkerHashSuffixTestDGD(
	t *testing.T,
	ctx context.Context,
	env *operatorenv.TestEnv,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) {
	t.Helper()
	config := env.RESTConfig()
	config.Impersonate.UserName = legacyGroveDGDSeeder
	config.Impersonate.Groups = []string{"system:masters"}
	legacyClient, err := client.New(config, client.Options{Scheme: env.Client().Scheme()})
	require.NoError(t, err)
	require.NoError(t, legacyClient.Create(ctx, dgd))
}

func startGroveWorkerHashSuffixTestController(t *testing.T, env *operatorenv.TestEnv) {
	t.Helper()
	config := env.OperatorConfig().DeepCopy()
	config.Namespace.Restricted = env.Namespace()
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}}
	env.StartManager(func(mgr ctrl.Manager) error {
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{
			SetupOptions: SetupOptions{
				Config:        config,
				RuntimeConfig: runtimeConfig,
			},
		})
	})
}

func newGroveWorkerHashSuffixTestDGD(namespace, name string) *nvidiacomv1beta1.DynamoGraphDeployment {
	component := func(name string, componentType nvidiacomv1beta1.ComponentType, image string) nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec {
		return nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{
			ComponentName: name,
			ComponentType: componentType,
			PodTemplate: &corev1.PodTemplateSpec{
				Spec: corev1.PodSpec{Containers: []corev1.Container{{
					Name:  consts.MainContainerName,
					Image: image,
				}}},
			},
		}
	}
	prefill := component("prefill", nvidiacomv1beta1.ComponentTypePrefill, "registry.example/dynamo-worker:1.4.0")
	prefill.PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "MODEL_MAX_LEN", Value: "4096"}}
	decode := component("decode", nvidiacomv1beta1.ComponentTypeDecode, "registry.example/dynamo-worker:1.4.0")
	frontend := component("frontend", nvidiacomv1beta1.ComponentTypeFrontend, "registry.example/dynamo-frontend:1.4.0")

	return &nvidiacomv1beta1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: nvidiacomv1beta1.DynamoGraphDeploymentSpec{
			BackendFramework: "vllm",
			Components:       []nvidiacomv1beta1.DynamoComponentDeploymentSharedSpec{prefill, decode, frontend},
		},
	}
}

func waitForGroveWorkerHashSuffixes(
	t *testing.T,
	ctx context.Context,
	env *operatorenv.TestEnv,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	want string,
) {
	t.Helper()
	dynamotesting.Eventually(t, func() (bool, string) {
		return checkGroveWorkerHashSuffixes(t, ctx, env, dgd, want)
	}, groveSuffixTestTimeout, groveSuffixTestInterval, "Grove worker cliques did not reach the expected worker suffix state")
}

func checkGroveWorkerHashSuffixes(
	t *testing.T,
	ctx context.Context,
	env *operatorenv.TestEnv,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	want string,
) (bool, string) {
	t.Helper()
	key := types.NamespacedName{
		Name:      dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components),
		Namespace: dgd.Namespace,
	}
	pcs := &grovev1alpha1.PodCliqueSet{}
	if err := env.Client().Get(ctx, key, pcs); err != nil {
		return false, fmt.Sprintf("get PodCliqueSet: %v", err)
	}
	seenWorkers := map[string]bool{"prefill": false, "decode": false}
	for _, clique := range pcs.Spec.Template.Cliques {
		component := clique.Labels[consts.KubeLabelDynamoComponent]
		if component != "prefill" && component != "decode" {
			continue
		}
		seenWorkers[component] = true
		main := findMainContainer(clique.Spec.PodSpec.Containers)
		if main == nil {
			return false, fmt.Sprintf("%s clique %q has no main container", component, clique.Name)
		}
		suffix := findEnv(main.Env, consts.DynamoNamespaceWorkerSuffixEnvVar)
		if want == "" {
			if suffix != nil {
				return false, fmt.Sprintf("%s clique %q unexpectedly has worker suffix %q", component, clique.Name, suffix.Value)
			}
			continue
		}
		if suffix == nil {
			return false, fmt.Sprintf("%s clique %q has no worker suffix", component, clique.Name)
		}
		if suffix.Value != want {
			return false, fmt.Sprintf("%s clique %q suffix=%q, want %q", component, clique.Name, suffix.Value, want)
		}
	}
	for component, found := range seenWorkers {
		if !found {
			return false, fmt.Sprintf("PodCliqueSet has no %s clique", component)
		}
	}
	return true, "Grove worker cliques have the expected worker suffix state"
}

func waitForGroveFrontendEnv(
	t *testing.T,
	ctx context.Context,
	env *operatorenv.TestEnv,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	name, want string,
) {
	t.Helper()
	key := types.NamespacedName{
		Name:      dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components),
		Namespace: dgd.Namespace,
	}
	dynamotesting.Eventually(t, func() (bool, string) {
		pcs := &grovev1alpha1.PodCliqueSet{}
		if err := env.Client().Get(ctx, key, pcs); err != nil {
			return false, fmt.Sprintf("get PodCliqueSet: %v", err)
		}
		for _, clique := range pcs.Spec.Template.Cliques {
			if clique.Labels[consts.KubeLabelDynamoComponent] != "frontend" {
				continue
			}
			main := findMainContainer(clique.Spec.PodSpec.Containers)
			if main == nil {
				return false, fmt.Sprintf("frontend clique %q has no main container", clique.Name)
			}
			env := findEnv(main.Env, name)
			if env == nil || env.Value != want {
				return false, fmt.Sprintf("frontend clique %q env %s = %v, want %q", clique.Name, name, env, want)
			}
			return true, "frontend update reconciled"
		}
		return false, "PodCliqueSet has no frontend clique"
	}, groveSuffixTestTimeout, groveSuffixTestInterval, "frontend update was not reconciled")
}

func findMainContainer(containers []corev1.Container) *corev1.Container {
	for i := range containers {
		if containers[i].Name == consts.MainContainerName {
			return &containers[i]
		}
	}
	return nil
}
