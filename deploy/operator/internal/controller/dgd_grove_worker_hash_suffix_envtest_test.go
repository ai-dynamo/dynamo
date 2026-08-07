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
)

const (
	groveSuffixTestTimeout  = 10 * time.Second
	groveSuffixTestInterval = 100 * time.Millisecond
)

func TestGroveWorkerHashSuffixDefaultedForNewDGD(t *testing.T) {
	ctx := context.Background()
	env := newGroveWorkerHashSuffixTestEnv(t)
	startGroveWorkerHashSuffixTestController(t, env)

	dgd := newGroveWorkerHashSuffixTestDGD(env.Namespace(), "new-without-pcs")
	require.NoError(t, env.Client().Create(ctx, dgd))

	enabled := waitForGroveWorkerHashSuffixEnabled(t, ctx, env, dgd.Name)
	wantHash, err := dynamo.ComputeDGDWorkersSpecHash(enabled)
	require.NoError(t, err)
	waitForGroveWorkerHashSuffixes(t, ctx, env, enabled, wantHash)
}

func newGroveWorkerHashSuffixTestEnv(t *testing.T) *operatorenv.TestEnv {
	t.Helper()
	return operatorenv.New(operatorenv.Options{
		Admission:     operatorenv.AdmissionWebhooks{Mutating: true, Validating: true},
		SetupWebhooks: setupProductionWebhooks,
		RuntimeConfig: &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}},
	}).RunT(t)
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

func waitForGroveWorkerHashSuffixEnabled(
	t *testing.T,
	ctx context.Context,
	env *operatorenv.TestEnv,
	name string,
) *nvidiacomv1beta1.DynamoGraphDeployment {
	t.Helper()
	key := types.NamespacedName{Name: name, Namespace: env.Namespace()}
	dynamotesting.Eventually(t, func() (bool, string) {
		dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
		if err := env.Client().Get(ctx, key, dgd); err != nil {
			return false, fmt.Sprintf("get DGD: %v", err)
		}
		annotations := dgd.GetAnnotations()
		if annotations[consts.AnnotationGroveWorkerHashSuffixEnabled] != "true" {
			return false, "worker hash suffix is not enabled"
		}
		return true, "Grove worker hash suffix enabled"
	}, groveSuffixTestTimeout, groveSuffixTestInterval, "Grove worker hash suffix was not enabled")

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, env.Client().Get(ctx, key, dgd))
	return dgd
}

func getGroveWorkerHashSuffixTestPCS(
	t *testing.T,
	ctx context.Context,
	env *operatorenv.TestEnv,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) *grovev1alpha1.PodCliqueSet {
	t.Helper()
	key := types.NamespacedName{
		Name:      dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components),
		Namespace: dgd.Namespace,
	}
	pcs := &grovev1alpha1.PodCliqueSet{}
	dynamotesting.Eventually(t, func() (bool, string) {
		if err := env.Client().Get(ctx, key, pcs); err != nil {
			return false, fmt.Sprintf("get PodCliqueSet: %v", err)
		}
		return true, "PodCliqueSet exists"
	}, groveSuffixTestTimeout, groveSuffixTestInterval, "Grove PodCliqueSet was not created")
	return pcs
}

func waitForGroveWorkerHashSuffixes(
	t *testing.T,
	ctx context.Context,
	env *operatorenv.TestEnv,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
	want string,
) *grovev1alpha1.PodCliqueSet {
	t.Helper()
	key := types.NamespacedName{
		Name:      dynamo.PCSNameForDGD(dgd.Name, dgd.Spec.Components),
		Namespace: dgd.Namespace,
	}
	pcs := &grovev1alpha1.PodCliqueSet{}
	dynamotesting.Eventually(t, func() (bool, string) {
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
		return true, "Grove worker cliques have the expected suffix"
	}, groveSuffixTestTimeout, groveSuffixTestInterval, "Grove worker cliques did not receive the expected suffix")
	return pcs
}

func findMainContainer(containers []corev1.Container) *corev1.Container {
	for i := range containers {
		if containers[i].Name == consts.MainContainerName {
			return &containers[i]
		}
	}
	return nil
}
