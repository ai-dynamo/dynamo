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
	"k8s.io/client-go/util/retry"
	ctrl "sigs.k8s.io/controller-runtime"
)

const (
	groveSuffixTestTimeout  = 10 * time.Second
	groveSuffixTestInterval = 100 * time.Millisecond
)

func TestGroveWorkerHashSuffixAdoptsExistingPCSBeforeSuffixingWorkerUpdate(t *testing.T) {
	ctx := context.Background()
	env := sharedEnv.ForTest(t)
	dgd := newGroveWorkerHashSuffixTestDGD(env.Namespace(), "adopt-existing")
	require.NoError(t, env.Client().Create(ctx, dgd))

	created := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, env.Client().Get(ctx, clientKey(dgd), created))
	legacyPCS := renderGroveWorkerHashSuffixTestPCS(t, ctx, env, created)
	require.NoError(t, env.Client().Create(ctx, legacyPCS))

	startGroveWorkerHashSuffixTestController(t, env)

	adopted := waitForGroveWorkerHashSuffixAdoption(t, ctx, env, created.Name)
	adoptedGeneration := adopted.Annotations[consts.AnnotationGroveWorkerHashSuffixAdoptedGeneration]
	require.Equal(t, fmt.Sprint(created.Generation), adoptedGeneration)
	assertGroveWorkerHashSuffixes(t, getGroveWorkerHashSuffixTestPCS(t, ctx, env, adopted), "")

	updateGroveWorkerHashSuffixTestDGD(t, ctx, env, adopted.Name, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
		frontend := dgd.GetComponentByName("frontend")
		require.NotNil(t, frontend)
		frontend.PodTemplate.Spec.Containers[0].Env = append(frontend.PodTemplate.Spec.Containers[0].Env,
			corev1.EnvVar{Name: "FRONTEND_CONFIG_REVISION", Value: "2"})
	})
	stillAdopted := waitForGroveWorkerHashSuffixAdoption(t, ctx, env, created.Name)
	require.Equal(t, adoptedGeneration, stillAdopted.Annotations[consts.AnnotationGroveWorkerHashSuffixAdoptedGeneration])
	assertGroveWorkerHashSuffixes(t, getGroveWorkerHashSuffixTestPCS(t, ctx, env, stillAdopted), "")

	updateGroveWorkerHashSuffixTestDGD(t, ctx, env, stillAdopted.Name, func(dgd *nvidiacomv1beta1.DynamoGraphDeployment) {
		prefill := dgd.GetComponentByName("prefill")
		require.NotNil(t, prefill)
		prefill.PodTemplate.Spec.Containers[0].Env[0].Value = "8192"
	})

	enabled := waitForGroveWorkerHashSuffixEnabled(t, ctx, env, created.Name)
	wantHash, err := dynamo.ComputeDGDWorkersSpecHash(enabled)
	require.NoError(t, err)
	assertGroveWorkerHashSuffixes(t, waitForGroveWorkerHashSuffixes(t, ctx, env, enabled, wantHash), wantHash)
}

func TestGroveWorkerHashSuffixEnablesNewDGDWithoutPCS(t *testing.T) {
	ctx := context.Background()
	env := sharedEnv.ForTest(t)
	startGroveWorkerHashSuffixTestController(t, env)

	dgd := newGroveWorkerHashSuffixTestDGD(env.Namespace(), "new-without-pcs")
	require.NoError(t, env.Client().Create(ctx, dgd))

	enabled := waitForGroveWorkerHashSuffixEnabled(t, ctx, env, dgd.Name)
	wantHash, err := dynamo.ComputeDGDWorkersSpecHash(enabled)
	require.NoError(t, err)
	assertGroveWorkerHashSuffixes(t, waitForGroveWorkerHashSuffixes(t, ctx, env, enabled, wantHash), wantHash)
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

func updateGroveWorkerHashSuffixTestDGD(
	t *testing.T,
	ctx context.Context,
	env *operatorenv.TestEnv,
	name string,
	mutate func(*nvidiacomv1beta1.DynamoGraphDeployment),
) {
	t.Helper()
	key := types.NamespacedName{Name: name, Namespace: env.Namespace()}
	require.NoError(t, retry.RetryOnConflict(retry.DefaultBackoff, func() error {
		dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
		if err := env.Client().Get(ctx, key, dgd); err != nil {
			return err
		}
		mutate(dgd)
		return env.Client().Update(ctx, dgd)
	}))
}

func renderGroveWorkerHashSuffixTestPCS(
	t *testing.T,
	ctx context.Context,
	env *operatorenv.TestEnv,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) *grovev1alpha1.PodCliqueSet {
	t.Helper()
	rendered, err := groveRenderDeployment(dgd, nil)
	require.NoError(t, err)
	pcs, err := dynamo.GenerateGrovePodCliqueSet(
		ctx,
		rendered,
		env.OperatorConfig(),
		&commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true}},
		env.Client(),
		nil,
		nil,
		nil,
		nil,
	)
	require.NoError(t, err)
	return pcs
}

func waitForGroveWorkerHashSuffixAdoption(
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
		wantHash, err := dynamo.ComputeDGDWorkersSpecHash(dgd)
		if err != nil {
			return false, fmt.Sprintf("compute worker hash: %v", err)
		}
		annotations := dgd.GetAnnotations()
		if annotations[consts.AnnotationGroveWorkerHashSuffixEnabled] != "" {
			return false, "worker hash suffix was enabled instead of adopted"
		}
		if annotations[consts.AnnotationGroveWorkerHashSuffixAdoptedGeneration] == "" {
			return false, "adopted generation is not recorded"
		}
		if annotations[consts.AnnotationGroveWorkerHashSuffixAdoptedHashV2] != wantHash {
			return false, "adopted hash does not match the canonical worker hash"
		}
		return true, "Grove deployment adopted without suffixing workers"
	}, groveSuffixTestTimeout, groveSuffixTestInterval, "Grove deployment was not adopted")

	dgd := &nvidiacomv1beta1.DynamoGraphDeployment{}
	require.NoError(t, env.Client().Get(ctx, key, dgd))
	return dgd
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
		if annotations[consts.AnnotationGroveWorkerHashSuffixAdoptedGeneration] != "" ||
			annotations[consts.AnnotationGroveWorkerHashSuffixAdoptedHashV2] != "" {
			return false, "temporary adoption annotations remain after suffixing is enabled"
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

func assertGroveWorkerHashSuffixes(t *testing.T, pcs *grovev1alpha1.PodCliqueSet, want string) {
	t.Helper()
	seenWorkers := map[string]bool{"prefill": false, "decode": false}
	for _, clique := range pcs.Spec.Template.Cliques {
		component := clique.Labels[consts.KubeLabelDynamoComponent]
		main := findMainContainer(clique.Spec.PodSpec.Containers)
		require.NotNil(t, main, "clique %q has no main container", clique.Name)
		suffix := findEnv(main.Env, consts.DynamoNamespaceWorkerSuffixEnvVar)
		switch component {
		case "prefill", "decode":
			seenWorkers[component] = true
			if want == "" {
				require.Nil(t, suffix, "%s clique %q unexpectedly has a worker suffix", component, clique.Name)
				continue
			}
			require.NotNil(t, suffix, "%s clique %q has no worker suffix", component, clique.Name)
			require.Equal(t, want, suffix.Value, "%s clique %q has the wrong worker suffix", component, clique.Name)
		case "frontend":
			require.Nil(t, suffix, "frontend clique %q unexpectedly has a worker suffix", clique.Name)
		}
	}
	for component, found := range seenWorkers {
		require.True(t, found, "PodCliqueSet has no %s clique", component)
	}
}

func findMainContainer(containers []corev1.Container) *corev1.Container {
	for i := range containers {
		if containers[i].Name == consts.MainContainerName {
			return &containers[i]
		}
	}
	return nil
}

func clientKey(object metav1.Object) types.NamespacedName {
	return types.NamespacedName{Name: object.GetName(), Namespace: object.GetNamespace()}
}
