/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dynamo

import (
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	gmsruntime "github.com/ai-dynamo/dynamo/deploy/operator/internal/gms"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// customMainContainerName is the non-default main-container name used across
// these tests.
const customMainContainerName = "engine"

func generateForBetaComponent(t *testing.T, component *v1beta1.DynamoComponentDeploymentSharedSpec) *corev1.PodSpec {
	t.Helper()
	podSpec, err := GenerateBasePodSpec(
		component,
		BackendFrameworkVLLM,
		&mockSecretsRetriever{},
		"test-deployment",
		"default",
		RoleMain,
		1,
		&configv1alpha1.OperatorConfiguration{},
		commonconsts.MultinodeDeploymentTypeGrove,
		"test-service",
		nil, // No checkpoint info in tests
		nil, // Use default deployer
	)
	if err != nil {
		t.Fatalf("GenerateBasePodSpec() unexpected error: %v", err)
	}
	return podSpec
}

func TestGenerateBasePodSpec_MainContainerNameDefault(t *testing.T) {
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "worker",
		ComponentType: v1beta1.ComponentType(commonconsts.ComponentTypeWorker),
	}
	podSpec := generateForBetaComponent(t, component)

	if podSpec.Containers[0].Name != commonconsts.MainContainerName {
		t.Errorf("default main container name = %q, want %q", podSpec.Containers[0].Name, commonconsts.MainContainerName)
	}
}

func TestGenerateBasePodSpec_CustomMainContainerName(t *testing.T) {
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:             "worker",
		ComponentType:             v1beta1.ComponentType(commonconsts.ComponentTypeWorker),
		MainContainerNameOverride: customMainContainerName,
		PodTemplate: &corev1.PodTemplateSpec{
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name:  customMainContainerName,
						Image: "custom-image:v1",
						Env:   []corev1.EnvVar{{Name: "FOO", Value: "bar"}},
					},
					{
						Name:  "extra-sidecar",
						Image: "sidecar-image:v1",
					},
				},
			},
		},
	}
	podSpec := generateForBetaComponent(t, component)

	main := podSpec.Containers[0]
	if main.Name != customMainContainerName {
		t.Fatalf("main container name = %q, want %q", main.Name, customMainContainerName)
	}
	if main.Image != "custom-image:v1" {
		t.Errorf("main container image = %q, want merged user image %q", main.Image, "custom-image:v1")
	}
	env := map[string]string{}
	for _, e := range main.Env {
		env[e.Name] = e.Value
	}
	if env["FOO"] != "bar" {
		t.Errorf("main container env FOO = %q, want %q (user env must merge into renamed main)", env["FOO"], "bar")
	}

	names := make([]string, 0, len(podSpec.Containers))
	for _, c := range podSpec.Containers {
		names = append(names, c.Name)
		if c.Name == commonconsts.MainContainerName {
			t.Errorf("no container should be named %q when mainContainerName overrides it; containers = %v", commonconsts.MainContainerName, names)
		}
	}
	if len(podSpec.Containers) != 2 || podSpec.Containers[1].Name != "extra-sidecar" {
		t.Errorf("sidecar split incorrect, containers = %v", names)
	}
}

func TestGenerateBasePodSpec_CustomMainContainerNameDiscoveryEnv(t *testing.T) {
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:             "worker",
		ComponentType:             v1beta1.ComponentType(commonconsts.ComponentTypeWorker),
		MainContainerNameOverride: customMainContainerName,
		PodTemplate: &corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: string(configv1alpha1.KubeDiscoveryModeContainer),
				},
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: customMainContainerName, Image: "custom-image:v1"}},
			},
		},
	}
	podSpec := generateForBetaComponent(t, component)

	env := map[string]string{}
	for _, e := range podSpec.Containers[0].Env {
		env[e.Name] = e.Value
	}
	if env["CONTAINER_NAME"] != customMainContainerName {
		t.Errorf("CONTAINER_NAME env = %q, want %q", env["CONTAINER_NAME"], customMainContainerName)
	}
	if env[commonconsts.DynamoMainContainerEnvVar] != customMainContainerName {
		t.Errorf("%s env = %q, want %q", commonconsts.DynamoMainContainerEnvVar, env[commonconsts.DynamoMainContainerEnvVar], customMainContainerName)
	}
}

func TestGenerateBasePodSpec_DefaultNameOmitsMainContainerEnv(t *testing.T) {
	// With the default main-container name the env var is NOT injected, so
	// pod specs of existing deployments stay byte-identical across operator
	// upgrades (the runtime falls back to "main" when the env var is unset).
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "worker",
		ComponentType: v1beta1.ComponentType(commonconsts.ComponentTypeWorker),
		PodTemplate: &corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: string(configv1alpha1.KubeDiscoveryModeContainer),
				},
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{Name: commonconsts.MainContainerName, Image: "custom-image:v1"}},
			},
		},
	}
	podSpec := generateForBetaComponent(t, component)

	for _, e := range podSpec.Containers[0].Env {
		if e.Name == commonconsts.DynamoMainContainerEnvVar {
			t.Errorf("%s must not be injected for the default main-container name (got %q)", commonconsts.DynamoMainContainerEnvVar, e.Value)
		}
	}
	env := map[string]string{}
	for _, e := range podSpec.Containers[0].Env {
		env[e.Name] = e.Value
	}
	if env["CONTAINER_NAME"] != commonconsts.MainContainerName {
		t.Errorf("CONTAINER_NAME env = %q, want %q", env["CONTAINER_NAME"], commonconsts.MainContainerName)
	}
}

func TestGenerateBasePodSpec_CustomMainNameWithFrontendSidecar(t *testing.T) {
	// The frontend sidecar is its own container: its CONTAINER_NAME must be
	// the sidecar's name (not "main"), and DYN_MAIN_CONTAINER_NAME must be
	// consistent across containers so only the real main container claims
	// pod-level identity.
	sidecarName := "sidecar-frontend"
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:             "worker",
		ComponentType:             v1beta1.ComponentType(commonconsts.ComponentTypeWorker),
		MainContainerNameOverride: customMainContainerName,
		FrontendSidecar:           &sidecarName,
		PodTemplate: &corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: string(configv1alpha1.KubeDiscoveryModeContainer),
				},
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: customMainContainerName, Image: "custom-image:v1"},
					{Name: sidecarName, Image: "sidecar-image:v1"},
				},
			},
		},
	}
	podSpec := generateForBetaComponent(t, component)

	byName := map[string]map[string]string{}
	for _, c := range podSpec.Containers {
		env := map[string]string{}
		for _, e := range c.Env {
			env[e.Name] = e.Value
		}
		byName[c.Name] = env
	}

	main, ok := byName[customMainContainerName]
	if !ok {
		t.Fatalf("main container %q not rendered; containers: %v", customMainContainerName, byName)
	}
	sidecar, ok := byName[sidecarName]
	if !ok {
		t.Fatalf("sidecar %q not rendered; containers: %v", sidecarName, byName)
	}

	if main["CONTAINER_NAME"] != customMainContainerName {
		t.Errorf("main CONTAINER_NAME = %q, want %q", main["CONTAINER_NAME"], customMainContainerName)
	}
	if sidecar["CONTAINER_NAME"] != sidecarName {
		t.Errorf("sidecar CONTAINER_NAME = %q, want %q (must not claim the main identity)", sidecar["CONTAINER_NAME"], sidecarName)
	}
	for name, env := range byName {
		if env[commonconsts.DynamoMainContainerEnvVar] != customMainContainerName {
			t.Errorf("container %q %s = %q, want %q", name, commonconsts.DynamoMainContainerEnvVar, env[commonconsts.DynamoMainContainerEnvVar], customMainContainerName)
		}
	}
}

func TestMainContainerAnnotationStamping(t *testing.T) {
	dgd := &v1beta1.DynamoGraphDeployment{}

	// generateAnnotations feeds Grove PodClique pod annotations: the stamp must
	// land there so Grove-managed pods carry the discovery annotation.
	custom := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:             "worker",
		MainContainerNameOverride: customMainContainerName,
	}
	annotationsForClique, err := generateAnnotations(custom, dgd, "worker")
	if err != nil {
		t.Fatalf("generateAnnotations: %v", err)
	}
	if got := annotationsForClique[commonconsts.KubeAnnotationDynamoMainContainerName]; got != customMainContainerName {
		t.Errorf("grove flow: annotation = %q, want %q", got, customMainContainerName)
	}

	// Default-named components stay unstamped so their rendered pods remain
	// byte-identical.
	dflt := &v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker"}
	annotationsForClique, err = generateAnnotations(dflt, dgd, "worker")
	if err != nil {
		t.Fatalf("generateAnnotations: %v", err)
	}
	if got, ok := annotationsForClique[commonconsts.KubeAnnotationDynamoMainContainerName]; ok {
		t.Errorf("grove flow: default name must not be stamped, got %q", got)
	}

	dcd := &v1beta1.DynamoComponentDeployment{}
	dcd.Spec.DynamoComponentDeploymentSharedSpec = v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:             "worker",
		MainContainerNameOverride: customMainContainerName,
	}
	annotations := GetDCDKubeAnnotations(dcd)
	if got := annotations[commonconsts.KubeAnnotationDynamoMainContainerName]; got != customMainContainerName {
		t.Errorf("DCD flow: annotation = %q, want %q", got, customMainContainerName)
	}

	dcdDefault := &v1beta1.DynamoComponentDeployment{}
	dcdDefault.Spec.DynamoComponentDeploymentSharedSpec = v1beta1.DynamoComponentDeploymentSharedSpec{ComponentName: "worker"}
	if got, ok := GetDCDKubeAnnotations(dcdDefault)[commonconsts.KubeAnnotationDynamoMainContainerName]; ok {
		t.Errorf("DCD flow: default name must not be stamped, got %q", got)
	}

	// A stale or user-supplied value of the operator-owned annotation is
	// removed when the effective name is the default.
	stale := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "worker",
		PodTemplate: &corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDynamoMainContainerName: "stale-engine",
				},
			},
		},
	}
	annotationsForClique, err = generateAnnotations(stale, dgd, "worker")
	if err != nil {
		t.Fatalf("generateAnnotations: %v", err)
	}
	if got, ok := annotationsForClique[commonconsts.KubeAnnotationDynamoMainContainerName]; ok {
		t.Errorf("stale annotation must be removed for default names, got %q", got)
	}
}

func TestGenerateBasePodSpec_UserCannotOverrideIdentityEnv(t *testing.T) {
	// Discovery identity env vars are operator-owned: user podTemplate env
	// entries of the same name are overwritten after the merge.
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:             "worker",
		ComponentType:             v1beta1.ComponentType(commonconsts.ComponentTypeWorker),
		MainContainerNameOverride: customMainContainerName,
		PodTemplate: &corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: string(configv1alpha1.KubeDiscoveryModeContainer),
				},
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{
					Name:  customMainContainerName,
					Image: "custom-image:v1",
					Env: []corev1.EnvVar{
						{Name: "CONTAINER_NAME", Value: "evil"},
						{Name: commonconsts.DynamoMainContainerEnvVar, Value: "evil"},
						{Name: "DYN_KUBE_DISCOVERY_MODE", Value: "evil"},
					},
				}},
			},
		},
	}
	podSpec := generateForBetaComponent(t, component)

	env := map[string]string{}
	for _, e := range podSpec.Containers[0].Env {
		env[e.Name] = e.Value
	}
	if env["CONTAINER_NAME"] != customMainContainerName {
		t.Errorf("CONTAINER_NAME = %q, want operator value %q", env["CONTAINER_NAME"], customMainContainerName)
	}
	if env[commonconsts.DynamoMainContainerEnvVar] != customMainContainerName {
		t.Errorf("%s = %q, want operator value %q", commonconsts.DynamoMainContainerEnvVar, env[commonconsts.DynamoMainContainerEnvVar], customMainContainerName)
	}
	if env["DYN_KUBE_DISCOVERY_MODE"] != string(configv1alpha1.KubeDiscoveryModeContainer) {
		t.Errorf("DYN_KUBE_DISCOVERY_MODE = %q, want operator value", env["DYN_KUBE_DISCOVERY_MODE"])
	}
}

func TestGenerateBasePodSpec_DefaultNameStripsUserMainContainerEnv(t *testing.T) {
	// A default-named component never has the operator inject
	// DYN_MAIN_CONTAINER_NAME, but a user podTemplate value for that reserved
	// key must still be removed: otherwise the runtime would treat a foreign
	// name as main while the (unstamped) pod tells remote watchers to fall
	// back to "main", splitting the component's discovery identity.
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName: "worker",
		ComponentType: v1beta1.ComponentType(commonconsts.ComponentTypeWorker),
		PodTemplate: &corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: string(configv1alpha1.KubeDiscoveryModeContainer),
				},
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{{
					Name:  commonconsts.MainContainerName,
					Image: "custom-image:v1",
					Env: []corev1.EnvVar{
						{Name: commonconsts.DynamoMainContainerEnvVar, Value: "evil"},
					},
				}},
			},
		},
	}
	podSpec := generateForBetaComponent(t, component)

	for _, e := range podSpec.Containers[0].Env {
		if e.Name == commonconsts.DynamoMainContainerEnvVar {
			t.Errorf("%s must be stripped for a default-named component (got %q)", commonconsts.DynamoMainContainerEnvVar, e.Value)
		}
	}
}

func TestGenerateBasePodSpec_DefaultNameFrontendSidecarStripsUserMainContainerEnv(t *testing.T) {
	// Same protection on the generated frontend sidecar: a default-named
	// component never sets DYN_MAIN_CONTAINER_NAME, so a user value on the
	// sidecar must be removed rather than left to claim pod-level identity.
	sidecarName := "sidecar-frontend"
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:   "worker",
		ComponentType:   v1beta1.ComponentType(commonconsts.ComponentTypeWorker),
		FrontendSidecar: &sidecarName,
		PodTemplate: &corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: string(configv1alpha1.KubeDiscoveryModeContainer),
				},
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{Name: commonconsts.MainContainerName, Image: "custom-image:v1"},
					{Name: sidecarName, Image: "sidecar-image:v1", Env: []corev1.EnvVar{
						{Name: commonconsts.DynamoMainContainerEnvVar, Value: "evil"},
					}},
				},
			},
		},
	}
	podSpec := generateForBetaComponent(t, component)

	for _, c := range podSpec.Containers {
		for _, e := range c.Env {
			if e.Name == commonconsts.DynamoMainContainerEnvVar {
				t.Errorf("container %q: %s must be stripped for a default-named component (got %q)", c.Name, commonconsts.DynamoMainContainerEnvVar, e.Value)
			}
		}
	}
}

// TestGenerateBasePodSpec_CustomMainNameSmoke is a breadth-first smoke test for
// the custom main-container name contract. It renders a single component with a
// custom mainContainerNameOverride and several operator features that inject or
// clone containers keyed on the main container (intra-pod GMS server sidecar,
// generated frontend sidecar, container-scoped discovery) plus a plain user
// sidecar, then asserts the whole rendered pod stays coherent.
//
// Its purpose is to fail loudly if a feature resolves the main container by the
// literal "main" instead of GetMainContainerName(): such a feature would either
// leak a container named "main", inject its sidecar off the wrong container, or
// drop the injection entirely, and one of the assertions below would catch it.
func TestGenerateBasePodSpec_CustomMainNameSmoke(t *testing.T) {
	const (
		mainName     = "engine"
		frontendName = commonconsts.FrontendSidecarContainerName
		userSidecar  = "user-sidecar"
		mainImage    = "engine:v1"
	)
	component := &v1beta1.DynamoComponentDeploymentSharedSpec{
		ComponentName:             "worker",
		ComponentType:             v1beta1.ComponentType(commonconsts.ComponentTypeWorker),
		MainContainerNameOverride: mainName,
		FrontendSidecar:           k8sPtr(frontendName),
		Experimental: &v1beta1.ExperimentalSpec{
			GPUMemoryService: &v1beta1.GPUMemoryServiceSpec{Mode: v1beta1.GMSModeIntraPod},
		},
		PodTemplate: &corev1.PodTemplateSpec{
			ObjectMeta: metav1.ObjectMeta{
				Annotations: map[string]string{
					commonconsts.KubeAnnotationDynamoKubeDiscoveryMode: string(configv1alpha1.KubeDiscoveryModeContainer),
				},
			},
			Spec: corev1.PodSpec{
				Containers: []corev1.Container{
					{
						Name:  mainName,
						Image: mainImage,
						Resources: corev1.ResourceRequirements{
							Limits: corev1.ResourceList{
								corev1.ResourceName(commonconsts.KubeResourceGPUNvidia): resource.MustParse("1"),
							},
						},
					},
					{Name: frontendName, Image: "frontend:v1"},
					{Name: userSidecar, Image: "sidecar:v1"},
				},
			},
		},
	}
	podSpec := generateForBetaComponent(t, component)

	// The custom-named main container must be first (positional Containers[0]
	// consumers depend on it) and named the override.
	if len(podSpec.Containers) == 0 || podSpec.Containers[0].Name != mainName {
		t.Fatalf("Containers[0] = %q, want the custom main %q; containers: %v", containerNames(podSpec.Containers), mainName, containerNames(podSpec.Containers))
	}

	// No container anywhere may be literally named "main": a stray "main" means
	// some feature auto-generated or mislabeled the main container instead of
	// honoring the override.
	for _, c := range podSpec.Containers {
		if c.Name == commonconsts.MainContainerName {
			t.Errorf("a container is literally named %q despite mainContainerNameOverride=%q; containers: %v", commonconsts.MainContainerName, mainName, containerNames(podSpec.Containers))
		}
	}
	for _, c := range podSpec.InitContainers {
		if c.Name == commonconsts.MainContainerName {
			t.Errorf("an init container is literally named %q despite the override", commonconsts.MainContainerName)
		}
	}

	// Intra-pod GMS injects its server sidecar keyed on the main container: it
	// must exist and derive its image from the custom-named main, proving the
	// injection found the right container rather than a "main" lookup.
	gmsServer := findInitContainerByName(podSpec, gmsruntime.ServerContainerName)
	if gmsServer == nil {
		t.Fatalf("intra-pod GMS server sidecar %q not injected for a custom-named main; init containers: %v", gmsruntime.ServerContainerName, containerNames(podSpec.InitContainers))
	}
	if gmsServer.Image != mainImage {
		t.Errorf("GMS server image = %q, want the custom main's image %q (injection keyed off the wrong container)", gmsServer.Image, mainImage)
	}

	byName := map[string]corev1.Container{}
	for _, c := range podSpec.Containers {
		byName[c.Name] = c
	}

	// The main container carries its own identity envs at the custom name.
	mainEnv := envMap(byName[mainName].Env)
	if mainEnv["CONTAINER_NAME"] != mainName {
		t.Errorf("main CONTAINER_NAME = %q, want %q", mainEnv["CONTAINER_NAME"], mainName)
	}
	if mainEnv[commonconsts.DynamoMainContainerEnvVar] != mainName {
		t.Errorf("main %s = %q, want %q", commonconsts.DynamoMainContainerEnvVar, mainEnv[commonconsts.DynamoMainContainerEnvVar], mainName)
	}

	// The generated frontend sidecar holds its own identity, not the main's, but
	// still agrees on which container is main.
	fe, ok := byName[frontendName]
	if !ok {
		t.Fatalf("frontend sidecar %q not rendered; containers: %v", frontendName, containerNames(podSpec.Containers))
	}
	feEnv := envMap(fe.Env)
	if feEnv["CONTAINER_NAME"] != frontendName {
		t.Errorf("frontend sidecar CONTAINER_NAME = %q, want %q (must not claim the main identity)", feEnv["CONTAINER_NAME"], frontendName)
	}
	if feEnv[commonconsts.DynamoMainContainerEnvVar] != mainName {
		t.Errorf("frontend sidecar %s = %q, want %q", commonconsts.DynamoMainContainerEnvVar, feEnv[commonconsts.DynamoMainContainerEnvVar], mainName)
	}
}

func k8sPtr[T any](v T) *T { return &v }

func containerNames(containers []corev1.Container) []string {
	names := make([]string, 0, len(containers))
	for _, c := range containers {
		names = append(names, c.Name)
	}
	return names
}

func envMap(env []corev1.EnvVar) map[string]string {
	m := make(map[string]string, len(env))
	for _, e := range env {
		m[e.Name] = e.Value
	}
	return m
}
