//go:build clustertest && upgradecontract

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/clusterenv"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	ctrl "sigs.k8s.io/controller-runtime"
)

// upgradeCompatibilityHooks adds scenario-specific checks to the shared
// compatibility takeover. Both hooks may be nil.
type upgradeCompatibilityHooks struct {
	// beforeTakeover checks the released snapshot before the candidate starts.
	beforeTakeover func(t *testing.T, dgd *nvidiacomv1beta1.DynamoGraphDeployment, before []unstructured.Unstructured)
	// afterTakeover checks the converged graph and may exercise further edits.
	afterTakeover func(t *testing.T, env *clusterenv.TestEnv, namespace string, after []unstructured.Unstructured)
}

// upgradeCompatibilityScenarioHooks is keyed by the scenario directory name under
// testdata/upgrade/compatibility. Scenarios without an entry run the shared checks only.
var upgradeCompatibilityScenarioHooks = map[string]upgradeCompatibilityHooks{
	"missing-origin": {
		beforeTakeover: func(t *testing.T, dgd *nvidiacomv1beta1.DynamoGraphDeployment, _ []unstructured.Unstructured) {
			require.NotContains(t, dgd.Annotations, consts.KubeAnnotationDynamoOperatorOriginVersion)
		},
		afterTakeover: testUpgradeMissingOriginAfterTakeover,
	},
	"frontend-sidecar-probes": {
		beforeTakeover: func(t *testing.T, dgd *nvidiacomv1beta1.DynamoGraphDeployment, before []unstructured.Unstructured) {
			require.Equal(t, "1.6.0", dgd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion])
			requireMergedSidecarProbes(t, before)
		},
		afterTakeover: func(t *testing.T, _ *clusterenv.TestEnv, _ string, after []unstructured.Unstructured) {
			t.Log("Keep the probe fields that only the 1.6.0 field merge supplied")
			requireMergedSidecarProbes(t, after)
		},
	},
}

// runUpgradeCompatibilityScenarios replays every snapshot under
// testdata/upgrade/compatibility as a subtest named after its directory.
func runUpgradeCompatibilityScenarios(t *testing.T) {
	inputs, err := upgradeInputs("compatibility")
	require.NoError(t, err)
	require.NotEmpty(t, inputs)
	for _, input := range inputs {
		t.Run(filepath.Base(filepath.Dir(input)), func(t *testing.T) { testUpgradeCompatibility(t, input) })
	}
}

// testUpgradeCompatibility replays one released compatibility snapshot through
// production takeover and the independent safety assertion.
func testUpgradeCompatibility(t *testing.T, input string) {
	name := filepath.Base(filepath.Dir(input))
	hooks := upgradeCompatibilityScenarioHooks[name]

	t.Log("Read the released snapshot seeded before current schema and admission")
	objects, err := golden.ReadManifests(input)
	require.NoError(t, err)
	require.NotEmpty(t, objects)
	namespace := objects[0].GetNamespace()
	env := clusterTestEnv.RunTInNamespace(t, namespace)
	env.BlockWorkloads()
	before := snapshotBaseline(t, objects, captureUpgrade(t, env))
	dgd := getOnlyClusterDGD(t, t.Context(), env.Client(), namespace)
	rollout := newDGDWorkerRolloutReconciler(env.Client(), nil)
	transition, err := rollout.planUnsupportedWorkerHashTransition(dgd)
	require.NoError(t, err)
	require.False(t, transition.hashChanged)
	if hooks.beforeTakeover != nil {
		hooks.beforeTakeover(t, dgd, before)
	}

	t.Log("Run production takeover and preserve workload identity, templates and origin")
	baseline := completedUpgradeReconciles(t)
	env.StartManager(func(mgr ctrl.Manager) error {
		opts := SetupOptions{Config: clusterTestRestrictedConfig(namespace), RuntimeConfig: &common.RuntimeConfig{Gate: features.Gates{Grove: true}}}
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: opts})
	})
	require.Eventually(t, func() bool { return completedUpgradeReconciles(t) > baseline }, 90*time.Second, 200*time.Millisecond)
	after := captureUpgrade(t, env)
	require.NoError(t, upgradeSafetyError(transition.hashChanged, before, after))

	t.Log("Compare the complete owned graph with the reviewed golden")
	golden.MatchCapturedManifests(t, after, filepath.Join(filepath.Dir(input), "output.yaml"))

	t.Log("Capture the released takeover state for the next release regeneration")
	if root := os.Getenv("DYNAMO_UPGRADE_CAPTURE_DIR"); root != "" && !t.Failed() {
		path := filepath.Join(root, "compatibility", name, "input.yaml")
		require.NoError(t, os.MkdirAll(filepath.Dir(path), 0755))
		require.NoError(t, golden.WriteManifests(path, after))
	}

	if hooks.afterTakeover != nil {
		hooks.afterTakeover(t, env, namespace, after)
	}
}

// testUpgradeMissingOriginAfterTakeover keeps the legacy-installation checks: the
// safety assertion rejects an invented origin, and a worker edit still reaches the
// requested Grove generation without materializing one.
func testUpgradeMissingOriginAfterTakeover(t *testing.T, env *clusterenv.TestEnv, namespace string, after []unstructured.Unstructured) {
	t.Log("The independent safety check rejects inventing a historical origin")
	changed := make([]unstructured.Unstructured, len(after))
	for i := range after {
		changed[i] = *after[i].DeepCopy()
		if changed[i].GetKind() == upgradeDynamoGraphDeploymentKind {
			annotations := changed[i].GetAnnotations()
			annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = "1.7.0"
			changed[i].SetAnnotations(annotations)
		}
	}
	require.ErrorContains(t, upgradeSafetyError(false, after, changed), "historical origin changed")

	t.Log("A valid worker edit still reaches the requested Grove generation and keeps origin absent")
	current := getOnlyClusterDGD(t, t.Context(), env.Client(), namespace)
	oldHash := current.Annotations[consts.AnnotationCurrentWorkerHashV2]
	for i := range current.Spec.Components {
		if current.Spec.Components[i].ComponentName == "worker" {
			container := &current.Spec.Components[i].PodTemplate.Spec.Containers[0]
			container.Env = append(container.Env, corev1.EnvVar{Name: "WORKER_REVISION", Value: "2"})
		}
	}
	desiredHash, err := dynamo.ComputeDGDWorkersSpecHash(current)
	require.NoError(t, err)
	require.NotEqual(t, oldHash, desiredHash)
	require.NoError(t, env.Client().Update(t.Context(), current))
	require.EventuallyWithT(t, func(c *assert.CollectT) {
		live := getOnlyClusterDGD(t, t.Context(), env.Client(), namespace)
		assert.NotContains(c, live.Annotations, consts.KubeAnnotationDynamoOperatorOriginVersion)
		assert.Equal(c, desiredHash, live.Annotations[consts.AnnotationCurrentWorkerHashV2])
		found := false
		for _, object := range captureUpgrade(t, env) {
			if object.GetKind() != upgradePodCliqueSetKind {
				continue
			}
			templates, err := upgradeWorkerTemplates(object)
			if !assert.NoError(c, err) {
				continue
			}
			for _, value := range templates {
				template := value.(map[string]any)
				labels := template["labels"].(map[string]any)
				if labels["nvidia.com/dynamo-component"] != "worker" || labels["nvidia.com/dynamo-worker-hash"] != desiredHash {
					continue
				}
				found = found || upgradeTemplateHasEdit(map[string]any{"spec": template["podSpec"]})
			}
		}
		assert.True(c, found, "worker edit did not reach the requested immediate workload generation")
	}, 60*time.Second, 200*time.Millisecond)
}

// requireMergedSidecarProbes checks the two sidecar probe fields that the 1.6.0
// field merge filled in and that complete-value replacement would drop.
func requireMergedSidecarProbes(t *testing.T, objects []unstructured.Unstructured) {
	t.Helper()
	for _, object := range objects {
		if object.GetKind() != upgradePodCliqueSetKind {
			continue
		}
		templates, err := upgradeWorkerTemplates(object)
		require.NoError(t, err)
		for _, value := range templates {
			template := value.(map[string]any)
			labels, _ := template["labels"].(map[string]any)
			if labels["nvidia.com/dynamo-component"] != "worker" {
				continue
			}
			podSpec := template["podSpec"].(map[string]any)
			for _, item := range podSpec["containers"].([]any) {
				container := item.(map[string]any)
				if container["name"] != "sidecar-frontend" {
					continue
				}
				readiness, _, err := unstructured.NestedInt64(container, "readinessProbe", "timeoutSeconds")
				require.NoError(t, err)
				liveness, _, err := unstructured.NestedInt64(container, "livenessProbe", "initialDelaySeconds")
				require.NoError(t, err)
				require.Equal(t, int64(3), readiness, "readiness timeoutSeconds merged by 1.6.0 was dropped")
				require.Equal(t, int64(15), liveness, "liveness initialDelaySeconds merged by 1.6.0 was dropped")
				return
			}
		}
	}
	require.Fail(t, "worker clique has no frontend sidecar container")
}
