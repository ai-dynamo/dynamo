//go:build clustertest && upgradecontract

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package controller

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	ctrl "sigs.k8s.io/controller-runtime"
)

const (
	upgradeFrontendSidecarProbesInput  = "testdata/upgrade/compatibility/frontend-sidecar-probes/input.yaml"
	upgradeFrontendSidecarProbesOrigin = "1.6.0"
	upgradeFrontendSidecarName         = "sidecar-frontend"
)

// testUpgradeFrontendSidecarProbes takes over a 1.6.0-origin worker whose frontend
// sidecar authored partial probes. Released 1.6.0 field-merged them with its
// frontend defaults; the candidate must keep that rendered template.
func testUpgradeFrontendSidecarProbes(t *testing.T) {
	t.Log("Read the released 1.6.0 snapshot seeded before current schema and admission")
	objects, err := golden.ReadManifests(upgradeFrontendSidecarProbesInput)
	require.NoError(t, err)
	require.NotEmpty(t, objects)
	namespace := objects[0].GetNamespace()
	env := clusterTestEnv.RunTInNamespace(t, namespace)
	env.BlockWorkloads()
	before := captureUpgrade(t, env)
	dgd := getOnlyClusterDGD(t, t.Context(), env.Client(), namespace)
	require.Equal(t, upgradeFrontendSidecarProbesOrigin, dgd.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion])
	rollout := newDGDWorkerRolloutReconciler(env.Client(), nil)
	transition, err := rollout.planUnsupportedWorkerHashTransition(dgd)
	require.NoError(t, err)
	require.False(t, transition.hashChanged)
	requireMergedSidecarProbes(t, before)

	t.Log("Run production takeover and preserve workload identity, templates and origin")
	baseline := completedUpgradeReconciles(t)
	env.StartManager(func(mgr ctrl.Manager) error {
		opts := SetupOptions{Config: clusterTestRestrictedConfig(namespace), RuntimeConfig: &common.RuntimeConfig{Gate: features.Gates{Grove: true}}}
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: opts})
	})
	require.Eventually(t, func() bool { return completedUpgradeReconciles(t) > baseline }, 90*time.Second, 200*time.Millisecond)
	after := captureUpgrade(t, env)
	require.NoError(t, upgradeSafetyError(transition.hashChanged, before, after))

	t.Log("Keep the probe fields that only the 1.6.0 field merge supplied")
	requireMergedSidecarProbes(t, after)

	t.Log("Capture the released takeover state for the next release regeneration")
	if root := os.Getenv("DYNAMO_UPGRADE_CAPTURE_DIR"); root != "" && !t.Failed() {
		path := filepath.Join(root, "compatibility", "frontend-sidecar-probes", "input.yaml")
		require.NoError(t, os.MkdirAll(filepath.Dir(path), 0755))
		require.NoError(t, golden.WriteManifests(path, after))
	}
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
				if container["name"] != upgradeFrontendSidecarName {
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
