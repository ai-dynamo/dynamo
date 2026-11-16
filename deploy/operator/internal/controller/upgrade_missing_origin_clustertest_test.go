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
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	ctrl "sigs.k8s.io/controller-runtime"
)

const upgradeMissingOriginInput = "testdata/upgrade/compatibility/missing-origin/input.yaml"

func testUpgradeMissingOrigin(t *testing.T) {
	t.Log("Read the released snapshot seeded before current schema and admission")
	objects, err := golden.ReadManifests(upgradeMissingOriginInput)
	require.NoError(t, err)
	require.NotEmpty(t, objects)
	namespace := objects[0].GetNamespace()
	env := clusterTestEnv.RunTInNamespace(t, namespace)
	env.BlockWorkloads()
	before := captureUpgrade(t, env)
	dgd := getOnlyClusterDGD(t, t.Context(), env.Client(), namespace)
	require.NotContains(t, dgd.Annotations, consts.KubeAnnotationDynamoOperatorOriginVersion)
	rollout := newDGDWorkerRolloutReconciler(env.Client(), nil)
	transition, err := rollout.planUnsupportedWorkerHashTransition(dgd)
	require.NoError(t, err)
	require.False(t, transition.hashChanged)

	t.Log("Run production takeover and preserve workload identity, templates and absent origin")
	baseline := completedUpgradeReconciles(t)
	env.StartManager(func(mgr ctrl.Manager) error {
		opts := SetupOptions{Config: clusterTestRestrictedConfig(namespace), RuntimeConfig: &common.RuntimeConfig{Gate: features.Gates{Grove: true}}}
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{SetupOptions: opts})
	})
	require.Eventually(t, func() bool { return completedUpgradeReconciles(t) > baseline }, 90*time.Second, 200*time.Millisecond)
	after := captureUpgrade(t, env)
	require.NoError(t, upgradeSafetyError(transition.hashChanged, before, after))

	t.Log("Capture the released takeover state before the intentional worker edit")
	if root := os.Getenv("DYNAMO_UPGRADE_CAPTURE_DIR"); root != "" && !t.Failed() {
		path := filepath.Join(root, "compatibility", "missing-origin", "input.yaml")
		require.NoError(t, os.MkdirAll(filepath.Dir(path), 0755))
		require.NoError(t, golden.WriteManifests(path, after))
	}

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
