/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/client-go/tools/events"
)

func TestDisaggregatedSetRealLWSCRDValidationAndConvergence(t *testing.T) {
	t.Log("Start envtest with Dynamo and the pinned real LWS CRDs")
	operatorRoot, err := filepath.Abs(filepath.Join("..", ".."))
	require.NoError(t, err)
	cmd := exec.Command("go", "list", "-m", "-f", "{{.Dir}}", "sigs.k8s.io/lws")
	cmd.Dir = operatorRoot
	output, err := cmd.Output()
	require.NoError(t, err)
	lwsModuleRoot := strings.TrimSpace(string(output))

	env := operatorenv.New(operatorenv.Options{
		SetupWebhooks: setupProductionWebhooks,
		CRDDirectoryPaths: []string{
			filepath.Join(operatorRoot, "config", "crd", "bases"),
			filepath.Join(lwsModuleRoot, "config", "crd", "bases"),
		},
	})
	testEnv := env.RunT(t)

	t.Log("Create a two-role DisaggregatedSet through the real API schema")
	dgd := newEnvtestDSHappyPathDGD("real-crd-convergence")
	dgd.Namespace = testEnv.Namespace()
	dcds, err := dynamo.GenerateDynamoComponentsDeployments(dgd, nil, nil, dynamo.RollingUpdateContext{})
	require.NoError(t, err)
	selection, reason := selectDisaggregatedSetComponents(dgd)
	require.Empty(t, reason)

	runtimeConfig := &commoncontroller.RuntimeConfig{
		Gate:         features.Gates{LWS: true, DisaggregatedSet: true},
		Capabilities: features.Capabilities{DisaggregatedSetAPI: true},
	}
	reconciler := &DynamoGraphDeploymentReconciler{
		Client:        testEnv.Client(),
		Recorder:      events.NewFakeRecorder(10),
		Config:        &configv1alpha1.OperatorConfiguration{},
		RuntimeConfig: runtimeConfig,
	}
	workloads := reconciler.newDisaggregatedSetWorkloadsReconciler(
		newDGDWorkerRolloutReconciler(reconciler.Client, reconciler.Recorder),
	)
	desired, err := workloads.generateDisaggregatedSet(t.Context(), dgd, dcds, selection)
	require.NoError(t, err)
	_, found, err := unstructured.NestedFieldNoCopy(desired.Object, "spec", "slices")
	require.NoError(t, err)
	require.False(t, found, "DGD does not expose slice cardinality before the grouping API lands")
	modified, current, err := workloads.syncDisaggregatedSet(t.Context(), dgd, desired)
	require.NoError(t, err)
	require.True(t, modified)
	slices, found, err := unstructured.NestedInt64(current.Object, "spec", "slices")
	require.NoError(t, err)
	require.True(t, found)
	require.Equal(t, int64(1), slices, "the LWS v0.10 CRD defaults the transitional pathway to one slice")

	t.Log("Reconcile identical desired state after API defaulting")
	desired, err = workloads.generateDisaggregatedSet(t.Context(), dgd, dcds, selection)
	require.NoError(t, err)
	modified, _, err = workloads.syncDisaggregatedSet(t.Context(), dgd, desired)
	require.NoError(t, err)
	require.False(t, modified, "an identical desired DisaggregatedSet must converge after API defaulting")
}
