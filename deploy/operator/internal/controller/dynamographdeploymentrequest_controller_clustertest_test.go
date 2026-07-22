//go:build clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"fmt"
	"os"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/golden"
	batchv1 "k8s.io/api/batch/v1"
	corev1 "k8s.io/api/core/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const clusterTestProfilerImageEnv = "DYNAMO_CLUSTERTEST_PROFILER_IMAGE"

func TestClusterDynamoGraphDeploymentRequestProfilesAndCreatesWorkloadManifests(t *testing.T) {
	ctx := t.Context()
	profilerImage := clusterTestRequiredEnv(t, clusterTestProfilerImageEnv)
	t.Log("Create an isolated namespace in the explicitly unlocked cluster")
	env := clusterTestEnv.RunT(t)

	t.Log("Block ReplicaSets while allowing the profiler Job Pod to run")
	env.BlockReplicaSets()

	t.Log("Install the namespace-local identity and permissions used by the profiling Job")
	clusterTestCreateProfilerRBAC(t, ctx, env.Client(), env.Namespace())
	if err := env.Client().Create(ctx, &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "hf-token-secret", Namespace: env.Namespace()},
		StringData: map[string]string{"HF_TOKEN": ""},
	}); err != nil {
		t.Fatalf("create profiler token secret: %v", err)
	}

	t.Log("Create the scale client and start the production DGDR-to-DGD controller chain with Grove enabled")
	operatorConfig := &configv1alpha1.OperatorConfiguration{}
	configv1alpha1.SetDefaultsOperatorConfiguration(operatorConfig)
	operatorConfig.Namespace.Restricted = env.Namespace()
	operatorConfig.GPU.DiscoveryEnabled = ptr.To(false)
	runtimeConfig := &commoncontroller.RuntimeConfig{Gate: features.Gates{Grove: true, LWS: true}}
	scaleClient, err := env.ScaleClient()
	if err != nil {
		t.Fatalf("create scale client: %v", err)
	}
	env.StartManager(func(mgr ctrl.Manager) error {
		if err := SetupDynamoGraphDeploymentRequest(mgr, DynamoGraphDeploymentRequestSetupOptions{
			SetupOptions: SetupOptions{
				Config:        operatorConfig,
				RuntimeConfig: runtimeConfig,
			},
		}); err != nil {
			return err
		}
		return SetupDynamoGraphDeployment(mgr, DynamoGraphDeploymentSetupOptions{
			SetupOptions: SetupOptions{
				Config:        operatorConfig,
				RuntimeConfig: runtimeConfig,
			},
			ScaleClient: scaleClient,
		})
	})

	t.Log("Create a rapid DGDR that runs AIConfigurator with simulated GPU performance data")
	dgdr := &nvidiacomv1beta1.DynamoGraphDeploymentRequest{}
	clusterTestReadInput(t, env.Namespace(), "testdata/dgdr-profiler/input.yaml", dgdr)
	dgdr.Spec.Image = profilerImage
	if err := env.Client().Create(ctx, dgdr); err != nil {
		t.Fatalf("create DGDR: %v", err)
	}

	t.Log("Wait for the real profiling Job Pod to complete")
	var completedDGDR nvidiacomv1beta1.DynamoGraphDeploymentRequest
	clusterTestEventually(t, 20*time.Minute, "DGDR to record a completed profiling Job", func(ctx context.Context) (bool, error) {
		if err := env.Client().Get(ctx, client.ObjectKeyFromObject(dgdr), &completedDGDR); err != nil {
			return false, err
		}
		if completedDGDR.Status.ProfilingJobName == "" {
			if completedDGDR.Status.Phase == nvidiacomv1beta1.DGDRPhaseFailed {
				return false, fmt.Errorf("DGDR failed before creating a profiling Job: %v", completedDGDR.Status.Conditions)
			}
			return false, nil
		}
		var job batchv1.Job
		if err := env.Client().Get(ctx, types.NamespacedName{
			Name: completedDGDR.Status.ProfilingJobName, Namespace: env.Namespace(),
		}, &job); err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		if job.Status.Failed > 0 {
			return false, fmt.Errorf("profiling Job failed: %v", job.Status.Conditions)
		}
		return job.Status.Succeeded == 1, nil
	})

	t.Log("Verify profiling output was consumed and autoApply created a DGD")
	var dgd nvidiacomv1beta1.DynamoGraphDeployment
	clusterTestEventually(t, 2*time.Minute, "DGD to be created", func(ctx context.Context) (bool, error) {
		if err := env.Client().Get(ctx, types.NamespacedName{
			Name: dgdr.Name + "-dgd", Namespace: env.Namespace(),
		}, &dgd); err != nil {
			if apierrors.IsNotFound(err) {
				return false, nil
			}
			return false, err
		}
		return true, nil
	})
	if err := env.Client().Get(ctx, client.ObjectKeyFromObject(dgdr), &completedDGDR); err != nil {
		t.Fatalf("get completed DGDR: %v", err)
	}
	if completedDGDR.Status.ProfilingResults == nil || completedDGDR.Status.ProfilingResults.SelectedConfig == nil {
		t.Fatal("DGDR has no selected profiling configuration")
	}

	t.Log("Match the generated DGD and its terminal Grove manifest")
	golden.MatchManifests(t, env.Client(), env.Namespace(), "testdata/dgdr-profiler/output.yaml")
}

func clusterTestRequiredEnv(t *testing.T, name string) string {
	t.Helper()
	value := os.Getenv(name)
	if value == "" {
		t.Fatalf("%s must be set for cluster tests", name)
	}
	return value
}

func clusterTestCreateProfilerRBAC(t *testing.T, ctx context.Context, k8sClient client.Client, namespace string) {
	t.Helper()
	objects := []client.Object{
		&corev1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: ServiceAccountProfilingJob, Namespace: namespace}},
		&rbacv1.Role{
			ObjectMeta: metav1.ObjectMeta{Name: ServiceAccountProfilingJob, Namespace: namespace},
			Rules: []rbacv1.PolicyRule{
				{APIGroups: []string{""}, Resources: []string{"configmaps"}, Verbs: []string{"create", "get", "update", "patch", "delete"}},
				{APIGroups: []string{nvidiacomv1beta1.GroupVersion.Group}, Resources: []string{"dynamographdeploymentrequests"}, Verbs: []string{"get"}},
				{APIGroups: []string{nvidiacomv1beta1.GroupVersion.Group}, Resources: []string{"dynamographdeployments"}, Verbs: []string{"get", "create", "delete", "list", "watch"}},
				{APIGroups: []string{""}, Resources: []string{"pods"}, Verbs: []string{"get", "list", "create", "delete"}},
				{APIGroups: []string{""}, Resources: []string{"pods/log"}, Verbs: []string{"get"}},
			},
		},
		&rbacv1.RoleBinding{
			ObjectMeta: metav1.ObjectMeta{Name: ServiceAccountProfilingJob, Namespace: namespace},
			RoleRef: rbacv1.RoleRef{
				APIGroup: rbacv1.GroupName,
				Kind:     "Role",
				Name:     ServiceAccountProfilingJob,
			},
			Subjects: []rbacv1.Subject{{
				Kind:      "ServiceAccount",
				Name:      ServiceAccountProfilingJob,
				Namespace: namespace,
			}},
		},
	}
	for _, object := range objects {
		if err := k8sClient.Create(ctx, object); err != nil {
			t.Fatalf("create profiler RBAC object %T: %v", object, err)
		}
	}
}

func clusterTestEventually(
	t *testing.T,
	timeout time.Duration,
	description string,
	condition func(context.Context) (bool, error),
) {
	t.Helper()
	ctx, cancel := context.WithTimeout(t.Context(), timeout)
	defer cancel()
	if err := wait.PollUntilContextCancel(ctx, time.Second, true, condition); err != nil {
		t.Fatalf("wait for %s: %v", description, err)
	}
}
