/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package clusterenv

import (
	"path/filepath"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/clientcmd"
	clientcmdapi "k8s.io/client-go/tools/clientcmd/api"
)

func TestRequireEnv(t *testing.T) {
	t.Log("Set a cluster-test input in the process environment")
	t.Setenv("DYNAMO_CLUSTERTEST_REQUIRED_TEST", "configured")

	t.Log("Return the configured value")
	if value := RequireEnv(t, "DYNAMO_CLUSTERTEST_REQUIRED_TEST"); value != "configured" {
		t.Fatalf("RequireEnv() = %q, want configured", value)
	}
}

func TestLoadRESTConfigRequiresUnlockedContext(t *testing.T) {
	t.Log("Clear the explicit cluster-test context")
	t.Setenv(ContextEnvVar, "")

	t.Log("Reject loading cluster credentials without an unlocked context")
	_, err := loadRESTConfig()
	if err == nil || !strings.Contains(err.Error(), ContextEnvVar+" must be set") {
		t.Fatalf("loadRESTConfig() error = %v, want missing %s error", err, ContextEnvVar)
	}
}

func TestLoadRESTConfigSelectsUnlockedContext(t *testing.T) {
	t.Log("Write a kubeconfig whose current context differs from the unlocked context")
	kubeconfig := writeKubeconfig(t, clientcmdapi.Config{
		CurrentContext: "other",
		Clusters: map[string]*clientcmdapi.Cluster{
			"allowed-cluster": {Server: "https://allowed.example"},
			"other-cluster":   {Server: "https://other.example"},
		},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{
			"allowed-user": {},
			"other-user":   {},
		},
		Contexts: map[string]*clientcmdapi.Context{
			"allowed": {Cluster: "allowed-cluster", AuthInfo: "allowed-user"},
			"other":   {Cluster: "other-cluster", AuthInfo: "other-user"},
		},
	})
	t.Setenv("KUBECONFIG", kubeconfig)
	t.Setenv(ContextEnvVar, "allowed")

	t.Log("Load credentials for the explicitly unlocked context")
	config, err := loadRESTConfig()
	if err != nil {
		t.Fatalf("loadRESTConfig(): %v", err)
	}
	if config.Host != "https://allowed.example" {
		t.Fatalf("REST host = %q, want unlocked context host", config.Host)
	}
}

func TestLoadRESTConfigRejectsUnknownContext(t *testing.T) {
	t.Log("Write a kubeconfig without the context named by the unlock variable")
	kubeconfig := writeKubeconfig(t, clientcmdapi.Config{
		CurrentContext: "other",
		Clusters: map[string]*clientcmdapi.Cluster{
			"other-cluster": {Server: "https://other.example"},
		},
		AuthInfos: map[string]*clientcmdapi.AuthInfo{"other-user": {}},
		Contexts: map[string]*clientcmdapi.Context{
			"other": {Cluster: "other-cluster", AuthInfo: "other-user"},
		},
	})
	t.Setenv("KUBECONFIG", kubeconfig)
	t.Setenv(ContextEnvVar, "missing")

	t.Log("Reject an unlock context that is absent from the kubeconfig")
	_, err := loadRESTConfig()
	if err == nil || !strings.Contains(err.Error(), `context "missing"`) {
		t.Fatalf("loadRESTConfig() error = %v, want unknown context error", err)
	}
}

func TestWorkloadBlockQuotaLeavesWorkloadManifestsWritable(t *testing.T) {
	t.Log("Build the quota installed before a manifest-producing controller starts")
	quota := workloadBlockQuota("test-namespace")

	t.Log("Verify only downstream ReplicaSets and Pods are blocked")
	if quota.Namespace != "test-namespace" {
		t.Fatalf("quota namespace = %q, want test-namespace", quota.Namespace)
	}
	if len(quota.Spec.Hard) != 2 {
		t.Fatalf("quota has %d hard limits, want 2", len(quota.Spec.Hard))
	}
	for _, name := range []corev1.ResourceName{"count/replicasets.apps", corev1.ResourcePods} {
		quantity, found := quota.Spec.Hard[name]
		if !found {
			t.Fatalf("quota does not limit %q", name)
		}
		if !quantity.IsZero() {
			t.Fatalf("quota limit %q = %s, want 0", name, quantity.String())
		}
	}
}

func TestReplicaSetBlockQuotaAllowsJobPods(t *testing.T) {
	t.Log("Build the quota used while a Job-backed controller test is running")
	quota := replicaSetBlockQuota("test-namespace")

	t.Log("Block only ReplicaSets so the profiler Job can still create its Pod")
	if len(quota.Spec.Hard) != 1 {
		t.Fatalf("quota has %d hard limits, want 1", len(quota.Spec.Hard))
	}
	if _, found := quota.Spec.Hard[corev1.ResourcePods]; found {
		t.Fatal("replica-set-only quota also limits Pods")
	}
	if quantity, found := quota.Spec.Hard["count/replicasets.apps"]; !found || !quantity.IsZero() {
		t.Fatalf("ReplicaSet quota = %s, found=%t, want 0", quantity.String(), found)
	}
}

func TestQuotaInitializedRequiresEveryHardLimit(t *testing.T) {
	expected := workloadBlockQuota("test-namespace").Spec.Hard
	quota := &corev1.ResourceQuota{}

	t.Log("Reject a quota status before the quota controller publishes its hard limits")
	if quotaInitialized(quota, expected) {
		t.Fatal("empty quota status was considered initialized")
	}

	t.Log("Accept the quota after the controller publishes every expected limit")
	quota.Status.Hard = expected.DeepCopy()
	if !quotaInitialized(quota, expected) {
		t.Fatal("complete quota status was not considered initialized")
	}
}

func writeKubeconfig(t *testing.T, config clientcmdapi.Config) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "config")
	if err := clientcmd.WriteToFile(config, path); err != nil {
		t.Fatalf("write kubeconfig: %v", err)
	}
	return path
}
