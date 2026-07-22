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

func TestLoadRESTConfigRequiresUnlockedContext(t *testing.T) {
	t.Setenv(ContextEnvVar, "")

	_, err := loadRESTConfig()

	if err == nil || !strings.Contains(err.Error(), ContextEnvVar+" must be set") {
		t.Fatalf("loadRESTConfig() error = %v, want missing %s error", err, ContextEnvVar)
	}
}

func TestLoadRESTConfigSelectsUnlockedContext(t *testing.T) {
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

	config, err := loadRESTConfig()
	if err != nil {
		t.Fatalf("loadRESTConfig(): %v", err)
	}
	if config.Host != "https://allowed.example" {
		t.Fatalf("REST host = %q, want unlocked context host", config.Host)
	}
}

func TestLoadRESTConfigRejectsUnknownContext(t *testing.T) {
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
