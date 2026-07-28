/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package operatorenv

import (
	"strings"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"k8s.io/client-go/rest"
	"k8s.io/utils/ptr"
)

func TestWebhookInstallOptionsSelectsValidatingConfiguration(t *testing.T) {
	install, err := webhookInstallOptions(Options{
		Admission: AdmissionWebhooks{Validating: true},
	})
	if err != nil {
		t.Fatalf("render webhook install options: %v", err)
	}

	if len(install.MutatingWebhooks) != 0 {
		t.Fatalf("mutating webhook configurations = %d, want 0", len(install.MutatingWebhooks))
	}
	if len(install.ValidatingWebhooks) == 0 {
		t.Fatal("validating webhook configurations = 0, want at least 1")
	}
}

func TestWebhookSetupIsRequired(t *testing.T) {
	_, err := startRuntime(normalizeOptions(Options{}))

	if err == nil || !strings.Contains(err.Error(), "SetupWebhooks is required") {
		t.Fatalf("startRuntime() error = %v, want missing SetupWebhooks error", err)
	}
}

func TestRESTConfigReturnsCopy(t *testing.T) {
	env := &TestEnv{rt: &runtimeEnv{config: &rest.Config{Host: "https://original.example"}}}

	config := env.RESTConfig()
	config.Host = "https://modified.example"

	if env.rt.config.Host != "https://original.example" {
		t.Fatalf("shared REST config host = %q, want original value", env.rt.config.Host)
	}
}

func TestDefaultOperatorConfigPreservesExplicitGPUDiscovery(t *testing.T) {
	config := &configv1alpha1.OperatorConfiguration{
		GPU: configv1alpha1.GPUConfiguration{
			DiscoveryEnabled: ptr.To(true),
		},
	}

	got := defaultOperatorConfig(config)

	if got.GPU.DiscoveryEnabled == nil || !*got.GPU.DiscoveryEnabled {
		t.Fatalf("GPU discovery enabled = %v, want true", got.GPU.DiscoveryEnabled)
	}
}

func TestDefaultRuntimeConfigDerivesStaticFeatureGates(t *testing.T) {
	clusterWide := &configv1alpha1.OperatorConfiguration{
		GPU: configv1alpha1.GPUConfiguration{DiscoveryEnabled: ptr.To(false)},
	}
	clusterWideRuntime := defaultRuntimeConfig(clusterWide)

	if !clusterWideRuntime.Gate.Enabled(features.GPUDiscovery) {
		t.Fatal("cluster-wide GPU discovery gate = false, want true")
	}

	restricted := clusterWide.DeepCopy()
	restricted.Namespace.Restricted = "test-namespace"
	restricted.Checkpoint.Enabled = true
	restrictedRuntime := defaultRuntimeConfig(restricted)

	if restrictedRuntime.Gate.Enabled(features.GPUDiscovery) {
		t.Fatal("namespace-restricted GPU discovery gate = true, want false")
	}
	if !restrictedRuntime.Gate.Enabled(features.Checkpoint) {
		t.Fatal("checkpoint gate = false, want true")
	}
}
