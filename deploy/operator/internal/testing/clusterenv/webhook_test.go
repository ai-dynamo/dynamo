/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package clusterenv

import (
	"testing"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
)

func TestPointAdmissionAtProxyPreservesPath(t *testing.T) {
	t.Log("Build a service-backed registration and a generated proxy location")
	path := "/validate-example"
	configuration := &admissionregistrationv1.WebhookClientConfig{
		Service: &admissionregistrationv1.ServiceReference{Path: &path},
	}
	proxy := &proxyRuntime{namespace: "proxy-namespace", service: "proxy-service"}

	t.Log("Redirect the registration while preserving its handler path")
	if err := pointAdmissionAtProxy(configuration, proxy, []byte("ca")); err != nil {
		t.Fatalf("point admission registration at proxy: %v", err)
	}
	if configuration.Service.Namespace != proxy.namespace || configuration.Service.Name != proxy.service {
		t.Fatalf("service reference = %s/%s, want %s/%s", configuration.Service.Namespace, configuration.Service.Name, proxy.namespace, proxy.service)
	}
	if configuration.Service.Path == nil || *configuration.Service.Path != path {
		t.Fatalf("service path = %v, want %q", configuration.Service.Path, path)
	}
	if string(configuration.CABundle) != "ca" {
		t.Fatalf("CA bundle = %q, want ca", configuration.CABundle)
	}
}

func TestPointAdmissionAtProxyRejectsURLRegistration(t *testing.T) {
	t.Log("Reject registrations without a service path that can be preserved")
	configuration := &admissionregistrationv1.WebhookClientConfig{}

	if err := pointAdmissionAtProxy(configuration, &proxyRuntime{}, nil); err == nil {
		t.Fatal("URL-only admission registration was accepted")
	}
}
