/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package clusterenv

import (
	"errors"
	"testing"
	"time"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	"sigs.k8s.io/controller-runtime/pkg/webhook"
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

func TestWebhookRuntimeStopAfterManagerExitedBeforeServing(t *testing.T) {
	t.Log("Represent a webhook manager that exited before its server became ready")
	managerErr := errors.New("manager failed")
	runtime := &webhookRuntime{
		cancel:     func() {},
		done:       make(chan struct{}),
		managerErr: managerErr,
	}
	close(runtime.done)
	server := webhook.NewServer(webhook.Options{})

	t.Log("Observe the manager failure while waiting for the webhook server")
	err := waitForWebhookServer(t.Context(), server, runtime, time.Second)
	if !errors.Is(err, managerErr) {
		t.Fatalf("wait for webhook server error = %v, want %v", err, managerErr)
	}

	t.Log("Stop without waiting for a second delivery of the manager result")
	err = runtime.stop()
	if !errors.Is(err, managerErr) {
		t.Fatalf("stop webhook runtime error = %v, want %v", err, managerErr)
	}
}
