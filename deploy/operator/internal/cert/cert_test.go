/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package cert

import (
	"context"
	"crypto/x509"
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/go-logr/logr"
	certrotator "github.com/open-policy-agent/cert-controller/pkg/rotator"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

// fakeCertProvisioner captures the rotator config passed to AddRotator and
// optionally simulates readiness by closing the IsReady channel.
type fakeCertProvisioner struct {
	called        bool
	capturedArgs  *certrotator.CertRotator
	simulateReady bool
	err           error
}

func (f *fakeCertProvisioner) AddRotator(_ ctrl.Manager, rotator *certrotator.CertRotator) error {
	f.called = true
	f.capturedArgs = rotator
	if f.simulateReady && rotator.IsReady != nil {
		close(rotator.IsReady)
	}
	return f.err
}

const (
	testSecretName  = "webhook-cert"
	testServiceName = "my-operator-webhook-service"
	testNamespace   = "test-ns"
)

func newScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	_ = corev1.AddToScheme(s)
	_ = admissionregistrationv1.AddToScheme(s)
	_ = apiextensionsv1.AddToScheme(s)
	return s
}

func newTestCertManager(cl *fake.ClientBuilder, cfg *configv1alpha1.WebhookServer) *CertManager {
	return &CertManager{
		client:    cl.Build(),
		cfg:       cfg,
		namespace: testNamespace,
		ready:     make(chan struct{}),
		logger:    logr.Discard(),
	}
}

func newTestInjector(cl *fake.ClientBuilder, cfg *configv1alpha1.OperatorConfiguration) *CABundleInjector {
	return &CABundleInjector{
		client:       cl.Build(),
		cfg:          cfg,
		namespace:    testNamespace,
		logger:       logr.Discard(),
		pollInterval: defaultCABundlePollInterval,
	}
}

func TestCreatePlaceholderSecretIfNotExists_CreatesWhenMissing(t *testing.T) {
	cfg := &configv1alpha1.WebhookServer{SecretName: testSecretName}
	cm := newTestCertManager(fake.NewClientBuilder().WithScheme(newScheme()), cfg)
	ctx := context.Background()

	if err := cm.createPlaceholderSecretIfNotExists(ctx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	secret := &corev1.Secret{}
	if err := cm.client.Get(ctx, types.NamespacedName{Namespace: testNamespace, Name: testSecretName}, secret); err != nil {
		t.Fatalf("secret should exist: %v", err)
	}

	if secret.Type != corev1.SecretTypeTLS {
		t.Errorf("expected TLS secret type, got %s", secret.Type)
	}
	if secret.Labels[partOfLabel] != partOfValue {
		t.Errorf("expected label %s=%s, got %s", partOfLabel, partOfValue, secret.Labels[partOfLabel])
	}
	for _, key := range []string{defaultCertName, defaultKeyName, defaultCACertName, defaultCAKeyName} {
		if _, ok := secret.Data[key]; !ok {
			t.Errorf("expected %s key in secret data", key)
		}
	}
}

func TestCreatePlaceholderSecretIfNotExists_NoopWhenExists(t *testing.T) {
	existing := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: testNamespace,
			Name:      testSecretName,
		},
		Type: corev1.SecretTypeTLS,
		Data: map[string][]byte{
			defaultCertName:   []byte("existing-cert"),
			defaultKeyName:    []byte("existing-key"),
			defaultCACertName: []byte("existing-ca"),
		},
	}
	cfg := &configv1alpha1.WebhookServer{SecretName: testSecretName}
	cm := newTestCertManager(fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(existing), cfg)
	ctx := context.Background()

	if err := cm.createPlaceholderSecretIfNotExists(ctx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	secret := &corev1.Secret{}
	if err := cm.client.Get(ctx, types.NamespacedName{Namespace: testNamespace, Name: testSecretName}, secret); err != nil {
		t.Fatalf("secret should exist: %v", err)
	}
	if string(secret.Data[defaultCertName]) != "existing-cert" {
		t.Error("existing secret data should not be overwritten")
	}
}

func TestCertManager_ManualModeClosesChannelImmediately(t *testing.T) {
	cfg := &configv1alpha1.WebhookServer{
		CertProvisionMode: configv1alpha1.CertProvisionModeManual,
		CertDir:           "/tmp/certs",
		SecretName:        testSecretName,
	}
	cm := newTestCertManager(fake.NewClientBuilder().WithScheme(newScheme()), cfg)

	if err := cm.SetupAndRunOnce(context.Background(), nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	select {
	case <-cm.ready:
	case <-time.After(time.Second):
		t.Fatal("ready channel should be closed immediately in manual mode")
	}
}

func TestCertManager_AutoModeConfiguresRotator(t *testing.T) {
	cfg := &configv1alpha1.WebhookServer{
		CertProvisionMode: configv1alpha1.CertProvisionModeAuto,
		CertDir:           "/tmp/certs",
		SecretName:        testSecretName,
		ServiceName:       testServiceName,
	}
	prov := &fakeCertProvisioner{simulateReady: true}
	cm := newTestCertManager(fake.NewClientBuilder().WithScheme(newScheme()), cfg)
	cm.provisioner = prov

	if err := cm.SetupAndRunOnce(context.Background(), nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !prov.called {
		t.Fatal("expected provisioner.AddRotator to be called")
	}

	extKeyUsages := []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}
	expected := &certrotator.CertRotator{
		SecretKey: types.NamespacedName{
			Namespace: testNamespace,
			Name:      testSecretName,
		},
		CertDir:            "/tmp/certs",
		CAName:             certificateAuthorityName,
		CAOrganization:     certificateAuthorityOrganization,
		IsReady:            cm.ready,
		DNSName:            fmt.Sprintf("%s.%s.svc", testServiceName, testNamespace),
		ExtKeyUsages:       &extKeyUsages,
		CaCertDuration:     defaultCACertValidityDuration,
		ServerCertDuration: defaultServerCertValidity,
		LookaheadInterval:  defaultLookaheadInterval,
		CertName:           defaultCertName,
		KeyName:            defaultKeyName,
		ExtraDNSNames: []string{
			testServiceName,
			fmt.Sprintf("%s.%s", testServiceName, testNamespace),
			fmt.Sprintf("%s.%s.svc.cluster.local", testServiceName, testNamespace),
		},
		EnableReadinessCheck: true,
	}

	if !reflect.DeepEqual(prov.capturedArgs, expected) {
		t.Errorf("rotator config mismatch\ngot:  %+v\nwant: %+v", prov.capturedArgs, expected)
	}

	// Verify placeholder secret was created
	secret := &corev1.Secret{}
	if err := cm.client.Get(context.Background(), types.NamespacedName{Namespace: testNamespace, Name: testSecretName}, secret); err != nil {
		t.Fatalf("webhook TLS secret should exist: %v", err)
	}
	for _, key := range []string{defaultCertName, defaultKeyName, defaultCACertName, defaultCAKeyName} {
		if len(secret.Data[key]) == 0 {
			t.Errorf("expected %s to be populated before rotator registration", key)
		}
	}
}

func TestCertManager_AutoModeProvisionerError(t *testing.T) {
	cfg := &configv1alpha1.WebhookServer{
		CertProvisionMode: configv1alpha1.CertProvisionModeAuto,
		SecretName:        testSecretName,
		ServiceName:       testServiceName,
	}
	prov := &fakeCertProvisioner{err: fmt.Errorf("rotator setup failed")}
	cm := newTestCertManager(fake.NewClientBuilder().WithScheme(newScheme()), cfg)
	cm.provisioner = prov

	err := cm.SetupAndRunOnce(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error from provisioner")
	}
	if !prov.called {
		t.Fatal("expected provisioner.AddRotator to be called")
	}
}

func TestInjectIntoValidatingWebhooks(t *testing.T) {
	wc := &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "test-validating",
			Labels: map[string]string{partOfLabel: partOfValue, operatorNamespaceLabel: testNamespace},
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name:                    "test.webhook.io",
				ClientConfig:            admissionregistrationv1.WebhookClientConfig{},
				SideEffects:             ptr.To(admissionregistrationv1.SideEffectClassNone),
				AdmissionReviewVersions: []string{"v1"},
			},
		},
	}

	cfg := &configv1alpha1.OperatorConfiguration{}
	cfg.Server.Webhook.SecretName = testSecretName
	injector := newTestInjector(fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(wc), cfg)
	ctx := context.Background()

	caBundle := []byte("test-ca-data")
	if err := injector.injectIntoValidatingWebhooks(ctx, caBundle); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	updated := &admissionregistrationv1.ValidatingWebhookConfiguration{}
	if err := injector.client.Get(ctx, types.NamespacedName{Name: "test-validating"}, updated); err != nil {
		t.Fatalf("failed to get webhook config: %v", err)
	}
	if string(updated.Webhooks[0].ClientConfig.CABundle) != "test-ca-data" {
		t.Errorf("expected CA bundle to be injected, got %q", string(updated.Webhooks[0].ClientConfig.CABundle))
	}
}

func TestInjectIntoValidatingWebhooks_SkipsNonMatchingLabels(t *testing.T) {
	wc := &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "other-validating",
			Labels: map[string]string{"app.kubernetes.io/part-of": "other-operator"},
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name:                    "other.webhook.io",
				ClientConfig:            admissionregistrationv1.WebhookClientConfig{},
				SideEffects:             ptr.To(admissionregistrationv1.SideEffectClassNone),
				AdmissionReviewVersions: []string{"v1"},
			},
		},
	}

	cfg := &configv1alpha1.OperatorConfiguration{}
	injector := &CABundleInjector{
		client:       fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(wc).Build(),
		cfg:          cfg,
		namespace:    "my-ns",
		logger:       logr.Discard(),
		pollInterval: defaultCABundlePollInterval,
	}
	ctx := context.Background()

	if err := injector.injectIntoValidatingWebhooks(ctx, []byte("test-ca")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	updated := &admissionregistrationv1.ValidatingWebhookConfiguration{}
	if err := injector.client.Get(ctx, types.NamespacedName{Name: "other-validating"}, updated); err != nil {
		t.Fatalf("failed to get webhook config: %v", err)
	}
	if updated.Webhooks[0].ClientConfig.CABundle != nil {
		t.Error("non-matching webhook config should not be patched")
	}
}

func TestInjectIntoValidatingWebhooks_SkipsDifferentNamespace(t *testing.T) {
	wc := &admissionregistrationv1.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "other-ns-validating",
			Labels: map[string]string{partOfLabel: partOfValue, operatorNamespaceLabel: "other-ns"},
		},
		Webhooks: []admissionregistrationv1.ValidatingWebhook{
			{
				Name:                    "test.webhook.io",
				ClientConfig:            admissionregistrationv1.WebhookClientConfig{},
				SideEffects:             ptr.To(admissionregistrationv1.SideEffectClassNone),
				AdmissionReviewVersions: []string{"v1"},
			},
		},
	}

	cfg := &configv1alpha1.OperatorConfiguration{}
	injector := &CABundleInjector{
		client:       fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(wc).Build(),
		cfg:          cfg,
		namespace:    "my-ns",
		logger:       logr.Discard(),
		pollInterval: defaultCABundlePollInterval,
	}
	ctx := context.Background()

	if err := injector.injectIntoValidatingWebhooks(ctx, []byte("test-ca")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	updated := &admissionregistrationv1.ValidatingWebhookConfiguration{}
	if err := injector.client.Get(ctx, types.NamespacedName{Name: "other-ns-validating"}, updated); err != nil {
		t.Fatalf("failed to get webhook config: %v", err)
	}
	if updated.Webhooks[0].ClientConfig.CABundle != nil {
		t.Error("webhook config from different operator namespace should not be patched")
	}
}

func TestInjectIntoMutatingWebhooks(t *testing.T) {
	wc := &admissionregistrationv1.MutatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "test-mutating",
			Labels: map[string]string{partOfLabel: partOfValue, operatorNamespaceLabel: testNamespace},
		},
		Webhooks: []admissionregistrationv1.MutatingWebhook{
			{
				Name:                    "mutate.webhook.io",
				ClientConfig:            admissionregistrationv1.WebhookClientConfig{},
				SideEffects:             ptr.To(admissionregistrationv1.SideEffectClassNone),
				AdmissionReviewVersions: []string{"v1"},
			},
		},
	}

	cfg := &configv1alpha1.OperatorConfiguration{}
	cfg.Server.Webhook.SecretName = testSecretName
	injector := newTestInjector(fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(wc), cfg)
	ctx := context.Background()

	caBundle := []byte("test-ca-data")
	if err := injector.injectIntoMutatingWebhooks(ctx, caBundle); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	updated := &admissionregistrationv1.MutatingWebhookConfiguration{}
	if err := injector.client.Get(ctx, types.NamespacedName{Name: "test-mutating"}, updated); err != nil {
		t.Fatalf("failed to get webhook config: %v", err)
	}
	if string(updated.Webhooks[0].ClientConfig.CABundle) != "test-ca-data" {
		t.Errorf("expected CA bundle to be injected, got %q", string(updated.Webhooks[0].ClientConfig.CABundle))
	}
}

func TestEnsureCRDConversionWebhooks(t *testing.T) {
	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: dgdrCRDName,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "nvidia.com",
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   "dynamographdeploymentrequests",
				Singular: "dynamographdeploymentrequest",
				Kind:     "DynamoGraphDeploymentRequest",
			},
			Scope: apiextensionsv1.NamespaceScoped,
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{Name: "v1alpha1", Served: true, Storage: true},
				{Name: "v1beta1", Served: true, Storage: false},
			},
		},
	}

	cfg := &configv1alpha1.OperatorConfiguration{}
	cfg.Server.Webhook.ServiceName = testServiceName
	injector := newTestInjector(fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(crd), cfg)
	ctx := context.Background()

	if err := injector.configureCRDConversionWebhooks(ctx, []byte("test-ca")); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	updated := &apiextensionsv1.CustomResourceDefinition{}
	if err := injector.client.Get(ctx, types.NamespacedName{Name: dgdrCRDName}, updated); err != nil {
		t.Fatalf("failed to get CRD: %v", err)
	}

	if updated.Spec.Conversion == nil {
		t.Fatal("expected conversion config to be set")
	}
	if updated.Spec.Conversion.Strategy != apiextensionsv1.WebhookConverter {
		t.Errorf("expected Webhook strategy, got %s", updated.Spec.Conversion.Strategy)
	}
	if updated.Spec.Conversion.Webhook.ClientConfig.Service.Name != testServiceName {
		t.Errorf("expected service name %s, got %s",
			testServiceName,
			updated.Spec.Conversion.Webhook.ClientConfig.Service.Name)
	}
	if updated.Spec.Conversion.Webhook.ClientConfig.Service.Namespace != testNamespace {
		t.Errorf("expected service namespace %s, got %s",
			testNamespace,
			updated.Spec.Conversion.Webhook.ClientConfig.Service.Namespace)
	}
	path := "/convert" //nolint:goconst
	if updated.Spec.Conversion.Webhook.ClientConfig.Service.Path == nil ||
		*updated.Spec.Conversion.Webhook.ClientConfig.Service.Path != path {
		t.Errorf("expected service path %s, got %v", path,
			updated.Spec.Conversion.Webhook.ClientConfig.Service.Path)
	}
	if string(updated.Spec.Conversion.Webhook.ClientConfig.CABundle) != "test-ca" {
		t.Errorf("expected CA bundle, got %q", string(updated.Spec.Conversion.Webhook.ClientConfig.CABundle))
	}
}

func TestEnsureCRDConversionWebhooksWithoutCABundle(t *testing.T) {
	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: dgdCRDName,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "nvidia.com",
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   "dynamographdeployments",
				Singular: "dynamographdeployment",
				Kind:     "DynamoGraphDeployment",
			},
			Scope: apiextensionsv1.NamespaceScoped,
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{Name: "v1alpha1", Served: true, Storage: true},
				{Name: "v1beta1", Served: true, Storage: false},
			},
		},
	}

	cfg := &configv1alpha1.OperatorConfiguration{}
	cfg.Server.Webhook.ServiceName = testServiceName
	injector := newTestInjector(fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(crd), cfg)
	ctx := context.Background()

	if err := injector.EnsureCRDConversionWebhooks(ctx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	updated := &apiextensionsv1.CustomResourceDefinition{}
	if err := injector.client.Get(ctx, types.NamespacedName{Name: dgdCRDName}, updated); err != nil {
		t.Fatalf("failed to get CRD: %v", err)
	}
	if updated.Spec.Conversion == nil {
		t.Fatal("expected conversion config to be set")
	}
	if len(updated.Spec.Conversion.Webhook.ClientConfig.CABundle) != 0 {
		t.Errorf("expected empty CA bundle, got %q", string(updated.Spec.Conversion.Webhook.ClientConfig.CABundle))
	}
}

func TestEnsureCRDConversionWebhooksPreservesExistingCABundle(t *testing.T) {
	path := "/old-convert"
	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: dgdCRDName,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "nvidia.com",
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   "dynamographdeployments",
				Singular: "dynamographdeployment",
				Kind:     "DynamoGraphDeployment",
			},
			Scope: apiextensionsv1.NamespaceScoped,
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{Name: "v1alpha1", Served: true, Storage: true},
				{Name: "v1beta1", Served: true, Storage: false},
			},
			Conversion: &apiextensionsv1.CustomResourceConversion{
				Strategy: apiextensionsv1.WebhookConverter,
				Webhook: &apiextensionsv1.WebhookConversion{
					ClientConfig: &apiextensionsv1.WebhookClientConfig{
						Service: &apiextensionsv1.ServiceReference{
							Name:      "old-service",
							Namespace: "old-namespace",
							Path:      &path,
						},
						CABundle: []byte("existing-ca"),
					},
				},
			},
		},
	}

	cfg := &configv1alpha1.OperatorConfiguration{}
	cfg.Server.Webhook.ServiceName = testServiceName
	injector := newTestInjector(fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(crd), cfg)
	ctx := context.Background()

	if err := injector.EnsureCRDConversionWebhooks(ctx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	updated := &apiextensionsv1.CustomResourceDefinition{}
	if err := injector.client.Get(ctx, types.NamespacedName{Name: dgdCRDName}, updated); err != nil {
		t.Fatalf("failed to get CRD: %v", err)
	}
	if string(updated.Spec.Conversion.Webhook.ClientConfig.CABundle) != "existing-ca" {
		t.Errorf("expected existing CA bundle to be preserved, got %q",
			string(updated.Spec.Conversion.Webhook.ClientConfig.CABundle))
	}
	if updated.Spec.Conversion.Webhook.ClientConfig.Service.Name != testServiceName {
		t.Errorf("expected service name %s, got %s",
			testServiceName,
			updated.Spec.Conversion.Webhook.ClientConfig.Service.Name)
	}
	if *updated.Spec.Conversion.Webhook.ClientConfig.Service.Path != "/convert" { //nolint:goconst
		t.Errorf("expected service path %s, got %s",
			"/convert", //nolint:goconst
			*updated.Spec.Conversion.Webhook.ClientConfig.Service.Path)
	}
}

func TestInjectCRDConversionCA_ReadsCABundleAndPatchesDGD(t *testing.T) {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: testNamespace,
			Name:      testSecretName,
		},
		Data: map[string][]byte{
			"tls.crt": []byte("cert-data"),
			"tls.key": []byte("key-data"),
			"ca.crt":  []byte("manual-ca"),
		},
	}
	crd := &apiextensionsv1.CustomResourceDefinition{
		ObjectMeta: metav1.ObjectMeta{
			Name: dgdCRDName,
		},
		Spec: apiextensionsv1.CustomResourceDefinitionSpec{
			Group: "nvidia.com",
			Names: apiextensionsv1.CustomResourceDefinitionNames{
				Plural:   "dynamographdeployments",
				Singular: "dynamographdeployment",
				Kind:     "DynamoGraphDeployment",
			},
			Scope: apiextensionsv1.NamespaceScoped,
			Versions: []apiextensionsv1.CustomResourceDefinitionVersion{
				{Name: "v1alpha1", Served: true, Storage: true},
				{Name: "v1beta1", Served: true, Storage: false},
			},
		},
	}

	cfg := &configv1alpha1.OperatorConfiguration{}
	cfg.Server.Webhook.SecretName = testSecretName
	cfg.Server.Webhook.ServiceName = testServiceName
	injector := newTestInjector(fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(secret, crd), cfg)
	ctx := context.Background()

	if err := injector.InjectCRDConversionCA(ctx); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	updated := &apiextensionsv1.CustomResourceDefinition{}
	if err := injector.client.Get(ctx, types.NamespacedName{Name: dgdCRDName}, updated); err != nil {
		t.Fatalf("failed to get CRD: %v", err)
	}

	if updated.Spec.Conversion == nil {
		t.Fatal("expected conversion config to be set")
	}
	if updated.Spec.Conversion.Strategy != apiextensionsv1.WebhookConverter {
		t.Errorf("expected Webhook strategy, got %s", updated.Spec.Conversion.Strategy)
	}
	if updated.Spec.Conversion.Webhook.ClientConfig.Service.Name != testServiceName {
		t.Errorf("expected service name %s, got %s",
			testServiceName,
			updated.Spec.Conversion.Webhook.ClientConfig.Service.Name)
	}
	if updated.Spec.Conversion.Webhook.ClientConfig.Service.Namespace != testNamespace {
		t.Errorf("expected service namespace %s, got %s",
			testNamespace,
			updated.Spec.Conversion.Webhook.ClientConfig.Service.Namespace)
	}
	path := "/convert" //nolint:goconst
	if updated.Spec.Conversion.Webhook.ClientConfig.Service.Path == nil ||
		*updated.Spec.Conversion.Webhook.ClientConfig.Service.Path != path {
		t.Errorf("expected service path %s, got %v", path,
			updated.Spec.Conversion.Webhook.ClientConfig.Service.Path)
	}
	if string(updated.Spec.Conversion.Webhook.ClientConfig.CABundle) != "manual-ca" {
		t.Errorf("expected CA bundle, got %q", string(updated.Spec.Conversion.Webhook.ClientConfig.CABundle))
	}
}

func TestInjectCRDConversionCA_WaitsWhenSecretNotFound(t *testing.T) {
	cfg := &configv1alpha1.OperatorConfiguration{}
	cfg.Server.Webhook.SecretName = testSecretName
	cfg.Server.Webhook.ServiceName = testServiceName
	injector := newTestInjector(fake.NewClientBuilder().WithScheme(newScheme()), cfg)
	ctx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()

	err := injector.InjectCRDConversionCA(ctx)
	if err == nil {
		t.Fatal("expected context timeout while waiting for missing secret")
	}
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("expected deadline exceeded, got: %v", err)
	}
}

func TestEnsureCRDConversionWebhooks_SkipsWhenCRDNotFound(t *testing.T) {
	cfg := &configv1alpha1.OperatorConfiguration{}
	cfg.Server.Webhook.ServiceName = testServiceName
	injector := newTestInjector(fake.NewClientBuilder().WithScheme(newScheme()), cfg)
	ctx := context.Background()

	if err := injector.configureCRDConversionWebhooks(ctx, []byte("test-ca")); err != nil {
		t.Fatalf("expected no error when CRD not found, got: %v", err)
	}
}

func TestReadCABundle(t *testing.T) {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: testNamespace,
			Name:      testSecretName,
		},
		Data: map[string][]byte{
			"tls.crt": []byte("cert-data"),
			"tls.key": []byte("key-data"),
			"ca.crt":  []byte("ca-data"),
		},
	}
	cfg := &configv1alpha1.OperatorConfiguration{}
	cfg.Server.Webhook.SecretName = testSecretName
	injector := newTestInjector(fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(secret), cfg)

	ca, err := injector.readCABundle(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if string(ca) != "ca-data" {
		t.Errorf("expected ca-data, got %q", string(ca))
	}
}

func TestReadCABundle_ErrorOnMissingCA(t *testing.T) {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: testNamespace,
			Name:      testSecretName,
		},
		Data: map[string][]byte{
			"tls.crt": []byte("cert-data"),
			"tls.key": []byte("key-data"),
		},
	}
	cfg := &configv1alpha1.OperatorConfiguration{}
	cfg.Server.Webhook.SecretName = testSecretName
	injector := newTestInjector(fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(secret), cfg)

	if _, err := injector.readCABundle(context.Background()); err == nil {
		t.Fatal("expected error when ca.crt is missing")
	}
}
