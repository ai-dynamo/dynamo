/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package cert

import (
	"context"
	"fmt"
	"os"
	"strings"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/go-logr/logr"
	certrotator "github.com/open-policy-agent/cert-controller/pkg/rotator"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	certificateAuthorityName         = "Dynamo-Webhook-CA"
	certificateAuthorityOrganization = "NVIDIA"
	namespaceFile                    = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
	dgdrCRDName                      = "dynamographdeploymentrequests.nvidia.com"
	partOfLabel                      = "app.kubernetes.io/part-of"
	partOfValue                      = "dynamo-operator"
	operatorNamespaceLabel           = "nvidia.com/dynamo-operator-namespace"
)

// CertManager manages webhook TLS certificate lifecycle.
// In auto mode it uses the OPA cert-controller for generation and rotation.
// In manual mode it expects externally provided certificates and signals
// readiness immediately.
type CertManager struct {
	client    client.Client
	cfg       *configv1alpha1.WebhookServer
	namespace string
	ready     chan struct{}
	logger    logr.Logger
}

// NewCertManager creates a CertManager. The client should be a direct
// (non-cached) client because the manager's cache isn't started yet when
// Setup is called. Only used to create the placeholder secret in auto mode;
// RBAC is the actual access boundary.
func NewCertManager(cl client.Client, cfg *configv1alpha1.WebhookServer) (*CertManager, error) {
	ns, err := getOperatorNamespace()
	if err != nil {
		return nil, fmt.Errorf("reading operator namespace: %w", err)
	}
	return &CertManager{
		client:    cl,
		cfg:       cfg,
		namespace: ns,
		ready:     make(chan struct{}),
		logger:    ctrl.Log.WithName("cert-manager"),
	}, nil
}

// Setup configures certificate management and adds the cert-controller to
// the manager (auto mode) or closes the ready channel immediately (manual mode).
func (cm *CertManager) Setup(ctx context.Context, mgr ctrl.Manager) error {
	switch cm.cfg.CertProvisionMode {
	case configv1alpha1.CertProvisionModeManual:
		cm.logger.Info("Using externally provided certificates (manual mode)",
			"certDir", cm.cfg.CertDir, "secretName", cm.cfg.SecretName)
		close(cm.ready)
		return nil

	case configv1alpha1.CertProvisionModeAuto:
		return cm.setupAutoProvisioning(ctx, mgr)

	default:
		return fmt.Errorf("unsupported cert provision mode: %q", cm.cfg.CertProvisionMode)
	}
}

// WaitReady blocks until certificates have been written to the cert directory.
func (cm *CertManager) WaitReady() {
	cm.logger.Info("Waiting for webhook certificates to be ready")
	<-cm.ready
	cm.logger.Info("Webhook certificates are ready")
}

// IsReady returns true if certificates are available (non-blocking).
func (cm *CertManager) IsReady() bool {
	select {
	case <-cm.ready:
		return true
	default:
		return false
	}
}

func (cm *CertManager) setupAutoProvisioning(ctx context.Context, mgr ctrl.Manager) error {
	if err := cm.createPlaceholderSecretIfNotExists(ctx); err != nil {
		return fmt.Errorf("ensuring webhook TLS secret exists: %w", err)
	}

	dnsName := fmt.Sprintf("%s.%s.svc", cm.cfg.ServiceName, cm.namespace)

	cm.logger.Info("Auto-provisioning certificates using cert-controller",
		"secretName", cm.cfg.SecretName, "dnsName", dnsName)

	rotator := &certrotator.CertRotator{
		SecretKey: types.NamespacedName{
			Namespace: cm.namespace,
			Name:      cm.cfg.SecretName,
		},
		CertDir:        cm.cfg.CertDir,
		CAName:         certificateAuthorityName,
		CAOrganization: certificateAuthorityOrganization,
		IsReady:        cm.ready,
		DNSName:        dnsName,
		ExtraDNSNames: []string{
			cm.cfg.ServiceName,
			fmt.Sprintf("%s.%s", cm.cfg.ServiceName, cm.namespace),
			fmt.Sprintf("%s.%s.svc.cluster.local", cm.cfg.ServiceName, cm.namespace),
		},
		EnableReadinessCheck:   true,
		RestartOnSecretRefresh: true,
	}
	return certrotator.AddRotator(mgr, rotator)
}

// createPlaceholderSecretIfNotExists creates the webhook TLS secret if it does
// not already exist. The OPA cert-controller can only Update existing secrets,
// not Create them. If the secret already exists it is left untouched.
func (cm *CertManager) createPlaceholderSecretIfNotExists(ctx context.Context) error {
	err := cm.client.Get(ctx, types.NamespacedName{Namespace: cm.namespace, Name: cm.cfg.SecretName}, &corev1.Secret{})
	if !apierrors.IsNotFound(err) {
		return err
	}

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: cm.namespace,
			Name:      cm.cfg.SecretName,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
				"app.kubernetes.io/component":  "webhook",
				partOfLabel:                    partOfValue,
			},
		},
		Type: corev1.SecretTypeTLS,
		Data: map[string][]byte{
			"tls.crt": {},
			"tls.key": {},
			"ca.crt":  {},
		},
	}
	if err := cm.client.Create(ctx, secret); err != nil {
		if apierrors.IsAlreadyExists(err) {
			return nil
		}
		return fmt.Errorf("creating webhook TLS secret: %w", err)
	}

	cm.logger.Info("Created webhook TLS secret", "namespace", cm.namespace, "name", cm.cfg.SecretName)
	return nil
}

// CABundleInjector discovers webhook configurations owned by this operator
// instance and patches them with the CA bundle from the cert secret.
type CABundleInjector struct {
	client    client.Client
	cfg       *configv1alpha1.OperatorConfiguration
	namespace string
	logger    logr.Logger
}

// NewCABundleInjector creates a CABundleInjector. The client should be the
// manager's cached client (available after mgr.Start).
func NewCABundleInjector(cl client.Client, cfg *configv1alpha1.OperatorConfiguration) (*CABundleInjector, error) {
	ns, err := getOperatorNamespace()
	if err != nil {
		return nil, fmt.Errorf("reading operator namespace: %w", err)
	}
	return &CABundleInjector{
		client:    cl,
		cfg:       cfg,
		namespace: ns,
		logger:    ctrl.Log.WithName("ca-bundle-injector"),
	}, nil
}

// InjectAll reads the CA bundle from the cert secret and injects it into all
// webhook configurations owned by this operator instance (scoped by namespace
// label), and into the DGDR CRD conversion webhook.
func (i *CABundleInjector) InjectAll(ctx context.Context) error {
	caBundle, err := i.readCABundle(ctx)
	if err != nil {
		return fmt.Errorf("reading CA bundle from secret %s/%s: %w", i.namespace, i.cfg.Server.Webhook.SecretName, err)
	}

	if err := i.injectIntoValidatingWebhooks(ctx, caBundle); err != nil {
		return err
	}
	if err := i.injectIntoMutatingWebhooks(ctx, caBundle); err != nil {
		return err
	}
	if err := i.ensureCRDConversion(ctx, caBundle); err != nil {
		return err
	}

	i.logger.Info("CA bundle injected into all webhook configurations")
	return nil
}

func (i *CABundleInjector) readCABundle(ctx context.Context) ([]byte, error) {
	secret := &corev1.Secret{}
	if err := i.client.Get(ctx, types.NamespacedName{Namespace: i.namespace, Name: i.cfg.Server.Webhook.SecretName}, secret); err != nil {
		return nil, err
	}
	ca, ok := secret.Data["ca.crt"]
	if !ok || len(ca) == 0 {
		return nil, fmt.Errorf("ca.crt not found or empty in secret %s/%s", i.namespace, i.cfg.Server.Webhook.SecretName)
	}
	return ca, nil
}

func (i *CABundleInjector) webhookLabels() client.MatchingLabels {
	return client.MatchingLabels{
		partOfLabel:            partOfValue,
		operatorNamespaceLabel: i.namespace,
	}
}

func (i *CABundleInjector) injectIntoValidatingWebhooks(ctx context.Context, caBundle []byte) error {
	list := &admissionregistrationv1.ValidatingWebhookConfigurationList{}
	if err := i.client.List(ctx, list, i.webhookLabels()); err != nil {
		return fmt.Errorf("listing validating webhook configurations: %w", err)
	}
	for idx := range list.Items {
		wc := &list.Items[idx]
		for j := range wc.Webhooks {
			wc.Webhooks[j].ClientConfig.CABundle = caBundle
		}
		if err := i.client.Update(ctx, wc); err != nil {
			return fmt.Errorf("patching validating webhook config %s: %w", wc.Name, err)
		}
		i.logger.Info("Injected CA bundle into ValidatingWebhookConfiguration", "name", wc.Name)
	}
	return nil
}

func (i *CABundleInjector) injectIntoMutatingWebhooks(ctx context.Context, caBundle []byte) error {
	list := &admissionregistrationv1.MutatingWebhookConfigurationList{}
	if err := i.client.List(ctx, list, i.webhookLabels()); err != nil {
		return fmt.Errorf("listing mutating webhook configurations: %w", err)
	}
	for idx := range list.Items {
		wc := &list.Items[idx]
		for j := range wc.Webhooks {
			wc.Webhooks[j].ClientConfig.CABundle = caBundle
		}
		if err := i.client.Update(ctx, wc); err != nil {
			return fmt.Errorf("patching mutating webhook config %s: %w", wc.Name, err)
		}
		i.logger.Info("Injected CA bundle into MutatingWebhookConfiguration", "name", wc.Name)
	}
	return nil
}

// ensureCRDConversion patches the DGDR CRD with the conversion webhook config,
// setting the caBundle and service reference to this operator's webhook service.
func (i *CABundleInjector) ensureCRDConversion(ctx context.Context, caBundle []byte) error {
	crd := &apiextensionsv1.CustomResourceDefinition{}
	if err := i.client.Get(ctx, types.NamespacedName{Name: dgdrCRDName}, crd); err != nil {
		if apierrors.IsNotFound(err) {
			i.logger.Info("DGDR CRD not found, skipping conversion webhook setup")
			return nil
		}
		return fmt.Errorf("getting CRD %s: %w", dgdrCRDName, err)
	}

	path := "/convert"
	crd.Spec.Conversion = &apiextensionsv1.CustomResourceConversion{
		Strategy: apiextensionsv1.WebhookConverter,
		Webhook: &apiextensionsv1.WebhookConversion{
			ClientConfig: &apiextensionsv1.WebhookClientConfig{
				Service: &apiextensionsv1.ServiceReference{
					Name:      i.cfg.Server.Webhook.ServiceName,
					Namespace: i.namespace,
					Path:      &path,
				},
				CABundle: caBundle,
			},
			ConversionReviewVersions: []string{"v1"},
		},
	}

	if err := i.client.Update(ctx, crd); err != nil {
		return fmt.Errorf("patching CRD %s conversion config: %w", dgdrCRDName, err)
	}
	i.logger.Info("Configured CRD conversion webhook", "crd", dgdrCRDName)
	return nil
}

func getOperatorNamespace() (string, error) {
	data, err := os.ReadFile(namespaceFile)
	if err != nil {
		return "", fmt.Errorf("reading namespace from %s: %w", namespaceFile, err)
	}
	ns := strings.TrimSpace(string(data))
	if len(ns) == 0 {
		return "", fmt.Errorf("operator namespace is empty")
	}
	return ns, nil
}
