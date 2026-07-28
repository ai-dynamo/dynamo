/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package cert

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/go-logr/logr"
	certrotator "github.com/open-policy-agent/cert-controller/pkg/rotator"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	corev1 "k8s.io/api/core/v1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	certificateAuthorityName         = "Dynamo-Webhook-CA"
	certificateAuthorityOrganization = "NVIDIA"
	namespaceFile                    = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
	dgdrCRDName                      = "dynamographdeploymentrequests.nvidia.com"
	dgdCRDName                       = "dynamographdeployments.nvidia.com"
	dcdCRDName                       = "dynamocomponentdeployments.nvidia.com"
	dgdsaCRDName                     = "dynamographdeploymentscalingadapters.nvidia.com"
	defaultCABundlePollInterval      = 500 * time.Millisecond
	defaultCertName                  = "tls.crt"
	defaultKeyName                   = "tls.key"
	defaultCACertName                = "ca.crt"
	defaultCAKeyName                 = "ca.key"
	defaultCACertValidityDuration    = 10 * 365 * 24 * time.Hour
	defaultServerCertValidity        = 365 * 24 * time.Hour
	defaultLookaheadInterval         = 90 * 24 * time.Hour
	partOfLabel                      = "app.kubernetes.io/part-of"
	partOfValue                      = "dynamo-operator"
	operatorNamespaceLabel           = "nvidia.com/dynamo-operator-namespace"
	defaultMountedCertPollInterval   = 500 * time.Millisecond
)

// convertibleCRDs is the list of CRDs whose conversion webhook this operator
// patches at startup. Each of these resources has at least two API versions
// with distinct shapes (see api/v1alpha1/*_conversion.go) and relies on the
// operator's /convert endpoint to translate between them.
var convertibleCRDs = []string{
	dgdrCRDName,
	dgdCRDName,
	dcdCRDName,
	dgdsaCRDName,
}

// CertProvisioner abstracts the mechanism that adds a certificate rotator to
// the controller-runtime manager. The default implementation delegates to the
// OPA cert-controller; tests can substitute a stub.
type CertProvisioner interface {
	AddRotator(mgr ctrl.Manager, rotator *certrotator.CertRotator) error
}

// opaCertProvisioner is the production implementation backed by the OPA
// cert-controller library.
type opaCertProvisioner struct{}

func (opaCertProvisioner) AddRotator(mgr ctrl.Manager, rotator *certrotator.CertRotator) error {
	return certrotator.AddRotator(mgr, rotator)
}

// CertManager manages webhook TLS certificate lifecycle.
// In auto mode it uses a CertProvisioner for generation and rotation.
// In manual mode it expects externally provided certificates and signals
// readiness immediately.
type CertManager struct {
	client      client.Client
	cfg         *configv1alpha1.WebhookServer
	namespace   string
	ready       chan struct{}
	logger      logr.Logger
	provisioner CertProvisioner
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
		client:      cl,
		cfg:         cfg,
		namespace:   ns,
		ready:       make(chan struct{}),
		logger:      ctrl.Log.WithName("cert-manager"),
		provisioner: opaCertProvisioner{},
	}, nil
}

// SetupAndRunOnce configures certificate management before the manager starts.
// Auto mode runs the first certificate refresh synchronously through the direct
// client, then adds the cert-controller to the not-yet-started manager for
// later rotation. Manual mode expects externally provided certificates.
func (cm *CertManager) SetupAndRunOnce(ctx context.Context, mgr ctrl.Manager) error {
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

// WaitForMountedCertificate waits until the webhook server certificate and key
// are available through the mounted Secret volume. The Secret API object may be
// created or updated before kubelet has projected valid files into the pod.
func (cm *CertManager) WaitForMountedCertificate(ctx context.Context) error {
	return cm.waitForMountedCertificate(ctx, defaultMountedCertPollInterval)
}

func (cm *CertManager) waitForMountedCertificate(ctx context.Context, pollInterval time.Duration) error {
	certPath := filepath.Join(cm.cfg.CertDir, defaultCertName)
	keyPath := filepath.Join(cm.cfg.CertDir, defaultKeyName)
	var lastErr error
	err := wait.PollUntilContextCancel(ctx, pollInterval, true, func(ctx context.Context) (bool, error) {
		if _, err := tls.LoadX509KeyPair(certPath, keyPath); err != nil {
			lastErr = err
			cm.logger.Info("Waiting for webhook TLS certificate files",
				"cert", certPath, "key", keyPath, "error", err.Error())
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		if lastErr != nil {
			return fmt.Errorf("waiting for webhook TLS certificate files %s and %s: %w (last read error: %v)",
				certPath, keyPath, err, lastErr)
		}
		return fmt.Errorf("waiting for webhook TLS certificate files %s and %s: %w", certPath, keyPath, err)
	}

	cm.logger.Info("Webhook TLS certificate files are ready", "cert", certPath, "key", keyPath)
	return nil
}

func (cm *CertManager) setupAutoProvisioning(ctx context.Context, mgr ctrl.Manager) error {
	if err := cm.createPlaceholderSecretIfNotExists(ctx); err != nil {
		return fmt.Errorf("ensuring webhook TLS secret exists: %w", err)
	}

	rotator := cm.newCertRotator()
	if err := cm.bootstrapCertSecret(ctx, rotator); err != nil {
		return fmt.Errorf("bootstrapping webhook TLS secret: %w", err)
	}

	cm.logger.Info("Auto-provisioning certificates using cert-controller",
		"secretName", cm.cfg.SecretName, "dnsName", rotator.DNSName)

	return cm.provisioner.AddRotator(mgr, rotator)
}

func (cm *CertManager) newCertRotator() *certrotator.CertRotator {
	dnsName := fmt.Sprintf("%s.%s.svc", cm.cfg.ServiceName, cm.namespace)
	extKeyUsages := []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth}
	return &certrotator.CertRotator{
		SecretKey: types.NamespacedName{
			Namespace: cm.namespace,
			Name:      cm.cfg.SecretName,
		},
		CertDir:            cm.cfg.CertDir,
		CAName:             certificateAuthorityName,
		CAOrganization:     certificateAuthorityOrganization,
		IsReady:            cm.ready,
		DNSName:            dnsName,
		ExtKeyUsages:       &extKeyUsages,
		CaCertDuration:     defaultCACertValidityDuration,
		ServerCertDuration: defaultServerCertValidity,
		LookaheadInterval:  defaultLookaheadInterval,
		CertName:           defaultCertName,
		KeyName:            defaultKeyName,
		ExtraDNSNames: []string{
			cm.cfg.ServiceName,
			fmt.Sprintf("%s.%s", cm.cfg.ServiceName, cm.namespace),
			fmt.Sprintf("%s.%s.svc.cluster.local", cm.cfg.ServiceName, cm.namespace),
		},
		EnableReadinessCheck: true,
		// RestartOnSecretRefresh is intentionally false (default). Startup
		// bootstraps the secret before mgr.Start; later rotations must not exit
		// immediately after updating the secret and race kubelet projection.
	}
}

func (cm *CertManager) bootstrapCertSecret(ctx context.Context, rotator *certrotator.CertRotator) error {
	secret := &corev1.Secret{}
	if err := cm.client.Get(ctx, rotator.SecretKey, secret); err != nil {
		return fmt.Errorf("getting webhook TLS secret: %w", err)
	}

	if secret.Data == nil || !validCACert(rotator, secret) {
		cm.logger.Info("Refreshing webhook CA and server certificate before manager start")
		return cm.refreshCertSecret(ctx, rotator, secret, true)
	}
	if !validServerCert(rotator, secret) {
		cm.logger.Info("Refreshing webhook server certificate before manager start")
		return cm.refreshCertSecret(ctx, rotator, secret, false)
	}

	cm.logger.Info("Webhook certificates are already valid before manager start")
	return nil
}

func (cm *CertManager) refreshCertSecret(
	ctx context.Context,
	rotator *certrotator.CertRotator,
	secret *corev1.Secret,
	refreshCA bool,
) error {
	now := time.Now()
	begin := now.Add(-1 * time.Hour)
	var caArtifacts *certrotator.KeyPairArtifacts
	var err error
	if refreshCA {
		caArtifacts, err = rotator.CreateCACert(begin, now.Add(rotator.CaCertDuration))
	} else {
		caArtifacts, err = buildCAArtifactsFromSecret(secret)
	}
	if err != nil {
		return err
	}

	cert, key, err := rotator.CreateCertPEM(caArtifacts, begin, now.Add(rotator.ServerCertDuration))
	if err != nil {
		return err
	}
	if secret.Data == nil {
		secret.Data = make(map[string][]byte)
	}
	secret.Data[defaultCACertName] = caArtifacts.CertPEM
	secret.Data[defaultCAKeyName] = caArtifacts.KeyPEM
	secret.Data[rotator.CertName] = cert
	secret.Data[rotator.KeyName] = key
	if err := cm.client.Update(ctx, secret); err != nil {
		return fmt.Errorf("updating webhook TLS secret: %w", err)
	}
	return nil
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
			defaultCertName:   {},
			defaultKeyName:    {},
			defaultCACertName: {},
			defaultCAKeyName:  {},
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
	client       client.Client
	cfg          *configv1alpha1.OperatorConfiguration
	namespace    string
	logger       logr.Logger
	pollInterval time.Duration
}

// CABundleInjectorOption configures optional CABundleInjector behavior.
type CABundleInjectorOption func(*CABundleInjector)

// WithCABundlePollInterval overrides the CA bundle polling interval.
func WithCABundlePollInterval(interval time.Duration) CABundleInjectorOption {
	return func(i *CABundleInjector) {
		i.pollInterval = interval
	}
}

// NewCABundleInjector creates a CABundleInjector. Use a direct client before
// mgr.Start and the manager client after its cache is running.
func NewCABundleInjector(cl client.Client, cfg *configv1alpha1.OperatorConfiguration, opts ...CABundleInjectorOption) (*CABundleInjector, error) {
	ns, err := getOperatorNamespace()
	if err != nil {
		return nil, fmt.Errorf("reading operator namespace: %w", err)
	}
	injector := &CABundleInjector{
		client:       cl,
		cfg:          cfg,
		namespace:    ns,
		logger:       ctrl.Log.WithName("ca-bundle-injector"),
		pollInterval: defaultCABundlePollInterval,
	}
	for _, opt := range opts {
		opt(injector)
	}
	return injector, nil
}

// InjectAll reads the CA bundle from the cert secret and injects it into all
// webhook configurations owned by this operator instance (scoped by namespace
// label), and into the multi-version CRD conversion webhooks.
func (i *CABundleInjector) InjectAll(ctx context.Context) error {
	caBundle, err := i.readCABundle(ctx)
	if err != nil {
		return fmt.Errorf("reading CA bundle from secret %s/%s: %w", i.namespace, i.cfg.Server.Webhook.SecretName, err)
	}

	if err := i.injectAdmission(ctx, caBundle); err != nil {
		return err
	}
	if err := i.ensureCRDConversionCA(ctx, caBundle); err != nil {
		return err
	}

	i.logger.Info("CA bundle injected into all webhook configurations")
	return nil
}

// InjectAdmission reads the CA bundle from the cert secret and injects it only
// into admission webhook configurations owned by this operator instance.
func (i *CABundleInjector) InjectAdmission(ctx context.Context) error {
	caBundle, err := i.readCABundle(ctx)
	if err != nil {
		return fmt.Errorf("reading CA bundle from secret %s/%s: %w", i.namespace, i.cfg.Server.Webhook.SecretName, err)
	}

	if err := i.injectAdmission(ctx, caBundle); err != nil {
		return err
	}

	i.logger.Info("CA bundle injected into admission webhook configurations")
	return nil
}

func (i *CABundleInjector) injectAdmission(ctx context.Context, caBundle []byte) error {
	if err := i.injectIntoValidatingWebhooks(ctx, caBundle); err != nil {
		return err
	}
	return i.injectIntoMutatingWebhooks(ctx, caBundle)
}

// InjectCRDConversionCA reads the CA bundle from the cert secret and patches it
// into the CRD conversion webhook configurations.
func (i *CABundleInjector) InjectCRDConversionCA(ctx context.Context) error {
	caBundle, err := i.waitForCABundle(ctx)
	if err != nil {
		return err
	}
	if err := i.ensureCRDConversionCA(ctx, caBundle); err != nil {
		return err
	}
	i.logger.Info("CRD conversion webhook CA bundle injected")
	return nil
}

func (i *CABundleInjector) waitForCABundle(ctx context.Context) ([]byte, error) {
	var caBundle []byte
	err := wait.PollUntilContextCancel(ctx, i.pollInterval, true, func(ctx context.Context) (bool, error) {
		secret := &corev1.Secret{}
		err := i.client.Get(ctx, types.NamespacedName{Namespace: i.namespace, Name: i.cfg.Server.Webhook.SecretName}, secret)
		if apierrors.IsNotFound(err) {
			i.logger.Info("Waiting for webhook CA bundle",
				"namespace", i.namespace,
				"secret", i.cfg.Server.Webhook.SecretName,
				"error", err.Error())
			return false, nil
		}
		if err != nil {
			return false, err
		}
		ca, ok := secret.Data[defaultCACertName]
		if !ok || len(ca) == 0 {
			i.logger.Info("Waiting for webhook CA bundle",
				"namespace", i.namespace,
				"secret", i.cfg.Server.Webhook.SecretName,
				"error", "ca.crt not found or empty")
			return false, nil
		}
		caBundle = ca
		return true, nil
	})
	if err != nil {
		return nil, fmt.Errorf("waiting for CA bundle in secret %s/%s: %w", i.namespace, i.cfg.Server.Webhook.SecretName, err)
	}
	return caBundle, nil
}

func (i *CABundleInjector) readCABundle(ctx context.Context) ([]byte, error) {
	secret := &corev1.Secret{}
	if err := i.client.Get(ctx, types.NamespacedName{Namespace: i.namespace, Name: i.cfg.Server.Webhook.SecretName}, secret); err != nil {
		return nil, err
	}
	ca, ok := secret.Data[defaultCACertName]
	if !ok || len(ca) == 0 {
		return nil, fmt.Errorf("%s not found or empty in secret %s/%s", defaultCACertName, i.namespace, i.cfg.Server.Webhook.SecretName)
	}
	return ca, nil
}

func validCACert(rotator *certrotator.CertRotator, secret *corev1.Secret) bool {
	valid, err := certrotator.ValidCert(
		secret.Data[defaultCACertName],
		secret.Data[defaultCACertName],
		secret.Data[defaultCAKeyName],
		rotator.CAName,
		nil,
		time.Now().Add(rotator.LookaheadInterval),
	)
	return err == nil && valid
}

func validServerCert(rotator *certrotator.CertRotator, secret *corev1.Secret) bool {
	valid, err := certrotator.ValidCert(
		secret.Data[defaultCACertName],
		secret.Data[rotator.CertName],
		secret.Data[rotator.KeyName],
		rotator.DNSName,
		rotator.ExtKeyUsages,
		time.Now().Add(rotator.LookaheadInterval),
	)
	return err == nil && valid
}

func buildCAArtifactsFromSecret(secret *corev1.Secret) (*certrotator.KeyPairArtifacts, error) {
	caPEM, ok := secret.Data[defaultCACertName]
	if !ok {
		return nil, fmt.Errorf("webhook TLS secret is missing %s", defaultCACertName)
	}
	keyPEM, ok := secret.Data[defaultCAKeyName]
	if !ok {
		return nil, fmt.Errorf("webhook TLS secret is missing %s", defaultCAKeyName)
	}

	caDER, _ := pem.Decode(caPEM)
	if caDER == nil {
		return nil, fmt.Errorf("failed to decode %s", defaultCACertName)
	}
	caCert, err := x509.ParseCertificate(caDER.Bytes)
	if err != nil {
		return nil, fmt.Errorf("parsing %s: %w", defaultCACertName, err)
	}

	keyDER, _ := pem.Decode(keyPEM)
	if keyDER == nil {
		return nil, fmt.Errorf("failed to decode %s", defaultCAKeyName)
	}
	caKey, err := x509.ParsePKCS1PrivateKey(keyDER.Bytes)
	if err != nil {
		return nil, fmt.Errorf("parsing %s: %w", defaultCAKeyName, err)
	}

	return &certrotator.KeyPairArtifacts{
		Cert:    caCert,
		Key:     caKey,
		CertPEM: caPEM,
		KeyPEM:  keyPEM,
	}, nil
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
		original := wc.DeepCopy()
		for j := range wc.Webhooks {
			wc.Webhooks[j].ClientConfig.CABundle = caBundle
		}
		if err := i.client.Patch(ctx, wc, client.MergeFrom(original)); err != nil {
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
		original := wc.DeepCopy()
		for j := range wc.Webhooks {
			wc.Webhooks[j].ClientConfig.CABundle = caBundle
		}
		if err := i.client.Patch(ctx, wc, client.MergeFrom(original)); err != nil {
			return fmt.Errorf("patching mutating webhook config %s: %w", wc.Name, err)
		}
		i.logger.Info("Injected CA bundle into MutatingWebhookConfiguration", "name", wc.Name)
	}
	return nil
}

// ensureCRDConversionCA patches the CA bundle on the conversion webhooks that
// are already present in the CRD manifests. Missing CRDs are tolerated with an
// info-level log so that a standalone operator image can be brought up before
// the CRDs are installed (helm install idempotency).
func (i *CABundleInjector) ensureCRDConversionCA(ctx context.Context, caBundle []byte) error {
	for _, name := range convertibleCRDs {
		if err := i.patchCRDConversionCA(ctx, name, caBundle); err != nil {
			return err
		}
	}
	return nil
}

func (i *CABundleInjector) patchCRDConversionCA(ctx context.Context, crdName string, caBundle []byte) error {
	crd := &apiextensionsv1.CustomResourceDefinition{}
	if err := i.client.Get(ctx, types.NamespacedName{Name: crdName}, crd); err != nil {
		if apierrors.IsNotFound(err) {
			i.logger.Info("CRD not found, skipping conversion webhook CA injection", "crd", crdName)
			return nil
		}
		return fmt.Errorf("getting CRD %s: %w", crdName, err)
	}

	original := crd.DeepCopy()
	if crd.Spec.Conversion == nil ||
		crd.Spec.Conversion.Strategy != apiextensionsv1.WebhookConverter ||
		crd.Spec.Conversion.Webhook == nil ||
		crd.Spec.Conversion.Webhook.ClientConfig == nil ||
		crd.Spec.Conversion.Webhook.ClientConfig.Service == nil {
		return fmt.Errorf("CRD %s is missing conversion webhook configuration; regenerate and apply CRD manifests", crdName)
	}
	crd.Spec.Conversion.Webhook.ClientConfig.CABundle = caBundle

	if err := i.client.Patch(ctx, crd, client.MergeFrom(original)); err != nil {
		return fmt.Errorf("patching CRD %s conversion CA bundle: %w", crdName, err)
	}
	i.logger.Info("Injected CA bundle into CRD conversion webhook", "crd", crdName)
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
