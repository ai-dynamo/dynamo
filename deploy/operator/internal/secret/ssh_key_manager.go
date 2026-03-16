/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package secret

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"crypto/x509"
	"encoding/pem"
	"fmt"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/go-logr/logr"
	"golang.org/x/crypto/ssh"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	rsaKeyBits     = 2048
	privateKeyFile = "private.key"
	publicKeyFile  = "private.key.pub"
)

// KeyPairGenerator abstracts SSH key pair generation for testability.
type KeyPairGenerator interface {
	Generate() (privateKeyPEM, publicKeyAuthorized []byte, err error)
}

// rsaKeyPairGenerator generates 2048-bit RSA key pairs, producing the same
// output format as ssh-keygen -t rsa -b 2048.
type rsaKeyPairGenerator struct{}

func (rsaKeyPairGenerator) Generate() ([]byte, []byte, error) {
	key, err := rsa.GenerateKey(rand.Reader, rsaKeyBits)
	if err != nil {
		return nil, nil, fmt.Errorf("generating RSA key: %w", err)
	}

	privPEM := pem.EncodeToMemory(&pem.Block{
		Type:  "RSA PRIVATE KEY",
		Bytes: x509.MarshalPKCS1PrivateKey(key),
	})

	sshPub, err := ssh.NewPublicKey(&key.PublicKey)
	if err != nil {
		return nil, nil, fmt.Errorf("converting to SSH public key: %w", err)
	}

	return privPEM, ssh.MarshalAuthorizedKey(sshPub), nil
}

// SSHKeyManager ensures the MPI SSH key pair secret exists and can replicate
// it to target namespaces for cross-namespace deployments. Keys are generated
// lazily on first use.
type SSHKeyManager struct {
	client    client.Client
	cfg       configv1alpha1.MPIConfiguration
	generator KeyPairGenerator
	logger    logr.Logger
}

// NewSSHKeyManager creates an SSHKeyManager backed by the production RSA
// key generator. The client should be the manager's cached client.
func NewSSHKeyManager(cl client.Client, cfg configv1alpha1.MPIConfiguration) *SSHKeyManager {
	return &SSHKeyManager{
		client:    cl,
		cfg:       cfg,
		generator: rsaKeyPairGenerator{},
		logger:    ctrl.Log.WithName("ssh-key-manager"),
	}
}

// EnsureAndReplicate ensures the SSH key pair secret exists in the source
// namespace (generating keys if needed) and replicates it to the target
// namespace if different from the source.
func (m *SSHKeyManager) EnsureAndReplicate(ctx context.Context, targetNamespace string) error {
	if err := m.ensureSourceSecret(ctx); err != nil {
		return fmt.Errorf("ensuring SSH key secret in %s: %w", m.cfg.SSHSecretNamespace, err)
	}

	if targetNamespace == m.cfg.SSHSecretNamespace {
		return nil
	}

	return m.replicateToNamespace(ctx, targetNamespace)
}

// secretExists returns true if the SSH key secret already exists in the given namespace.
func (m *SSHKeyManager) secretExists(ctx context.Context, namespace string) (bool, error) {
	key := types.NamespacedName{Namespace: namespace, Name: m.cfg.SSHSecretName}
	if err := m.client.Get(ctx, key, &corev1.Secret{}); err == nil {
		return true, nil
	} else if apierrors.IsNotFound(err) {
		return false, nil
	} else {
		return false, err
	}
}

func (m *SSHKeyManager) ensureSourceSecret(ctx context.Context) error {
	exists, err := m.secretExists(ctx, m.cfg.SSHSecretNamespace)
	if err != nil {
		return err
	}
	if exists {
		return nil
	}

	privKey, pubKey, err := m.generator.Generate()
	if err != nil {
		return fmt.Errorf("generating SSH key pair: %w", err)
	}

	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: m.cfg.SSHSecretNamespace,
			Name:      m.cfg.SSHSecretName,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
			},
		},
		Data: map[string][]byte{
			privateKeyFile: privKey,
			publicKeyFile:  pubKey,
		},
	}

	if err := m.client.Create(ctx, secret); err != nil {
		if apierrors.IsAlreadyExists(err) {
			return nil
		}
		return fmt.Errorf("creating SSH key secret: %w", err)
	}

	m.logger.Info("Created MPI SSH key pair secret",
		"namespace", m.cfg.SSHSecretNamespace, "name", m.cfg.SSHSecretName)
	return nil
}

func (m *SSHKeyManager) replicateToNamespace(ctx context.Context, targetNamespace string) error {
	exists, err := m.secretExists(ctx, targetNamespace)
	if err != nil {
		return fmt.Errorf("checking target secret: %w", err)
	}
	if exists {
		return nil
	}

	source := &corev1.Secret{}
	sourceKey := types.NamespacedName{Namespace: m.cfg.SSHSecretNamespace, Name: m.cfg.SSHSecretName}
	if err := m.client.Get(ctx, sourceKey, source); err != nil {
		return fmt.Errorf("reading source secret: %w", err)
	}

	replica := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: targetNamespace,
			Name:      m.cfg.SSHSecretName,
			Labels: map[string]string{
				"app.kubernetes.io/managed-by": "dynamo-operator",
			},
		},
		Type: source.Type,
		Data: source.Data,
	}

	if err := m.client.Create(ctx, replica); err != nil {
		if apierrors.IsAlreadyExists(err) {
			return nil
		}
		return fmt.Errorf("replicating secret to %s: %w", targetNamespace, err)
	}

	m.logger.Info("Replicated MPI SSH key secret",
		"source", m.cfg.SSHSecretNamespace, "target", targetNamespace)
	return nil
}
