/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package secret

import (
	"context"
	"crypto/x509"
	"encoding/pem"
	"fmt"
	"reflect"
	"testing"

	"github.com/go-logr/logr"
	"golang.org/x/crypto/ssh"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
)

const (
	testSecretName = "mpi-ssh-secret"
	testSourceNS   = "operator-ns"
	testTargetNS   = "workload-ns"
	testPrivateKey = "fake-private-key"
	testPublicKey  = "fake-public-key"
)

type fakeKeyPairGenerator struct {
	called bool
	err    error
}

func (f *fakeKeyPairGenerator) Generate() ([]byte, []byte, error) {
	f.called = true
	if f.err != nil {
		return nil, nil, f.err
	}
	return []byte(testPrivateKey), []byte(testPublicKey), nil
}

func newScheme() *runtime.Scheme {
	s := runtime.NewScheme()
	_ = corev1.AddToScheme(s)
	return s
}

func newTestManager(builder *fake.ClientBuilder, gen KeyPairGenerator) *SSHKeyManager {
	return &SSHKeyManager{
		client: builder.Build(),
		cfg: configv1alpha1.MPIConfiguration{
			SSHSecretName:      testSecretName,
			SSHSecretNamespace: testSourceNS,
		},
		generator: gen,
		logger:    logr.Discard(),
	}
}

func TestRSAKeyPairGenerator(t *testing.T) {
	gen := rsaKeyPairGenerator{}
	privPEM, pubAuthorized, err := gen.Generate()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	block, _ := pem.Decode(privPEM)
	if block == nil {
		t.Fatal("expected PEM block in private key")
	}
	if block.Type != "RSA PRIVATE KEY" {
		t.Errorf("expected PEM type RSA PRIVATE KEY, got %s", block.Type)
	}
	rsaKey, err := x509.ParsePKCS1PrivateKey(block.Bytes)
	if err != nil {
		t.Fatalf("failed to parse private key: %v", err)
	}
	if rsaKey.N.BitLen() != rsaKeyBits {
		t.Errorf("expected %d-bit key, got %d", rsaKeyBits, rsaKey.N.BitLen())
	}

	_, _, _, _, err = ssh.ParseAuthorizedKey(pubAuthorized)
	if err != nil {
		t.Fatalf("failed to parse public key as authorized_keys format: %v", err)
	}
}

func TestEnsureAndReplicate_CreatesSourceWhenMissing(t *testing.T) {
	gen := &fakeKeyPairGenerator{}
	mgr := newTestManager(fake.NewClientBuilder().WithScheme(newScheme()), gen)
	ctx := context.Background()

	if err := mgr.EnsureAndReplicate(ctx, testSourceNS); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !gen.called {
		t.Fatal("expected generator to be called")
	}

	secret := &corev1.Secret{}
	if err := mgr.client.Get(ctx, types.NamespacedName{Namespace: testSourceNS, Name: testSecretName}, secret); err != nil {
		t.Fatalf("source secret should exist: %v", err)
	}
	if !reflect.DeepEqual(secret.Data[privateKeyFile], []byte(testPrivateKey)) {
		t.Error("private key data mismatch")
	}
	if !reflect.DeepEqual(secret.Data[publicKeyFile], []byte(testPublicKey)) {
		t.Error("public key data mismatch")
	}
	if secret.Labels["app.kubernetes.io/managed-by"] != "dynamo-operator" {
		t.Error("expected managed-by label")
	}
}

func TestEnsureAndReplicate_SkipsExistingSource(t *testing.T) {
	existing := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: testSourceNS,
			Name:      testSecretName,
		},
		Data: map[string][]byte{
			privateKeyFile: []byte("existing-private"),
			publicKeyFile:  []byte("existing-public"),
		},
	}
	gen := &fakeKeyPairGenerator{}
	mgr := newTestManager(fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(existing), gen)
	ctx := context.Background()

	if err := mgr.EnsureAndReplicate(ctx, testSourceNS); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if gen.called {
		t.Error("generator should not be called when source secret exists")
	}
}

func TestEnsureAndReplicate_ReplicatesToTargetNamespace(t *testing.T) {
	source := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: testSourceNS,
			Name:      testSecretName,
		},
		Data: map[string][]byte{
			privateKeyFile: []byte("source-private"),
			publicKeyFile:  []byte("source-public"),
		},
	}
	gen := &fakeKeyPairGenerator{}
	mgr := newTestManager(fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(source), gen)
	ctx := context.Background()

	if err := mgr.EnsureAndReplicate(ctx, testTargetNS); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	replica := &corev1.Secret{}
	if err := mgr.client.Get(ctx, types.NamespacedName{Namespace: testTargetNS, Name: testSecretName}, replica); err != nil {
		t.Fatalf("replica secret should exist: %v", err)
	}
	if !reflect.DeepEqual(replica.Data[privateKeyFile], []byte("source-private")) {
		t.Error("replica private key should match source")
	}
	if !reflect.DeepEqual(replica.Data[publicKeyFile], []byte("source-public")) {
		t.Error("replica public key should match source")
	}
}

func TestEnsureAndReplicate_SameNamespaceSkipsReplicate(t *testing.T) {
	gen := &fakeKeyPairGenerator{}
	mgr := newTestManager(fake.NewClientBuilder().WithScheme(newScheme()), gen)
	ctx := context.Background()

	if err := mgr.EnsureAndReplicate(ctx, testSourceNS); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Only the source secret should exist, no replica
	list := &corev1.SecretList{}
	if err := mgr.client.List(ctx, list); err != nil {
		t.Fatalf("failed to list secrets: %v", err)
	}
	if len(list.Items) != 1 {
		t.Errorf("expected exactly 1 secret (source only), got %d", len(list.Items))
	}
}

func TestEnsureAndReplicate_SkipsExistingReplica(t *testing.T) {
	source := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Namespace: testSourceNS, Name: testSecretName},
		Data:       map[string][]byte{privateKeyFile: []byte("key"), publicKeyFile: []byte("pub")},
	}
	existingReplica := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Namespace: testTargetNS, Name: testSecretName},
		Data:       map[string][]byte{privateKeyFile: []byte("old-key"), publicKeyFile: []byte("old-pub")},
	}
	gen := &fakeKeyPairGenerator{}
	mgr := newTestManager(fake.NewClientBuilder().WithScheme(newScheme()).WithObjects(source, existingReplica), gen)
	ctx := context.Background()

	if err := mgr.EnsureAndReplicate(ctx, testTargetNS); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	replica := &corev1.Secret{}
	if err := mgr.client.Get(ctx, types.NamespacedName{Namespace: testTargetNS, Name: testSecretName}, replica); err != nil {
		t.Fatalf("failed to get replica: %v", err)
	}
	if !reflect.DeepEqual(replica.Data[privateKeyFile], []byte("old-key")) {
		t.Error("existing replica should not be overwritten")
	}
}

func TestEnsureAndReplicate_GeneratorError(t *testing.T) {
	gen := &fakeKeyPairGenerator{err: fmt.Errorf("keygen failed")}
	mgr := newTestManager(fake.NewClientBuilder().WithScheme(newScheme()), gen)

	err := mgr.EnsureAndReplicate(context.Background(), testSourceNS)
	if err == nil {
		t.Fatal("expected error from generator")
	}
	if !gen.called {
		t.Fatal("expected generator to be called")
	}
}
