/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package clusterenv

import (
	"crypto/ecdsa"
	"crypto/elliptic"
	cryptorand "crypto/rand"
	"crypto/x509"
	"crypto/x509/pkix"
	"encoding/pem"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"time"
)

type servingCertificate struct {
	directory string
	caBundle  []byte
}

func newServingCertificate(service, namespace string) (*servingCertificate, error) {
	now := time.Now()
	caKey, err := ecdsa.GenerateKey(elliptic.P256(), cryptorand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generate webhook CA key: %w", err)
	}
	caTemplate := &x509.Certificate{
		SerialNumber:          big.NewInt(1),
		Subject:               pkix.Name{CommonName: "clusterenv webhook CA"},
		NotBefore:             now.Add(-time.Minute),
		NotAfter:              now.Add(24 * time.Hour),
		KeyUsage:              x509.KeyUsageCertSign | x509.KeyUsageDigitalSignature,
		BasicConstraintsValid: true,
		IsCA:                  true,
	}
	caDER, err := x509.CreateCertificate(cryptorand.Reader, caTemplate, caTemplate, &caKey.PublicKey, caKey)
	if err != nil {
		return nil, fmt.Errorf("create webhook CA certificate: %w", err)
	}

	serverKey, err := ecdsa.GenerateKey(elliptic.P256(), cryptorand.Reader)
	if err != nil {
		return nil, fmt.Errorf("generate webhook serving key: %w", err)
	}
	serviceDNS := service + "." + namespace
	serverTemplate := &x509.Certificate{
		SerialNumber: big.NewInt(2),
		Subject:      pkix.Name{CommonName: serviceDNS + ".svc"},
		NotBefore:    now.Add(-time.Minute),
		NotAfter:     now.Add(24 * time.Hour),
		KeyUsage:     x509.KeyUsageDigitalSignature | x509.KeyUsageKeyEncipherment,
		ExtKeyUsage:  []x509.ExtKeyUsage{x509.ExtKeyUsageServerAuth},
		DNSNames: []string{
			service,
			serviceDNS,
			serviceDNS + ".svc",
			serviceDNS + ".svc.cluster.local",
		},
	}
	serverDER, err := x509.CreateCertificate(cryptorand.Reader, serverTemplate, caTemplate, &serverKey.PublicKey, caKey)
	if err != nil {
		return nil, fmt.Errorf("create webhook serving certificate: %w", err)
	}

	directory, err := os.MkdirTemp("", "clusterenv-webhook-certs-")
	if err != nil {
		return nil, fmt.Errorf("create webhook certificate directory: %w", err)
	}
	caPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: caDER})
	serverPEM := pem.EncodeToMemory(&pem.Block{Type: "CERTIFICATE", Bytes: serverDER})
	serverKeyDER, err := x509.MarshalPKCS8PrivateKey(serverKey)
	if err != nil {
		_ = os.RemoveAll(directory)
		return nil, fmt.Errorf("marshal webhook serving key: %w", err)
	}
	serverKeyPEM := pem.EncodeToMemory(&pem.Block{Type: "PRIVATE KEY", Bytes: serverKeyDER})
	if err := os.WriteFile(filepath.Join(directory, "tls.crt"), serverPEM, 0o600); err != nil {
		_ = os.RemoveAll(directory)
		return nil, fmt.Errorf("write webhook serving certificate: %w", err)
	}
	if err := os.WriteFile(filepath.Join(directory, "tls.key"), serverKeyPEM, 0o600); err != nil {
		_ = os.RemoveAll(directory)
		return nil, fmt.Errorf("write webhook serving key: %w", err)
	}
	return &servingCertificate{directory: directory, caBundle: caPEM}, nil
}

func (c *servingCertificate) stop() error {
	return os.RemoveAll(c.directory)
}
