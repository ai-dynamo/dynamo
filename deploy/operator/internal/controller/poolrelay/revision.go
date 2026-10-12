/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package poolrelay

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// relayConnectionRevision uses a private per-Relay key so public revision hashes
// cannot be used to guess low-entropy credentials. Only referenced data affects
// the revision: metadata and unrelated keys never restart the workload.
func (r *relayReconciler) relayConnectionRevision(ctx context.Context, relay *v1alpha1.DynamoPoolRelay, connection relayConnection) (string, error) {
	keyName := relayResourceName(relay.Name) + "-revision-key"
	var secret corev1.Secret
	err := r.Get(ctx, client.ObjectKey{Namespace: relay.Namespace, Name: keyName}, &secret)
	if apierrors.IsNotFound(err) {
		key := make([]byte, 32)
		if _, err := rand.Read(key); err != nil {
			return "", err
		}
		secret = corev1.Secret{ObjectMeta: metav1.ObjectMeta{Name: keyName, Namespace: relay.Namespace}, Data: map[string][]byte{"key": key}}
		if err := controllerutil.SetControllerReference(relay, &secret, r.Scheme()); err != nil {
			return "", err
		}
		if err := r.Create(ctx, &secret); err != nil {
			return "", err
		}
	} else if err != nil {
		return "", err
	}
	if !metav1.IsControlledBy(&secret, relay) {
		return "", fmt.Errorf("connection revision Secret is not owned by this Relay")
	}
	if len(secret.Data["key"]) != 32 {
		return "", fmt.Errorf("connection revision Secret has an invalid key")
	}
	return relayContentRevision(ctx, r.Client, relay.Namespace, connection, secret.Data["key"])
}

// relayContentRevision hashes a canonical stream using JSON framing to avoid
// concatenation ambiguity. Secret data is consumed in-memory and never returned.
func relayContentRevision(ctx context.Context, reader client.Reader, namespace string, connection relayConnection, key []byte) (string, error) {
	digest := hmac.New(sha256.New, key)
	encoder := json.NewEncoder(digest)
	if err := encoder.Encode(connection); err != nil {
		return "", err
	}
	for _, entry := range connection.Env {
		if entry.ValueFrom == nil {
			continue
		}
		value, err := relayEnvValue(ctx, reader, namespace, entry)
		if err != nil {
			return "", err
		}
		if err := encoder.Encode(value); err != nil {
			return "", err
		}
	}
	for _, volume := range connection.Volumes {
		if ref := volume.Secret; ref != nil {
			var object corev1.Secret
			if err := reader.Get(ctx, client.ObjectKey{Namespace: namespace, Name: ref.SecretName}, &object); err != nil {
				return "", fmt.Errorf("transport Secret volume is unavailable")
			}
			for _, item := range ref.Items {
				value, ok := object.Data[item.Key]
				if !ok {
					return "", fmt.Errorf("transport Secret volume key is missing")
				}
				if err := encoder.Encode(value); err != nil {
					return "", err
				}
			}
		} else if ref := volume.ConfigMap; ref != nil {
			var object corev1.ConfigMap
			if err := reader.Get(ctx, client.ObjectKey{Namespace: namespace, Name: ref.Name}, &object); err != nil {
				return "", fmt.Errorf("transport ConfigMap volume is unavailable")
			}
			for _, item := range ref.Items {
				value, ok := object.BinaryData[item.Key]
				if !ok {
					text, exists := object.Data[item.Key]
					if !exists {
						return "", fmt.Errorf("transport ConfigMap volume key is missing")
					}
					value = []byte(text)
				}
				if err := encoder.Encode(value); err != nil {
					return "", err
				}
			}
		}
	}
	return hex.EncodeToString(digest.Sum(nil)), nil
}
