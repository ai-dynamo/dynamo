/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package controller

import (
	"context"
	"fmt"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features/compatibility"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// +kubebuilder:rbac:groups=core,resources=secrets,verbs=get;list;watch;create;delete

// dgdSSHKeysReconciler owns multinode MPI SSH key material: the shared namespace
// replica and, for gated DGDs, the DGD's own key pair.
type dgdSSHKeysReconciler struct {
	client  client.Client
	manager *secret.SSHKeyManager
}

func newDGDSSHKeysReconciler(kubeClient client.Client, manager *secret.SSHKeyManager) *dgdSSHKeysReconciler {
	return &dgdSSHKeysReconciler{client: kubeClient, manager: manager}
}

func (r *dgdSSHKeysReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	if r.manager == nil || !dgd.HasAnyMultinodeComponent() {
		return nil
	}

	// Authored volumes and DGDs created before the gate still reference the shared replica.
	if err := r.manager.EnsureAndReplicate(ctx, dgd.Namespace); err != nil {
		log.FromContext(ctx).Error(err, "Failed to ensure MPI SSH key secret", "namespace", dgd.Namespace)
		return fmt.Errorf("failed to ensure MPI SSH key secret: %w", err)
	}
	if !compatibility.TRTLLMPerDGDMPISSHKey.Enabled(dgd.Annotations) {
		return nil
	}
	if err := r.ensureDGDSecret(ctx, dgd); err != nil {
		log.FromContext(ctx).Error(err, "Failed to ensure the DGD's MPI SSH key secret", "namespace", dgd.Namespace)
		return fmt.Errorf("failed to ensure the DGD's MPI SSH key secret: %w", err)
	}
	return nil
}

// ensureDGDSecret creates the DGD's immutable MPI SSH key pair Secret once. It replaces only
// a Secret left by an earlier same-name DGD, so a recreated DGD gets new keys.
func (r *dgdSSHKeysReconciler) ensureDGDSecret(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) error {
	key := client.ObjectKey{Namespace: dgd.Namespace, Name: dynamo.MPISSHSecretName(dgd.Name)}
	existing := &corev1.Secret{}
	err := r.client.Get(ctx, key, existing)
	switch {
	case err == nil && metav1.IsControlledBy(existing, dgd):
		return nil
	case err == nil && controlledByEarlierDGD(existing, dgd):
		uid := existing.UID
		if err := r.client.Delete(ctx, existing, client.Preconditions{UID: &uid}); err != nil && !apierrors.IsNotFound(err) {
			return fmt.Errorf("deleting Secret %s left by an earlier DGD: %w", key, err)
		}
	case err == nil:
		return fmt.Errorf("secret %s exists and is not controlled by this DGD", key)
	case !apierrors.IsNotFound(err):
		return err
	}

	// Generate a new key pair under the shared Secret's key names, so the launch scripts are unchanged.
	privateKey, publicKey, err := secret.GenerateSSHKeyPair()
	if err != nil {
		return fmt.Errorf("generating SSH key pair: %w", err)
	}
	desired := &corev1.Secret{
		Name:      key.Name,
		Namespace: key.Namespace,
		Labels: map[string]string{
			"app.kubernetes.io/managed-by": "dynamo-operator",
		},
		Type: corev1.SecretTypeOpaque,
		// The key pair is never rewritten, so nobody may swap it in place either.
		Immutable: new(true),
		Data: map[string][]byte{
			"private.key":     privateKey,
			"private.key.pub": publicKey,
		},
	}
	if err := controllerutil.SetControllerReference(dgd, desired, r.client.Scheme()); err != nil {
		return err
	}

	// An earlier reconcile may have created it before the cache observed it; check that one's owner.
	err = r.client.Create(ctx, desired)
	if !apierrors.IsAlreadyExists(err) {
		return err
	}
	created := &corev1.Secret{}
	if err := r.client.Get(ctx, key, created); err != nil {
		return err
	}
	if !metav1.IsControlledBy(created, dgd) {
		return fmt.Errorf("secret %s exists and is not controlled by this DGD", key)
	}
	return nil
}

// controlledByEarlierDGD reports whether a DGD with this DGD's name but another UID controls object.
func controlledByEarlierDGD(object metav1.Object, dgd *nvidiacomv1beta1.DynamoGraphDeployment) bool {
	owner := metav1.GetControllerOf(object)
	if owner == nil {
		return false
	}
	group, err := schema.ParseGroupVersion(owner.APIVersion)
	return err == nil && group.Group == nvidiacomv1beta1.GroupVersion.Group &&
		owner.Kind == "DynamoGraphDeployment" && owner.Name == dgd.Name && owner.UID != dgd.UID
}
