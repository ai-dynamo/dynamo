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
	"strings"

	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dynamo"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features/compatibility"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/secret"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
	"sigs.k8s.io/controller-runtime/pkg/log"
)

// dgdSSHKeysReconciler owns multinode MPI SSH key material: the shared namespace
// replica and, for gated DGDs, the DGD's own key pair.
type dgdSSHKeysReconciler struct {
	client     client.Client
	apiReader  client.Reader
	sharedName string
	manager    *secret.SSHKeyManager
}

// A nil apiReader uses kubeClient for reconcilers without a manager cache.
func newDGDSSHKeysReconciler(kubeClient client.Client, apiReader client.Reader, sharedName string, manager *secret.SSHKeyManager) *dgdSSHKeysReconciler {
	if apiReader == nil {
		apiReader = kubeClient
	}
	return &dgdSSHKeysReconciler{client: kubeClient, apiReader: apiReader, sharedName: sharedName, manager: manager}
}

func (r *dgdSSHKeysReconciler) Reconcile(
	ctx context.Context,
	dgd *nvidiacomv1beta1.DynamoGraphDeployment,
) error {
	if r.manager == nil {
		return nil
	}

	// Authored volumes and injected backend volumes determine which keys are needed.
	graphKey, sharedKey, err := r.requiredSecrets(dgd)
	if err != nil {
		return err
	}
	if sharedKey {
		if err := r.manager.EnsureAndReplicate(ctx, dgd.Namespace); err != nil {
			return fmt.Errorf("failed to ensure MPI SSH key secret: %w", err)
		}
	}
	if !graphKey {
		return nil
	}
	if err := r.ensureDGDSecret(ctx, dgd); err != nil {
		log.FromContext(ctx).Error(err, "Failed to ensure the DGD's MPI SSH key secret", "namespace", dgd.Namespace)
		return fmt.Errorf("failed to ensure the DGD's MPI SSH key secret: %w", err)
	}
	return nil
}

// requiredSecrets follows serving volumes and the authored volumes retained by automatic snapshots.
func (r *dgdSSHKeysReconciler) requiredSecrets(dgd *nvidiacomv1beta1.DynamoGraphDeployment) (graphKey, sharedKey bool, err error) {
	gated := compatibility.TRTLLMPerDGDMPISSHKey.Enabled(dgd.Annotations)
	for i := range dgd.Spec.Components {
		component := &dgd.Spec.Components[i]
		if component.ManagedByExternalController() {
			continue
		}
		backend, err := dynamo.BackendFrameworkForComponent(component, dgd)
		if err != nil {
			return false, false, err
		}

		// Generated TRT-LLM volumes use the shared key only when the gate leaves them alone.
		if backend == dynamo.BackendFrameworkTRTLLM && component.GetNumberOfNodes() > 1 && dynamo.GetComponentDynamoSidecar(component) == nil {
			graphKey = graphKey || gated
			sharedKey = sharedKey || !gated
		}

		// Repoint a copy so legacy authored references still request the shared replica.
		for _, template := range dynamo.ComponentPodTemplates(component) {
			pod := template.Spec.DeepCopy()
			graphKey = dynamo.RepointMPISSHSecret(pod, dgd.UID, dgd.Annotations, r.sharedName) || graphKey
			sharedKey = podReferencesMPISSHSecret(pod, r.sharedName) || sharedKey
		}

		// Automatic snapshots retain the leader/common template's authored Secret names.
		if checkpoint := dynamo.GetCheckpoint(component); checkpoint != nil && (checkpoint.CheckpointRef == nil || strings.TrimSpace(*checkpoint.CheckpointRef) == "") {
			effective, err := dynamo.EffectiveComponentForRole(component, dynamo.RoleLeader)
			if err != nil {
				return false, false, err
			}
			pod := &corev1.PodSpec{}
			if effective.PodTemplate != nil {
				pod = effective.PodTemplate.Spec.DeepCopy()
			}
			if checkpoint.Job != nil && checkpoint.Job.PodTemplate != nil {
				pod.Volumes = mergeNamedSlice(pod.Volumes, checkpoint.Job.PodTemplate.Spec.Volumes, func(v corev1.Volume) string { return v.Name })
			}
			sharedKey = podReferencesMPISSHSecret(pod, r.sharedName) || sharedKey
		}
	}
	return graphKey, sharedKey, nil
}

// podReferencesMPISSHSecret also finds references inside projected volumes.
func podReferencesMPISSHSecret(pod *corev1.PodSpec, name string) bool {
	for _, volume := range pod.Volumes {
		if volume.Secret != nil && volume.Secret.SecretName == name {
			return true
		}
		if volume.Projected != nil {
			for _, source := range volume.Projected.Sources {
				if source.Secret != nil && source.Secret.Name == name {
					return true
				}
			}
		}
	}
	return false
}

// ensureDGDSecret creates the DGD's immutable key once and never adopts another owner's Secret.
func (r *dgdSSHKeysReconciler) ensureDGDSecret(ctx context.Context, dgd *nvidiacomv1beta1.DynamoGraphDeployment) error {
	key := client.ObjectKey{Namespace: dgd.Namespace, Name: dynamo.MPISSHSecretName(dgd.UID)}
	existing := &corev1.Secret{}
	err := r.client.Get(ctx, key, existing)
	if err == nil {
		if !metav1.IsControlledBy(existing, dgd) {
			return fmt.Errorf("secret %s exists and is not controlled by this DGD", key)
		}
		return nil
	}
	if !apierrors.IsNotFound(err) {
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

	// Ownership must be checked before publishing Pods that mount the key. No Secret watch
	// can prove ownership while the cache is stale after a competing create.
	created := &corev1.Secret{}
	if err := r.apiReader.Get(ctx, key, created); err != nil {
		return err
	}
	if !metav1.IsControlledBy(created, dgd) {
		return fmt.Errorf("secret %s exists and is not controlled by this DGD", key)
	}
	return nil
}
