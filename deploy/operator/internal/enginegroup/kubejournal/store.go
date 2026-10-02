/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// Package kubejournal persists adapter-private reconciliation evidence in Kubernetes.
package kubejournal

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/util/retry"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const stateDataKey = "state.json"

// Store persists one typed state document in a ConfigMap owned by an Engine Group.
// Callers remain responsible for validating their own state machine.
type Store struct {
	Client    client.Client
	Namespace string
	Name      string
	Owner     metav1.OwnerReference
}

// NewStore constructs a store with a DNS-safe deterministic name.
func NewStore(
	kubeClient client.Client,
	namespace string,
	groupName string,
	groupUID types.UID,
	suffix string,
) Store {
	controller := true
	blockOwnerDeletion := true
	return Store{
		Client:    kubeClient,
		Namespace: namespace,
		Name:      objectName(groupName, suffix),
		Owner: metav1.OwnerReference{
			APIVersion:         "nvidia.com/v1beta1",
			Kind:               "DynamoGraphDeploymentEngineGroup",
			Name:               groupName,
			UID:                groupUID,
			Controller:         &controller,
			BlockOwnerDeletion: &blockOwnerDeletion,
		},
	}
}

// Load decodes the current state. found is false only when the journal does not exist.
func (s Store) Load(ctx context.Context, out any) (found bool, err error) {
	if err := s.validate(); err != nil {
		return false, err
	}
	journal := &corev1.ConfigMap{}
	if err := s.Client.Get(ctx, client.ObjectKey{Namespace: s.Namespace, Name: s.Name}, journal); err != nil {
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		return false, fmt.Errorf("get journal %s/%s: %w", s.Namespace, s.Name, err)
	}
	if !ownedBy(journal, s.Owner) {
		return false, fmt.Errorf("journal %s/%s is not owned by Engine Group UID %s", s.Namespace, s.Name, s.Owner.UID)
	}
	payload, present := journal.Data[stateDataKey]
	if !present || payload == "" {
		return false, fmt.Errorf("journal %s/%s has no %s payload", s.Namespace, s.Name, stateDataKey)
	}
	if err := json.Unmarshal([]byte(payload), out); err != nil {
		return false, fmt.Errorf("decode journal %s/%s: %w", s.Namespace, s.Name, err)
	}
	return true, nil
}

// Save atomically replaces the journal payload while preserving Kubernetes metadata.
func (s Store) Save(ctx context.Context, value any) error {
	if err := s.validate(); err != nil {
		return err
	}
	payload, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("encode journal %s/%s: %w", s.Namespace, s.Name, err)
	}

	key := client.ObjectKey{Namespace: s.Namespace, Name: s.Name}
	return retry.RetryOnConflict(retry.DefaultRetry, func() error {
		journal := &corev1.ConfigMap{}
		err := s.Client.Get(ctx, key, journal)
		if apierrors.IsNotFound(err) {
			journal = &corev1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Namespace:       s.Namespace,
					Name:            s.Name,
					OwnerReferences: []metav1.OwnerReference{s.Owner},
				},
				Data: map[string]string{stateDataKey: string(payload)},
			}
			if err := s.Client.Create(ctx, journal); err != nil {
				return fmt.Errorf("create journal %s/%s: %w", s.Namespace, s.Name, err)
			}
			return nil
		}
		if err != nil {
			return fmt.Errorf("get journal %s/%s for update: %w", s.Namespace, s.Name, err)
		}
		if !ownedBy(journal, s.Owner) {
			return fmt.Errorf("journal %s/%s is not owned by Engine Group UID %s", s.Namespace, s.Name, s.Owner.UID)
		}
		if journal.Data == nil {
			journal.Data = make(map[string]string, 1)
		}
		journal.Data[stateDataKey] = string(payload)
		if err := s.Client.Update(ctx, journal); err != nil {
			return fmt.Errorf("update journal %s/%s: %w", s.Namespace, s.Name, err)
		}
		return nil
	})
}

func ownedBy(object metav1.Object, owner metav1.OwnerReference) bool {
	for _, reference := range object.GetOwnerReferences() {
		if reference.APIVersion == owner.APIVersion && reference.Kind == owner.Kind &&
			reference.Name == owner.Name && reference.UID == owner.UID {
			return true
		}
	}
	return false
}

func (s Store) validate() error {
	if s.Client == nil {
		return fmt.Errorf("journal Kubernetes client is required")
	}
	if s.Namespace == "" || s.Name == "" {
		return fmt.Errorf("journal namespace and name are required")
	}
	if s.Owner.Name == "" || s.Owner.UID == "" {
		return fmt.Errorf("journal owner name and UID are required")
	}
	return nil
}

func objectName(groupName, suffix string) string {
	candidate := groupName + "-" + suffix
	if len(candidate) <= 63 {
		return candidate
	}
	digest := sha256.Sum256([]byte(candidate))
	hash := hex.EncodeToString(digest[:6])
	prefixLength := 63 - len(suffix) - len(hash) - 2
	if prefixLength < 1 {
		return hash + "-" + suffix[:min(len(suffix), 63-len(hash)-1)]
	}
	return groupName[:min(len(groupName), prefixLength)] + "-" + hash + "-" + suffix
}
