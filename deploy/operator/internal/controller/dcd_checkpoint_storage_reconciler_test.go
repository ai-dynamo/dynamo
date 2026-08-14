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
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	snapshotprotocol "github.com/ai-dynamo/dynamo/deploy/snapshot/protocol"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	k8serrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
)

func TestDCDCheckpointStorageReconciler(t *testing.T) {
	testScheme := runtime.NewScheme()
	require.NoError(t, corev1.AddToScheme(testScheme))

	storage := configv1alpha1.CheckpointStorageConfiguration{
		Type: snapshotprotocol.StorageTypePVC,
		PVC: configv1alpha1.CheckpointPVCConfig{
			PVCName:    "checkpoint-storage",
			BasePath:   "/checkpoints",
			Create:     true,
			Size:       "1Gi",
			AccessMode: string(corev1.ReadWriteMany),
		},
	}

	tests := []struct {
		name              string
		gate              features.Gates
		checkpointEnabled bool
		wantPVC           bool
	}{
		{
			name:              "creates storage for a checkpoint-enabled component",
			gate:              features.Gates{Checkpoint: true},
			checkpointEnabled: true,
			wantPVC:           true,
		},
		{
			name:              "does nothing when the checkpoint feature is disabled",
			checkpointEnabled: true,
		},
		{
			name: "does nothing when the component has no enabled checkpoint",
			gate: features.Gates{Checkpoint: true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Reconcile namespace checkpoint storage independently from workload rendering")
			kubeClient := fake.NewClientBuilder().WithScheme(testScheme).Build()
			dcd := &nvidiacomv1beta1.DynamoComponentDeployment{
				ObjectMeta: metav1.ObjectMeta{Name: "worker", Namespace: "default"},
			}
			if tt.checkpointEnabled {
				dcd.Spec.Experimental = &nvidiacomv1beta1.ExperimentalSpec{
					Checkpoint: &nvidiacomv1beta1.ComponentCheckpointConfig{Enabled: true},
				}
			}

			reconciler := newDCDCheckpointStorageReconciler(kubeClient, storage, tt.gate)
			require.NoError(t, reconciler.Reconcile(context.Background(), dcd))

			pvc := &corev1.PersistentVolumeClaim{}
			err := kubeClient.Get(context.Background(), types.NamespacedName{
				Name:      storage.PVC.PVCName,
				Namespace: dcd.Namespace,
			}, pvc)
			if tt.wantPVC {
				require.NoError(t, err)
				return
			}
			require.True(t, k8serrors.IsNotFound(err), "expected no PVC, got %v", err)
		})
	}
}
