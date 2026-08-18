//go:build !clustertest

/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"context"
	"errors"
	"path/filepath"
	"strings"
	"testing"
	"time"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	commoncontroller "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/client-go/tools/events"
	"k8s.io/utils/ptr"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	controllerconfig "sigs.k8s.io/controller-runtime/pkg/config"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

func TestSetupDynamoCheckpointWithSnapshotAPIAvailability(t *testing.T) {
	tests := []struct {
		name              string
		checkpointEnabled bool
		crdPaths          []string
		wantGateEnabled   bool
		wantGateError     bool
		verifyFinalizer   bool
	}{
		{
			name:            "API absent and checkpoint disabled",
			crdPaths:        []string{filepath.Join("..", "..", "config", "crd", "bases")},
			wantGateEnabled: false,
			verifyFinalizer: true,
		},
		{
			name: "API present and checkpoint disabled",
			crdPaths: []string{
				filepath.Join("..", "..", "config", "crd", "bases"),
				filepath.Join("testing", "nvidia"),
			},
			wantGateEnabled: false,
		},
		{
			name:              "API present and checkpoint enabled",
			checkpointEnabled: true,
			crdPaths: []string{
				filepath.Join("..", "..", "config", "crd", "bases"),
				filepath.Join("testing", "nvidia"),
			},
			wantGateEnabled: true,
		},
		{
			name:              "API absent and checkpoint enabled",
			checkpointEnabled: true,
			crdPaths:          []string{filepath.Join("..", "..", "config", "crd", "bases")},
			wantGateError:     true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Start an API server with the scenario's CRDs installed")
			testEnv := operatorenv.New(operatorenv.Options{
				CRDDirectoryPaths: tt.crdPaths,
				Config: &configv1alpha1.OperatorConfiguration{
					Checkpoint: configv1alpha1.CheckpointConfiguration{Enabled: tt.checkpointEnabled},
				},
				SetupWebhooks: func(ctrl.Manager, operatorenv.WebhookSetupOptions) error {
					return nil
				},
			}).RunT(t)

			t.Log("Create a manager with a short controller cache-sync timeout")
			mgr, err := ctrl.NewManager(testEnv.RESTConfig(), ctrl.Options{
				Scheme:  testEnv.Scheme(),
				Metrics: metricsserver.Options{BindAddress: "0"},
				Controller: controllerconfig.Controller{
					CacheSyncTimeout:   time.Second,
					SkipNameValidation: ptr.To(true),
				},
			})
			if err != nil {
				t.Fatalf("create manager: %v", err)
			}

			t.Log("Resolve feature gates from configuration and cluster API availability")
			gates, err := features.New(context.Background(), mgr, testEnv.OperatorConfig())
			if tt.wantGateError {
				if err == nil {
					t.Fatal("resolve feature gates succeeded, want unavailable PodSnapshot API error")
				}
				return
			}
			if err != nil {
				t.Fatalf("resolve feature gates: %v", err)
			}
			if got := gates.Enabled(features.Checkpoint); got != tt.wantGateEnabled {
				t.Fatalf("Checkpoint gate = %v, want %v", got, tt.wantGateEnabled)
			}
			testEnv.RuntimeConfig().Gate = gates

			t.Log("Register the DynamoCheckpoint controller with watches selected by its feature gate")
			if err := SetupDynamoCheckpoint(mgr, SetupOptions{
				Config:        testEnv.OperatorConfig(),
				RuntimeConfig: testEnv.RuntimeConfig(),
			}); err != nil {
				t.Fatalf("setup DynamoCheckpoint controller: %v", err)
			}

			var deletingCheckpoint client.ObjectKey
			if tt.verifyFinalizer {
				t.Log("Create and delete a finalized checkpoint before starting the disabled controller")
				checkpoint := &nvidiacomv1alpha1.DynamoCheckpoint{
					ObjectMeta: metav1.ObjectMeta{Name: "finalizer-cleanup", Namespace: testEnv.Namespace()},
					Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
						Identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
							Model: "test-model", BackendFramework: "vllm",
						},
						Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
							PodTemplateSpec: corev1.PodTemplateSpec{
								Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "test"}}},
							},
						},
					},
				}
				commoncontroller.AddFinalizer(checkpoint)
				if err := testEnv.Client().Create(context.Background(), checkpoint); err != nil {
					t.Fatalf("create finalized checkpoint: %v", err)
				}
				if err := testEnv.Client().Delete(context.Background(), checkpoint); err != nil {
					t.Fatalf("delete finalized checkpoint: %v", err)
				}
				deletingCheckpoint = client.ObjectKeyFromObject(checkpoint)
			}

			t.Log("Start the manager and require it to remain available beyond cache synchronization")
			ctx, cancel := context.WithCancel(context.Background())
			done := make(chan error, 1)
			go func() {
				done <- mgr.Start(ctx)
			}()
			if tt.verifyFinalizer {
				t.Log("Require the disabled controller to remove the finalizer and complete deletion")
				deadline := time.Now().Add(5 * time.Second)
				for time.Now().Before(deadline) {
					checkpoint := &nvidiacomv1alpha1.DynamoCheckpoint{}
					err := testEnv.Client().Get(context.Background(), deletingCheckpoint, checkpoint)
					if apierrors.IsNotFound(err) {
						break
					}
					if err != nil {
						t.Fatalf("get deleting checkpoint: %v", err)
					}
					time.Sleep(50 * time.Millisecond)
				}
				checkpoint := &nvidiacomv1alpha1.DynamoCheckpoint{}
				if err := testEnv.Client().Get(context.Background(), deletingCheckpoint, checkpoint); !apierrors.IsNotFound(err) {
					t.Fatalf("checkpoint still exists after finalizer cleanup: %v", err)
				}
				cancel()
			} else {
				select {
				case err := <-done:
					cancel()
					t.Fatalf("manager exited during startup: %v", err)
				case <-time.After(2 * time.Second):
					cancel()
				}
			}
			if err := <-done; err != nil && !errors.Is(err, context.Canceled) {
				t.Fatalf("stop manager: %v", err)
			}
		})
	}
}

func TestDisabledCheckpointReconcileWithoutSnapshotAPI(t *testing.T) {
	t.Log("Start an API server without the PodSnapshot CRD")
	testEnv := operatorenv.New(operatorenv.Options{
		CRDDirectoryPaths: []string{filepath.Join("..", "..", "config", "crd", "bases")},
		Config: &configv1alpha1.OperatorConfiguration{
			Checkpoint: configv1alpha1.CheckpointConfiguration{Enabled: false},
		},
		SetupWebhooks: func(ctrl.Manager, operatorenv.WebhookSetupOptions) error {
			return nil
		},
	}).RunT(t)

	t.Log("Create an existing checkpoint in the Creating phase")
	checkpoint := &nvidiacomv1alpha1.DynamoCheckpoint{
		ObjectMeta: metav1.ObjectMeta{Name: "creating-checkpoint", Namespace: testEnv.Namespace()},
		Spec: nvidiacomv1alpha1.DynamoCheckpointSpec{
			Identity: nvidiacomv1alpha1.DynamoCheckpointIdentity{
				Model: "test-model", BackendFramework: "vllm",
			},
			Job: nvidiacomv1alpha1.DynamoCheckpointJobConfig{
				PodTemplateSpec: corev1.PodTemplateSpec{
					Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "main", Image: "test"}}},
				},
			},
		},
	}
	if err := testEnv.Client().Create(context.Background(), checkpoint); err != nil {
		t.Fatalf("create checkpoint: %v", err)
	}
	checkpoint.Status.Phase = nvidiacomv1alpha1.DynamoCheckpointPhaseCreating
	checkpoint.Status.JobName = "checkpoint-job"
	if err := testEnv.Client().Status().Update(context.Background(), checkpoint); err != nil {
		t.Fatalf("set checkpoint status: %v", err)
	}

	t.Log("Reconcile with the Checkpoint gate disabled")
	recorder := events.NewFakeRecorder(1)
	reconciler := &CheckpointReconciler{
		Client:        testEnv.Client(),
		Config:        testEnv.OperatorConfig(),
		RuntimeConfig: testEnv.RuntimeConfig(),
		Recorder:      recorder,
	}
	_, err := reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKeyFromObject(checkpoint),
	})
	if err != nil {
		t.Fatalf("disabled reconcile accessed the unavailable PodSnapshot API: %v", err)
	}

	t.Log("Observe the disabled status and warning event")
	stored := &nvidiacomv1alpha1.DynamoCheckpoint{}
	if err := testEnv.Client().Get(context.Background(), client.ObjectKeyFromObject(checkpoint), stored); err != nil {
		t.Fatalf("get reconciled checkpoint: %v", err)
	}
	if stored.Status.Phase != nvidiacomv1alpha1.DynamoCheckpointPhaseCreating {
		t.Fatalf("checkpoint phase = %q, want %q", stored.Status.Phase, nvidiacomv1alpha1.DynamoCheckpointPhaseCreating)
	}
	if stored.Status.Message != checkpointDisabledMessage {
		t.Fatalf("checkpoint message = %q, want %q", stored.Status.Message, checkpointDisabledMessage)
	}
	select {
	case event := <-recorder.Events:
		if !strings.Contains(event, "CheckpointDisabled") {
			t.Fatalf("event = %q, want CheckpointDisabled warning", event)
		}
	default:
		t.Fatal("CheckpointDisabled warning event was not emitted")
	}

	t.Log("Reconcile the disabled checkpoint again without emitting a duplicate event")
	_, err = reconciler.Reconcile(context.Background(), ctrl.Request{
		NamespacedName: client.ObjectKeyFromObject(checkpoint),
	})
	if err != nil {
		t.Fatalf("repeat disabled reconcile: %v", err)
	}
	select {
	case event := <-recorder.Events:
		t.Fatalf("repeat reconcile emitted duplicate event %q", event)
	default:
	}
}
