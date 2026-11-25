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

package dgdrrunpublisher

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"time"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/cache"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/manager"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"
	"sigs.k8s.io/controller-runtime/pkg/source"
)

// Config is everything one publisher needs to know about its run and pod.
type Config struct {
	Namespace string
	RunName   string
	// RunUID is the UID of the run this pod serves. The run controller knows it when it
	// builds the Job; the publisher uses it to establish Candidate ownership.
	RunUID           string
	PodName          string
	SweeperContainer string
	SnapshotDir      string
	// MinPollInterval and MaxPollInterval bound the snapshot check interval, which backs
	// off from the minimum to the maximum while the input is unchanged.
	MinPollInterval time.Duration
	MaxPollInterval time.Duration
}

func (c Config) validate() error {
	switch {
	case c.Namespace == "", c.RunName == "", c.RunUID == "", c.PodName == "", c.SnapshotDir == "":
		return errors.New("namespace, run name, run uid, pod name and snapshot dir are required")
	case c.MinPollInterval <= 0 || c.MaxPollInterval < c.MinPollInterval:
		return errors.New("poll intervals must be positive and the maximum at least the minimum")
	}
	return nil
}

// NewScheme registers every type the publisher reads or writes.
func NewScheme() (*runtime.Scheme, error) {
	scheme := runtime.NewScheme()
	for _, add := range []func(*runtime.Scheme) error{clientgoscheme.AddToScheme, v1beta1.AddToScheme, v1beta2.AddToScheme} {
		if err := add(scheme); err != nil {
			return nil, err
		}
	}
	return scheme, nil
}

// cacheOptions limits the manager's informers to one namespace and selects this Run and
// Pod by name. Candidates remain namespace-scoped because their labels are mutable;
// reconciliation establishes membership from their controller owner reference.
func cacheOptions(c Config) cache.Options {
	return cache.Options{
		DefaultNamespaces: map[string]cache.Config{c.Namespace: {}},
		ByObject: map[client.Object]cache.ByObject{
			&v1beta2.DynamoGraphDeploymentRun{}: {
				Field: fields.OneTermEqualSelector("metadata.name", c.RunName),
			},
			&v1beta2.DynamoGraphDeploymentCandidate{}: {},
			&corev1.Pod{}: {
				Field: fields.OneTermEqualSelector("metadata.name", c.PodName),
			},
		},
	}
}

// Run starts the controller manager and returns the publisher's outcome (see the package
// comment for its meaning and ExitCode for the exit code), or ctx's error if the process
// is told to stop first.
func Run(ctx context.Context, restConfig *rest.Config, cfg Config) error {
	if err := cfg.validate(); err != nil {
		return err
	}
	scheme, err := NewScheme()
	if err != nil {
		return fmt.Errorf("building scheme: %w", err)
	}
	shutdownTimeout := 5 * time.Second
	mgr, err := ctrl.NewManager(restConfig, ctrl.Options{
		Scheme:                  scheme,
		Cache:                   cacheOptions(cfg),
		Metrics:                 metricsserver.Options{BindAddress: "0"},
		LeaderElection:          false,
		GracefulShutdownTimeout: &shutdownTimeout,
	})
	if err != nil {
		return fmt.Errorf("creating manager: %w", err)
	}

	key := types.NamespacedName{Namespace: cfg.Namespace, Name: cfg.RunName}
	events := make(chan event.GenericEvent, 1)
	store := &InputStore{}
	observer := &Observer{
		Store:       store,
		Path:        SnapshotPath(cfg.SnapshotDir),
		MinInterval: cfg.MinPollInterval,
		MaxInterval: cfg.MaxPollInterval,
		OnError: func(err error) {
			ctrl.Log.WithName("input-observer").Error(err, "reading snapshot")
		},
		Enqueue: func() {
			// One pending event is enough: the workqueue deduplicates by key anyway.
			select {
			case events <- event.GenericEvent{Object: runKeyObject(key)}:
			default:
			}
		},
	}
	completion := NewCompletion()
	reconciler := &Reconciler{
		Client: mgr.GetClient(),
		Writer: &clientWriter{
			Client:    mgr.GetClient(),
			Reader:    mgr.GetAPIReader(),
			Namespace: cfg.Namespace,
			RunName:   cfg.RunName,
			RunUID:    cfg.RunUID,
		},
		Namespace:        cfg.Namespace,
		RunName:          cfg.RunName,
		RunUID:           cfg.RunUID,
		PodName:          cfg.PodName,
		SweeperContainer: cfg.SweeperContainer,
		Observer:         observer,
		Store:            store,
		Completion:       completion,
	}

	if err := setupController(mgr, reconciler, key, events); err != nil {
		return fmt.Errorf("building controller: %w", err)
	}
	if err := mgr.Add(manager.RunnableFunc(observer.Run)); err != nil {
		return fmt.Errorf("adding input observer: %w", err)
	}

	runCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	go func() {
		select {
		case <-completion.Done():
			cancel() // the outcome is known: stop the manager and exit
		case <-runCtx.Done():
		}
	}()
	startErr := mgr.Start(runCtx)

	if completion.Finished() {
		return completion.Err()
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	if startErr != nil {
		return fmt.Errorf("manager stopped: %w", startErr)
	}
	return errors.New("manager stopped before the run finished")
}

// setupController centralizes the production event wiring: every Kubernetes and
// local-input event reaches the one Run key.
func setupController(mgr manager.Manager, reconciler reconcile.Reconciler, key types.NamespacedName, events <-chan event.GenericEvent) error {
	podToRun := handler.EnqueueRequestsFromMapFunc(func(context.Context, client.Object) []reconcile.Request {
		return []reconcile.Request{{NamespacedName: key}}
	})
	return ctrl.NewControllerManagedBy(mgr).
		Named("dgdr-run-publisher").
		For(&v1beta2.DynamoGraphDeploymentRun{}).
		Owns(&v1beta2.DynamoGraphDeploymentCandidate{}).
		Watches(&corev1.Pod{}, podToRun).
		WatchesRawSource(source.Channel(events, &handler.EnqueueRequestForObject{})).
		WithOptions(controller.Options{MaxConcurrentReconciles: 1}).
		Complete(reconciler)
}

// SnapshotPath is where the Sweeper writes the snapshot inside the shared directory.
func SnapshotPath(dir string) string {
	return filepath.Join(dir, SnapshotFileName)
}

// runKeyObject is the object carried by enqueue events; only its name and namespace
// matter to the handler.
func runKeyObject(key types.NamespacedName) client.Object {
	run := &v1beta2.DynamoGraphDeploymentRun{}
	run.Namespace, run.Name = key.Namespace, key.Name
	return run
}
