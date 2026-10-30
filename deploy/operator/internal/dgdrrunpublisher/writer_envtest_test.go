//go:build !clustertest

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
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
)

var publisherEnv = operatorenv.New(operatorenv.Options{
	Admission:       operatorenv.AdmissionWebhooks{Mutating: true, Validating: true},
	OperatorVersion: "1.6.0",
	SetupWebhooks: func(mgr ctrl.Manager, opts operatorenv.WebhookSetupOptions) error {
		return webhooksetup.Setup(mgr, webhooksetup.Options{
			Config: opts.OperatorConfig, RuntimeConfig: opts.RuntimeConfig,
			OperatorVersion: opts.OperatorVersion, OperatorPrincipal: opts.OperatorPrincipal,
		})
	},
})

func TestMain(m *testing.M) {
	os.Exit(publisherEnv.RunM(m))
}

// These tests send the publisher's real server-side-apply payloads to an API server that
// has the v1beta2 CRDs installed. The reconciler tests use a fake writer; this is the
// check that the payloads are accepted by the real schema, that the publisher owns what
// it writes, and that the resourceVersion constraints behave as the reconciler assumes.

type writerFixture struct {
	t      *testing.T
	ctx    context.Context
	client client.Client
	config *rest.Config
	run    *v1beta2.DynamoGraphDeploymentRun
	writer *clientWriter
}

func newWriterFixture(t *testing.T) *writerFixture {
	t.Helper()
	env := publisherEnv.ForTest(t)
	ctx := context.Background()
	c := env.Client()

	metric := v1beta2.ObjectiveMetric("ThroughputPerGPU")
	run := &v1beta2.DynamoGraphDeploymentRun{
		Namespace: env.Namespace(), Name: "publisher-run",
		Spec: v1beta2.DynamoGraphDeploymentRunSpec{
			NativeSearchInput: v1beta2.NativeSearchInput{
				Model:    &v1beta2.ModelSpec{Name: "Qwen/Qwen3-0.6B"},
				Backends: []v1beta2.Backend{v1beta2.BackendVLLM},
				Hardware: &v1beta2.HardwareSpec{
					GPU:            &v1beta2.GPUHardwareSpec{SKU: new("h200_sxm"), Budget: new(int32(8))},
					NumGPUsPerNode: new(int32(8)),
				},
				Workload: &v1beta2.WorkloadSpec{Static: &v1beta2.StaticWorkloadSpec{}},
				Objective: &v1beta2.ObjectiveSpec{
					Mode: v1beta2.ObjectiveModeOptimize, Metric: &metric, SLA: &v1beta2.ObjectiveSLASpec{},
				},
				Recommendation: &v1beta2.RecommendationSpec{MaxCandidates: 1},
			},
		},
	}
	if err := c.Create(ctx, run); err != nil {
		t.Fatalf("create run: %v", err)
	}
	return &writerFixture{
		t:      t,
		ctx:    ctx,
		client: c,
		config: env.RESTConfig(),
		run:    run,
		writer: &clientWriter{
			Client:    c,
			Reader:    c,
			Namespace: run.Namespace,
			RunName:   run.Name,
			RunUID:    string(run.UID),
		},
	}
}

func (f *writerFixture) name(id string) string {
	return CandidateName(f.run.Name, string(f.run.UID), id)
}

func (f *writerFixture) candidate(name string) *v1beta2.DynamoGraphDeploymentCandidate {
	f.t.Helper()
	var got v1beta2.DynamoGraphDeploymentCandidate
	if err := f.client.Get(f.ctx, types.NamespacedName{Namespace: f.run.Namespace, Name: name}, &got); err != nil {
		f.t.Fatalf("get candidate %s: %v", name, err)
	}
	return &got
}

func (f *writerFixture) currentRun() *v1beta2.DynamoGraphDeploymentRun {
	f.t.Helper()
	var got v1beta2.DynamoGraphDeploymentRun
	if err := f.client.Get(f.ctx, client.ObjectKeyFromObject(f.run), &got); err != nil {
		f.t.Fatalf("get run: %v", err)
	}
	return &got
}

func envtestCandidate() dgdcreconcile.DesiredCandidate {
	return dgdcreconcile.DesiredCandidate{
		ID:         "a",
		Spec:       validManifest + "---\napiVersion: v1\nkind: ConfigMap\nmetadata:\n  name: companion\n",
		Parameters: map[string]any{"tp": 2},
		Metrics:    map[string]any{"throughput": 12.5},
		Validity: metav1.Condition{Type: v1beta2.DGDCConditionValid, Status: metav1.ConditionTrue,
			Reason: "ValidationSucceeded", LastTransitionTime: metav1.Now()},
	}
}

func foreignCandidate(t *testing.T, namespace, name string, candidate dgdcreconcile.DesiredCandidate) *v1beta2.DynamoGraphDeploymentCandidate {
	t.Helper()
	dgd, _, err := decodeManifest(candidate.ID, candidate.Spec)
	if err != nil {
		t.Fatal(err)
	}
	spec, err := json.Marshal(dgd.Object["spec"])
	if err != nil {
		t.Fatal(err)
	}
	return &v1beta2.DynamoGraphDeploymentCandidate{
		Namespace: namespace,
		Name:      name,
		Labels:    map[string]string{"owner": "foreign"},
		Spec:      runtime.RawExtension{Raw: spec},
	}
}

type writeInterceptClient struct {
	client.Client
	beforeCreate func()
	beforeApply  func()
	createError  error
}

func (c *writeInterceptClient) Create(ctx context.Context, object client.Object, options ...client.CreateOption) error {
	if c.createError != nil {
		return c.createError
	}
	if c.beforeCreate != nil {
		hook := c.beforeCreate
		c.beforeCreate = nil
		hook()
	}
	return c.Client.Create(ctx, object, options...)
}

func (c *writeInterceptClient) Apply(ctx context.Context, configuration runtime.ApplyConfiguration, options ...client.ApplyOption) error {
	if c.beforeApply != nil {
		hook := c.beforeApply
		c.beforeApply = nil
		hook()
	}
	return c.Client.Apply(ctx, configuration, options...)
}

type reconcileRecorder struct {
	delegate *Reconciler
	count    atomic.Int64
}

func (r *reconcileRecorder) Reconcile(ctx context.Context, request ctrl.Request) (ctrl.Result, error) {
	result, err := r.delegate.Reconcile(ctx, request)
	r.count.Add(1)
	return result, err
}

func waitForReconcileQuiet(t *testing.T, recorder *reconcileRecorder, quiet, timeout time.Duration) {
	t.Helper()
	var last int64 = -1
	var stableSince time.Time
	require.Eventually(t, func() bool {
		current := recorder.count.Load()
		if current == 0 {
			return false
		}
		if current != last {
			last = current
			stableSince = time.Now()
			return false
		}
		return time.Since(stableSince) >= quiet
	}, timeout, 10*time.Millisecond, "reconciliation count did not become stable")
}

func startPublisherController(t *testing.T, f *writerFixture, snapshot *Snapshot) (*reconcileRecorder, *Completion, *InputStore, chan event.GenericEvent) {
	t.Helper()
	podName := "publisher-pod"
	pod := &corev1.Pod{
		Namespace: f.run.Namespace, Name: podName,
		Spec: corev1.PodSpec{Containers: []corev1.Container{{Name: "sweeper", Image: "example.invalid/sweeper"}}},
	}
	if err := f.client.Create(f.ctx, pod); err != nil {
		t.Fatalf("create pod: %v", err)
	}

	scheme, err := NewScheme()
	if err != nil {
		t.Fatal(err)
	}
	manager, err := ctrl.NewManager(f.config, ctrl.Options{
		Scheme:         scheme,
		Cache:          cacheOptions(Config{Namespace: f.run.Namespace, RunName: f.run.Name, RunUID: string(f.run.UID), PodName: podName}),
		Metrics:        metricsserver.Options{BindAddress: "0"},
		LeaderElection: false,
	})
	if err != nil {
		t.Fatalf("create manager: %v", err)
	}
	store := &InputStore{}
	store.accept(snapshot)
	completion := NewCompletion()
	reconciler := &Reconciler{
		Client: manager.GetClient(),
		Writer: &clientWriter{
			Client:    manager.GetClient(),
			Reader:    manager.GetAPIReader(),
			Namespace: f.run.Namespace,
			RunName:   f.run.Name,
			RunUID:    string(f.run.UID),
		},
		Namespace:        f.run.Namespace,
		RunName:          f.run.Name,
		RunUID:           string(f.run.UID),
		PodName:          podName,
		SweeperContainer: "sweeper",
		Store:            store,
		Completion:       completion,
	}
	recorder := &reconcileRecorder{delegate: reconciler}
	events := make(chan event.GenericEvent, 1)
	key := types.NamespacedName{Namespace: f.run.Namespace, Name: f.run.Name}
	if err := setupController(manager, recorder, key, events); err != nil {
		t.Fatalf("setup controller: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		done <- manager.Start(ctx)
	}()
	t.Cleanup(func() {
		cancel()
		select {
		case err := <-done:
			if err != nil && !errors.Is(err, context.Canceled) {
				t.Errorf("publisher stopped: %v", err)
			}
		case <-time.After(5 * time.Second):
			t.Error("publisher did not stop")
		}
	})
	return recorder, completion, store, events
}

func hasApplyManager(fields []metav1.ManagedFieldsEntry, subresource string) bool {
	for _, entry := range fields {
		if entry.Manager == fieldOwner && entry.Operation == metav1.ManagedFieldsOperationApply && entry.Subresource == subresource {
			return true
		}
	}
	return false
}

func TestEnvtestCandidatePreparation(t *testing.T) {
	f := newWriterFixture(t)
	const overridePrefix = `{"apiVersion":"nvidia.com/v1beta1","kind":"DynamoGraphDeployment",`
	tests := []struct {
		name      string
		manifest  string
		override  string
		reason    string
		wantSpec  string
		wantLabel string
	}{
		{name: "valid DGD keeps its pre-defaulting spec", manifest: validManifest, reason: "ValidationSucceeded"},
		{name: "unknown field is retained", manifest: validManifest + "  unknownField: 9007199254740993\n", reason: "ValidationFailed", wantSpec: `"unknownField":9007199254740993`},
		{name: "wrong type is retained", manifest: "apiVersion: nvidia.com/v1beta1\nkind: DynamoGraphDeployment\nspec:\n  components: not-a-list\n", reason: "ValidationFailed", wantSpec: `"components":"not-a-list"`},
		{name: "DGD webhook rejects an ambiguous runtime image", manifest: strings.ReplaceAll(validManifest, "runtime:1.6.0", "runtime:custom"), reason: "ValidationFailed", wantSpec: `"image":"registry.example/runtime:custom"`},
		{name: "shared override applies without recomputing metrics", manifest: validManifest, override: overridePrefix + `"spec":{"components":[{"name":"worker","replicas":3}]}}`, reason: "ValidationSucceeded", wantSpec: `"replicas":3`},
		{name: "missing target applies nothing", manifest: validManifest, override: overridePrefix + `"spec":{"backendFramework":"sglang","components":[{"name":"missing","replicas":3}]}}`, reason: "OverrideTargetNotFound", wantSpec: `"backendFramework":"vllm"`},
		{name: "metadata is copied to Candidate", manifest: validManifest, override: overridePrefix + `"metadata":{"labels":{"custom":"kept"},"annotations":{"custom":"kept"}}}`, reason: "ValidationSucceeded", wantLabel: "kept"},
		{name: "identity override fails atomically", manifest: validManifest, override: overridePrefix + `"metadata":{"namespace":"other"},"spec":{"backendFramework":"sglang"}}`, reason: "OverrideFailed", wantSpec: `"backendFramework":"vllm"`},
	}
	for i, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Log("Prepare a generated DGD through the production override and admission paths")
			run := f.run.DeepCopy()
			if test.override != "" {
				run.Spec.Overrides = &v1beta2.OverridesSpec{DGD: &runtime.RawExtension{Raw: []byte(test.override)}}
			}
			candidate := envtestCandidate()
			candidate.ID = fmt.Sprintf("case-%d", i)
			candidate.Spec = test.manifest
			wire, err := json.Marshal(Snapshot{SchemaVersion: SnapshotSchemaVersion,
				Run: SnapshotRun{Phase: PhaseRunning},
				Candidates: []SnapshotCandidate{{ID: candidate.ID, Outcome: OutcomeMaterialized,
					Manifest: candidate.Spec, Parameters: candidate.Parameters, Metrics: candidate.Metrics}},
			})
			require.NoError(t, err)
			parsed, err := ParseSnapshot(wire)
			require.NoError(t, err, "readable DGD content errors must not fail the snapshot protocol")
			candidate = Desire(run.Name, string(run.UID), parsed).Candidates[0]
			prepared, err := f.writer.PrepareCandidate(f.ctx, run, candidate)
			require.NoError(t, err)
			require.Equal(t, test.reason, prepared.Validity.Reason)

			t.Log("Persist the pre-dry-run payload and its original evaluation facts")
			name := f.name(candidate.ID)
			stored, err := f.writer.CreateCandidate(f.ctx, run, name, prepared)
			require.NoError(t, err)
			require.NoError(t, f.writer.PopulateStatus(f.ctx, name, stored.ResourceVersion, prepared))
			got := f.candidate(name)
			require.Contains(t, string(got.Status.Parameters.Raw), `"tp":2`)
			require.Contains(t, string(got.Status.Metrics.Raw), `"throughput":12.5`)
			require.Equal(t, test.reason, meta.FindStatusCondition(got.Status.Conditions, v1beta2.DGDCConditionValid).Reason)
			if test.wantSpec != "" {
				require.Contains(t, string(got.Spec.Raw), test.wantSpec)
			}
			if test.wantLabel != "" {
				require.Equal(t, test.wantLabel, got.Labels["custom"])
				require.Equal(t, test.wantLabel, got.Annotations["custom"])
			}
			if i == 0 {
				var spec map[string]any
				require.NoError(t, json.Unmarshal(got.Spec.Raw, &spec))
				for _, component := range spec["components"].([]any) {
					require.NotContains(t, component.(map[string]any), "replicas", "dry-run defaults leaked into the stored payload")
				}
			}

			t.Log("Verify dry-run did not create a deployment")
			deployments := &unstructured.UnstructuredList{}
			deployments.SetAPIVersion("nvidia.com/v1beta1")
			deployments.SetKind("DynamoGraphDeploymentList")
			require.NoError(t, f.client.List(f.ctx, deployments, client.InNamespace(run.Namespace)))
			require.Empty(t, deployments.Items)

			t.Log("Reuse the persisted validity verdict after a publisher restart")
			restartedWriter := *f.writer
			restartedWriter.Client = &writeInterceptClient{Client: f.client, createError: errors.New("dry-run must not run again")}
			restarted := &Reconciler{Client: f.client, Writer: &restartedWriter, Namespace: run.Namespace}
			desired := DesiredState{Candidates: []dgdcreconcile.DesiredCandidate{candidate}, Names: map[string]string{candidate.ID: name}}
			require.NoError(t, restarted.prepareCandidates(f.ctx, run, &desired))
			require.Equal(t, got.Status.Conditions[0], desired.Candidates[0].Validity)
		})
	}

	t.Run("API outage remains retryable", func(t *testing.T) {
		t.Log("Fail the API create operation before a definitive content verdict exists")
		writer := *f.writer
		writer.Client = &writeInterceptClient{Client: f.client, createError: apierrors.NewServiceUnavailable("temporary outage")}
		_, err := writer.PrepareCandidate(f.ctx, f.run, envtestCandidate())
		require.True(t, apierrors.IsServiceUnavailable(err), "unexpected error: %v", err)
	})
}

func TestEnvtestCandidateApplyIsAcceptedAndIdempotent(t *testing.T) {
	f := newWriterFixture(t)
	name := f.name("a")

	stored, err := f.writer.CreateCandidate(f.ctx, f.run, name, envtestCandidate())
	if err != nil {
		t.Fatalf("create candidate: %v", err)
	}
	if stored.ResourceVersion == "" || stored.Populated {
		t.Fatalf("stored = %+v, want a resourceVersion and no status yet", stored)
	}

	got := f.candidate(name)
	if got.Labels[LabelRunUID] != string(f.run.UID) || got.Labels[LabelCandidateID] != "a" {
		t.Errorf("labels = %v", got.Labels)
	}
	if owner := metav1.GetControllerOf(got); owner == nil || owner.UID != f.run.UID || owner.Kind != "DynamoGraphDeploymentRun" {
		t.Errorf("controller owner = %+v, want the run", owner)
	}
	if !strings.Contains(got.Annotations[AnnotationAdditionalResources], "kind: ConfigMap") {
		t.Errorf("companion annotation = %q", got.Annotations[AnnotationAdditionalResources])
	}
	var spec map[string]any
	if err := json.Unmarshal(got.Spec.Raw, &spec); err != nil {
		t.Fatal(err)
	}
	if _, found := spec["parameters"]; found {
		t.Fatal("simulation parameters must not be injected into the DGD spec")
	}
	if got.Status.Metrics != nil || got.Status.Parameters != nil {
		t.Errorf("candidate apply wrote through the status subresource: %+v", got.Status)
	}
	if !hasApplyManager(got.ManagedFields, "") {
		t.Errorf("no apply manager %q on the candidate: %+v", fieldOwner, got.ManagedFields)
	}

	again, err := f.writer.CreateCandidate(f.ctx, f.run, name, envtestCandidate())
	if err != nil {
		t.Fatalf("re-apply candidate: %v", err)
	}
	if again.ResourceVersion != stored.ResourceVersion {
		t.Errorf("re-applying the same candidate changed resourceVersion %s -> %s", stored.ResourceVersion, again.ResourceVersion)
	}

	drifted := f.candidate(name)
	delete(drifted.Labels, LabelRunUID)
	if err := f.client.Update(f.ctx, drifted); err != nil {
		t.Fatalf("remove run label: %v", err)
	}
	if _, err := f.writer.CreateCandidate(f.ctx, f.run, name, envtestCandidate()); err != nil {
		t.Fatalf("repair candidate labels: %v", err)
	}
	if got := f.candidate(name); got.Labels[LabelRunUID] != string(f.run.UID) {
		t.Fatalf("run label was not repaired: %v", got.Labels)
	}
}

func TestEnvtestCandidateApplyDoesNotAdoptAForeignObject(t *testing.T) {
	f := newWriterFixture(t)
	name := f.name("a")
	candidate := envtestCandidate()
	foreign := foreignCandidate(t, f.run.Namespace, name, candidate)
	if err := f.client.Create(f.ctx, foreign); err != nil {
		t.Fatalf("create foreign candidate: %v", err)
	}

	if _, err := f.writer.CreateCandidate(f.ctx, f.run, name, candidate); err == nil {
		t.Fatal("applying a candidate over a foreign object succeeded")
	}
	got := f.candidate(name)
	if owner := metav1.GetControllerOf(got); owner != nil {
		t.Fatalf("foreign candidate was adopted by %+v", owner)
	}
	if got.Labels["owner"] != "foreign" || got.Labels[LabelRunUID] != "" {
		t.Fatalf("foreign candidate was modified: labels = %v", got.Labels)
	}
}

func TestEnvtestCandidateCreateRaceDoesNotAdoptForeignObject(t *testing.T) {
	f := newWriterFixture(t)
	name := f.name("a")
	candidate := envtestCandidate()
	foreign := foreignCandidate(t, f.run.Namespace, name, candidate)
	intercept := &writeInterceptClient{Client: f.client}
	intercept.beforeCreate = func() {
		if err := f.client.Create(f.ctx, foreign); err != nil {
			t.Fatalf("create racing foreign candidate: %v", err)
		}
	}
	writer := *f.writer
	writer.Client = intercept

	if _, err := writer.CreateCandidate(f.ctx, f.run, name, candidate); err == nil {
		t.Fatal("candidate creation adopted the foreign object that won the name race")
	}
	got := f.candidate(name)
	if owner := metav1.GetControllerOf(got); owner != nil {
		t.Fatalf("racing foreign candidate was adopted by %+v", owner)
	}
	if got.Labels["owner"] != "foreign" || got.Labels[LabelRunUID] != "" {
		t.Fatalf("racing foreign candidate was modified: labels = %v", got.Labels)
	}
}

func TestEnvtestCandidateApplyRejectsDeleteRecreateRace(t *testing.T) {
	f := newWriterFixture(t)
	name := f.name("a")
	candidate := envtestCandidate()
	if _, err := f.writer.CreateCandidate(f.ctx, f.run, name, candidate); err != nil {
		t.Fatalf("create owned candidate: %v", err)
	}
	ownedUID := f.candidate(name).UID
	foreign := foreignCandidate(t, f.run.Namespace, name, candidate)
	intercept := &writeInterceptClient{Client: f.client}
	intercept.beforeApply = func() {
		current := f.candidate(name)
		if err := f.client.Delete(f.ctx, current); err != nil {
			t.Fatalf("delete owned candidate during apply race: %v", err)
		}
		if err := f.client.Create(f.ctx, foreign); err != nil {
			t.Fatalf("recreate foreign candidate during apply race: %v", err)
		}
	}
	writer := *f.writer
	writer.Client = intercept

	if _, err := writer.CreateCandidate(f.ctx, f.run, name, candidate); err == nil {
		t.Fatal("UID- and resourceVersion-constrained apply accepted a replacement object")
	}
	got := f.candidate(name)
	if got.UID == ownedUID {
		t.Fatalf("candidate UID did not change across delete/recreate: %s", got.UID)
	}
	if owner := metav1.GetControllerOf(got); owner != nil {
		t.Fatalf("replacement foreign candidate was adopted by %+v", owner)
	}
	if got.Labels["owner"] != "foreign" || got.Labels[LabelRunUID] != "" {
		t.Fatalf("replacement foreign candidate was modified: labels = %v", got.Labels)
	}
}

func TestEnvtestCandidateStatusIsWrittenOnceAndGuardedByResourceVersion(t *testing.T) {
	f := newWriterFixture(t)
	name := f.name("a")
	candidate := envtestCandidate()

	stored, err := f.writer.CreateCandidate(f.ctx, f.run, name, candidate)
	if err != nil {
		t.Fatalf("create candidate: %v", err)
	}
	if err := f.writer.PopulateStatus(f.ctx, name, stored.ResourceVersion, candidate); err != nil {
		t.Fatalf("populate status: %v", err)
	}
	got := f.candidate(name)
	if got.Status.Metrics == nil {
		t.Fatal("status.metrics is not set after PopulateStatus")
	}
	var metrics map[string]any
	if err := json.Unmarshal(got.Status.Metrics.Raw, &metrics); err != nil || metrics["throughput"] != 12.5 {
		t.Errorf("metrics = %s, err = %v", got.Status.Metrics.Raw, err)
	}
	var parameters map[string]any
	if got.Status.Parameters == nil {
		t.Fatal("status.parameters is not set after PopulateStatus")
	}
	if err := json.Unmarshal(got.Status.Parameters.Raw, &parameters); err != nil || parameters["tp"] != float64(2) {
		t.Errorf("parameters = %s, err = %v", got.Status.Parameters.Raw, err)
	}
	if !meta.IsStatusConditionTrue(got.Status.Conditions, v1beta2.DGDCConditionValid) {
		t.Fatalf("validity was not published: %+v", got.Status.Conditions)
	}
	if !hasApplyManager(got.ManagedFields, "status") {
		t.Errorf("no status apply manager %q: %+v", fieldOwner, got.ManagedFields)
	}

	// A decision made from the version before the status was written must conflict.
	err = f.writer.PopulateStatus(f.ctx, name, stored.ResourceVersion, candidate)
	if !apierrors.IsConflict(err) {
		t.Fatalf("populate with a stale resourceVersion: err = %v, want a conflict", err)
	}

	// Re-applying the candidate after its status exists reports it as populated.
	after, err := f.writer.CreateCandidate(f.ctx, f.run, name, candidate)
	if err != nil {
		t.Fatalf("re-apply candidate: %v", err)
	}
	if !after.Populated {
		t.Error("a candidate with status was reported as not populated")
	}

	// Even an evaluated candidate without metrics gets an empty object. Presence, not
	// contents, is the persisted marker that its status has been populated.
	empty := envtestCandidate()
	empty.ID = "empty"
	empty.Metrics = nil
	emptyName := f.name(empty.ID)
	emptyStored, err := f.writer.CreateCandidate(f.ctx, f.run, emptyName, empty)
	if err != nil {
		t.Fatalf("create candidate without metrics: %v", err)
	}
	if err := f.writer.PopulateStatus(f.ctx, emptyName, emptyStored.ResourceVersion, empty); err != nil {
		t.Fatalf("populate candidate without metrics: %v", err)
	}
	var emptyMetrics map[string]any
	got = f.candidate(emptyName)
	if got.Status.Metrics == nil {
		t.Fatal("candidate without metrics was not marked populated")
	}
	if err := json.Unmarshal(got.Status.Metrics.Raw, &emptyMetrics); err != nil || len(emptyMetrics) != 0 {
		t.Fatalf("metrics = %s, err = %v; want an empty object", got.Status.Metrics.Raw, err)
	}
}

func TestEnvtestRunStatusIsOwnedByThePublisher(t *testing.T) {
	f := newWriterFixture(t)
	first := time.Date(2026, 10, 22, 12, 0, 0, 0, time.UTC)

	staleVersion := f.currentRun().ResourceVersion
	if err := f.writer.UpdateRunStatus(f.ctx, staleVersion, RunStatus{
		Published:      2,
		LastProgress:   first,
		CandidateNames: []string{f.name("a"), f.name("b")},
	}); err != nil {
		t.Fatalf("update run status: %v", err)
	}
	run := f.currentRun()
	if len(run.Status.CandidateRefs) != 2 || run.Status.CandidateRefs[0].Name != f.name("a") {
		t.Fatalf("candidateRefs = %+v", run.Status.CandidateRefs)
	}
	if run.Status.Progress == nil || run.Status.Progress.Candidates == nil || run.Status.Progress.Candidates.Published != 2 {
		t.Fatalf("progress = %+v, want 2 published", run.Status.Progress)
	}
	if run.Status.LastProgressTime == nil || !run.Status.LastProgressTime.UTC().Equal(first) {
		t.Fatalf("lastProgressTime = %v, want %v", run.Status.LastProgressTime, first)
	}
	if !hasApplyManager(run.ManagedFields, "status") {
		t.Errorf("no status apply manager %q: %+v", fieldOwner, run.ManagedFields)
	}

	// A write based on the version read before the first write conflicts.
	err := f.writer.UpdateRunStatus(f.ctx, staleVersion, RunStatus{Published: 1, CandidateNames: []string{f.name("a")}})
	if !apierrors.IsConflict(err) {
		t.Fatalf("update with a stale resourceVersion: err = %v, want a conflict", err)
	}

	// Fields the publisher owns and omits are removed: this is why the reconciler carries
	// the existing progress time into every payload.
	if err := f.writer.UpdateRunStatus(f.ctx, run.ResourceVersion, RunStatus{Published: 1, CandidateNames: []string{f.name("a")}}); err != nil {
		t.Fatalf("shrinking update: %v", err)
	}
	run = f.currentRun()
	if len(run.Status.CandidateRefs) != 1 || run.Status.Progress.Candidates.Published != 1 {
		t.Errorf("status after shrinking = %+v", run.Status)
	}
	if run.Status.LastProgressTime != nil {
		t.Errorf("lastProgressTime = %v, want it removed when the payload omits it", run.Status.LastProgressTime)
	}
}

func TestEnvtestDeleteCandidateHonorsTheUIDPrecondition(t *testing.T) {
	f := newWriterFixture(t)
	name := f.name("a")
	if _, err := f.writer.CreateCandidate(f.ctx, f.run, name, envtestCandidate()); err != nil {
		t.Fatalf("create candidate: %v", err)
	}
	uid := f.candidate(name).UID

	if err := f.writer.DeleteCandidate(f.ctx, name, types.UID("not-the-uid")); err == nil {
		t.Fatal("deleting with the wrong UID succeeded; a recreated candidate would be lost")
	}
	f.candidate(name) // still there

	if err := f.writer.DeleteCandidate(f.ctx, name, uid); err != nil {
		t.Fatalf("delete with the right UID: %v", err)
	}
	var gone v1beta2.DynamoGraphDeploymentCandidate
	err := f.client.Get(f.ctx, types.NamespacedName{Namespace: f.run.Namespace, Name: name}, &gone)
	if !apierrors.IsNotFound(err) {
		t.Fatalf("candidate still exists after delete: %v", err)
	}
	if err := f.writer.DeleteCandidate(f.ctx, name, uid); err != nil {
		t.Fatalf("deleting a missing candidate must not fail: %v", err)
	}
}

func TestEnvtestProductionWatchesDriveRepairAndTerminalCompletion(t *testing.T) {
	f := newWriterFixture(t)
	candidate := envtestCandidate()
	snapshot, err := ParseSnapshot(snapshotJSON(t, PhaseRunning, cand{id: "a", metrics: candidate.Metrics}))
	if err != nil {
		t.Fatal(err)
	}
	desired := Desire(f.run.Name, string(f.run.UID), snapshot)

	t.Log("Arrange a fully converged Run and Candidate before starting the manager")
	name := f.name("a")
	stored, err := f.writer.CreateCandidate(f.ctx, f.run, name, candidate)
	if err != nil {
		t.Fatalf("create candidate: %v", err)
	}
	if err := f.writer.PopulateStatus(f.ctx, name, stored.ResourceVersion, candidate); err != nil {
		t.Fatalf("populate candidate: %v", err)
	}
	if err := f.writer.UpdateRunStatus(f.ctx, f.currentRun().ResourceVersion, desired.Status); err != nil {
		t.Fatalf("converge run status: %v", err)
	}

	t.Log("Start the production controller watches around an instrumented Reconciler")
	recorder, completion, store, events := startPublisherController(t, f, snapshot)
	waitForReconcileQuiet(t, recorder, 250*time.Millisecond, 10*time.Second)
	beforeUpdate := recorder.count.Load()

	t.Log("Clear only Candidate status after the controller is quiet")
	var current v1beta2.DynamoGraphDeploymentCandidate
	if err := f.client.Get(f.ctx, types.NamespacedName{Namespace: f.run.Namespace, Name: name}, &current); err != nil {
		t.Fatalf("get candidate: %v", err)
	}
	current.Status.Metrics = nil
	if err := f.client.Status().Update(f.ctx, &current); err != nil {
		t.Fatalf("clear candidate status: %v", err)
	}
	require.Eventually(t, func() bool { return recorder.count.Load() > beforeUpdate }, 10*time.Second, 25*time.Millisecond,
		"Candidate update did not trigger a new reconciliation")
	require.Eventually(t, func() bool {
		var repaired v1beta2.DynamoGraphDeploymentCandidate
		if err := f.client.Get(f.ctx, types.NamespacedName{Namespace: f.run.Namespace, Name: name}, &repaired); err != nil {
			return false
		}
		return repaired.Status.Metrics != nil
	}, 10*time.Second, 25*time.Millisecond, "candidate status update did not trigger repair")
	waitForReconcileQuiet(t, recorder, 250*time.Millisecond, 10*time.Second)

	t.Log("Publish a terminal replacement and let write echoes drive completion")
	terminal, err := ParseSnapshot(snapshotJSON(t, PhaseSucceeded, cand{id: "b", metrics: candidate.Metrics}))
	if err != nil {
		t.Fatal(err)
	}
	if !store.accept(terminal) {
		t.Fatal("terminal snapshot was not accepted")
	}
	beforeTerminal := recorder.count.Load()
	events <- event.GenericEvent{Object: runKeyObject(types.NamespacedName{Namespace: f.run.Namespace, Name: f.run.Name})}
	select {
	case <-completion.Done():
	case <-time.After(10 * time.Second):
		t.Fatal("terminal completion did not observe its replacement Candidate, Run status, and deletion through the cache")
	}
	if err := completion.Err(); err != nil {
		t.Fatalf("terminal completion: %v", err)
	}
	if got := recorder.count.Load() - beforeTerminal; got < 2 {
		t.Fatalf("terminal reconciliations = %d, want a write pass and a cache-observation pass", got)
	}

	t.Log("Verify the state acknowledged by the publisher is present in the API server")
	if got := f.candidate(f.name("b")); got.Status.Metrics == nil {
		t.Fatal("candidate metrics were not populated before completion")
	}
	run := f.currentRun()
	if len(run.Status.CandidateRefs) != 1 || run.Status.CandidateRefs[0].Name != f.name("b") {
		t.Fatalf("candidateRefs = %+v, want the terminal candidate", run.Status.CandidateRefs)
	}
	var removed v1beta2.DynamoGraphDeploymentCandidate
	err = f.client.Get(f.ctx, types.NamespacedName{Namespace: f.run.Namespace, Name: f.name("a")}, &removed)
	if !apierrors.IsNotFound(err) {
		t.Fatalf("obsolete candidate still exists after completion: %v", err)
	}
}
