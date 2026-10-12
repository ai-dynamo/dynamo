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
	"time"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdrrunpublisher/placeholderapi"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/controller/controllerutil"
)

// writer is every write the reconciler makes to the API server. It is a seam so the
// reconciler's behavior can be tested without server-side apply support in a fake client.
type writer interface {
	// CreateCandidate creates the candidate and returns its resourceVersion as the API
	// server has it. It must be idempotent.
	CreateCandidate(ctx context.Context, run *v1beta2.DynamoGraphDeploymentRun, name string, candidate dgdcreconcile.DesiredCandidate) (resourceVersion string, err error)
	// PopulateStatus records the candidate's evaluation facts and marks it Evaluated. The
	// write is constrained to resourceVersion, so a stale view of the candidate fails
	// with a conflict instead of rewriting status that was already populated.
	PopulateStatus(ctx context.Context, name, resourceVersion string, candidate dgdcreconcile.DesiredCandidate) error
	// UpdateRunStatus writes the publisher-owned run status fields, constrained to the
	// resourceVersion of the run the decision to write was based on.
	UpdateRunStatus(ctx context.Context, resourceVersion string, status RunStatus) error
	// DeleteCandidate deletes the candidate with the given UID precondition.
	DeleteCandidate(ctx context.Context, name string, uid types.UID) error
}

// clientWriter writes with server-side apply through a controller-runtime client.
type clientWriter struct {
	Client client.Client
	// Reader reads straight from the API server. A candidate's resourceVersion right
	// after its creation cannot come from the informer cache, which has not seen it yet.
	Reader    client.Reader
	Namespace string
	RunName   string
	RunUID    string
	// Now stamps the Evaluated condition; tests replace it.
	Now func() time.Time
}

var _ writer = (*clientWriter)(nil)

func (w *clientWriter) apply(ctx context.Context, object *unstructured.Unstructured, status bool) error {
	configuration := client.ApplyConfigurationFromUnstructured(object)
	if status {
		return w.Client.Status().Apply(ctx, configuration, client.FieldOwner(fieldOwner), client.ForceOwnership)
	}
	return w.Client.Apply(ctx, configuration, client.FieldOwner(fieldOwner), client.ForceOwnership)
}

// CreateCandidate applies the candidate. It is only called for a candidate missing from
// the cache, and re-applying the same content after a stale read is a no-op, so no spec
// comparison is needed.
func (w *clientWriter) CreateCandidate(ctx context.Context, run *v1beta2.DynamoGraphDeploymentRun, name string, candidate dgdcreconcile.DesiredCandidate) (string, error) {
	object, err := buildCandidateApply(w.Client.Scheme(), w.Namespace, w.RunUID, run, name, candidate)
	if err != nil {
		return "", err
	}
	if err := w.apply(ctx, object, false); err != nil {
		return "", err
	}
	var stored v1beta2.DynamoGraphDeploymentCandidate
	if err := w.Reader.Get(ctx, types.NamespacedName{Namespace: w.Namespace, Name: name}, &stored); err != nil {
		return "", err
	}
	return stored.ResourceVersion, nil
}

func (w *clientWriter) PopulateStatus(ctx context.Context, name, resourceVersion string, candidate dgdcreconcile.DesiredCandidate) error {
	now := time.Now
	if w.Now != nil {
		now = w.Now
	}
	object, err := buildCandidateStatusApply(w.Namespace, name, resourceVersion, candidate, now())
	if err != nil {
		return err
	}
	return w.apply(ctx, object, true)
}

func (w *clientWriter) UpdateRunStatus(ctx context.Context, resourceVersion string, status RunStatus) error {
	return w.apply(ctx, buildRunStatusApply(w.Namespace, w.RunName, resourceVersion, status), true)
}

func (w *clientWriter) DeleteCandidate(ctx context.Context, name string, uid types.UID) error {
	object := &v1beta2.DynamoGraphDeploymentCandidate{ObjectMeta: metav1.ObjectMeta{Namespace: w.Namespace, Name: name}}
	var options []client.DeleteOption
	if uid != "" {
		options = append(options, client.Preconditions{UID: &uid})
	}
	return client.IgnoreNotFound(w.Client.Delete(ctx, object, options...))
}

// buildCandidateApply is the server-side-apply payload that creates a candidate: labels
// and controller owner reference, the DGD spec with the resolved parameters, and the
// companion resources as an annotation. The status goes through the status subresource.
func buildCandidateApply(scheme *runtime.Scheme, namespace, runUID string, run *v1beta2.DynamoGraphDeploymentRun, name string, candidate dgdcreconcile.DesiredCandidate) (*unstructured.Unstructured, error) {
	spec, companions, err := decodeManifest(candidate)
	if err != nil {
		return nil, err
	}
	parameters, err := normalizeJSON(candidate.Parameters)
	if err != nil {
		return nil, err
	}
	object := &v1beta2.DynamoGraphDeploymentCandidate{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: namespace,
			Name:      name,
			Labels:    map[string]string{LabelRunUID: runUID, LabelCandidateID: candidate.ID},
		},
		Spec: v1beta2.DynamoGraphDeploymentCandidateSpec{DynamoGraphDeploymentSpec: *spec},
	}
	if companions != "" {
		object.Annotations = map[string]string{AnnotationAdditionalResources: companions}
	}
	if err := controllerutil.SetControllerReference(run, object, scheme); err != nil {
		return nil, err
	}
	content, err := runtime.DefaultUnstructuredConverter.ToUnstructured(object)
	if err != nil {
		return nil, err
	}
	delete(content, "status")
	unstructured.RemoveNestedField(content, "metadata", "creationTimestamp")
	if err := unstructured.SetNestedField(content, parameters, "spec", "parameters"); err != nil {
		return nil, err
	}
	apply := &unstructured.Unstructured{Object: content}
	apply.SetGroupVersionKind(v1beta2.GroupVersion.WithKind("DynamoGraphDeploymentCandidate"))
	return apply, nil
}

// withResourceVersion constrains a server-side apply to one version of the object: the
// API server rejects it with a conflict if the object has changed since.
func withResourceVersion(metadata map[string]any, resourceVersion string) map[string]any {
	if resourceVersion != "" {
		metadata["resourceVersion"] = resourceVersion
	}
	return metadata
}

// buildCandidateStatusApply is the status payload of a candidate: the metrics and the
// Evaluated condition. Candidate status is written once and then stays immutable, so the
// payload carries the resourceVersion the "not populated yet" decision was based on.
func buildCandidateStatusApply(namespace, name, resourceVersion string, candidate dgdcreconcile.DesiredCandidate, now time.Time) (*unstructured.Unstructured, error) {
	metrics, err := normalizeJSON(candidate.Metrics)
	if err != nil {
		return nil, err
	}
	apply := &unstructured.Unstructured{Object: map[string]any{
		"metadata": withResourceVersion(map[string]any{"namespace": namespace, "name": name}, resourceVersion),
		"status": map[string]any{
			"metrics": metrics,
			"conditions": []any{map[string]any{
				"type":               evaluatedCondition,
				"status":             string(metav1.ConditionTrue),
				"reason":             evaluatedCondition,
				"message":            "evaluated by the Sweeper",
				"lastTransitionTime": now.UTC().Format(time.RFC3339),
			}},
		},
	}}
	apply.SetGroupVersionKind(v1beta2.GroupVersion.WithKind("DynamoGraphDeploymentCandidate"))
	return apply, nil
}

// buildRunStatusApply is the status payload of the run. It lists every publisher-owned
// field it wants to keep: under server-side apply a field this manager owns and omits
// from the payload is removed. It carries the resourceVersion of the run the decision to
// write was based on, so a decision made from a stale cached run conflicts instead of
// overwriting a newer status.
func buildRunStatusApply(namespace, runName, resourceVersion string, status RunStatus) *unstructured.Unstructured {
	refs := make([]any, 0, len(status.CandidateNames))
	for _, name := range status.CandidateNames {
		refs = append(refs, map[string]any{"name": name})
	}
	fields := map[string]any{
		"message":       status.Message,
		"candidateRefs": refs,
	}
	if status.HasProgress {
		fields["progress"] = map[string]any{"round": int64(status.Round), "evaluated": int64(status.Evaluated)}
	}
	if !status.LastProgress.IsZero() {
		fields["lastProgressTime"] = status.LastProgress.UTC().Format(time.RFC3339)
	}
	apply := &unstructured.Unstructured{Object: map[string]any{
		"metadata": withResourceVersion(map[string]any{"namespace": namespace, "name": runName}, resourceVersion),
		"status":   fields,
	}}
	apply.SetGroupVersionKind(v1beta2.GroupVersion.WithKind("DynamoGraphDeploymentRun"))
	return apply
}

// normalizeJSON round-trips a value through JSON so it only holds the types unstructured
// objects allow (int64, float64, string, bool, maps, slices).
func normalizeJSON(value map[string]any) (map[string]any, error) {
	out := map[string]any{}
	if value == nil {
		return out, nil
	}
	raw, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &out); err != nil {
		return nil, err
	}
	return out, nil
}
