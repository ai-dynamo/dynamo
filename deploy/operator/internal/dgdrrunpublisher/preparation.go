/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dgdrrunpublisher

import (
	"context"
	"errors"
	"fmt"

	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	v1beta2 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta2"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdcreconcile"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/dgdoverride"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	utiljson "k8s.io/apimachinery/pkg/util/json"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/yaml"
)

// candidateStatusComplete recognizes the final evaluation and preparation result.
// status must not be nil. Both successful and invalid Candidates are complete.
func candidateStatusComplete(status *v1beta2.DynamoGraphDeploymentCandidateStatus) bool {
	condition := meta.FindStatusCondition(status.Conditions, v1beta2.DGDCConditionValid)
	return status.Parameters != nil && status.Metrics != nil && condition != nil &&
		(condition.Status == metav1.ConditionTrue || condition.Status == metav1.ConditionFalse)
}

// PrepareCandidate preserves the unpruned DGD and checks a separate copy through admission.
// run must be non-nil. Only definitive content errors become Valid=False; infrastructure
// errors return to reconciliation without caching a verdict.
func (w *clientWriter) PrepareCandidate(
	ctx context.Context,
	run *v1beta2.DynamoGraphDeploymentRun,
	candidate dgdcreconcile.DesiredCandidate,
) (dgdcreconcile.DesiredCandidate, error) {
	dgd, companions, err := decodeManifest(candidate.ID, candidate.Spec)
	if err != nil {
		return candidate, err
	}
	candidate.Validity = metav1.Condition{
		Type: v1beta2.DGDCConditionValid, Status: metav1.ConditionTrue,
		ObservedGeneration: 1, Reason: "ValidationSucceeded", Message: "Strict DGD dry-run succeeded",
		LastTransitionTime: metav1.Now(),
	}

	// The same immutable override applies independently to every selected backend and topology.
	if run.Spec.Overrides != nil && run.Spec.Overrides.DGD != nil {
		override := &unstructured.Unstructured{}
		if err := utiljson.Unmarshal(run.Spec.Overrides.DGD.Raw, &override.Object); err != nil {
			return candidate, fmt.Errorf("decode frozen DGD override: %w", err)
		}
		patched, err := dgdoverride.ApplyStrict(dgd, override)
		if err == nil {
			err = reservedCandidateMetadata(patched)
		}
		if err != nil {
			candidate.Validity.Status = metav1.ConditionFalse
			candidate.Validity.Reason = "OverrideFailed"
			if _, ok := errors.AsType[*dgdoverride.TargetNotFoundError](err); ok {
				candidate.Validity.Reason = "OverrideTargetNotFound"
			}
			candidate.Validity.Message = boundDiagnostic(err.Error())
			return candidate, nil
		}
		dgd = patched
	}

	// Serialize before the dry-run; the API client mutates the object passed to Create.
	contents, err := yaml.Marshal(dgd.Object)
	if err != nil {
		return candidate, fmt.Errorf("encode prepared DGD: %w", err)
	}
	candidate.Spec = string(contents)
	if companions != "" {
		candidate.Spec += "---\n" + companions
	}
	probe := dgd.DeepCopy()
	probe.SetGroupVersionKind(v1beta1.DynamoGraphDeploymentGVK)
	probe.SetNamespace(run.Namespace)
	probe.SetName("")
	probe.SetGenerateName("dgdr-validation-")
	err = w.Client.Create(ctx, probe, client.DryRunAll, client.FieldValidation("Strict"))
	if err == nil {
		return candidate, nil
	}
	if !apierrors.IsInvalid(err) && !apierrors.IsBadRequest(err) {
		return candidate, fmt.Errorf("strict DGD dry-run: %w", err)
	}
	candidate.Validity.Status = metav1.ConditionFalse
	candidate.Validity.Reason = "ValidationFailed"
	candidate.Validity.Message = boundDiagnostic(err.Error())
	return candidate, nil
}

// reservedCandidateMetadata prevents DGD metadata from replacing publisher transport or identity.
// dgd must not be nil. Other labels and annotations are preserved without interpretation.
func reservedCandidateMetadata(dgd *unstructured.Unstructured) error {
	for _, key := range []string{LabelRunUID, LabelCandidateID} {
		if _, found := dgd.GetLabels()[key]; found {
			return fmt.Errorf("metadata.labels[%s] is managed by the publisher", key)
		}
	}
	if _, found := dgd.GetAnnotations()[AnnotationAdditionalResources]; found {
		return fmt.Errorf("metadata.annotations[%s] is managed by the publisher", AnnotationAdditionalResources)
	}
	return nil
}

// prepareCandidates resolves each immutable point once and reuses persisted verdicts after restart.
// run and desired must not be nil. The cache is bounded by the selected Candidate projection.
func (r *Reconciler) prepareCandidates(ctx context.Context, run *v1beta2.DynamoGraphDeploymentRun, desired *DesiredState) error {
	if run.Spec.Recommendation == nil || len(desired.Candidates)+desired.Failed > int(run.Spec.Recommendation.MaxCandidates) {
		return violation("snapshot exceeds the frozen recommendation.maxCandidates")
	}
	if r.preparedCandidates == nil {
		r.preparedCandidates = map[string]dgdcreconcile.DesiredCandidate{}
	}

	// Drop obsolete entries before preparation can return a retryable error.
	live := make(map[string]bool, len(desired.Candidates))
	for _, candidate := range desired.Candidates {
		live[candidate.ID] = true
	}
	for id := range r.preparedCandidates {
		if !live[id] {
			delete(r.preparedCandidates, id)
		}
	}

	// Preserve successful preparation for retries within the current projection.
	for i, candidate := range desired.Candidates {
		prepared, found := r.preparedCandidates[candidate.ID]
		if !found {
			var existing v1beta2.DynamoGraphDeploymentCandidate
			err := r.Client.Get(ctx, types.NamespacedName{Namespace: r.Namespace, Name: desired.Names[candidate.ID]}, &existing)
			if err != nil && !apierrors.IsNotFound(err) {
				return err
			}
			owner := metav1.GetControllerOf(&existing)
			if err == nil && owner != nil && owner.UID == run.UID && candidateStatusComplete(&existing.Status) {
				prepared = candidate
				prepared.Validity = *meta.FindStatusCondition(existing.Status.Conditions, v1beta2.DGDCConditionValid)
				manifest := map[string]any{"apiVersion": "nvidia.com/v1beta1", "kind": dgdKind}
				var spec map[string]any
				if err := utiljson.Unmarshal(existing.Spec.Raw, &spec); err != nil {
					return err
				}
				manifest["spec"] = spec
				manifest["metadata"] = map[string]any{"labels": existing.Labels, "annotations": existing.Annotations}
				encoded, err := yaml.Marshal(manifest)
				if err != nil {
					return err
				}
				prepared.Spec = string(encoded)
			} else {
				prepared, err = r.Writer.PrepareCandidate(ctx, run, candidate)
				if err != nil {
					return err
				}
			}
			r.preparedCandidates[candidate.ID] = prepared
		}
		desired.Candidates[i] = prepared
		if prepared.Validity.Status == metav1.ConditionTrue {
			desired.Valid++
		} else {
			desired.Failed++
			if desired.FirstFailure == "" {
				desired.FirstFailure = boundDiagnostic(prepared.ID + ": " + prepared.Validity.Message)
			}
		}
	}
	return nil
}
