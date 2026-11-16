/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package searchui

import (
	"context"
	"errors"
	"fmt"
	"net/http"
	"strings"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

// DGDRequest asks for an independent DynamoGraphDeployment built from a Candidate.
type DGDRequest struct {
	CandidateNamespace string `json:"candidateNamespace"`
	CandidateName      string `json:"candidateName"`
	Namespace          string `json:"namespace"`
	Name               string `json:"dgdName"`
	DryRun             bool   `json:"dryRun"`
}

// DGDResult reports what was created, or what would be.
type DGDResult struct {
	Namespace  string `json:"namespace"`
	Name       string `json:"name"`
	DryRun     bool   `json:"dryRun"`
	Validation string `json:"validation"`
}

// statusError carries an HTTP status for the handler.
type statusError struct {
	status int
	msg    string
}

func (e *statusError) Error() string { return e.msg }

func failf(status int, format string, args ...any) error {
	return &statusError{status: status, msg: fmt.Sprintf(format, args...)}
}

func namespacedName(namespace, name string) types.NamespacedName {
	return types.NamespacedName{Namespace: namespace, Name: name}
}

// RenderDGD builds the DGD object for a Candidate: only the deployable spec is copied;
// evaluation parameters, Candidate metadata and status are left behind.
func RenderDGD(cand map[string]any, namespace, name string) (*unstructured.Unstructured, error) {
	spec := candidateDGDSpec(cand)
	if len(spec) == 0 {
		return nil, failf(http.StatusUnprocessableEntity, "candidate has no DGD spec")
	}
	dgd := &unstructured.Unstructured{Object: map[string]any{
		"metadata": map[string]any{
			"name":      name,
			"namespace": namespace,
			"annotations": map[string]any{
				annotationFromDGD: str(cand, "metadata", "namespace") + "/" + str(cand, "metadata", "name"),
			},
		},
		"spec": spec,
	}}
	dgd.SetGroupVersionKind(gvkDGD)
	return dgd, nil
}

// CreateDGD validates the request, runs a strict server-side dry run, and unless
// req.DryRun is set creates the DGD. A Candidate with Valid=False is refused; one without
// a Valid condition (older publisher) relies on the dry run alone.
func CreateDGD(ctx context.Context, cl client.Client, cand map[string]any, req DGDRequest) (DGDResult, error) {
	if req.Name == "" || req.Namespace == "" {
		return DGDResult{}, failf(http.StatusBadRequest, "dgdName and namespace are required")
	}
	if errs := validation.IsDNS1123Subdomain(req.Name); len(errs) > 0 {
		return DGDResult{}, failf(http.StatusBadRequest, "invalid DGD name %q: %s", req.Name, strings.Join(errs, "; "))
	}
	if errs := validation.IsDNS1123Label(req.Namespace); len(errs) > 0 {
		return DGDResult{}, failf(http.StatusBadRequest, "invalid namespace %q: %s", req.Namespace, strings.Join(errs, "; "))
	}
	valid, reason, message := candidateValidity(cand)
	if valid == "False" {
		return DGDResult{}, failf(http.StatusConflict, "candidate is Valid=False (%s): %s", reason, message)
	}

	dgd, err := RenderDGD(cand, req.Namespace, req.Name)
	if err != nil {
		return DGDResult{}, err
	}
	if err := cl.Create(ctx, dgd.DeepCopy(), client.DryRunAll); err != nil {
		return DGDResult{}, translateCreateError(err, req)
	}
	validation := "dry run passed"
	if valid == "NotEvaluated" {
		validation = "dry run passed (candidate had no Valid condition; validated now)"
	}
	if req.DryRun {
		return DGDResult{Namespace: req.Namespace, Name: req.Name, DryRun: true, Validation: validation}, nil
	}
	if err := cl.Create(ctx, dgd); err != nil {
		return DGDResult{}, translateCreateError(err, req)
	}
	return DGDResult{Namespace: req.Namespace, Name: req.Name, Validation: validation}, nil
}

func translateCreateError(err error, req DGDRequest) error {
	switch {
	case apierrors.IsAlreadyExists(err):
		return failf(http.StatusConflict, "a DynamoGraphDeployment named %q already exists in %s", req.Name, req.Namespace)
	case apierrors.IsForbidden(err):
		return failf(http.StatusForbidden, "not allowed to create DynamoGraphDeployments in %s: %v", req.Namespace, err)
	case apierrors.IsInvalid(err), apierrors.IsBadRequest(err):
		return failf(http.StatusUnprocessableEntity, "the API server rejected the DGD: %v", err)
	case apierrors.IsNotFound(err):
		return failf(http.StatusNotFound, "%v", err)
	}
	if _, ok := errors.AsType[*statusError](err); ok {
		return err
	}
	return failf(http.StatusBadGateway, "create DGD: %v", err)
}
