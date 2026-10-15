/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package features

import (
	"context"
	"fmt"
	"sync"
	"time"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const (
	// GroveMinimumVersion is the oldest Grove release that sets
	// GROVE_PCSG_POD_INDEX, which multinode rank aliases reference.
	GroveMinimumVersion = "v0.1.0-alpha.14"

	grovePodCliqueSetCRDName = "podcliquesets.grove.io"
	groveAPIVersion          = "v1alpha1"

	// groveCompatibilityTTL bounds how long a probe result is reused, so a
	// Grove upgrade is noticed without restarting the operator.
	groveCompatibilityTTL = 30 * time.Second
)

// GroveCompatibilityState classifies an external Grove installation.
type GroveCompatibilityState string

const (
	// GroveCompatible means the Grove CRDs are v0.1.0-alpha.14 or later.
	GroveCompatible GroveCompatibilityState = "Compatible"
	// GroveIncompatible means the Grove CRDs predate v0.1.0-alpha.14.
	GroveIncompatible GroveCompatibilityState = "Incompatible"
	// GroveUnverifiable means the operator could not read the Grove CRDs.
	GroveUnverifiable GroveCompatibilityState = "Unverifiable"
)

// GroveCompatibility is the result of one Grove compatibility probe.
type GroveCompatibility struct {
	State GroveCompatibilityState
	// Message is a user-facing explanation; empty when compatible.
	Message string
}

// GroveCompatibilityProbe checks whether the installed Grove CRDs are new
// enough for multinode rank aliases. It never modifies the cluster. A nil
// probe reports Compatible so callers that do not wire one keep working.
type GroveCompatibilityProbe struct {
	reader client.Reader
	now    func() time.Time

	mu        sync.Mutex
	cached    GroveCompatibility
	checkedAt time.Time
}

// NewGroveCompatibilityProbe returns a probe that reads CRDs through reader.
// Use a non-cached reader so the operator does not start a CRD informer.
func NewGroveCompatibilityProbe(reader client.Reader) *GroveCompatibilityProbe {
	return &GroveCompatibilityProbe{reader: reader, now: time.Now}
}

// Check returns the cached result when it is fresh and probes otherwise.
func (p *GroveCompatibilityProbe) Check(ctx context.Context) GroveCompatibility {
	if p == nil {
		return GroveCompatibility{State: GroveCompatible}
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	now := p.now()
	if !p.checkedAt.IsZero() && now.Sub(p.checkedAt) < groveCompatibilityTTL {
		return p.cached
	}
	p.cached = probeGroveCompatibility(ctx, p.reader)
	p.checkedAt = now
	return p.cached
}

func probeGroveCompatibility(ctx context.Context, reader client.Reader) GroveCompatibility {
	crd := &apiextensionsv1.CustomResourceDefinition{}
	if err := reader.Get(ctx, types.NamespacedName{Name: grovePodCliqueSetCRDName}, crd); err != nil {
		cause := err.Error()
		switch {
		case apierrors.IsForbidden(err):
			cause = "the operator lacks permission to read CustomResourceDefinitions"
		case apierrors.IsNotFound(err):
			cause = fmt.Sprintf("the %s CRD was not found", grovePodCliqueSetCRDName)
		}
		return GroveCompatibility{
			State: GroveUnverifiable,
			Message: fmt.Sprintf(
				"Grove compatibility could not be verified: %s. Multinode workers need the Grove controller and matching CRDs at %s or later; confirm the installed Grove version before creating multinode DGDs.",
				cause, GroveMinimumVersion,
			),
		}
	}
	if !podCliqueSetHasCliqueRollingUpdate(crd) {
		return GroveCompatibility{
			State: GroveIncompatible,
			Message: fmt.Sprintf(
				"Externally managed Grove is incompatible with this Dynamo release: the %s CRD predates %s (its %s schema has no spec.template.cliques[].rollingUpdate field). Upgrade the Grove controller and matching CRDs to %s or later, then the DGD reconciles automatically. Single-node DGDs are unaffected.",
				grovePodCliqueSetCRDName, GroveMinimumVersion, groveAPIVersion, GroveMinimumVersion,
			),
		}
	}
	return GroveCompatibility{State: GroveCompatible}
}

// podCliqueSetHasCliqueRollingUpdate reports whether the served v1alpha1
// schema has the per-clique rollingUpdate field. Grove added that field in
// the same release as GROVE_PCSG_POD_INDEX (absent through v0.1.0-alpha.13,
// present from v0.1.0-alpha.14). Grove releases carry no readable version
// label, so the schema is the reliable signal.
func podCliqueSetHasCliqueRollingUpdate(crd *apiextensionsv1.CustomResourceDefinition) bool {
	for i := range crd.Spec.Versions {
		version := &crd.Spec.Versions[i]
		if version.Name != groveAPIVersion || !version.Served || version.Schema == nil || version.Schema.OpenAPIV3Schema == nil {
			continue
		}
		spec, ok := version.Schema.OpenAPIV3Schema.Properties["spec"]
		if !ok {
			return false
		}
		template, ok := spec.Properties["template"]
		if !ok {
			return false
		}
		cliques, ok := template.Properties["cliques"]
		if !ok || cliques.Items == nil || cliques.Items.Schema == nil {
			return false
		}
		_, ok = cliques.Items.Schema.Properties["rollingUpdate"]
		return ok
	}
	return false
}
