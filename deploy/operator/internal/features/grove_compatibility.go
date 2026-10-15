// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package features

import (
	"context"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	appsv1 "k8s.io/api/apps/v1"
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
	// groveControllerName is the Deployment label value and container name
	// used by every Grove operator chart from v0.1.0-alpha.11 on.
	groveControllerName = "grove-operator"
	// groveMinimumAlpha is N in GroveMinimumVersion.
	groveMinimumAlpha = 14

	// groveCompatibilityTTL bounds how long a probe result is reused, so a
	// Grove upgrade is noticed without restarting the operator.
	groveCompatibilityTTL = 30 * time.Second
)

var groveAlphaTagPattern = regexp.MustCompile(`^v?0\.1\.0-alpha\.(\d+)(?:-rc\d+)?$`)

// GroveCompatibilityState classifies an external Grove installation.
type GroveCompatibilityState string

const (
	// GroveCompatible means the Grove CRDs are v0.1.0-alpha.14 or later and
	// no older Grove controller was identified.
	GroveCompatible GroveCompatibilityState = "Compatible"
	// GroveIncompatible means the Grove controller or CRDs predate
	// v0.1.0-alpha.14.
	GroveIncompatible GroveCompatibilityState = "Incompatible"
	// GroveUnverifiable means the Grove CRD is missing or not readable
	// (NotFound or Forbidden).
	GroveUnverifiable GroveCompatibilityState = "Unverifiable"
)

// GroveCompatibility is the result of one Grove compatibility probe.
type GroveCompatibility struct {
	State GroveCompatibilityState
	// Message is a user-facing explanation. When compatible it is empty or
	// notes that the controller version could not be identified.
	Message string
}

// GroveCompatibilityProbe checks whether the installed Grove controller and CRDs are new
// enough for multinode rank aliases. It is read-only; a nil probe reports Compatible.
type GroveCompatibilityProbe struct {
	reader client.Reader
	now    func() time.Time

	mu        sync.Mutex
	cached    GroveCompatibility
	checkedAt time.Time
}

// NewGroveCompatibilityProbe returns a probe that reads CRDs and Deployments through
// reader. Pass a non-cached reader so the operator starts no informers for them.
func NewGroveCompatibilityProbe(reader client.Reader) *GroveCompatibilityProbe {
	return &GroveCompatibilityProbe{reader: reader, now: time.Now}
}

// Check returns the fresh cached result or probes. A transient read error is never cached:
// Check returns the previous verdict, or the error, so one API hiccup cannot fail DGDs.
func (p *GroveCompatibilityProbe) Check(ctx context.Context) (GroveCompatibility, error) {
	if p == nil {
		return GroveCompatibility{State: GroveCompatible}, nil
	}
	p.mu.Lock()
	defer p.mu.Unlock()
	now := p.now()
	if !p.checkedAt.IsZero() && now.Sub(p.checkedAt) < groveCompatibilityTTL {
		return p.cached, nil
	}
	result, err := probeGroveCompatibility(ctx, p.reader)
	if err != nil {
		if !p.checkedAt.IsZero() {
			return p.cached, nil
		}
		return GroveCompatibility{}, err
	}
	p.cached = result
	p.checkedAt = now
	return p.cached, nil
}

func probeGroveCompatibility(ctx context.Context, reader client.Reader) (GroveCompatibility, error) {
	controllerTag, controllerOld, err := groveControllerVersion(ctx, reader)
	if err != nil {
		return GroveCompatibility{}, err
	}
	if controllerOld {
		return GroveCompatibility{
			State: GroveIncompatible,
			Message: fmt.Sprintf(
				"External Grove %s is incompatible with this Dynamo release: the Grove controller predates %s and does not set GROVE_PCSG_POD_INDEX. Upgrade the Grove controller and matching CRDs to %s or later, then the DGD reconciles automatically. Single-node DGDs are unaffected.",
				controllerTag, GroveMinimumVersion, GroveMinimumVersion,
			),
		}, nil
	}
	controllerNote := ""
	if controllerTag == "" {
		controllerNote = " The Grove controller version could not be identified (no grove-operator Deployment with a v0.1.0-alpha.N image tag was readable), so only the CRDs were checked."
	}

	crd := &apiextensionsv1.CustomResourceDefinition{}
	if err := reader.Get(ctx, types.NamespacedName{Name: grovePodCliqueSetCRDName}, crd); err != nil {
		var cause string
		switch {
		case apierrors.IsForbidden(err):
			cause = "the operator lacks permission to read CustomResourceDefinitions"
		case apierrors.IsNotFound(err):
			cause = fmt.Sprintf("the %s CRD was not found", grovePodCliqueSetCRDName)
		default:
			return GroveCompatibility{}, fmt.Errorf("read %s CRD: %w", grovePodCliqueSetCRDName, err)
		}
		return GroveCompatibility{
			State: GroveUnverifiable,
			Message: fmt.Sprintf(
				"Grove compatibility could not be verified: %s. Multinode workers need the Grove controller and matching CRDs at %s or later; confirm the installed Grove version before creating multinode DGDs.%s",
				cause, GroveMinimumVersion, controllerNote,
			),
		}, nil
	}
	if !podCliqueSetHasCliqueRollingUpdate(crd) {
		return GroveCompatibility{
			State: GroveIncompatible,
			Message: fmt.Sprintf(
				"Externally managed Grove is incompatible with this Dynamo release: the %s CRD predates %s (its %s schema has no spec.template.cliques[].rollingUpdate field). Upgrade the Grove controller and matching CRDs to %s or later, then the DGD reconciles automatically. Single-node DGDs are unaffected.%s",
				grovePodCliqueSetCRDName, GroveMinimumVersion, groveAPIVersion, GroveMinimumVersion, controllerNote,
			),
		}, nil
	}
	return GroveCompatibility{State: GroveCompatible, Message: strings.TrimSpace(controllerNote)}, nil
}

// groveControllerVersion returns the first release image tag of a Grove controller and whether
// it predates GroveMinimumVersion; "" means unreadable or digest, -dev, or custom tags only.
func groveControllerVersion(ctx context.Context, reader client.Reader) (string, bool, error) {
	deployments := &appsv1.DeploymentList{}
	if err := reader.List(ctx, deployments, client.MatchingLabels{"app.kubernetes.io/name": groveControllerName}); err != nil {
		if apierrors.IsNotFound(err) || apierrors.IsForbidden(err) {
			return "", false, nil
		}
		return "", false, fmt.Errorf("list %s Deployments: %w", groveControllerName, err)
	}
	identified := ""
	for i := range deployments.Items {
		for _, container := range deployments.Items[i].Spec.Template.Spec.Containers {
			if container.Name != groveControllerName {
				continue
			}
			tag := imageTag(container.Image)
			alpha, ok := groveAlphaNumber(tag)
			if !ok {
				continue
			}
			if alpha < groveMinimumAlpha {
				return tag, true, nil
			}
			identified = tag
		}
	}
	return identified, false, nil
}

// imageTag returns the tag of an image reference, or "" when it has none.
func imageTag(image string) string {
	if i := strings.Index(image, "@"); i >= 0 {
		image = image[:i]
	}
	if i := strings.LastIndex(image, "/"); i >= 0 {
		image = image[i+1:]
	}
	if i := strings.LastIndex(image, ":"); i >= 0 {
		return image[i+1:]
	}
	return ""
}

// groveAlphaNumber parses N from v0.1.0-alpha.N tags, including -rcM suffixes.
func groveAlphaNumber(tag string) (int, bool) {
	match := groveAlphaTagPattern.FindStringSubmatch(tag)
	if match == nil {
		return 0, false
	}
	n, err := strconv.Atoi(match[1])
	return n, err == nil
}

// podCliqueSetHasCliqueRollingUpdate reports whether served v1alpha1 has per-clique rollingUpdate,
// added in v0.1.0-alpha.14 with GROVE_PCSG_POD_INDEX; the CRDs carry no version label.
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
