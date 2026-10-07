// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"context"
	"errors"
	"fmt"
	"sync"

	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
)

const GrovePodCliqueSetCRDName = "podcliquesets.grove.io"

// ErrGroveCoherentUnsupported identifies a cluster whose PCS schema lacks Coherent support.
var ErrGroveCoherentUnsupported = errors.New("Grove Coherent updates require Grove v0.1.0-alpha.14 or later and its matching CRDs")

// GroveCoherentSupport shares one schema capability result across Grove workload controllers.
// Metadata is read from the informer; full schemas are fetched uncached and discarded.
// An enum proves schema support, not controller implementation support.
type GroveCoherentSupport struct {
	metadataReader client.Reader
	apiReader      client.Reader
	mutex          sync.Mutex
	capability     *groveCoherentCapability
}

type groveCoherentCapability struct {
	uid             types.UID
	resourceVersion string
	supported       bool
}

// NewGroveCoherentSupport constructs the manager-scoped capability checker.
// Both readers must be non-nil; apiReader must bypass the informer cache.
func NewGroveCoherentSupport(metadataReader, apiReader client.Reader) *GroveCoherentSupport {
	return &GroveCoherentSupport{metadataReader: metadataReader, apiReader: apiReader}
}

// Check rejects Coherent intent unsupported by the installed PCS API version.
// Call only for Coherent intent, before mutating workloads. The receiver must be non-nil.
func (s *GroveCoherentSupport) Check(ctx context.Context) error {
	// Serialize revision observation and refresh across both workload controllers.
	s.mutex.Lock()
	defer s.mutex.Unlock()

	// A metadata read reuses OnlyMetadata watches without starting a full-CRD informer.
	metadata := &metav1.PartialObjectMetadata{}
	metadata.SetGroupVersionKind(apiextensionsv1.SchemeGroupVersion.WithKind("CustomResourceDefinition"))
	key := client.ObjectKey{Name: GrovePodCliqueSetCRDName}
	if err := s.metadataReader.Get(ctx, key, metadata); err != nil {
		return fmt.Errorf("read Grove PodCliqueSet CRD metadata: %w", err)
	}

	// Both supported and unsupported results remain valid for this exact CRD revision.
	if s.capability != nil && s.capability.uid == metadata.UID && s.capability.resourceVersion == metadata.ResourceVersion {
		if !s.capability.supported {
			return ErrGroveCoherentUnsupported
		}
		return nil
	}

	// Schema discovery is uncached to avoid retaining every cluster CRD's schema.
	crd := &apiextensionsv1.CustomResourceDefinition{}
	if err := s.apiReader.Get(ctx, key, crd); err != nil {
		return fmt.Errorf("read Grove PodCliqueSet CRD schema: %w", err)
	}
	supported := grovePCSSchemaSupportsCoherent(crd)

	// Record the fetched revision, which may be newer than the informer observation.
	s.capability = &groveCoherentCapability{uid: crd.UID, resourceVersion: crd.ResourceVersion, supported: supported}
	if !supported {
		return ErrGroveCoherentUnsupported
	}
	return nil
}

// grovePCSSchemaSupportsCoherent checks the served version used by this operator.
// crd must be non-nil.
func grovePCSSchemaSupportsCoherent(crd *apiextensionsv1.CustomResourceDefinition) bool {
	// Require the exact strategy enum in the API version used to write PodCliqueSets.
	for _, version := range crd.Spec.Versions {
		if version.Name != grovev1alpha1.SchemeGroupVersion.Version || !version.Served || version.Schema == nil || version.Schema.OpenAPIV3Schema == nil {
			continue
		}
		spec := version.Schema.OpenAPIV3Schema.Properties["spec"]
		updateStrategy := spec.Properties["updateStrategy"]
		strategyType := updateStrategy.Properties["type"]
		for _, value := range strategyType.Enum {
			if string(value.Raw) == "\"Coherent\"" {
				return true
			}
		}
	}
	return false
}
