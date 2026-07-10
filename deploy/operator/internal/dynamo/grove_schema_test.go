// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"os"
	"path/filepath"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	controller_common "github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	grovecrds "github.com/ai-dynamo/grove/operator/api/core/v1alpha1/crds"
	"github.com/stretchr/testify/require"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsvalidation "k8s.io/apiextensions-apiserver/pkg/apiserver/validation"
	apitest "k8s.io/apiextensions-apiserver/pkg/test"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/utils/ptr"
)

type grovePodCliqueSetRequestValidator struct {
	schemaValidator apiextensionsvalidation.SchemaValidator
	celValidator    apitest.CELValidateFunc
}

func newGrovePodCliqueSetRequestValidator(t *testing.T) *grovePodCliqueSetRequestValidator {
	t.Helper()

	crdPath := filepath.Join(t.TempDir(), "grove.io_podcliquesets.yaml")
	require.NoError(t, os.WriteFile(crdPath, []byte(grovecrds.PodCliqueSetCRD()), 0o600))
	crd := apitest.MustLoadManifest[apiextensionsv1.CustomResourceDefinition](t, crdPath)

	versionName := grovev1alpha1.SchemeGroupVersion.Version
	var schema *apiextensionsv1.JSONSchemaProps
	for i := range crd.Spec.Versions {
		if crd.Spec.Versions[i].Name == versionName {
			schema = crd.Spec.Versions[i].Schema.OpenAPIV3Schema
			break
		}
	}
	require.NotNil(t, schema, "Grove CRD has no %s schema", versionName)

	var internalSchema apiextensions.JSONSchemaProps
	require.NoError(t, apiextensionsv1.Convert_v1_JSONSchemaProps_To_apiextensions_JSONSchemaProps(schema, &internalSchema, nil))
	schemaValidator, _, err := apiextensionsvalidation.NewSchemaValidator(&internalSchema)
	require.NoError(t, err)

	celValidator, ok := apitest.VersionValidatorsFromFile(t, crdPath)[versionName]
	require.True(t, ok, "Grove CRD has no %s CEL validator", versionName)
	return &grovePodCliqueSetRequestValidator{
		schemaValidator: schemaValidator,
		celValidator:    celValidator,
	}
}

func (v *grovePodCliqueSetRequestValidator) validate(t *testing.T, current, old *grovev1alpha1.PodCliqueSet) {
	t.Helper()

	toUnstructured := func(pcs *grovev1alpha1.PodCliqueSet) map[string]any {
		if pcs == nil {
			return nil
		}
		pcs = pcs.DeepCopy()
		pcs.APIVersion = grovev1alpha1.SchemeGroupVersion.String()
		pcs.Kind = "PodCliqueSet"
		obj, err := runtime.DefaultUnstructuredConverter.ToUnstructured(pcs)
		require.NoError(t, err)
		delete(obj, "status")
		return obj
	}

	currentObject := toUnstructured(current)
	oldObject := toUnstructured(old)
	var schemaErrs field.ErrorList
	if oldObject == nil {
		schemaErrs = apiextensionsvalidation.ValidateCustomResource(nil, currentObject, v.schemaValidator)
	} else {
		schemaErrs = apiextensionsvalidation.ValidateCustomResourceUpdate(nil, currentObject, oldObject, v.schemaValidator)
	}
	require.Empty(t, schemaErrs)
	require.Empty(t, v.celValidator(currentObject, oldObject))
}

func generateTopologyTestPodCliqueSet(t *testing.T, deploymentPack v1alpha1.TopologyDomain, multinode bool) *grovev1alpha1.PodCliqueSet {
	t.Helper()

	component := &v1alpha1.DynamoComponentDeploymentSharedSpec{
		ComponentType: "worker",
		Replicas:      ptr.To(int32(2)),
		TopologyConstraint: &v1alpha1.TopologyConstraint{
			PackDomain: "rack",
		},
	}
	if multinode {
		component.Multinode = &v1alpha1.MultinodeSpec{NodeCount: 2}
	}
	deployment := &v1alpha1.DynamoGraphDeployment{
		ObjectMeta: metav1.ObjectMeta{Name: "test-deploy", Namespace: "default"},
		Spec: v1alpha1.DynamoGraphDeploymentSpec{
			TopologyConstraint: &v1alpha1.SpecTopologyConstraint{
				TopologyProfile: "grove-topology",
				PackDomain:      deploymentPack,
			},
			Services: map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{"Worker": component},
		},
	}

	pcs, err := GenerateGrovePodCliqueSet(
		t.Context(),
		betaDGD(t, deployment),
		&configv1alpha1.OperatorConfiguration{},
		&controller_common.RuntimeConfig{},
		nil,
		&mockSecretsRetriever{},
		&RestartState{},
		nil,
		nil,
	)
	require.NoError(t, err)
	return pcs
}

func TestGeneratedGroveTopologyConstraintsValidateAgainstPinnedCRD(t *testing.T) {
	validator := newGrovePodCliqueSetRequestValidator(t)

	tests := []struct {
		name           string
		deploymentPack v1alpha1.TopologyDomain
		multinode      bool
	}{
		{name: "deployment and component constraints", deploymentPack: "zone"},
		{name: "service-only single-node constraint"},
		{name: "service-only multinode constraint", multinode: true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pcs := generateTopologyTestPodCliqueSet(t, tt.deploymentPack, tt.multinode)
			validator.validate(t, pcs, nil)
		})
	}
}

func TestLegacyGroveTopologyConstraintUpgradeValidatesAgainstPinnedCRD(t *testing.T) {
	validator := newGrovePodCliqueSetRequestValidator(t)

	tests := []struct {
		name       string
		modern     func(*testing.T) *grovev1alpha1.PodCliqueSet
		constraint func(*grovev1alpha1.PodCliqueSet) *grovev1alpha1.TopologyConstraint
		packDomain grovev1alpha1.TopologyDomain
	}{
		{
			name: "template",
			modern: func(t *testing.T) *grovev1alpha1.PodCliqueSet {
				return generateTopologyTestPodCliqueSet(t, "zone", false)
			},
			constraint: func(pcs *grovev1alpha1.PodCliqueSet) *grovev1alpha1.TopologyConstraint {
				return pcs.Spec.Template.TopologyConstraint
			},
			packDomain: "zone",
		},
		{
			name: "clique",
			modern: func(t *testing.T) *grovev1alpha1.PodCliqueSet {
				pcs := generateTopologyTestPodCliqueSet(t, "", false)
				require.Len(t, pcs.Spec.Template.Cliques, 1)
				return pcs
			},
			constraint: func(pcs *grovev1alpha1.PodCliqueSet) *grovev1alpha1.TopologyConstraint {
				return pcs.Spec.Template.Cliques[0].TopologyConstraint
			},
			packDomain: "rack",
		},
		{
			name: "scaling group",
			modern: func(t *testing.T) *grovev1alpha1.PodCliqueSet {
				pcs := generateTopologyTestPodCliqueSet(t, "", true)
				require.Len(t, pcs.Spec.Template.PodCliqueScalingGroupConfigs, 1)
				return pcs
			},
			constraint: func(pcs *grovev1alpha1.PodCliqueSet) *grovev1alpha1.TopologyConstraint {
				return pcs.Spec.Template.PodCliqueScalingGroupConfigs[0].TopologyConstraint
			},
			packDomain: "rack",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			modern := tt.modern(t)
			modernConstraint := tt.constraint(modern)
			require.NotNil(t, modernConstraint)

			legacy := modern.DeepCopy()
			*tt.constraint(legacy) = grovev1alpha1.TopologyConstraint{PackDomain: tt.packDomain}
			repaired := modern.DeepCopy()
			*tt.constraint(repaired) = grovev1alpha1.TopologyConstraint{
				TopologyName: modernConstraint.TopologyName,
				PackDomain:   tt.packDomain,
			}

			validator.validate(t, repaired, legacy)
			validator.validate(t, modern, repaired)
		})
	}
}
