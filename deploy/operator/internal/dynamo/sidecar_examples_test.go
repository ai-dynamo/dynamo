// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"os"
	"path/filepath"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	commonconsts "github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"sigs.k8s.io/yaml"
)

func TestSGLangMultinodeSidecarExamples(t *testing.T) {
	for _, mode := range []string{"tp", "dp"} {
		t.Run(mode, func(t *testing.T) {
			t.Log("Load the shipped example through the typed DGD API")
			data, err := os.ReadFile(filepath.Join("../../../../lib/sidecar/sglang/deploy", "multinode_"+mode+".yaml"))
			require.NoError(t, err)
			var dgd v1beta1.DynamoGraphDeployment
			require.NoError(t, yaml.UnmarshalStrict(data, &dgd))

			t.Log("Require an admissible explicit-role component and preserve each authored launch")
			for i := range dgd.Spec.Components {
				component := &dgd.Spec.Components[i]
				errs := ValidateComponentSidecarConfiguration(component, field.NewPath("spec", "components").Index(i), dgd.Spec.BackendFramework)
				require.Empty(t, errs)
				if component.Multinode == nil {
					continue
				}
				require.True(t, HasCompleteMultinodeRolePodTemplates(component))
				for _, role := range component.Roles {
					authored, err := EffectiveComponentForRole(component, Role(role.Name))
					require.NoError(t, err)
					for _, provider := range []commonconsts.MultinodeDeploymentType{commonconsts.MultinodeDeploymentTypeGrove, commonconsts.MultinodeDeploymentTypeLWS} {
						pod, err := GenerateBasePodSpec(component, BackendFrameworkSGLang, nil, dgd.Name, "test", Role(role.Name), component.GetNumberOfNodes(), &configv1alpha1.OperatorConfiguration{}, provider, component.ComponentName, nil, staticContainerGPUCount(1))
						require.NoError(t, err)
						require.Equal(t, GetMainContainer(authored).Command, pod.Containers[0].Command)
						require.Equal(t, GetMainContainer(authored).Args, pod.Containers[0].Args)
						require.Equal(t, role.Name == "leader" || mode == "dp", findRuntimeInitContainer(pod) != nil)
					}
				}
			}
		})
	}
}
