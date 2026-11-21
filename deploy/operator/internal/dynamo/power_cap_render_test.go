// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamo

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"testing"

	configv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/config/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/controller_common"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
)

func powerCapRenderInput(t *testing.T, provider, source, watts string) *v1beta1.DynamoGraphDeployment {
	t.Helper()
	image := "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.5.0"
	dgd := betaDGDWithRuntimeVersion(t, image, "")
	if source == "beta" {
		dgd = &v1beta1.DynamoGraphDeployment{Name: dgd.Name, Namespace: dgd.Namespace, Spec: *dgd.Spec.DeepCopy()}
	}
	if source == "alpha" || source == "alpha-inherited" {
		alpha := baseDGD(map[string]*v1alpha1.DynamoComponentDeploymentSharedSpec{
			"worker": {ComponentType: consts.ComponentTypeWorker, Replicas: new(int32(1))},
		})
		worker := alpha.Spec.Services["worker"]
		worker.ExtraPodMetadata = &v1alpha1.ExtraPodMetadata{Annotations: map[string]string{"example.com/keep": "yes"}}
		if watts != "" {
			worker.ExtraPodMetadata.Annotations[consts.KubeAnnotationGPUPowerLimit] = watts
		}
		if source == "alpha-inherited" {
			worker.Annotations = map[string]string{consts.KubeAnnotationGPUPowerLimit: "250"}
		}
		dgd = rawBetaDGD(t, alpha)
		dgd.Spec.Components[0].PodTemplate.Spec.Containers = []corev1.Container{{Name: consts.MainContainerName, Image: image}}
	} else {
		dgd.Spec.Components[0].PodTemplate.Annotations = map[string]string{"example.com/keep": "yes"}
		if watts != "" {
			dgd.Spec.Components[0].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = watts
		}
	}
	if dgd.Annotations == nil {
		dgd.Annotations = make(map[string]string)
	}
	dgd.Annotations[consts.KubeAnnotationWorkloadProvider] = provider
	return dgd
}

func renderPowerCapResources(t *testing.T, dgd *v1beta1.DynamoGraphDeployment) (string, *v1beta1.DynamoComponentDeployment, *grovev1alpha1.PodCliqueSet) {
	t.Helper()
	hash := mustComputeBetaDGDWorkersSpecHash(t, dgd)
	dcds, err := GenerateDynamoComponentsDeployments(dgd, nil, nil, RollingUpdateContext{NewWorkerHash: hash})
	require.NoError(t, err)
	pcs, err := GenerateGrovePodCliqueSet(t.Context(), dgd, nil, &configv1alpha1.OperatorConfiguration{}, &controller_common.RuntimeConfig{}, nil, &mockSecretsRetriever{}, nil, nil, true, nil)
	require.NoError(t, err)
	return hash, dcds["worker"], pcs
}

func powerCapRenderDigest(t *testing.T, resource any) string {
	t.Helper()
	data, err := json.Marshal(resource)
	require.NoError(t, err)
	return fmt.Sprintf("%x", sha256.Sum256(data))
}

func TestWorkerPowerCapUpgradeBaseline(t *testing.T) {
	for _, provider := range []string{consts.WorkloadProviderComponent, consts.WorkloadProviderGrove} {
		for _, source := range []string{"beta", "alpha"} {
			t.Run(provider+"/"+source, func(t *testing.T) {
				t.Log("Render capped and uncapped workers through the same path as the live operator")
				var result struct {
					DGD                                                *v1beta1.DynamoGraphDeployment
					DCD                                                *v1beta1.DynamoComponentDeployment `json:",omitempty"`
					PCS                                                *grovev1alpha1.PodCliqueSet        `json:",omitempty"`
					UncappedHash, UncappedDCDDigest, UncappedPCSDigest string
				}
				uncapped := powerCapRenderInput(t, provider, source, "")
				hash, dcd, pcs := renderPowerCapResources(t, uncapped)
				result.UncappedHash, result.UncappedDCDDigest, result.UncappedPCSDigest = hash, powerCapRenderDigest(t, dcd), powerCapRenderDigest(t, pcs)
				filename := provider + "-" + source + ".json"
				dir := os.Getenv("DYNAMO_POWER_CAP_BASELINE_DIR")
				if dir == "" {
					data, err := os.ReadFile(filepath.Join("../controller/testdata/power-cap-upgrade", filename))
					require.NoError(t, err)
					var baseline struct {
						UncappedHash, UncappedDCDDigest, UncappedPCSDigest string
					}
					require.NoError(t, json.Unmarshal(data, &baseline))
					require.Equal(t, baseline.UncappedHash, result.UncappedHash)
					require.Equal(t, baseline.UncappedDCDDigest, result.UncappedDCDDigest)
					require.Equal(t, baseline.UncappedPCSDigest, result.UncappedPCSDigest)
					return
				}

				t.Log("Capture the old resources for the upgrade regression")
				result.DGD = powerCapRenderInput(t, provider, source, "300")
				oldHash, oldDCD, oldPCS := renderPowerCapResources(t, result.DGD)
				result.DGD.Annotations[consts.AnnotationCurrentWorkerHashV2] = oldHash
				if provider == consts.WorkloadProviderGrove {
					result.PCS = oldPCS
				} else {
					result.DCD = oldDCD
				}
				data, err := json.MarshalIndent(result, "", "  ")
				require.NoError(t, err)
				require.NoError(t, os.WriteFile(filepath.Join(dir, filename), append(data, '\n'), 0644))
			})
		}
	}
}

func TestWorkerPowerCapRender(t *testing.T) {
	for _, provider := range []string{consts.WorkloadProviderComponent, consts.WorkloadProviderGrove} {
		for _, source := range []string{"beta", "alpha", "alpha-inherited", "graph-inherited"} {
			t.Run(provider+"/"+source, func(t *testing.T) {
				t.Log("Render an explicitly capped worker from its preserved DGD input")
				dgd := powerCapRenderInput(t, provider, source, "300")
				wantTemplateCap := ""
				if source == "alpha-inherited" || source == "graph-inherited" {
					wantTemplateCap = "300"
				}
				if source == "graph-inherited" {
					dgd.Spec.Annotations = map[string]string{consts.KubeAnnotationGPUPowerLimit: "250"}
				}
				saved := dgd.DeepCopy()
				hash, dcd, pcs := renderPowerCapResources(t, dgd)
				require.Equal(t, saved, dgd)
				const key = consts.KubeAnnotationGPUPowerLimit
				require.NotContains(t, dcd.Annotations, key)
				require.Equal(t, wantTemplateCap, GetDCDKubeAnnotations(dcd)[key])
				require.Equal(t, "yes", GetDCDKubeAnnotations(dcd)["example.com/keep"])
				require.Len(t, pcs.Spec.Template.Cliques, 1)
				clique := pcs.Spec.Template.Cliques[0]
				require.Equal(t, wantTemplateCap, clique.Annotations[key])
				require.Equal(t, "yes", clique.Annotations["example.com/keep"])
				require.Equal(t, dgd.Name, clique.Labels[consts.KubeLabelDynamoGraphDeploymentName])
				require.Equal(t, "worker", clique.Labels[consts.KubeLabelDynamoComponent])
				require.Equal(t, "worker", clique.Labels[consts.KubeLabelDynamoComponentType])
				if source == "alpha" {
					roundTrip := &v1alpha1.DynamoGraphDeployment{}
					require.NoError(t, roundTrip.ConvertFrom(dgd))
					require.Equal(t, "300", roundTrip.Spec.Services["worker"].ExtraPodMetadata.Annotations[key])
				}
				if wantTemplateCap != "" {
					require.Empty(t, DesiredWorkerPowerCaps(dgd))
					return
				}

				t.Log("Edit the desired cap repeatedly, then combine a cap edit with an image change")
				for _, watts := range []string{"350", "325", "340"} {
					dgd.Spec.Components[0].PodTemplate.Annotations[key] = watts
					newHash, newDCD, newPCS := renderPowerCapResources(t, dgd)
					require.Equal(t, hash, newHash)
					require.Equal(t, dcd, newDCD)
					require.Equal(t, pcs, newPCS)
				}
				dgd.Spec.Components[0].PodTemplate.Spec.Containers[0].Image = "nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0"
				imageHash := mustComputeBetaDGDWorkersSpecHash(t, dgd)
				require.NotEqual(t, hash, imageHash)
				dgd.Spec.Components[0].PodTemplate.Annotations[key] = "300"
				require.Equal(t, imageHash, mustComputeBetaDGDWorkersSpecHash(t, dgd))
			})
		}
	}
}

func TestWorkerPowerCapRenderGroveRoleTemplates(t *testing.T) {
	t.Log("Pair a capped prefill component with a decode component that supplies its own role templates")
	const capKey = consts.KubeAnnotationGPUPowerLimit
	dgd := powerCapRenderInput(t, consts.WorkloadProviderGrove, "beta", "300")
	dgd.Spec.Components[0].ComponentType = consts.ComponentTypePrefill
	decode := dgd.Spec.Components[0].DeepCopy()
	decode.ComponentName, decode.ComponentType = "decode", consts.ComponentTypeDecode
	decode.Multinode = &v1beta1.MultinodeSpec{NodeCount: 2}
	roleTemplate := decode.PodTemplate
	delete(roleTemplate.Annotations, capKey)
	decode.PodTemplate = nil
	decode.Roles = []v1beta1.ComponentRoleSpec{
		{Name: v1beta1.ComponentRoleLeader, PodTemplate: roleTemplate.DeepCopy()},
		{Name: v1beta1.ComponentRoleWorker, PodTemplate: roleTemplate.DeepCopy()},
	}
	dgd.Spec.Components = append(dgd.Spec.Components, *decode)
	require.Equal(t, map[string]string{"worker": "300"}, DesiredWorkerPowerCaps(dgd))

	t.Log("Keep power cap data out of Grove templates and preserve both decode role templates")
	_, _, pcs := renderPowerCapResources(t, dgd)
	require.Len(t, pcs.Spec.Template.Cliques, 3)
	countsByType := make(map[string]int)
	for _, clique := range pcs.Spec.Template.Cliques {
		require.NotContains(t, clique.Annotations, capKey)
		require.Equal(t, "yes", clique.Annotations["example.com/keep"])
		countsByType[clique.Labels[consts.KubeLabelDynamoComponentType]]++
	}
	require.Equal(t, map[string]int{"prefill": 1, "decode": 2}, countsByType)
}
