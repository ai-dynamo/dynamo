// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package validation_test

import (
	"fmt"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
)

// powerCapAdmissionCases extends the shared production schema/conversion/admission chain.
func powerCapAdmissionCases() []dgdAdmissionTestCase {
	var rows []dgdAdmissionTestCase
	for _, scenario := range []string{"grove", "component", "planner-old", "planner-new", "multinode-old", "multinode-new", "Pending", "InProgress", "Failed", "Completed", "None", "non-worker", "type-transition", "image-and-cap"} {
		old := betaDGDForAdmission(nil)
		old.Annotations = map[string]string{consts.KubeAnnotationWorkloadProvider: "component", consts.KubeAnnotationEnableGrove: "false"}
		setBetaWorkerPowerInputs(old, "300", "1", 2)
		old.Spec.Components[1].Multinode = nil
		if scenario == "grove" {
			old.Annotations[consts.KubeAnnotationWorkloadProvider] = "grove"
			delete(old.Annotations, consts.KubeAnnotationEnableGrove)
		}
		if scenario == "planner-old" {
			old.Spec.Components[0].ComponentType = "planner"
		}
		if scenario == "multinode-old" {
			old.Spec.Components[1].Multinode = &v1beta1.MultinodeSpec{NodeCount: 2}
		}
		if scenario == "non-worker" || scenario == "type-transition" {
			old.Spec.Components[1].ComponentType = "frontend"
		}
		switch scenario {
		case "Pending", "InProgress", "Failed", "Completed", "None":
			old.Status.RollingUpdate = &v1beta1.RollingUpdateStatus{Phase: v1beta1.RollingUpdatePhase(scenario)}
		}
		updated := old.DeepCopy()
		updated.Spec.Components[1].PodTemplate.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"

		if scenario == "planner-new" {
			planner := *updated.Spec.Components[0].DeepCopy()
			planner.ComponentName, planner.ComponentType = "planner", "planner"
			updated.Spec.Components = append(updated.Spec.Components, planner)
		}
		if scenario == "multinode-new" {
			updated.Spec.Components[1].Multinode = &v1beta1.MultinodeSpec{NodeCount: 2}
		}

		if scenario == "type-transition" {
			updated.Spec.Components[1].ComponentType = "worker"
		}
		if scenario == "image-and-cap" {
			updated.Spec.Components[1].PodTemplate.Spec.Containers[0].Image = "worker:v2"
		}
		if scenario == "None" {
			old.Status.RollingUpdate.Phase = ""
			updated.Status.RollingUpdate.Phase = ""
		}
		row := dgdAdmissionTestCase{name: "live power cap " + scenario, oldDeployment: old, deployment: updated}
		switch scenario {
		case "grove", "component", "Completed", "None", "image-and-cap":
		default:
			detail := apivalidation.FieldImmutableErrorMsg
			if scenario == "Pending" || scenario == "InProgress" || scenario == "Failed" {
				detail = fmt.Sprintf("cannot change power cap while rolling update is %s: workers may span generations; resolve the rollout and retry after Completed or None", scenario)
			}
			row.wantWebhookErrs = []string{`spec.components[1].podTemplate.metadata.annotations[dynamo.nvidia.com/gpu-power-limit]: Invalid value: "350": ` + detail}
			if scenario == "multinode-new" {
				row.wantWebhookErrs = append([]string{`spec.components[1].multinode: Invalid value: {"nodeCount":2}: cannot change node topology between single-node and multi-node after creation`}, row.wantWebhookErrs...)
			}
			if scenario == "planner-new" {
				row.wantWebhookErrs = append([]string{`spec.components: Forbidden: component topology is immutable and cannot be modified after creation: components added: [planner]`}, row.wantWebhookErrs...)
			}
			if scenario == "type-transition" {
				row.wantWebhookErrs = nil
				row.wantCELErr = `spec.components[1]: Invalid value: type is immutable after it is set`
			}
		}
		rows = append(rows, row)
	}
	return rows
}
