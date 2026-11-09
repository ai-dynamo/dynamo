// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package validation_test

import (
	"fmt"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
)

// powerCapAdmissionCases extends the shared production schema/conversion/admission chain.
func powerCapAdmissionCases() []dgdAdmissionTestCase {
	var rows []dgdAdmissionTestCase
	for _, scenario := range []string{
		"grove", "component", "planner-old", "planner-new", "multinode-old", "multinode-new",
		"Pending", "InProgress", "Failed", "Completed", "None", "non-worker", "type-transition", "image-and-cap",
		"lpx-mixed", "lpx-added", "lpx-removed", "graph-cap-old", "graph-cap-new", "auto-checkpoint",
		"legacy-v1-component", "legacy-v1-grove", "migrated-component", "migrated-grove",
		"no-hash-component", "no-hash-grove", "empty-v2-component", "empty-v2-grove",
	} {
		old := betaDGDForAdmission(nil)
		old.Annotations = map[string]string{consts.KubeAnnotationWorkloadProvider: "component", consts.KubeAnnotationEnableGrove: "false"}
		setBetaWorkerPowerInputs(old, "300", "1", 2)
		old.Spec.Components[1].Multinode = nil
		switch scenario {
		case "grove", "lpx-mixed", "lpx-added", "lpx-removed", "legacy-v1-grove", "migrated-grove", "no-hash-grove", "empty-v2-grove":
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

		// Stage-1 exclusions apply when either side of the update leaves the supported boundary.
		lpx := betaLPXDGDForAdmission(nil).Spec.Components[0]
		if scenario == "lpx-mixed" || scenario == "lpx-removed" {
			old.Spec.Components = append(old.Spec.Components, *lpx.DeepCopy())
		}
		if scenario == "graph-cap-old" {
			old.Spec.Annotations = map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}
		}
		if scenario == "auto-checkpoint" {
			old.Spec.Components[1].Experimental = &v1beta1.ExperimentalSpec{Checkpoint: &v1beta1.ComponentCheckpointConfig{Enabled: true}}
		}

		// A reconciled graph has recorded its v2 worker hash; these scenarios withhold, blank or bridge it.
		old.Annotations[consts.AnnotationCurrentWorkerHashV2] = "def456"
		switch scenario {
		case "legacy-v1-component", "legacy-v1-grove":
			old.Annotations[consts.AnnotationCurrentWorkerHash] = "abc123"
			delete(old.Annotations, consts.AnnotationCurrentWorkerHashV2)
		case "migrated-component", "migrated-grove":
			old.Annotations[consts.AnnotationCurrentWorkerHash] = "abc123"
		case "no-hash-component", "no-hash-grove":
			delete(old.Annotations, consts.AnnotationCurrentWorkerHashV2)
		case "empty-v2-component", "empty-v2-grove":
			old.Annotations[consts.AnnotationCurrentWorkerHash] = "abc123"
			old.Annotations[consts.AnnotationCurrentWorkerHashV2] = ""
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
		if scenario == "lpx-added" {
			updated.Spec.Components = append(updated.Spec.Components, *lpx.DeepCopy())
		}
		if scenario == "lpx-removed" {
			updated.Spec.Components = updated.Spec.Components[:2]
		}
		if scenario == "graph-cap-new" {
			updated.Spec.Annotations = map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}
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
		row.wantWebhookErrs, row.wantCELErr = powerCapAdmissionWant(scenario)
		rows = append(rows, row)
	}

	// A v1alpha1 service-annotation cap renders into a second worker without a podTemplate source.
	alphaOld := alphaDGDForAdmission(func(dgd *v1alpha1.DynamoGraphDeployment) {
		dgd.Annotations = map[string]string{consts.KubeAnnotationWorkloadProvider: "grove"}
		worker := dgd.Spec.Services["worker"]
		inherited := worker.DeepCopy()
		worker.ExtraPodMetadata = &v1alpha1.ExtraPodMetadata{Annotations: map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}}
		inherited.Annotations = map[string]string{consts.KubeAnnotationGPUPowerLimit: "300"}
		dgd.Spec.Services["inherited"] = inherited
	})
	alphaUpdated := alphaOld.DeepCopy()
	alphaUpdated.Spec.Services["worker"].ExtraPodMetadata.Annotations[consts.KubeAnnotationGPUPowerLimit] = "350"
	row := dgdAdmissionTestCase{name: "live power cap alpha-service-cap", oldDeployment: alphaOld, deployment: alphaUpdated}
	row.wantWebhookErrs, row.wantCELErr = powerCapAdmissionWant("alpha-service-cap")
	return append(rows, row)
}

// powerCapAdmissionWant is the expected admission oracle for one powerCapAdmissionCases scenario.
func powerCapAdmissionWant(scenario string) (webhookErrs []string, celErr string) {
	const capPath = `spec.components[1].podTemplate.metadata.annotations[dynamo.nvidia.com/gpu-power-limit]: Invalid value: "350": `
	const topologyErr = `spec.components: Forbidden: component topology is immutable and cannot be modified after creation: `
	detail := apivalidation.FieldImmutableErrorMsg
	switch scenario {
	case "grove", "component", "Completed", "None", "image-and-cap", "migrated-component", "migrated-grove":
		return nil, ""
	case "type-transition":
		return nil, `spec.components[1]: Invalid value: type is immutable after it is set`
	case "Pending", "InProgress", "Failed":
		detail = fmt.Sprintf("cannot change power cap while rolling update is %s: workers may span generations; resolve the rollout and retry after Completed or None", scenario)
	case "planner-old", "planner-new":
		detail += ": live power-cap updates are unsupported for graphs with a planner component"
	case "multinode-old", "multinode-new":
		detail += ": live power-cap updates require the grove workload provider or the component provider without multinode components"
	case "lpx-mixed", "lpx-added", "lpx-removed":
		detail += ": live power-cap updates are unsupported for graphs with LPX components"
	case "graph-cap-old", "graph-cap-new":
		detail += ": live power-cap updates are unsupported when spec.annotations sets dynamo.nvidia.com/gpu-power-limit"
	case "auto-checkpoint":
		detail += ": live power-cap updates are unsupported for worker worker with automatic checkpoints"
	case "alpha-service-cap":
		detail += ": live power-cap updates are unsupported when the v1alpha1 annotations of service inherited set dynamo.nvidia.com/gpu-power-limit"
	case "legacy-v1-component", "legacy-v1-grove", "no-hash-component", "no-hash-grove", "empty-v2-component", "empty-v2-grove":
		detail = "cannot change power cap until the operator records nvidia.com/current-worker-hash-v2: the worker hash is initialized or migrated on reconciliation; retry after the DGD is reconciled"
	}
	webhookErrs = []string{capPath + detail}
	switch scenario {
	case "multinode-new":
		webhookErrs = append([]string{`spec.components[1].multinode: Invalid value: {"nodeCount":2}: cannot change node topology between single-node and multi-node after creation`}, webhookErrs...)
	case "planner-new":
		webhookErrs = append([]string{topologyErr + "components added: [planner]"}, webhookErrs...)
	case "lpx-added":
		webhookErrs = append([]string{topologyErr + "components added: [lpx]"}, webhookErrs...)
	case "lpx-removed":
		webhookErrs = append([]string{topologyErr + "components removed: [lpx]"}, webhookErrs...)
	}
	return webhookErrs, ""
}
