// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package validation_test

import (
	"fmt"
	"strings"
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	"github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/stretchr/testify/require"
	apivalidation "k8s.io/apimachinery/pkg/api/validation"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func retuneDGDForAdmission(change func(*v1beta1.DynamoGraphDeployment)) *v1beta1.DynamoGraphDeployment {
	dgd := betaDGDForAdmission(nil)
	addBetaPlanner(dgd, "registry.example/planner:1.1.0", "1.1.0")
	dgd.Spec.Components[1], dgd.Spec.Components[2] = dgd.Spec.Components[2], dgd.Spec.Components[1]
	dgd.Spec.Components[0].ComponentName = "gateway"
	dgd.Spec.Components[1].Retune = &v1beta1.RetuneSpec{Enabled: true,
		TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "request-traces"},
	}
	if change != nil {
		change(dgd)
	}
	return dgd
}

func retuneDGDAdmissionCases(t *testing.T) []dgdAdmissionTestCase {
	t.Helper()
	var tests []dgdAdmissionTestCase
	retunePath := field.NewPath("spec", "components").Index(1).Child("retune")
	for _, alpha := range []bool{false, true} {
		rows := []dgdAdmissionTestCase{
			{name: "enabled with renamed frontend and no existing PVC", deployment: retuneDGDForAdmission(nil)},
			{name: "router accepts zero startup values", deployment: retuneDGDForAdmission(func(dgd *v1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[1].Retune.Router = &v1beta1.RetuneRouterSpec{OverlapScoreCredit: new(float64(0)), PrefillLoadScale: new(float64(0)), Temperature: new(float64(0))}
			})},
			{name: "negative router startup value", deployment: retuneDGDForAdmission(func(dgd *v1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[1].Retune.Router = &v1beta1.RetuneRouterSpec{Temperature: new(float64(-1))}
			}), wantSchemaErr: "spec.components[1].retune.router.temperature: Invalid value: -1: spec.components[1].retune.router.temperature in body should be greater than or equal to 0"},
			{name: "disabled without frontend or storage", deployment: retuneDGDForAdmission(func(dgd *v1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[1].Retune = &v1beta1.RetuneSpec{}
				dgd.Spec.Components = dgd.Spec.Components[1:]
			})},
			{name: "non-Planner component cannot declare even disabled Retune", deployment: retuneDGDForAdmission(func(dgd *v1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[1].ComponentType = v1beta1.ComponentTypeWorker
				dgd.Spec.Components[1].Retune.Enabled = false
			}), wantWebhookErrs: []string{field.Forbidden(retunePath, "may only be set when type is planner").Error()}},
			{name: "enabled requires trace storage", deployment: retuneDGDForAdmission(func(dgd *v1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[1].Retune.TraceStorage = nil
			}), wantWebhookErrs: []string{field.Required(retunePath.Child("traceStorage"), "must reference the trace PVC when Retune is enabled").Error()}},
			{name: "invalid trace PVC name", deployment: retuneDGDForAdmission(func(dgd *v1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[1].Retune.TraceStorage.ClaimName = "bad_claim"
			}), wantWebhookErrs: []string{field.Invalid(retunePath.Child("traceStorage", "claimName"), "bad_claim", apivalidation.NameIsDNSSubdomain("bad_claim", false)[0]).Error()}},
			{name: "missing frontend", deployment: retuneDGDForAdmission(func(dgd *v1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components = dgd.Spec.Components[1:]
			}), wantWebhookErrs: []string{field.Forbidden(field.NewPath("spec", "components").Index(0).Child("retune"), "enabled Retune requires exactly one frontend component selected by type").Error()}},
			{name: "aggregate missing frontend and trace storage", deployment: retuneDGDForAdmission(func(dgd *v1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[1].Retune.TraceStorage = nil
				dgd.Spec.Components = dgd.Spec.Components[1:]
			}), wantWebhookErrs: []string{
				field.Forbidden(field.NewPath("spec", "components").Index(0).Child("retune"), "enabled Retune requires exactly one frontend component selected by type").Error(),
				field.Required(field.NewPath("spec", "components").Index(0).Child("retune", "traceStorage"), "must reference the trace PVC when Retune is enabled").Error(),
			}},
			{name: "ambiguous frontends", deployment: retuneDGDForAdmission(func(dgd *v1beta1.DynamoGraphDeployment) {
				frontend := dgd.Spec.Components[0].DeepCopy()
				frontend.ComponentName = "z-gateway"
				dgd.Spec.Components = append(dgd.Spec.Components, *frontend)
			}), wantWebhookErrs: []string{field.Forbidden(retunePath, "enabled Retune requires exactly one frontend component selected by type").Error()}},
			{name: "multiple Planner writers", deployment: retuneDGDForAdmission(func(dgd *v1beta1.DynamoGraphDeployment) {
				planner := dgd.Spec.Components[1].DeepCopy()
				planner.ComponentName = "z-planner"
				dgd.Spec.Components = append(dgd.Spec.Components, *planner)
			}), wantWebhookErrs: []string{
				field.Forbidden(retunePath, "only one Planner component may enable Retune in a DGD").Error(),
				field.Forbidden(field.NewPath("spec", "components").Index(3).Child("retune"), "only one Planner component may enable Retune in a DGD").Error(),
			}},
			{name: "enable on update", oldDeployment: retuneDGDForAdmission(func(dgd *v1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[1].Retune = nil
			}), deployment: retuneDGDForAdmission(nil)},
			{name: "disable on update without removing ordinary scaling", oldDeployment: retuneDGDForAdmission(nil), deployment: retuneDGDForAdmission(func(dgd *v1beta1.DynamoGraphDeployment) {
				dgd.Spec.Components[1].Retune = nil
			})},
		}
		for _, row := range rows {
			row.name = fmt.Sprintf("Retune alpha=%t %s", alpha, row.name)
			if alpha {
				row.wantSchemaErr = strings.ReplaceAll(row.wantSchemaErr, "spec.components[1]", "spec.services.planner")
				converted := &v1alpha1.DynamoGraphDeployment{}
				require.NoError(t, converted.ConvertFrom(row.deployment.(*v1beta1.DynamoGraphDeployment)))
				row.deployment = converted
				if row.oldDeployment != nil {
					old := &v1alpha1.DynamoGraphDeployment{}
					require.NoError(t, old.ConvertFrom(row.oldDeployment.(*v1beta1.DynamoGraphDeployment)))
					row.oldDeployment = old
				}
			}
			tests = append(tests, row)
		}
	}
	return tests
}

func retuneDCDAdmissionCases(t *testing.T) []dcdAdmissionTestCase {
	t.Helper()
	var tests []dcdAdmissionTestCase
	for _, alpha := range []bool{false, true} {
		for _, graphOwned := range []bool{false, true} {
			for _, enabled := range []bool{false, true} {
				object := betaDCDForAdmission(nil)
				object.Spec.ComponentType = v1beta1.ComponentTypePlanner
				object.Spec.Retune = &v1beta1.RetuneSpec{Enabled: enabled,
					TraceStorage: &v1beta1.RetuneTraceStorageSpec{ClaimName: "request-traces"},
				}
				if graphOwned {
					object.OwnerReferences = []metav1.OwnerReference{{APIVersion: v1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment", Name: "graph", UID: "graph-uid", Controller: new(true)}}
				}
				row := dcdAdmissionTestCase{name: fmt.Sprintf("Retune alpha=%t graph=%t enabled=%t", alpha, graphOwned, enabled), deployment: object}
				if enabled && !graphOwned {
					row.wantWebhookErrs = []string{field.Forbidden(field.NewPath("spec", "retune"), "enabled Retune is supported only on DynamoGraphDeployment-managed components").Error()}
				}
				if alpha {
					converted := &v1alpha1.DynamoComponentDeployment{}
					require.NoError(t, converted.ConvertFrom(object))
					row.deployment = converted
				}
				tests = append(tests, row)
			}
		}
	}
	return tests
}
