// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package validation_test

import (
	"fmt"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/consts"
	"github.com/stretchr/testify/require"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

type environmentAdmissionCase struct {
	name       string
	env        []corev1.EnvVar
	oldEnv     []corev1.EnvVar
	update     bool
	duplicates []int
}

// environmentAdmissionCases shares the root-list contract, not resource fixtures.
func environmentAdmissionCases() []environmentAdmissionCase {
	duplicate := []corev1.EnvVar{{Name: "A", Value: "first"}, {Name: "A", Value: "do-not-leak-this-value"}}
	unique := []corev1.EnvVar{{Name: "A", Value: "first"}}
	return []environmentAdmissionCase{
		{name: "cross-layer override", env: unique},
		{name: "case-sensitive names", env: []corev1.EnvVar{{Name: "A", Value: "first"}, {Name: "a", Value: "second"}}},
		{name: "reject duplicate create", env: duplicate, duplicates: []int{1}},
		{name: "aggregate duplicate names", env: []corev1.EnvVar{{Name: "A"}, {Name: "A"}, {Name: "B"}, {Name: "B"}}, duplicates: []int{1, 3}},
		{name: "unrelated update ratchets unchanged list", oldEnv: duplicate, env: duplicate, update: true},
		{name: "reject duplicate introduction", oldEnv: unique, env: duplicate, update: true, duplicates: []int{1}},
		{name: "changed value revalidates whole list", oldEnv: duplicate, env: []corev1.EnvVar{{Name: "A", Value: "changed"}, duplicate[1]}, update: true, duplicates: []int{1}},
		{name: "changed source revalidates whole list", oldEnv: duplicate, env: []corev1.EnvVar{duplicate[0], {Name: "A", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Name: "private", Key: "token"}}}}, update: true, duplicates: []int{1}},
		{name: "changed order revalidates whole list", oldEnv: duplicate, env: []corev1.EnvVar{duplicate[1], duplicate[0]}, update: true, duplicates: []int{1}},
		{name: "new member revalidates whole list", oldEnv: duplicate, env: []corev1.EnvVar{duplicate[0], duplicate[1], {Name: "B"}}, update: true, duplicates: []int{1}},
		{name: "repair duplicate list", oldEnv: duplicate, env: unique, update: true},
		{name: "remove duplicate list", oldEnv: duplicate, update: true},
	}
}

func environmentDGDAdmissionCases(t *testing.T) []dgdAdmissionTestCase {
	t.Helper()
	var tests []dgdAdmissionTestCase
	for _, alpha := range []bool{false, true} {
		for _, envCase := range environmentAdmissionCases() {
			updated := betaDGDForAdmission(nil)
			updated.Spec.Env = envCase.env
			updated.Spec.Components[1].PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "A", Value: "specific"}}
			old := updated.DeepCopy()
			old.Spec.Env = envCase.oldEnv
			updated.Labels = map[string]string{"example.com/unrelated": "edit"}
			entry := dgdAdmissionTestCase{name: fmt.Sprintf("root env alpha=%t %s", alpha, envCase.name), deployment: updated, seedWithoutWebhook: envCase.update}
			envPath := field.NewPath("spec", "env")
			if envCase.update {
				entry.oldDeployment = old
			}
			if alpha {
				converted := &nvidiacomv1alpha1.DynamoGraphDeployment{}
				require.NoError(t, converted.ConvertFrom(updated))
				entry.deployment = converted
				if envCase.update {
					previous := &nvidiacomv1alpha1.DynamoGraphDeployment{}
					require.NoError(t, previous.ConvertFrom(old))
					entry.oldDeployment = previous
				}
				envPath = field.NewPath("spec", "envs")
			}
			for _, index := range envCase.duplicates {
				entry.wantWebhookErrs = append(entry.wantWebhookErrs, field.Duplicate(envPath.Index(index).Child("name"), envCase.env[index].Name).Error())
			}
			tests = append(tests, entry)
		}
	}
	return tests
}

func environmentDCDAdmissionCases(t *testing.T) []dcdAdmissionTestCase {
	t.Helper()
	var tests []dcdAdmissionTestCase
	for _, alpha := range []bool{false, true} {
		for _, envCase := range environmentAdmissionCases() {
			updated := betaDCDForAdmission(nil)
			updated.OwnerReferences = []metav1.OwnerReference{{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment", Name: "graph", UID: "graph-uid", Controller: new(true)}}
			updated.Spec.PodTemplate.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"}
			updated.Spec.PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "A", Value: "specific"}}
			updated.Spec.Env = envCase.env
			old := updated.DeepCopy()
			old.Spec.Env = envCase.oldEnv
			updated.Labels = map[string]string{"example.com/unrelated": "edit"}
			entry := dcdAdmissionTestCase{name: fmt.Sprintf("root env alpha=%t %s", alpha, envCase.name), deployment: updated, seedWithoutWebhook: envCase.update}
			if envCase.update {
				entry.oldDeployment = old
			}
			if alpha {
				converted := &nvidiacomv1alpha1.DynamoComponentDeployment{}
				require.NoError(t, converted.ConvertFrom(updated))
				entry.deployment = converted
				if envCase.update {
					previous := &nvidiacomv1alpha1.DynamoComponentDeployment{}
					require.NoError(t, previous.ConvertFrom(old))
					entry.oldDeployment = previous
				}
			}
			for _, index := range envCase.duplicates {
				entry.wantWebhookErrs = append(entry.wantWebhookErrs, field.Duplicate(field.NewPath("spec", "env").Index(index).Child("name"), envCase.env[index].Name).Error())
			}
			tests = append(tests, entry)
		}
	}

	for _, alpha := range []bool{false, true} {
		for _, origin := range []struct {
			name, object, template string
			child, mixedRoles      bool
			activate, reject       bool
		}{
			{name: "standalone current ignores legacy template", object: "1.8.0", template: "1.7.0", reject: true},
			{name: "standalone legacy ignores semantic template", object: "1.7.0", template: "1.8.0"},
			{name: "standalone missing ignores semantic template", template: "1.8.0"},
			{name: "standalone invalid ignores semantic template", object: "custom-build", template: "1.8.0"},
			{name: "standalone prerelease remains legacy", object: "1.8.0-rc.1", template: "1.8.0"},
			{name: "child legacy ignores semantic object", object: "1.8.0", template: "1.7.0", child: true},
			{name: "child missing ignores semantic object", object: "1.8.0", child: true},
			{name: "child current ignores legacy object", object: "1.7.0", template: "1.8.0", child: true, reject: true},
			{name: "child mixed roles select semantic contract", object: "1.7.0", template: "1.7.0", child: true, mixedRoles: true, reject: true},
			{name: "child activation revalidates unchanged duplicate list", object: "1.7.0", template: "1.8.0", child: true, activate: true, reject: true},
		} {
			updated := betaDCDForAdmission(nil)
			updated.Annotations = map[string]string{}
			if origin.object != "" {
				updated.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = origin.object
			}
			updated.Spec.PodTemplate.Annotations = map[string]string{}
			if origin.template != "" {
				updated.Spec.PodTemplate.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = origin.template
			}
			if origin.child {
				updated.OwnerReferences = []metav1.OwnerReference{{APIVersion: nvidiacomv1beta1.GroupVersion.String(), Kind: "DynamoGraphDeployment", Name: "graph", UID: "graph-uid", Controller: new(true)}}
			}
			if origin.mixedRoles {
				setBetaExplicitMultinodeRoleTemplates(&updated.Spec.DynamoComponentDeploymentSharedSpec, 2)
				updated.Spec.Roles[0].PodTemplate.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.7.0"}
				updated.Spec.Roles[1].PodTemplate.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"}
			}
			updated.Spec.Env = []corev1.EnvVar{{Name: "A", Value: "first"}, {Name: "A", Value: "second"}}
			old := updated.DeepCopy()
			old.Spec.Env = old.Spec.Env[:1]
			if origin.activate {
				old.Spec.Env = updated.Spec.Env
				old.Spec.PodTemplate.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = "1.7.0"
			}
			entry := dcdAdmissionTestCase{name: fmt.Sprintf("root env origin alpha=%t %s", alpha, origin.name), deployment: updated, oldDeployment: old, seedWithoutWebhook: true}
			seed := old.DeepCopy()
			seed.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.6.0"}
			if !origin.child {
				entry.oldBeforeUpdate = seed
			}
			if alpha {
				converted, previous, before := &nvidiacomv1alpha1.DynamoComponentDeployment{}, &nvidiacomv1alpha1.DynamoComponentDeployment{}, &nvidiacomv1alpha1.DynamoComponentDeployment{}
				require.NoError(t, converted.ConvertFrom(updated))
				require.NoError(t, previous.ConvertFrom(old))
				entry.deployment, entry.oldDeployment = converted, previous
				if !origin.child {
					require.NoError(t, before.ConvertFrom(seed))
					entry.oldBeforeUpdate = before
				}
			}
			if origin.reject {
				entry.wantWebhookErrs = []string{`spec.env[1].name: Duplicate value: "A"`}
			}
			tests = append(tests, entry)
		}
	}
	return tests
}
