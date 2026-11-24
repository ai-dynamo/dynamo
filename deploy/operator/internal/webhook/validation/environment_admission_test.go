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
	name                string
	env                 []corev1.EnvVar
	oldEnv              []corev1.EnvVar
	update              bool
	duplicates          []int
	unchangedDuplicates []int
	conflictingSources  []int
}

// environmentAdmissionCases shares the root-list contract, not resource fixtures.
func environmentAdmissionCases() []environmentAdmissionCase {
	duplicate := []corev1.EnvVar{{Name: "A", Value: "first"}, {Name: "A", Value: "do-not-leak-this-value"}}
	unique := []corev1.EnvVar{{Name: "A", Value: "first"}}
	conflict := []corev1.EnvVar{{Name: "PORT", Value: "do-not-leak-this-value", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Name: "settings", Key: "port"}}}}
	return []environmentAdmissionCase{
		{name: "cross-layer override", env: unique},
		{name: "case-sensitive names", env: []corev1.EnvVar{{Name: "A", Value: "first"}, {Name: "a", Value: "second"}}},
		{name: "reject duplicate create", env: duplicate, duplicates: []int{1}},
		{name: "aggregate duplicate names", env: []corev1.EnvVar{{Name: "A"}, {Name: "A"}, {Name: "B"}, {Name: "B"}}, duplicates: []int{1, 3}},
		{name: "reject order-dependent repeated name", env: []corev1.EnvVar{{Name: "X", Value: "first"}, {Name: "Y", Value: "$(X)"}, {Name: "X", Value: "last"}}, duplicates: []int{2}},
		{name: "unrelated update with unchanged duplicate list", oldEnv: duplicate, env: duplicate, update: true, unchangedDuplicates: []int{1}},
		{name: "reject duplicate introduction", oldEnv: unique, env: duplicate, update: true, duplicates: []int{1}},
		{name: "changed value revalidates whole list", oldEnv: duplicate, env: []corev1.EnvVar{{Name: "A", Value: "changed"}, duplicate[1]}, update: true, duplicates: []int{1}},
		{name: "changed source revalidates whole list", oldEnv: duplicate, env: []corev1.EnvVar{duplicate[0], {Name: "A", ValueFrom: &corev1.EnvVarSource{SecretKeyRef: &corev1.SecretKeySelector{Name: "private", Key: "token"}}}}, update: true, duplicates: []int{1}},
		{name: "changed order revalidates whole list", oldEnv: duplicate, env: []corev1.EnvVar{duplicate[1], duplicate[0]}, update: true, duplicates: []int{1}},
		{name: "new member revalidates whole list", oldEnv: duplicate, env: []corev1.EnvVar{duplicate[0], duplicate[1], {Name: "B"}}, update: true, duplicates: []int{1}},
		{name: "repair duplicate list", oldEnv: duplicate, env: unique, update: true},
		{name: "remove duplicate list", oldEnv: duplicate, update: true},
		{name: "reject value plus source create", env: conflict, conflictingSources: []int{0}},
		{name: "ratchet unchanged value plus source", env: conflict, oldEnv: conflict, update: true},
		{name: "reject value plus source introduction", env: conflict, oldEnv: unique, update: true, conflictingSources: []int{0}},
		{name: "repair value plus source", env: unique, oldEnv: conflict, update: true},
		{name: "empty value with source is valid", env: []corev1.EnvVar{{Name: "PORT", ValueFrom: &corev1.EnvVarSource{ConfigMapKeyRef: &corev1.ConfigMapKeySelector{Name: "settings", Key: "port"}}}}},
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
			for _, index := range envCase.conflictingSources {
				entry.wantWebhookErrs = append(entry.wantWebhookErrs, field.Forbidden(envPath.Index(index).Child("valueFrom"), "may not be specified when value is non-empty").Error())
			}
			tests = append(tests, entry)
		}
	}

	// Alpha service envs has no source-schema map-list validation before conversion.
	for _, envCase := range environmentAdmissionCases() {
		updated := alphaDGDForAdmission(nil)
		updated.Spec.Envs = []corev1.EnvVar{{Name: "GENERAL", Value: "default"}}
		updated.Spec.Services["worker"].Envs = envCase.env
		old := updated.DeepCopy()
		old.Spec.Services["worker"].Envs = envCase.oldEnv
		updated.Labels = map[string]string{"example.com/unrelated": "edit"}
		entry := dgdAdmissionTestCase{name: "alpha service envs " + envCase.name, deployment: updated, seedWithoutWebhook: envCase.update}
		if envCase.update {
			entry.oldDeployment = old
		}
		envPath := field.NewPath("spec", "services").Key("worker").Child("envs")
		for _, index := range envCase.duplicates {
			entry.wantWebhookErrs = append(entry.wantWebhookErrs, field.Duplicate(envPath.Index(index).Child("name"), envCase.env[index].Name).Error())
		}
		for _, index := range envCase.conflictingSources {
			entry.wantWebhookErrs = append(entry.wantWebhookErrs, field.Forbidden(envPath.Index(index).Child("valueFrom"), "may not be specified when value is non-empty").Error())
		}
		tests = append(tests, entry)
	}

	// Semantic-origin DGDs cannot retain a list that a replacement child must reject.
	for _, origin := range []struct {
		value  string
		reject bool
	}{
		{value: ""},
		{value: "1.7.0"},
		{value: "1.8.0", reject: true},
		{value: "1.8.1", reject: true},
	} {
		for _, alpha := range []bool{false, true} {
			for _, repair := range []bool{false, true} {
				old := betaDGDForAdmission(nil)
				old.Annotations = map[string]string{}
				if origin.value != "" {
					old.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = origin.value
				}
				old.Spec.Env = []corev1.EnvVar{{Name: "A", Value: "first"}, {Name: "A", Value: "second"}}
				old.Spec.Components[1].PodTemplate.Spec.Containers[0].Env = []corev1.EnvVar{{Name: "A", Value: "specific"}}
				seed := old.DeepCopy()
				seed.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.6.0"}
				updated := old.DeepCopy()
				updated.Labels = map[string]string{"example.com/unrelated": "edit"}
				if repair {
					updated.Spec.Env = updated.Spec.Env[:1]
				}
				entry := dgdAdmissionTestCase{name: fmt.Sprintf("root env retained duplicate origin=%q alpha=%t repair=%t", origin.value, alpha, repair), deployment: updated, oldDeployment: old, oldBeforeUpdate: seed, seedWithoutWebhook: true}
				envPath := field.NewPath("spec", "env")
				if alpha {
					converted, previous, before := &nvidiacomv1alpha1.DynamoGraphDeployment{}, &nvidiacomv1alpha1.DynamoGraphDeployment{}, &nvidiacomv1alpha1.DynamoGraphDeployment{}
					require.NoError(t, converted.ConvertFrom(updated))
					require.NoError(t, previous.ConvertFrom(old))
					require.NoError(t, before.ConvertFrom(seed))
					entry.deployment, entry.oldDeployment, entry.oldBeforeUpdate = converted, previous, before
					envPath = field.NewPath("spec", "envs")
				}
				if origin.reject && !repair {
					entry.wantWebhookErrs = []string{field.Duplicate(envPath.Index(1).Child("name"), "A").Error()}
				}
				tests = append(tests, entry)
			}
		}

		old := alphaDGDForAdmission(nil)
		old.Annotations = map[string]string{}
		if origin.value != "" {
			old.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = origin.value
		}
		old.Spec.Services["worker"].Envs = []corev1.EnvVar{{Name: "A", Value: "first"}, {Name: "A", Value: "second"}}
		seed := old.DeepCopy()
		seed.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.6.0"}
		updated := old.DeepCopy()
		updated.Labels = map[string]string{"example.com/unrelated": "edit"}
		entry := dgdAdmissionTestCase{name: fmt.Sprintf("alpha service envs retained duplicate origin=%q", origin.value), deployment: updated, oldDeployment: old, oldBeforeUpdate: seed, seedWithoutWebhook: true}
		if origin.reject {
			entry.wantWebhookErrs = []string{`spec.services[worker].envs[1].name: Duplicate value: "A"`}
		}
		tests = append(tests, entry)
	}
	return tests
}

func environmentDCDAdmissionCases(t *testing.T) []dcdAdmissionTestCase {
	t.Helper()
	var tests []dcdAdmissionTestCase
	for _, alpha := range []bool{false, true} {
		// CREATE assigns standalone origins using the same ownership rule as rendering.
		for _, owner := range []struct {
			name, apiVersion, kind string
			controller, graphChild bool
		}{
			{name: "standalone"},
			{name: "beta graph controller", apiVersion: nvidiacomv1beta1.GroupVersion.String(), kind: "DynamoGraphDeployment", controller: true, graphChild: true},
			{name: "alpha graph controller", apiVersion: nvidiacomv1alpha1.GroupVersion.String(), kind: "DynamoGraphDeployment", controller: true, graphChild: true},
			{name: "foreign graph controller", apiVersion: "example.com/v1", kind: "DynamoGraphDeployment", controller: true},
			{name: "non-controlling graph owner", apiVersion: nvidiacomv1beta1.GroupVersion.String(), kind: "DynamoGraphDeployment"},
			{name: "unrelated controller", apiVersion: nvidiacomv1beta1.GroupVersion.String(), kind: "OtherWorkload", controller: true},
		} {
			created := betaDCDForAdmission(nil)
			created.Spec.PodTemplate.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.7.0"}
			if owner.kind != "" {
				created.OwnerReferences = []metav1.OwnerReference{{APIVersion: owner.apiVersion, Kind: owner.kind, Name: "graph", UID: "graph-uid", Controller: new(owner.controller)}}
			}
			entry := dcdAdmissionTestCase{
				name: fmt.Sprintf("root env CREATE ownership alpha=%t %s", alpha, owner.name), deployment: created,
				wantOriginVersion: new("1.6.0"), wantPodAnnotations: created.Spec.PodTemplate.Annotations,
			}
			if owner.graphChild {
				entry.wantOriginVersion = new("")
			}
			if alpha {
				converted := &nvidiacomv1alpha1.DynamoComponentDeployment{}
				require.NoError(t, converted.ConvertFrom(created))
				entry.deployment = converted
			}
			tests = append(tests, entry)
		}

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
			for _, index := range append(envCase.duplicates, envCase.unchangedDuplicates...) {
				entry.wantWebhookErrs = append(entry.wantWebhookErrs, field.Duplicate(field.NewPath("spec", "env").Index(index).Child("name"), envCase.env[index].Name).Error())
			}
			for _, index := range envCase.conflictingSources {
				entry.wantWebhookErrs = append(entry.wantWebhookErrs, field.Forbidden(field.NewPath("spec", "env").Index(index).Child("valueFrom"), "may not be specified when value is non-empty").Error())
			}
			tests = append(tests, entry)
		}
	}

	for _, alpha := range []bool{false, true} {
		for _, origin := range []struct {
			name, object, template     string
			child, mixedRoles          bool
			nonControlling, foreign    bool
			activate, retained, reject bool
		}{
			{name: "standalone current ignores legacy template", object: "1.8.0", template: "1.7.0", reject: true},
			{name: "standalone current rejects retained duplicates", object: "1.8.0", template: "1.7.0", retained: true, reject: true},
			{name: "standalone legacy retains duplicates", object: "1.7.0", template: "1.8.0", retained: true},
			{name: "standalone legacy ignores semantic template", object: "1.7.0", template: "1.8.0"},
			{name: "standalone missing ignores semantic template", template: "1.8.0"},
			{name: "standalone invalid ignores semantic template", object: "custom-build", template: "1.8.0"},
			{name: "standalone prerelease remains legacy", object: "1.8.0-rc.1", template: "1.8.0"},
			{name: "child legacy ignores semantic object", object: "1.8.0", template: "1.7.0", child: true},
			{name: "child missing ignores semantic object", object: "1.8.0", child: true},
			{name: "child current ignores legacy object", object: "1.7.0", template: "1.8.0", child: true, reject: true},
			{name: "child current rejects retained duplicates", object: "1.7.0", template: "1.8.0", child: true, retained: true, reject: true},
			{name: "child legacy retains duplicates", object: "1.8.0", template: "1.7.0", child: true, retained: true},
			{name: "child mixed roles select semantic contract", object: "1.7.0", template: "1.7.0", child: true, mixedRoles: true, reject: true},
			{name: "child activation revalidates unchanged duplicate list", object: "1.7.0", template: "1.8.0", child: true, activate: true, reject: true},
			{name: "non-controlling DGD does not replace standalone origin", object: "1.8.0", template: "1.7.0", child: true, nonControlling: true, reject: true},
			{name: "foreign controlling DGD does not replace standalone origin", object: "1.8.0", template: "1.7.0", child: true, foreign: true, reject: true},
			{name: "non-controlling DGD does not enable template origin", object: "1.7.0", template: "1.8.0", child: true, nonControlling: true},
			{name: "foreign controlling DGD does not enable template origin", object: "1.7.0", template: "1.8.0", child: true, foreign: true},
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
				if origin.nonControlling {
					updated.OwnerReferences[0].Controller = new(false)
				}
				if origin.foreign {
					updated.OwnerReferences[0].APIVersion = "example.com/v1"
				}
			}
			if origin.mixedRoles {
				setBetaExplicitMultinodeRoleTemplates(&updated.Spec.DynamoComponentDeploymentSharedSpec, 2)
				updated.Spec.Roles[0].PodTemplate.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.7.0"}
				updated.Spec.Roles[1].PodTemplate.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.8.0"}
			}
			updated.Spec.Env = []corev1.EnvVar{{Name: "A", Value: "first"}, {Name: "A", Value: "second"}}
			old := updated.DeepCopy()
			old.Spec.Env = old.Spec.Env[:1]
			if origin.retained {
				old.Spec.Env = updated.Spec.Env
			}
			if origin.activate {
				old.Spec.Env = updated.Spec.Env
				old.Spec.PodTemplate.Annotations[consts.KubeAnnotationDynamoOperatorOriginVersion] = "1.7.0"
			}
			entry := dcdAdmissionTestCase{name: fmt.Sprintf("root env origin alpha=%t %s", alpha, origin.name), deployment: updated, oldDeployment: old, seedWithoutWebhook: true}
			seed := old.DeepCopy()
			seed.Annotations = map[string]string{consts.KubeAnnotationDynamoOperatorOriginVersion: "1.6.0"}
			standalone := !origin.child || origin.nonControlling || origin.foreign
			if standalone {
				entry.oldBeforeUpdate = seed
			}
			if alpha {
				converted, previous, before := &nvidiacomv1alpha1.DynamoComponentDeployment{}, &nvidiacomv1alpha1.DynamoComponentDeployment{}, &nvidiacomv1alpha1.DynamoComponentDeployment{}
				require.NoError(t, converted.ConvertFrom(updated))
				require.NoError(t, previous.ConvertFrom(old))
				entry.deployment, entry.oldDeployment = converted, previous
				if standalone {
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
