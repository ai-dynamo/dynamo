/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package dgdoverride

import (
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"sigs.k8s.io/yaml"
)

func TestApplyVersionMatrix(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name             string
		blueprint        string
		override         string
		wantAPIVersion   string
		wantArgs         []string
		wantPreservation func(*testing.T, *unstructured.Unstructured)
	}{
		{
			name:           "alpha blueprint with alpha override",
			blueprint:      alphaBlueprintYAML,
			override:       alphaOverrideYAML,
			wantAPIVersion: "nvidia.com/v1alpha1",
			wantArgs:       []string{"--base", "--override"},
			wantPreservation: func(t *testing.T, result *unstructured.Unstructured) {
				pvcs := mustNestedSlice(t, result.Object, "spec", "pvcs")
				require.Len(t, pvcs, 1)
				assert.Equal(t, "model-cache", pvcs[0].(map[string]interface{})["name"])
			},
		},
		{
			name:           "alpha blueprint with beta override",
			blueprint:      alphaBlueprintYAML,
			override:       betaOverrideYAML,
			wantAPIVersion: "nvidia.com/v1alpha1",
			wantArgs:       []string{"--override"},
			wantPreservation: func(t *testing.T, result *unstructured.Unstructured) {
				pvcs := mustNestedSlice(t, result.Object, "spec", "pvcs")
				require.Len(t, pvcs, 1)
				assert.Equal(t, "model-cache", pvcs[0].(map[string]interface{})["name"])
			},
		},
		{
			name:           "beta blueprint with alpha override",
			blueprint:      betaBlueprintYAML,
			override:       alphaOverrideYAML,
			wantAPIVersion: "nvidia.com/v1beta1",
			wantArgs:       []string{"--base", "--override"},
			wantPreservation: func(t *testing.T, result *unstructured.Unstructured) {
				worker := mustBetaWorker(t, result)
				assert.Equal(t, "sidecar", worker["frontendSidecar"])
				containers := mustNestedSlice(t, worker, "podTemplate", "spec", "containers")
				assert.NotNil(t, findNamedObject(t, containers, "sidecar"))
			},
		},
		{
			name:           "beta blueprint with beta override",
			blueprint:      betaBlueprintYAML,
			override:       betaOverrideYAML,
			wantAPIVersion: "nvidia.com/v1beta1",
			wantArgs:       []string{"--override"},
			wantPreservation: func(t *testing.T, result *unstructured.Unstructured) {
				worker := mustBetaWorker(t, result)
				assert.Equal(t, "sidecar", worker["frontendSidecar"])
				containers := mustNestedSlice(t, worker, "podTemplate", "spec", "containers")
				assert.NotNil(t, findNamedObject(t, containers, "sidecar"))
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			blueprint := mustObject(t, test.blueprint)
			override := mustObject(t, test.override)
			blueprintBefore := blueprint.DeepCopy()
			overrideBefore := override.DeepCopy()

			result, warnings, err := Apply(blueprint, override)
			require.NoError(t, err)
			assert.Empty(t, warnings)
			assert.Equal(t, test.wantAPIVersion, result.GetAPIVersion())
			assert.Equal(t, dgdKind, result.GetKind())
			assert.Equal(t, "generated", result.GetName())
			assert.Equal(t, "new-image", mainContainerImage(t, result))
			assert.Equal(t, test.wantArgs, mainContainerArgs(t, result))
			assert.Equal(t, map[string]string{
				"ADDED":  "added",
				"CHANGE": "new",
				"KEEP":   "keep",
			}, mainContainerEnv(t, result))
			test.wantPreservation(t, result)
			assert.Equal(t, blueprintBefore, blueprint, "Apply mutated the blueprint")
			assert.Equal(t, overrideBefore, override, "Apply mutated the override")
		})
	}
}

func TestApplyUsesStructuralListSemantics(t *testing.T) {
	t.Parallel()

	result, warnings, err := Apply(
		mustObject(t, betaBlueprintYAML),
		mustObject(t, betaOverrideYAML),
	)
	require.NoError(t, err)
	assert.Empty(t, warnings)

	components := mustNestedSlice(t, result.Object, "spec", "components")
	require.Len(t, components, 2, "component list should merge by name")
	assert.Equal(t, "Frontend", components[0].(map[string]interface{})["name"])
	assert.Equal(t, "Worker", components[1].(map[string]interface{})["name"])

	worker := mustBetaWorker(t, result)
	containers := mustNestedSlice(t, worker, "podTemplate", "spec", "containers")
	require.Len(t, containers, 2, "container list should merge by name")
	assert.NotNil(t, findNamedObject(t, containers, "sidecar"))
	main := findNamedObject(t, containers, "main")
	require.NotNil(t, main)
	assert.Equal(t, "new-image", main["image"])
	assert.Equal(t, []interface{}{"--override"}, main["args"], "atomic args list should be replaced")

	env := mustNestedSlice(t, main, "env")
	require.Len(t, env, 3, "environment variables should merge by name")
	assert.Equal(t, "keep", findNamedObject(t, env, "KEEP")["value"])
	assert.Equal(t, "new", findNamedObject(t, env, "CHANGE")["value"])
	assert.Equal(t, "added", findNamedObject(t, env, "ADDED")["value"])
}

func TestApplyUsesStructuralSetSemantics(t *testing.T) {
	t.Parallel()

	blueprint := mustObject(t, betaBlueprintYAML)
	updateBetaComponent(t, blueprint, "Worker", func(worker map[string]interface{}) {
		require.NoError(t, unstructured.SetNestedStringSlice(
			worker,
			[]string{"sidecar", "shared"},
			"experimental",
			"gpuMemoryService",
			"extraClientContainers",
		))
	})

	override := mustObject(t, betaOverrideYAML)
	updateBetaComponent(t, override, "Worker", func(worker map[string]interface{}) {
		require.NoError(t, unstructured.SetNestedStringSlice(
			worker,
			[]string{"metrics", "shared"},
			"experimental",
			"gpuMemoryService",
			"extraClientContainers",
		))
	})

	result, warnings, err := Apply(blueprint, override)
	require.NoError(t, err)
	assert.Empty(t, warnings)

	worker := mustBetaWorker(t, result)
	clients := mustNestedStringSlice(
		t,
		worker,
		"experimental",
		"gpuMemoryService",
		"extraClientContainers",
	)
	assert.ElementsMatch(t, []string{"sidecar", "shared", "metrics"}, clients)
}

func TestApplyAllowsNullInPreservedUnknownFields(t *testing.T) {
	t.Parallel()

	result, warnings, err := Apply(mustObject(t, betaBlueprintYAML), mustObject(t, `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
spec:
  components:
  - name: Worker
    eppConfig:
      config:
        customPluginConfig:
          optionalValue: null
`))
	require.NoError(t, err)
	assert.Empty(t, warnings)

	worker := mustBetaWorker(t, result)
	value, found, err := unstructured.NestedFieldNoCopy(
		worker,
		"eppConfig",
		"config",
		"customPluginConfig",
		"optionalValue",
	)
	require.NoError(t, err)
	require.True(t, found)
	assert.Nil(t, value)
}

func TestApplySanitizesMetadataAndUnknownTopology(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name                string
		blueprint           string
		override            string
		wantTopologyWarning string
	}{
		{
			name:                "alpha",
			blueprint:           alphaBlueprintYAML,
			wantTopologyWarning: "spec.services.Missing: ignored because the generated blueprint has no such service",
			override: `
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: user-name
  namespace: other
  finalizers: [do-not-copy]
  labels:
    added: "true"
    base: "false"
  annotations:
    added: "true"
    base: "false"
spec:
  services:
    Missing:
      extraPodSpec:
        mainContainer:
          image: ignored
`,
		},
		{
			name:                "beta",
			blueprint:           betaBlueprintYAML,
			wantTopologyWarning: "spec.components[name=Missing]: ignored because the generated blueprint has no such component",
			override: `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: user-name
  namespace: other
  finalizers: [do-not-copy]
  labels:
    added: "true"
    base: "false"
  annotations:
    added: "true"
    base: "false"
spec:
  components:
  - name: Missing
    podTemplate:
      spec:
        containers:
        - name: main
          image: ignored
`,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			override := mustObject(t, test.override)
			overrideBefore := override.DeepCopy()
			result, warnings, err := Apply(
				mustObject(t, test.blueprint),
				override,
			)
			require.NoError(t, err)
			require.Len(t, warnings, 2)
			assert.Equal(t, "metadata: ignored identity/runtime fields: finalizers, name, namespace", warnings[0].String())
			assert.Equal(t, test.wantTopologyWarning, warnings[1].String())

			assert.Equal(t, "generated", result.GetName())
			assert.Equal(t, "default", result.GetNamespace())
			assert.Empty(t, result.GetFinalizers())
			assert.Equal(t, map[string]string{"added": "true", "base": "false"}, result.GetLabels())
			assert.Equal(t, "false", result.GetAnnotations()["base"])
			assert.Equal(t, "true", result.GetAnnotations()["added"])
			assert.Equal(t, "old-image", mainContainerImage(t, result))
			assert.Equal(t, overrideBefore, override, "Apply mutated the override")
		})
	}
}

func TestApplyProtectsConversionAnnotations(t *testing.T) {
	t.Parallel()

	override := mustObject(t, alphaOverrideYAML)
	require.NoError(t, unstructured.SetNestedStringMap(
		override.Object,
		map[string]string{
			"nvidia.com/dgd-future": "malicious-future-value",
			"nvidia.com/dgd-spec":   "malicious-preservation-value",
			"user.example/setting":  "allowed",
		},
		"metadata",
		"annotations",
	))

	result, warnings, err := Apply(mustObject(t, betaBlueprintYAML), override)
	require.NoError(t, err)
	require.Len(t, warnings, 1)
	assert.Equal(
		t,
		"metadata.annotations: ignored reserved operator keys: nvidia.com/dgd-future, nvidia.com/dgd-spec",
		warnings[0].String(),
	)

	worker := mustBetaWorker(t, result)
	assert.Equal(t, "sidecar", worker["frontendSidecar"], "beta-only data was corrupted during round trip")
	assert.Equal(t, "allowed", result.GetAnnotations()["user.example/setting"])
	assert.NotEqual(t, "malicious-preservation-value", result.GetAnnotations()["nvidia.com/dgd-spec"])
	assert.NotContains(t, result.GetAnnotations(), "nvidia.com/dgd-future")
}

func TestApplyRejectsInvalidInput(t *testing.T) {
	t.Parallel()

	validBlueprint := mustObject(t, betaBlueprintYAML)
	validOverride := mustObject(t, betaOverrideYAML)

	tests := []struct {
		name      string
		blueprint func(*testing.T) *unstructured.Unstructured
		override  func(*testing.T) *unstructured.Unstructured
		wantError string
	}{
		{
			name:      "nil blueprint",
			blueprint: func(*testing.T) *unstructured.Unstructured { return nil },
			override:  func(*testing.T) *unstructured.Unstructured { return validOverride.DeepCopy() },
			wantError: "blueprint must not be nil",
		},
		{
			name:      "nil override",
			blueprint: func(*testing.T) *unstructured.Unstructured { return validBlueprint.DeepCopy() },
			override:  func(*testing.T) *unstructured.Unstructured { return nil },
			wantError: "override must not be nil",
		},
		{
			name:      "missing override version",
			blueprint: func(*testing.T) *unstructured.Unstructured { return validBlueprint.DeepCopy() },
			override: func(t *testing.T) *unstructured.Unstructured {
				return mustObject(t, `kind: DynamoGraphDeployment`)
			},
			wantError: `got apiVersion "" kind "DynamoGraphDeployment"`,
		},
		{
			name:      "unsupported override version",
			blueprint: func(*testing.T) *unstructured.Unstructured { return validBlueprint.DeepCopy() },
			override: func(t *testing.T) *unstructured.Unstructured {
				return mustObject(t, "apiVersion: nvidia.com/v2\nkind: DynamoGraphDeployment")
			},
			wantError: `got apiVersion "nvidia.com/v2"`,
		},
		{
			name:      "wrong kind",
			blueprint: func(*testing.T) *unstructured.Unstructured { return validBlueprint.DeepCopy() },
			override: func(t *testing.T) *unstructured.Unstructured {
				return mustObject(t, "apiVersion: nvidia.com/v1beta1\nkind: ConfigMap")
			},
			wantError: `kind "ConfigMap"`,
		},
		{
			name:      "explicit null",
			blueprint: func(*testing.T) *unstructured.Unstructured { return validBlueprint.DeepCopy() },
			override: func(t *testing.T) *unstructured.Unstructured {
				return mustObject(t, `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
spec:
  backendFramework: null
`)
			},
			wantError: "override spec.backendFramework must not be null",
		},
		{
			name:      "explicit null in typed field below preserve unknown object",
			blueprint: func(*testing.T) *unstructured.Unstructured { return validBlueprint.DeepCopy() },
			override: func(t *testing.T) *unstructured.Unstructured {
				return mustObject(t, `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
spec:
  components:
  - name: Worker
    eppConfig:
      config:
        apiVersion: null
`)
			},
			wantError: "override spec.components[0].eppConfig.config.apiVersion must not be null",
		},
		{
			name:      "status override",
			blueprint: func(*testing.T) *unstructured.Unstructured { return validBlueprint.DeepCopy() },
			override: func(t *testing.T) *unstructured.Unstructured {
				return mustObject(t, `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
status:
  state: successful
`)
			},
			wantError: "override status is not supported",
		},
		{
			name:      "beta component missing merge key",
			blueprint: func(*testing.T) *unstructured.Unstructured { return validBlueprint.DeepCopy() },
			override: func(t *testing.T) *unstructured.Unstructured {
				return mustObject(t, `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
spec:
  components:
  - replicas: 2
`)
			},
			wantError: "spec.components[0].name must be a non-empty string",
		},
		{
			name:      "duplicate beta component merge key",
			blueprint: func(*testing.T) *unstructured.Unstructured { return validBlueprint.DeepCopy() },
			override: func(t *testing.T) *unstructured.Unstructured {
				return mustObject(t, `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
spec:
  components:
  - name: Worker
    replicas: 2
  - name: Worker
    replicas: 3
`)
			},
			wantError: "duplicate",
		},
		{
			name:      "invalid alpha worker args",
			blueprint: func(t *testing.T) *unstructured.Unstructured { return mustObject(t, alphaBlueprintYAML) },
			override: func(t *testing.T) *unstructured.Unstructured {
				return mustObject(t, `
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
spec:
  services:
    Worker:
      extraPodSpec:
        mainContainer:
          args: [valid, 7]
`)
			},
			wantError: "must be a list of strings",
		},
		{
			name:      "unknown beta field",
			blueprint: func(*testing.T) *unstructured.Unstructured { return validBlueprint.DeepCopy() },
			override: func(t *testing.T) *unstructured.Unstructured {
				return mustObject(t, `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
spec:
  notARealField: true
`)
			},
			wantError: "notARealField",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			t.Parallel()

			_, _, err := Apply(test.blueprint(t), test.override(t))
			require.Error(t, err)
			assert.Contains(t, err.Error(), test.wantError)
		})
	}
}

func TestApplyEmptyOverrideIsNoOp(t *testing.T) {
	t.Parallel()

	blueprint := mustObject(t, betaBlueprintYAML)
	result, warnings, err := Apply(blueprint, mustObject(t, `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
`))
	require.NoError(t, err)
	assert.Empty(t, warnings)
	assert.Equal(t, blueprint, result)
}

func mustObject(t *testing.T, document string) *unstructured.Unstructured {
	t.Helper()
	content := map[string]interface{}{}
	require.NoError(t, yaml.Unmarshal([]byte(document), &content))
	return &unstructured.Unstructured{Object: content}
}

func mustNestedSlice(t *testing.T, object map[string]interface{}, fields ...string) []interface{} {
	t.Helper()
	value, found, err := unstructured.NestedSlice(object, fields...)
	require.NoError(t, err)
	require.True(t, found, "missing field %s", strings.Join(fields, "."))
	return value
}

func mustNestedStringSlice(t *testing.T, object map[string]interface{}, fields ...string) []string {
	t.Helper()
	value, found, err := unstructured.NestedStringSlice(object, fields...)
	require.NoError(t, err)
	require.True(t, found, "missing field %s", strings.Join(fields, "."))
	return value
}

func mustBetaWorker(t *testing.T, object *unstructured.Unstructured) map[string]interface{} {
	t.Helper()
	components := mustNestedSlice(t, object.Object, "spec", "components")
	component := findNamedObject(t, components, "Worker")
	require.NotNil(t, component, "missing Worker component")
	return component
}

func updateBetaComponent(
	t *testing.T,
	object *unstructured.Unstructured,
	name string,
	update func(map[string]interface{}),
) {
	t.Helper()
	components := mustNestedSlice(t, object.Object, "spec", "components")
	component := findNamedObject(t, components, name)
	require.NotNil(t, component, "missing component %q", name)
	update(component)
	require.NoError(t, unstructured.SetNestedSlice(object.Object, components, "spec", "components"))
}

func findNamedObject(t *testing.T, values []interface{}, name string) map[string]interface{} {
	t.Helper()
	for i, value := range values {
		object, ok := value.(map[string]interface{})
		require.True(t, ok, "item %d is %T, not an object", i, value)
		if object["name"] == name {
			return object
		}
	}
	return nil
}

func mainContainer(t *testing.T, object *unstructured.Unstructured) map[string]interface{} {
	t.Helper()
	switch object.GetAPIVersion() {
	case "nvidia.com/v1alpha1":
		container, found, err := unstructured.NestedMap(
			object.Object,
			"spec",
			"services",
			"Worker",
			"extraPodSpec",
			"mainContainer",
		)
		require.NoError(t, err)
		require.True(t, found)
		return container
	case "nvidia.com/v1beta1":
		worker := mustBetaWorker(t, object)
		containers := mustNestedSlice(t, worker, "podTemplate", "spec", "containers")
		container := findNamedObject(t, containers, "main")
		require.NotNil(t, container)
		return container
	default:
		t.Fatalf("unexpected apiVersion %q", object.GetAPIVersion())
		return nil
	}
}

func mainContainerImage(t *testing.T, object *unstructured.Unstructured) string {
	t.Helper()
	image, ok := mainContainer(t, object)["image"].(string)
	require.True(t, ok)
	return image
}

func mainContainerArgs(t *testing.T, object *unstructured.Unstructured) []string {
	t.Helper()
	args := mainContainer(t, object)["args"]
	values, ok := args.([]interface{})
	require.True(t, ok, "args has type %T", args)
	result := make([]string, len(values))
	for i, value := range values {
		result[i], ok = value.(string)
		require.True(t, ok, "args[%d] has type %T", i, value)
	}
	return result
}

func mainContainerEnv(t *testing.T, object *unstructured.Unstructured) map[string]string {
	t.Helper()
	result := map[string]string{}
	if object.GetAPIVersion() == "nvidia.com/v1alpha1" {
		service, found, err := unstructured.NestedMap(object.Object, "spec", "services", "Worker")
		require.NoError(t, err)
		require.True(t, found)
		env, found, err := unstructured.NestedSlice(service, "envs")
		require.NoError(t, err)
		if found {
			addEnvValues(t, result, env)
		}
	}
	env, found, err := unstructured.NestedSlice(mainContainer(t, object), "env")
	require.NoError(t, err)
	if found {
		addEnvValues(t, result, env)
	}
	return result
}

func addEnvValues(t *testing.T, result map[string]string, env []interface{}) {
	t.Helper()
	for _, item := range env {
		entry := item.(map[string]interface{})
		result[entry["name"].(string)] = entry["value"].(string)
	}
}

const alphaBlueprintYAML = `
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
metadata:
  name: generated
  namespace: default
  labels:
    base: "true"
  annotations:
    base: "true"
spec:
  backendFramework: vllm
  pvcs:
  - name: model-cache
  services:
    Frontend:
      componentType: frontend
    Worker:
      componentType: worker
      extraPodSpec:
        mainContainer:
          name: main
          image: old-image
          args: [--base]
          env:
          - name: KEEP
            value: keep
          - name: CHANGE
            value: old
`

const alphaOverrideYAML = `
apiVersion: nvidia.com/v1alpha1
kind: DynamoGraphDeployment
spec:
  services:
    Worker:
      extraPodSpec:
        mainContainer:
          image: new-image
          args: [--override]
          env:
          - name: CHANGE
            value: new
          - name: ADDED
            value: added
`

const betaBlueprintYAML = `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
metadata:
  name: generated
  namespace: default
  labels:
    base: "true"
  annotations:
    base: "true"
spec:
  backendFramework: vllm
  components:
  - name: Frontend
    type: frontend
  - name: Worker
    type: worker
    frontendSidecar: sidecar
    podTemplate:
      spec:
        containers:
        - name: main
          image: old-image
          args: [--base]
          env:
          - name: KEEP
            value: keep
          - name: CHANGE
            value: old
        - name: sidecar
          image: sidecar-image
`

const betaOverrideYAML = `
apiVersion: nvidia.com/v1beta1
kind: DynamoGraphDeployment
spec:
  components:
  - name: Worker
    podTemplate:
      spec:
        containers:
        - name: main
          image: new-image
          args: [--override]
          env:
          - name: CHANGE
            value: new
          - name: ADDED
            value: added
`
