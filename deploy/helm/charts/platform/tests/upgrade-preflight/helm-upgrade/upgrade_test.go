/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package upgrade

import (
	"context"
	"fmt"
	"io"
	"path/filepath"
	"reflect"
	"runtime"
	"strings"
	"testing"
	"time"

	"helm.sh/helm/v3/pkg/action"
	"helm.sh/helm/v3/pkg/chart"
	"helm.sh/helm/v3/pkg/chart/loader"
	"helm.sh/helm/v3/pkg/chartutil"
	"helm.sh/helm/v3/pkg/kube"
	"helm.sh/helm/v3/pkg/kube/fake"
	"helm.sh/helm/v3/pkg/release"
	"helm.sh/helm/v3/pkg/storage"
	"helm.sh/helm/v3/pkg/storage/driver"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/yaml"
)

func TestUpgradePreflightDefaultsMatch(t *testing.T) {
	t.Log("Compare the complete parsed parent and operator defaults before Helm coalesces them")
	parent, err := loader.Load(filepath.Dir(filepath.Dir(operatorChartDirectory())))
	if err != nil {
		t.Fatal(err)
	}
	operator, err := loader.Load(operatorChartDirectory())
	if err != nil {
		t.Fatal(err)
	}
	parentDefaults := parent.Values["dynamo-operator"].(map[string]interface{})["upgradePreflight"]
	operatorDefaults, found := operator.Values["upgradePreflight"]
	if !found || !reflect.DeepEqual(parentDefaults, operatorDefaults) {
		t.Fatalf("upgradePreflight defaults differ: platform=%#v operator=%#v", parentDefaults, operatorDefaults)
	}
}

func TestUpgradeReuseValues(t *testing.T) {
	for _, scenario := range []struct {
		name     string
		values   map[string]interface{}
		deadline int64
		override string
	}{
		{name: "no overrides", values: map[string]interface{}{}, deadline: 120},
		{name: "policy only", values: map[string]interface{}{
			"dynamo-operator": map[string]interface{}{"upgradePreflight": map[string]interface{}{
				"checks": map[string]interface{}{"upgradePath": map[string]interface{}{"unsupportedUpgradeOverride": "1.7"}},
			}},
		}, deadline: 120, override: "1.7"},
		{name: "sparse job and policy", values: map[string]interface{}{
			"dynamo-operator": map[string]interface{}{"upgradePreflight": map[string]interface{}{
				"job":    map[string]interface{}{"activeDeadlineSeconds": 300},
				"checks": map[string]interface{}{"upgradePath": map[string]interface{}{"unsupportedUpgradeOverride": "1.7"}},
			}},
		}, deadline: 300, override: "1.7"},
	} {
		t.Run(scenario.name, func(t *testing.T) {
			t.Log("Upgrade a stored 1.6 release using the target chart and ReuseValues=true")
			target := loadUpgradeChart(t)
			upgraded, err := runReuseValuesUpgrade(t, target, scenario.values)
			if err != nil {
				t.Fatal(err)
			}

			t.Log("Check the rendered preflight Job defaults and explicit overrides")
			var jobs []unstructured.Unstructured
			for _, hook := range upgraded.Hooks {
				objects, err := decodeObjects(hook.Manifest)
				if err != nil {
					t.Fatal(err)
				}
				for _, object := range objects {
					if object.GetKind() == "Job" && strings.HasSuffix(object.GetName(), "-check-upgrade-path") {
						jobs = append(jobs, object)
					}
				}
			}
			if len(jobs) != 1 {
				t.Fatalf("expected one upgrade-path Job, got %d", len(jobs))
			}
			deadline, found, err := unstructured.NestedInt64(jobs[0].Object, "spec", "activeDeadlineSeconds")
			if err != nil || !found || deadline != scenario.deadline {
				t.Fatalf("deadline=%d found=%t err=%v; want %d", deadline, found, err, scenario.deadline)
			}
			containers, found, err := unstructured.NestedSlice(jobs[0].Object, "spec", "template", "spec", "containers")
			if err != nil || !found || len(containers) != 1 {
				t.Fatalf("expected one container: found=%t err=%v containers=%#v", found, err, containers)
			}
			container := containers[0].(map[string]interface{})
			if container["image"] != "docker.io/library/python:3.12-slim" {
				t.Fatalf("image=%v; want the default Python image", container["image"])
			}
			args := container["args"].([]interface{})
			want := []interface{}{target.Dependencies()[0].Metadata.AppVersion, scenario.override, ""}
			if len(args) != 6 || !reflect.DeepEqual(args[3:], want) {
				t.Fatalf("args=%#v; want target version and policy overrides %#v", args, want)
			}
		})
	}
}

func TestUpgradeReuseValuesWithoutSubchartDefaultsFails(t *testing.T) {
	t.Log("Remove the fix and reproduce the original missing checks error")
	target := loadUpgradeChart(t)
	delete(target.Dependencies()[0].Values, "upgradePreflight")
	_, err := runReuseValuesUpgrade(t, target, map[string]interface{}{})
	if err == nil || !strings.Contains(err.Error(), "nil pointer evaluating interface {}.checks") {
		t.Fatalf("expected the original nil checks error without subchart defaults, got %v", err)
	}
}

func loadUpgradeChart(t *testing.T) *chart.Chart {
	t.Helper()

	// Load the production templates and only the dependency needed by this preflight.
	parent, err := loader.Load(filepath.Dir(filepath.Dir(operatorChartDirectory())))
	if err != nil {
		t.Fatal(err)
	}
	operator, err := loader.Load(operatorChartDirectory())
	if err != nil {
		t.Fatal(err)
	}
	if err := retainTemplate(parent, "templates/upgrade-preflight/upgrade-path.yaml"); err != nil {
		t.Fatal(err)
	}
	var helpers []*chart.File
	for _, template := range operator.Templates {
		if strings.HasPrefix(filepath.Base(template.Name), "_") {
			helpers = append(helpers, template)
		}
	}
	operator.Templates = helpers
	parent.SetDependencies(operator)
	parent.Metadata.Dependencies = []*chart.Dependency{{Name: operator.Name(), Version: operator.Metadata.Version}}
	return parent
}

func runReuseValuesUpgrade(t *testing.T, target *chart.Chart, values map[string]interface{}) (*release.Release, error) {
	t.Helper()

	// Rebuild the stored 1.6 chart from frozen release values, without new preflight keys.
	parentValues, err := chartutil.ReadValuesFile("testdata/release-1.6.0/platform-values.yaml")
	if err != nil {
		t.Fatal(err)
	}
	operatorValues, err := chartutil.ReadValuesFile("testdata/release-1.6.0/operator-values.yaml")
	if err != nil {
		t.Fatal(err)
	}
	oldChart := &chart.Chart{Metadata: &chart.Metadata{APIVersion: "v2", Name: "dynamo-platform", Version: "1.6.0"}, Values: parentValues}
	oldChart.SetDependencies(&chart.Chart{Metadata: &chart.Metadata{APIVersion: "v2", Name: "dynamo-operator", Version: "1.6.0", AppVersion: "1.6.0"}, Values: operatorValues})
	oldValues, err := chartutil.CoalesceValues(oldChart, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, found := oldValues["dynamo-operator"].(map[string]interface{})["upgradePreflight"]; found {
		t.Fatal("the stored 1.6 defaults must predate upgradePreflight")
	}
	stored := storage.Init(driver.NewMemory())
	if err := stored.Create(&release.Release{
		Name: "upgrade-test", Namespace: "dynamo-system", Version: 1,
		Chart: oldChart, Config: map[string]interface{}{}, Info: &release.Info{Status: release.StatusDeployed},
	}); err != nil {
		t.Fatal(err)
	}

	// Run Helm's real upgrade action; fake only Kubernetes I/O in client dry-run mode.
	capabilities := chartutil.DefaultCapabilities.Copy()
	capabilities.KubeVersion = chartutil.KubeVersion{Version: "v1.32.0", Major: "1", Minor: "32"}
	configuration := &action.Configuration{
		Releases: stored, Capabilities: capabilities, Log: t.Logf,
		KubeClient: &fake.FailingKubeClient{DummyResources: kube.ResourceList{}},
	}
	upgrade := action.NewUpgrade(configuration)
	upgrade.ReuseValues = true
	upgrade.DryRunOption = "client"
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	return upgrade.RunWithContext(ctx, "upgrade-test", target, values)
}

func operatorChartDirectory() string {
	_, file, _, _ := runtime.Caller(0)
	return filepath.Join(filepath.Dir(file), "..", "..", "..", "components", "operator")
}

func retainTemplate(parent *chart.Chart, name string) error {
	var templates []*chart.File
	found := false
	for _, template := range parent.Templates {
		if template.Name == name {
			found = true
			templates = append(templates, template)
		} else if strings.HasPrefix(filepath.Base(template.Name), "_") {
			templates = append(templates, template)
		}
	}
	parent.Templates = templates
	if !found {
		return fmt.Errorf("missing template %s", name)
	}
	return nil
}

func decodeObjects(manifest string) ([]unstructured.Unstructured, error) {
	decoder := yaml.NewYAMLOrJSONDecoder(strings.NewReader(manifest), 4096)
	var objects []unstructured.Unstructured
	for {
		var object unstructured.Unstructured
		if err := decoder.Decode(&object); err != nil {
			if err == io.EOF {
				return objects, nil
			}
			return nil, err
		}
		if len(object.Object) != 0 {
			objects = append(objects, object)
		}
	}
}
