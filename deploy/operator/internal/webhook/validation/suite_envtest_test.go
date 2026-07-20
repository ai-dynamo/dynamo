/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package validation_test

import (
	"context"
	"fmt"
	"os"
	"slices"
	"strings"
	"sync"
	"testing"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/features"
	"github.com/ai-dynamo/dynamo/deploy/operator/internal/testing/operatorenv"
	webhooksetup "github.com/ai-dynamo/dynamo/deploy/operator/internal/webhook/setup"
	grovev1alpha1 "github.com/ai-dynamo/grove/operator/api/core/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
)

const (
	admissionOperatorPrincipal = "system:serviceaccount:dynamo-system:dynamo-operator"
	legacySeedUsername         = "operatorenv-legacy-seeder"
)

var (
	admissionGate = &mutableFeatureGate{}
	admissionEnv  = operatorenv.New(operatorenv.Options{
		Admission: operatorenv.AdmissionWebhooks{
			Validating:  true,
			BypassUsers: []string{legacySeedUsername},
		},
		Conversion:        true,
		SetupWebhooks:     setupAdmissionWebhooks,
		OperatorVersion:   "1.1.0",
		OperatorPrincipal: admissionOperatorPrincipal,
	})
)

func TestMain(m *testing.M) {
	os.Exit(admissionEnv.RunM(m))
}

func setupAdmissionWebhooks(mgr ctrl.Manager, opts operatorenv.WebhookSetupOptions) error {
	return webhooksetup.Setup(mgr, webhooksetup.Options{
		Config:            opts.OperatorConfig,
		RuntimeConfig:     opts.RuntimeConfig,
		OperatorVersion:   opts.OperatorVersion,
		OperatorPrincipal: opts.OperatorPrincipal,
		Gate:              admissionGate,
	})
}

type mutableFeatureGate struct {
	mu    sync.RWMutex
	gates features.Gates
}

func (g *mutableFeatureGate) Enabled(name features.Name) bool {
	g.mu.RLock()
	defer g.mu.RUnlock()
	return g.gates.Enabled(name)
}

func (g *mutableFeatureGate) set(gates features.Gates) {
	g.mu.Lock()
	defer g.mu.Unlock()
	g.gates = gates
}

type admissionTestCase struct {
	object             runtime.Object
	oldObject          runtime.Object
	oldBeforeUpdate    runtime.Object
	mutateObject       func(*testing.T, map[string]any)
	gates              features.Gates
	seedGates          *features.Gates
	seedWithoutWebhook bool
	withoutTopology    bool
	username           string

	wantSchemaError   string
	wantCELError      string
	wantWebhookErrors []string
	wantWarnings      []string
	notWantError      string
}

func runAdmissionTest(t *testing.T, test admissionTestCase) {
	t.Helper()

	t.Log("Create an isolated namespace in the shared operator environment")
	env := admissionEnv.ForTest(t)
	warnings := &warningRecorder{}
	resourceClient := newAdmissionResourceClient(t, env, test.object, test.username, warnings)

	if !test.withoutTopology {
		t.Log("Install the test-owned cluster topology used by DGD validation")
		createTestClusterTopology(t, env)
	}

	admissionGate.set(test.gates)
	current, originalNamespace := admissionObject(t, test.object, env.Namespace(), test.mutateObject)

	var err error
	if test.oldObject == nil {
		t.Log("Submit the create request through the Kubernetes API server")
		_, err = resourceClient.Create(t.Context(), current, metav1.CreateOptions{})
	} else {
		t.Log("Create the old resource state through the Kubernetes API server")
		seedGates := test.gates
		if test.seedGates != nil {
			seedGates = *test.seedGates
		}
		admissionGate.set(seedGates)
		seedClient := resourceClient
		if test.seedWithoutWebhook {
			seedClient = newAdmissionResourceClient(t, env, test.oldObject, legacySeedUsername, warnings)
		}
		old := seedAdmissionObject(t, seedClient, test, env.Namespace())

		t.Log("Submit the update request through the Kubernetes API server")
		admissionGate.set(test.gates)
		warnings.clear()
		current.SetResourceVersion(old.GetResourceVersion())
		_, err = resourceClient.Update(t.Context(), current, metav1.UpdateOptions{})
	}

	t.Log("Compare the API server admission result with the table expectations")
	wantErrors := expectedAdmissionErrors(t, test)
	wantErrors = rewriteExpectedNamespace(wantErrors, originalNamespace, env.Namespace())
	assertAdmissionErrors(t, err, wantErrors, test.notWantError)
	wantWarnings := rewriteExpectedNamespace(test.wantWarnings, originalNamespace, env.Namespace())
	if got := warnings.list(); !slices.Equal(got, wantWarnings) {
		t.Fatalf("warnings = %v, want %v", got, wantWarnings)
	}
}

func newAdmissionResourceClient(
	t *testing.T,
	env *operatorenv.TestEnv,
	object runtime.Object,
	username string,
	warnings rest.WarningHandler,
) dynamic.ResourceInterface {
	t.Helper()
	config := env.RESTConfig()
	config.WarningHandler = warnings
	if username != "" {
		config.Impersonate.UserName = username
		config.Impersonate.Groups = []string{"system:masters"}
	}
	client, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatalf("create dynamic client: %v", err)
	}
	gvk := admissionGVK(t, object)
	resource := schema.GroupVersionResource{
		Group:    gvk.Group,
		Version:  gvk.Version,
		Resource: admissionResource(t, gvk.Kind),
	}
	return client.Resource(resource).Namespace(env.Namespace())
}

func admissionResource(t *testing.T, kind string) string {
	t.Helper()
	switch kind {
	case "DynamoComponentDeployment":
		return "dynamocomponentdeployments"
	case "DynamoGraphDeployment":
		return "dynamographdeployments"
	case "DynamoGraphDeploymentRequest":
		return "dynamographdeploymentrequests"
	default:
		t.Fatalf("unsupported admission kind %q", kind)
		return ""
	}
}

func admissionGVK(t *testing.T, object runtime.Object) schema.GroupVersionKind {
	t.Helper()
	if gvk := object.GetObjectKind().GroupVersionKind(); !gvk.Empty() {
		return gvk
	}
	switch object.(type) {
	case *nvidiacomv1alpha1.DynamoComponentDeployment:
		return nvidiacomv1alpha1.DynamoComponentDeploymentGVK
	case *nvidiacomv1beta1.DynamoComponentDeployment:
		return nvidiacomv1beta1.DynamoComponentDeploymentGVK
	case *nvidiacomv1alpha1.DynamoGraphDeployment:
		return nvidiacomv1alpha1.DynamoGraphDeploymentGVK
	case *nvidiacomv1beta1.DynamoGraphDeployment:
		return nvidiacomv1beta1.DynamoGraphDeploymentGVK
	case *nvidiacomv1alpha1.DynamoGraphDeploymentRequest:
		return nvidiacomv1alpha1.DynamoGraphDeploymentRequestGVK
	case *nvidiacomv1beta1.DynamoGraphDeploymentRequest:
		return nvidiacomv1beta1.DynamoGraphDeploymentRequestGVK
	default:
		t.Fatalf("unsupported admission object type %T", object)
		return schema.GroupVersionKind{}
	}
}

func admissionObject(
	t *testing.T,
	object runtime.Object,
	namespace string,
	mutate func(*testing.T, map[string]any),
) (*unstructured.Unstructured, string) {
	t.Helper()
	value, err := runtime.DefaultUnstructuredConverter.ToUnstructured(object)
	if err != nil {
		t.Fatalf("convert %T to unstructured: %v", object, err)
	}
	gvk := admissionGVK(t, object)
	value["apiVersion"] = gvk.GroupVersion().String()
	value["kind"] = gvk.Kind
	delete(value, "status")
	metadata, ok := value["metadata"].(map[string]any)
	if !ok {
		t.Fatalf("%T metadata is missing or not an object", object)
	}
	originalNamespace, _ := metadata["namespace"].(string)
	metadata["namespace"] = namespace
	delete(metadata, "resourceVersion")
	delete(metadata, "uid")
	delete(metadata, "generation")
	delete(metadata, "creationTimestamp")
	delete(metadata, "managedFields")
	if mutate != nil {
		mutate(t, value)
	}
	return &unstructured.Unstructured{Object: value}, originalNamespace
}

func seedAdmissionObject(
	t *testing.T,
	resourceClient dynamic.ResourceInterface,
	test admissionTestCase,
	namespace string,
) *unstructured.Unstructured {
	t.Helper()
	seedObject := test.oldObject
	if test.oldBeforeUpdate != nil {
		seedObject = test.oldBeforeUpdate
	}
	seed, _ := admissionObject(t, seedObject, namespace, nil)
	created, err := resourceClient.Create(t.Context(), seed, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("create old resource state: %v", err)
	}
	if test.oldBeforeUpdate != nil {
		old, _ := admissionObject(t, test.oldObject, namespace, nil)
		old.SetResourceVersion(created.GetResourceVersion())
		created, err = resourceClient.Update(t.Context(), old, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("update old resource state: %v", err)
		}
	}
	status, found, err := unstructured.NestedMap(mustUnstructured(t, test.oldObject), "status")
	if err != nil {
		t.Fatalf("read old resource status: %v", err)
	}
	prunedStatus, nonZero := pruneZeroValues(status)
	if found && nonZero {
		if err := unstructured.SetNestedMap(created.Object, prunedStatus.(map[string]any), "status"); err != nil {
			t.Fatalf("set old resource status: %v", err)
		}
		created, err = resourceClient.UpdateStatus(t.Context(), created, metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("update old resource status: %v", err)
		}
	}
	return created
}

func pruneZeroValues(value any) (any, bool) {
	switch value := value.(type) {
	case nil:
		return nil, false
	case bool:
		return value, value
	case string:
		return value, value != ""
	case int64:
		return value, value != 0
	case float64:
		return value, value != 0
	case []any:
		result := make([]any, 0, len(value))
		for _, item := range value {
			if pruned, ok := pruneZeroValues(item); ok {
				result = append(result, pruned)
			}
		}
		return result, len(result) > 0
	case map[string]any:
		result := make(map[string]any, len(value))
		for key, item := range value {
			if pruned, ok := pruneZeroValues(item); ok {
				result[key] = pruned
			}
		}
		return result, len(result) > 0
	default:
		return value, true
	}
}

func mustUnstructured(t *testing.T, object runtime.Object) map[string]any {
	t.Helper()
	value, err := runtime.DefaultUnstructuredConverter.ToUnstructured(object)
	if err != nil {
		t.Fatalf("convert %T to unstructured: %v", object, err)
	}
	return value
}

func createTestClusterTopology(t *testing.T, env *operatorenv.TestEnv) {
	t.Helper()
	client, err := dynamic.NewForConfig(env.RESTConfig())
	if err != nil {
		t.Fatalf("create topology client: %v", err)
	}
	resource := client.Resource(schema.GroupVersionResource{
		Group: "grove.io", Version: "v1alpha1", Resource: "clustertopologybindings",
	})
	topology := &grovev1alpha1.ClusterTopologyBinding{
		TypeMeta:   metav1.TypeMeta{APIVersion: grovev1alpha1.SchemeGroupVersion.String(), Kind: "ClusterTopologyBinding"},
		ObjectMeta: metav1.ObjectMeta{Name: "grove-topology"},
		Spec: grovev1alpha1.ClusterTopologyBindingSpec{
			Levels: []grovev1alpha1.TopologyLevel{
				{Domain: grovev1alpha1.TopologyDomainZone, Key: "topology.kubernetes.io/zone"},
				{Domain: grovev1alpha1.TopologyDomainRack, Key: "nvidia.com/rack"},
			},
		},
	}
	value, err := runtime.DefaultUnstructuredConverter.ToUnstructured(topology)
	if err != nil {
		t.Fatalf("convert cluster topology: %v", err)
	}
	if _, err := resource.Create(t.Context(), &unstructured.Unstructured{Object: value}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("create cluster topology: %v", err)
	}
	t.Cleanup(func() {
		if err := resource.Delete(context.Background(), topology.Name, metav1.DeleteOptions{}); err != nil {
			t.Errorf("delete cluster topology: %v", err)
		}
	})
}

func expectedAdmissionErrors(t *testing.T, test admissionTestCase) []string {
	t.Helper()
	categories := 0
	if test.wantSchemaError != "" {
		categories++
	}
	if test.wantCELError != "" {
		categories++
	}
	if len(test.wantWebhookErrors) != 0 {
		categories++
	}
	if categories > 1 {
		t.Fatal("one admission stage cannot have multiple error expectations")
	}
	if test.wantSchemaError != "" {
		return []string{test.wantSchemaError}
	}
	if test.wantCELError != "" {
		return []string{test.wantCELError}
	}
	return test.wantWebhookErrors
}

func assertAdmissionErrors(t *testing.T, err error, want []string, notWant string) {
	t.Helper()
	if len(want) == 0 {
		if err != nil {
			t.Fatalf("admission error = %v, want none", err)
		}
		return
	}
	if err == nil {
		t.Fatalf("admission error = nil, want %v", want)
	}
	if notWant != "" && strings.Contains(err.Error(), notWant) {
		t.Fatalf("admission error = %q, must not contain %q", err, notWant)
	}
	status, ok := err.(interface{ Status() metav1.Status })
	if !ok {
		t.Fatalf("admission error = %T %v, want Kubernetes status error", err, err)
	}
	details := status.Status().Details
	if details == nil {
		t.Fatalf("admission error = %v, want field causes", err)
	}
	got := make([]string, 0, len(details.Causes))
	for _, cause := range details.Causes {
		if (cause.Field == "" || cause.Field == "<nil>") && strings.Contains(cause.Message, "some validation rules were not checked because the object was invalid") {
			continue
		}
		if cause.Field == "" {
			t.Fatalf("admission cause = %#v, want field path", cause)
		}
		message := strings.Replace(cause.Message, `Invalid value: "object": `, "Invalid value: ", 1)
		message = strings.Replace(message, `Invalid value: "array": `, "Invalid value: ", 1)
		got = append(got, fmt.Sprintf("%s: %s", cause.Field, message))
	}
	if !slices.Equal(got, want) {
		t.Fatalf("admission errors = %v, want %v; full error: %v", got, want, err)
	}
}

func rewriteExpectedNamespace(values []string, oldNamespace, newNamespace string) []string {
	if oldNamespace == "" || oldNamespace == newNamespace {
		return values
	}
	rewritten := make([]string, len(values))
	for i, value := range values {
		rewritten[i] = strings.ReplaceAll(value, oldNamespace+"-", newNamespace+"-")
	}
	return rewritten
}

type warningRecorder struct {
	mu       sync.Mutex
	warnings []string
}

func (r *warningRecorder) HandleWarningHeader(code int, _ string, message string) {
	if code != 299 {
		return
	}
	if strings.HasPrefix(message, "nvidia.com/v1alpha1 ") && strings.Contains(message, " is deprecated; use nvidia.com/v1beta1 ") {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.warnings = append(r.warnings, message)
}

func (r *warningRecorder) clear() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.warnings = nil
}

func (r *warningRecorder) list() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.warnings)
}
