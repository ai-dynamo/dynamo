/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package validation

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	goruntime "runtime"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	nvidiacomv1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	nvidiacomv1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	authenticationv1 "k8s.io/api/authentication/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/wait"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/envtest"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"
	ctrlwebhook "sigs.k8s.io/controller-runtime/pkg/webhook"
	"sigs.k8s.io/controller-runtime/pkg/webhook/admission"
)

var apiAdmissionTestEnvironment *admissionTestEnvironment

func TestMain(m *testing.M) {
	testEnvironment, err := startAdmissionTestEnvironment()
	if err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "start admission test environment: %v\n", err)
		os.Exit(1)
	}
	apiAdmissionTestEnvironment = testEnvironment

	code := m.Run()
	if err := testEnvironment.Stop(); err != nil {
		_, _ = fmt.Fprintf(os.Stderr, "stop admission test environment: %v\n", err)
		if code == 0 {
			code = 1
		}
	}
	os.Exit(code)
}

type admissionTestEnvironment struct {
	environment *envtest.Environment
	client      dynamic.Interface
	cancel      context.CancelFunc
	managerDone <-chan error
	warnings    *admissionWarningCollector
	dcdBeta     *validatorSlot
	dgdBeta     *validatorSlot
}

func startAdmissionTestEnvironment() (*admissionTestEnvironment, error) {
	operatorRoot, err := admissionTestOperatorRoot()
	if err != nil {
		return nil, err
	}
	validatingWebhook, err := renderHelmValidatingWebhook(operatorRoot)
	if err != nil {
		return nil, err
	}
	scheme := runtime.NewScheme()
	if err := nvidiacomv1alpha1.AddToScheme(scheme); err != nil {
		return nil, fmt.Errorf("add v1alpha1 scheme: %w", err)
	}
	if err := nvidiacomv1beta1.AddToScheme(scheme); err != nil {
		return nil, fmt.Errorf("add v1beta1 scheme: %w", err)
	}

	testEnvironment := &envtest.Environment{
		Scheme:                scheme,
		CRDDirectoryPaths:     []string{filepath.Join(operatorRoot, "config", "crd", "bases")},
		ErrorIfCRDPathMissing: true,
		BinaryAssetsDirectory: filepath.Join(
			operatorRoot,
			"bin",
			"k8s",
			fmt.Sprintf("1.30.0-%s-%s", goruntime.GOOS, goruntime.GOARCH),
		),
		WebhookInstallOptions: envtest.WebhookInstallOptions{
			ValidatingWebhooks: []*admissionregistrationv1.ValidatingWebhookConfiguration{validatingWebhook},
		},
	}
	config, err := testEnvironment.Start()
	if err != nil {
		return nil, err
	}

	webhookServer := ctrlwebhook.NewServer(ctrlwebhook.Options{
		Host:    testEnvironment.WebhookInstallOptions.LocalServingHost,
		Port:    testEnvironment.WebhookInstallOptions.LocalServingPort,
		CertDir: testEnvironment.WebhookInstallOptions.LocalServingCertDir,
	})
	manager, err := ctrl.NewManager(config, ctrl.Options{
		Scheme:        scheme,
		Metrics:       metricsserver.Options{BindAddress: "0"},
		WebhookServer: webhookServer,
	})
	if err != nil {
		_ = testEnvironment.Stop()
		return nil, fmt.Errorf("create webhook manager: %w", err)
	}

	for _, object := range []runtime.Object{
		&nvidiacomv1beta1.DynamoGraphDeployment{},
		&nvidiacomv1beta1.DynamoComponentDeployment{},
		&nvidiacomv1beta1.DynamoGraphDeploymentRequest{},
		&nvidiacomv1beta1.DynamoGraphDeploymentScalingAdapter{},
	} {
		if err := ctrl.NewWebhookManagedBy(manager, object).Complete(); err != nil {
			_ = testEnvironment.Stop()
			return nil, fmt.Errorf("register conversion webhook for %T: %w", object, err)
		}
	}

	harness := &admissionTestEnvironment{
		environment: testEnvironment,
		warnings:    &admissionWarningCollector{},
		dcdBeta:     newValidatorSlot(),
		dgdBeta:     newValidatorSlot(),
	}
	dcdRegistration := NewDynamoComponentDeploymentHandler()
	dcdRegistration.registerWithManager(
		manager,
		&nvidiacomv1beta1.DynamoComponentDeployment{},
		dynamoComponentDeploymentV1Beta1WebhookPath,
		harness.dcdBeta,
	)
	dgdRegistration := &DynamoGraphDeploymentHandler{}
	dgdRegistration.registerWithManager(
		manager,
		&nvidiacomv1beta1.DynamoGraphDeployment{},
		dynamoGraphDeploymentV1Beta1WebhookPath,
		harness.dgdBeta,
	)
	clientConfig := rest.CopyConfig(config)
	clientConfig.WarningHandler = harness.warnings
	harness.client, err = dynamic.NewForConfig(clientConfig)
	if err != nil {
		_ = testEnvironment.Stop()
		return nil, fmt.Errorf("create dynamic client: %w", err)
	}

	managerContext, cancel := context.WithCancel(context.Background())
	managerDone := make(chan error, 1)
	harness.cancel = cancel
	harness.managerDone = managerDone
	go func() {
		managerDone <- manager.Start(managerContext)
	}()
	if err := wait.PollUntilContextTimeout(
		context.Background(),
		10*time.Millisecond,
		10*time.Second,
		true,
		func(context.Context) (bool, error) {
			return webhookServer.StartedChecker()(nil) == nil, nil
		},
	); err != nil {
		cancel()
		_ = testEnvironment.Stop()
		return nil, fmt.Errorf("wait for webhook server: %w", err)
	}

	return harness, nil
}

func (e *admissionTestEnvironment) Stop() error {
	e.cancel()
	managerErr := <-e.managerDone
	environmentErr := e.environment.Stop()
	if managerErr != nil {
		return fmt.Errorf("stop webhook manager: %w", managerErr)
	}
	if environmentErr != nil {
		return fmt.Errorf("stop envtest: %w", environmentErr)
	}
	return nil
}

func admissionTestOperatorRoot() (string, error) {
	_, filename, _, ok := goruntime.Caller(0)
	if !ok {
		return "", fmt.Errorf("resolve admission test source path")
	}
	return filepath.Clean(filepath.Join(filepath.Dir(filename), "..", "..", "..")), nil
}

func renderHelmValidatingWebhook(operatorRoot string) (*admissionregistrationv1.ValidatingWebhookConfiguration, error) {
	helmBinary := os.Getenv("HELM")
	if helmBinary == "" {
		var err error
		helmBinary, err = exec.LookPath("helm")
		if err != nil {
			return nil, fmt.Errorf("find helm binary: %w", err)
		}
	}
	chartPath := filepath.Join(operatorRoot, "..", "helm", "charts", "platform", "components", "operator")
	command := exec.Command(
		helmBinary,
		"template",
		"admission-test",
		chartPath,
		"--namespace", "default",
		"--show-only", "templates/webhook-configuration.yaml",
		"--set", "discoveryBackend=kubernetes",
	)
	output, err := command.CombinedOutput()
	if err != nil {
		return nil, fmt.Errorf("render Helm webhook configuration: %w: %s", err, output)
	}

	decoder := utilyaml.NewYAMLOrJSONDecoder(bytes.NewReader(output), 4096)
	var validatingWebhooks []*admissionregistrationv1.ValidatingWebhookConfiguration
	for {
		var document map[string]any
		if err := decoder.Decode(&document); err != nil {
			if err == io.EOF {
				break
			}
			return nil, fmt.Errorf("decode rendered Helm webhook configuration: %w", err)
		}
		if document["kind"] != "ValidatingWebhookConfiguration" {
			continue
		}
		encoded, err := json.Marshal(document)
		if err != nil {
			return nil, fmt.Errorf("encode rendered validating webhook configuration: %w", err)
		}
		webhook := &admissionregistrationv1.ValidatingWebhookConfiguration{}
		if err := json.Unmarshal(encoded, webhook); err != nil {
			return nil, fmt.Errorf("decode rendered validating webhook configuration: %w", err)
		}
		validatingWebhooks = append(validatingWebhooks, webhook)
	}
	if len(validatingWebhooks) != 1 {
		return nil, fmt.Errorf("rendered Helm chart contains %d validating webhook configurations, want 1", len(validatingWebhooks))
	}
	return validatingWebhooks[0], nil
}

func (e *admissionTestEnvironment) useDynamoComponentDeploymentHandler(handler *DynamoComponentDeploymentHandler) {
	e.dcdBeta.Set(handler)
}

func (e *admissionTestEnvironment) allowDynamoComponentDeployments() {
	e.dcdBeta.Set(allowingValidator{})
}

func (e *admissionTestEnvironment) useDynamoGraphDeploymentHandler(handler *DynamoGraphDeploymentHandler) {
	e.dgdBeta.Set(handler)
}

func (e *admissionTestEnvironment) allowDynamoGraphDeployments() {
	e.dgdBeta.Set(allowingValidator{})
}

func (e *admissionTestEnvironment) Admit(
	t *testing.T,
	oldObject runtime.Object,
	object runtime.Object,
	mutateRequest func(*testing.T, map[string]any),
	userInfo *authenticationv1.UserInfo,
	allow func(),
	configure func(),
) ([]string, error) {
	t.Helper()

	request := admissionUnstructured(t, object)
	if mutateRequest != nil {
		mutateRequest(t, request)
	}
	resourceInfo, err := admissionTarget(request)
	if err != nil {
		t.Fatal(err)
	}
	e.ensureNamespace(t, resourceInfo.namespace)
	resource := resourceInfo.ForClient(e.client)

	if oldObject != nil {
		if admissionSourceVersion(t, oldObject) != admissionSourceVersion(t, object) {
			t.Fatal("old and current source versions differ")
		}
		allow()
		oldRequest := admissionUnstructured(t, oldObject)
		seedRequest := (&unstructured.Unstructured{Object: oldRequest}).DeepCopy()

		// Restart is update-only, so reproduce its two-step API lifecycle when seeding old state.
		_, hasRestart, err := unstructured.NestedFieldNoCopy(seedRequest.Object, "spec", "restart")
		if err != nil {
			t.Fatalf("inspect old admission restart: %v", err)
		}
		if hasRestart {
			unstructured.RemoveNestedField(seedRequest.Object, "spec", "restart")
		}
		created, err := resource.Create(t.Context(), seedRequest, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("seed old admission object: %v", err)
		}
		e.cleanup(t, resource, created.GetName())
		if hasRestart {
			desiredOld := &unstructured.Unstructured{Object: oldRequest}
			desiredOld.SetResourceVersion(created.GetResourceVersion())
			created, err = resource.Update(t.Context(), desiredOld, metav1.UpdateOptions{})
			if err != nil {
				t.Fatalf("seed old admission restart: %v", err)
			}
		}
		created = e.updateStatus(t, resource, created, oldObject)
		if err := unstructured.SetNestedField(request, created.GetResourceVersion(), "metadata", "resourceVersion"); err != nil {
			t.Fatalf("set update resource version: %v", err)
		}
	}

	configure()
	client, err := e.clientForUser(userInfo)
	if err != nil {
		t.Fatalf("create admission client: %v", err)
	}
	resource = resourceInfo.ForClient(client)
	e.warnings.Reset()
	if oldObject == nil {
		created, err := resource.Create(t.Context(), &unstructured.Unstructured{Object: request}, metav1.CreateOptions{})
		if err == nil {
			e.cleanup(t, resource, created.GetName())
		}
		return e.warnings.List(), err
	}
	_, err = resource.Update(t.Context(), &unstructured.Unstructured{Object: request}, metav1.UpdateOptions{})
	return e.warnings.List(), err
}

type admissionResourceTarget struct {
	groupVersionResource schema.GroupVersionResource
	namespace            string
}

func (r admissionResourceTarget) ForClient(client dynamic.Interface) dynamic.ResourceInterface {
	return client.Resource(r.groupVersionResource).Namespace(r.namespace)
}

func admissionTarget(object map[string]any) (admissionResourceTarget, error) {
	apiVersion, _, err := unstructured.NestedString(object, "apiVersion")
	if err != nil || apiVersion == "" {
		return admissionResourceTarget{}, fmt.Errorf("admission object has no apiVersion")
	}
	groupVersion, err := schema.ParseGroupVersion(apiVersion)
	if err != nil {
		return admissionResourceTarget{}, fmt.Errorf("parse admission apiVersion %q: %w", apiVersion, err)
	}
	kind, _, err := unstructured.NestedString(object, "kind")
	if err != nil || kind == "" {
		return admissionResourceTarget{}, fmt.Errorf("admission object has no kind")
	}
	var resource string
	switch kind {
	case "DynamoComponentDeployment":
		resource = "dynamocomponentdeployments"
	case "DynamoGraphDeployment":
		resource = "dynamographdeployments"
	default:
		return admissionResourceTarget{}, fmt.Errorf("unsupported admission kind %q", kind)
	}
	namespace, _, err := unstructured.NestedString(object, "metadata", "namespace")
	if err != nil || namespace == "" {
		return admissionResourceTarget{}, fmt.Errorf("admission object has no namespace")
	}
	groupVersionResource := groupVersion.WithResource(resource)
	return admissionResourceTarget{
		groupVersionResource: groupVersionResource,
		namespace:            namespace,
	}, nil
}

func (e *admissionTestEnvironment) updateStatus(
	t *testing.T,
	resource dynamic.ResourceInterface,
	created *unstructured.Unstructured,
	oldObject runtime.Object,
) *unstructured.Unstructured {
	t.Helper()
	oldWithStatus, err := runtime.DefaultUnstructuredConverter.ToUnstructured(oldObject)
	if err != nil {
		t.Fatalf("convert old admission status: %v", err)
	}
	status, found := oldWithStatus["status"]
	if !found || !hasNonZeroAdmissionStatus(status) {
		return created
	}
	withStatus := created.DeepCopy()
	withStatus.Object["status"] = status
	updated, err := resource.UpdateStatus(t.Context(), withStatus, metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("seed old admission status: %v", err)
	}
	return updated
}

func hasNonZeroAdmissionStatus(value any) bool {
	switch value := value.(type) {
	case nil:
		return false
	case map[string]any:
		for _, child := range value {
			if hasNonZeroAdmissionStatus(child) {
				return true
			}
		}
		return false
	case []any:
		for _, child := range value {
			if hasNonZeroAdmissionStatus(child) {
				return true
			}
		}
		return false
	case string:
		return value != ""
	case bool:
		return value
	case int:
		return value != 0
	case int8:
		return value != 0
	case int16:
		return value != 0
	case int32:
		return value != 0
	case int64:
		return value != 0
	case uint:
		return value != 0
	case uint8:
		return value != 0
	case uint16:
		return value != 0
	case uint32:
		return value != 0
	case uint64:
		return value != 0
	case float32:
		return value != 0
	case float64:
		return value != 0
	default:
		return true
	}
}

func (e *admissionTestEnvironment) ensureNamespace(t *testing.T, namespace string) {
	t.Helper()
	namespaces := e.client.Resource(schema.GroupVersionResource{Version: "v1", Resource: "namespaces"})
	if _, err := namespaces.Get(t.Context(), namespace, metav1.GetOptions{}); err == nil {
		return
	} else if !apierrors.IsNotFound(err) {
		t.Fatalf("get admission namespace %q: %v", namespace, err)
	}
	_, err := namespaces.Create(t.Context(), &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "v1",
		"kind":       "Namespace",
		"metadata": map[string]any{
			"name": namespace,
		},
	}}, metav1.CreateOptions{})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatalf("create admission namespace %q: %v", namespace, err)
	}
}

func (e *admissionTestEnvironment) cleanup(t *testing.T, resource dynamic.ResourceInterface, name string) {
	t.Helper()
	t.Cleanup(func() {
		if err := resource.Delete(context.Background(), name, metav1.DeleteOptions{}); err != nil {
			if apierrors.IsNotFound(err) {
				return
			}
			t.Errorf("delete admitted object %q: %v", name, err)
		}
	})
}

func assertAdmissionErrors(t *testing.T, err error, schemaErr, celErr string, webhookErrs []string) {
	t.Helper()

	want := webhookErrs
	if schemaErr != "" {
		if celErr != "" || len(webhookErrs) != 0 {
			t.Fatal("schema rejection cannot have downstream expectations")
		}
		want = []string{schemaErr}
	} else if celErr != "" {
		if len(webhookErrs) != 0 {
			t.Fatal("CEL rejection cannot have webhook expectations")
		}
		want = []string{celErr}
	}

	if len(want) == 0 {
		if err != nil {
			t.Fatalf("admission error = %v, want none", err)
		}
		return
	}
	if err == nil {
		t.Fatalf("admission errors = nil, want %v", want)
	}
	statusErr, ok := err.(*apierrors.StatusError)
	if !ok || !apierrors.IsInvalid(err) {
		t.Fatalf("error = %T %v, want typed Kubernetes invalid error", err, err)
	}
	if statusErr.ErrStatus.Details == nil {
		t.Fatalf("error = %v, want typed field causes", err)
	}

	causes := statusErr.ErrStatus.Details.Causes
	got := make([]string, 0, len(causes))
	for _, cause := range causes {
		// Ignore the API server's unfielded cascade notice after a primary schema error.
		if strings.Contains(cause.Message, "some validation rules were not checked because the object was invalid") {
			continue
		}
		if cause.Field == "" {
			t.Fatalf("error cause = %#v, want an exact field path", cause)
		}
		got = append(got, fmt.Sprintf("%s: %s", cause.Field, cause.Message))
	}
	if !slices.Equal(got, want) {
		t.Fatalf("admission errors = %v, want %v", got, want)
	}
}

func assertBetaValidationErrors(t *testing.T, err error, want []string) {
	t.Helper()
	assertAdmissionErrors(t, err, "", "", want)
}

func (e *admissionTestEnvironment) clientForUser(userInfo *authenticationv1.UserInfo) (dynamic.Interface, error) {
	if userInfo == nil {
		return e.client, nil
	}
	if userInfo.UID != "" || len(userInfo.Extra) != 0 {
		return nil, fmt.Errorf("envtest admission users support username and groups only")
	}
	groups := append([]string{}, userInfo.Groups...)
	groups = append(groups, "system:masters")
	user, err := e.environment.AddUser(envtest.User{Name: userInfo.Username, Groups: groups}, nil)
	if err != nil {
		return nil, err
	}
	config := rest.CopyConfig(user.Config())
	config.WarningHandler = e.warnings
	return dynamic.NewForConfig(config)
}

type admissionWarningCollector struct {
	lock     sync.Mutex
	warnings []string
}

func (c *admissionWarningCollector) HandleWarningHeader(_ int, _ string, text string) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.warnings = append(c.warnings, text)
}

func (c *admissionWarningCollector) Reset() {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.warnings = nil
}

func (c *admissionWarningCollector) List() []string {
	c.lock.Lock()
	defer c.lock.Unlock()
	warnings := make([]string, 0, len(c.warnings))
	for _, warning := range c.warnings {
		// The matrices assert webhook warnings, not Kubernetes' built-in API deprecation warning.
		if strings.HasPrefix(warning, "nvidia.com/v1alpha1 ") && strings.Contains(warning, " is deprecated; use nvidia.com/v1beta1 ") {
			continue
		}
		warnings = append(warnings, warning)
	}
	return warnings
}

type validatorSlot struct {
	lock      sync.RWMutex
	validator admission.CustomValidator
}

func newValidatorSlot() *validatorSlot {
	return &validatorSlot{validator: allowingValidator{}}
}

func (s *validatorSlot) Set(validator admission.CustomValidator) {
	s.lock.Lock()
	defer s.lock.Unlock()
	s.validator = validator
}

func (s *validatorSlot) ValidateCreate(ctx context.Context, object runtime.Object) (admission.Warnings, error) {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.validator.ValidateCreate(ctx, object)
}

func (s *validatorSlot) ValidateUpdate(ctx context.Context, oldObject, object runtime.Object) (admission.Warnings, error) {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.validator.ValidateUpdate(ctx, oldObject, object)
}

func (s *validatorSlot) ValidateDelete(ctx context.Context, object runtime.Object) (admission.Warnings, error) {
	s.lock.RLock()
	defer s.lock.RUnlock()
	return s.validator.ValidateDelete(ctx, object)
}

type allowingValidator struct{}

func (allowingValidator) ValidateCreate(context.Context, runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

func (allowingValidator) ValidateUpdate(context.Context, runtime.Object, runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

func (allowingValidator) ValidateDelete(context.Context, runtime.Object) (admission.Warnings, error) {
	return nil, nil
}

var _ admission.CustomValidator = &validatorSlot{}
var _ admission.CustomValidator = allowingValidator{}
var _ rest.WarningHandler = &admissionWarningCollector{}
