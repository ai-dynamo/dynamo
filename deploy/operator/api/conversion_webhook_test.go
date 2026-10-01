/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package api

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	webhookconversion "sigs.k8s.io/controller-runtime/pkg/webhook/conversion"

	v1alpha1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1alpha1"
	v1beta1 "github.com/ai-dynamo/dynamo/deploy/operator/api/v1beta1"
)

func TestConversionWebhook_PartialObjectsRetainMetadata(t *testing.T) {
	t.Log("Set up the conversion webhook")
	scheme := runtime.NewScheme()
	if err := v1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1alpha1 to scheme: %v", err)
	}
	if err := v1beta1.AddToScheme(scheme); err != nil {
		t.Fatalf("add v1beta1 to scheme: %v", err)
	}
	handler := webhookconversion.NewWebhookHandler(scheme, webhookconversion.NewRegistry())

	alphaVersion, betaVersion := v1alpha1.GroupVersion.String(), v1beta1.GroupVersion.String()
	for _, test := range []struct {
		kind, sourceVersion, destinationVersion string
	}{
		{"DynamoGraphDeployment", alphaVersion, betaVersion},
		{"DynamoComponentDeployment", alphaVersion, betaVersion},
		{"DynamoGraphDeploymentRequest", alphaVersion, betaVersion},
		{"DynamoGraphDeployment", betaVersion, alphaVersion},
	} {
		t.Run(test.kind+"/"+test.destinationVersion, func(t *testing.T) {
			t.Logf("Build a partial %s object with an empty metadata envelope", test.sourceVersion)
			source := map[string]any{
				"apiVersion": test.sourceVersion,
				"kind":       test.kind,
				"metadata":   map[string]any{},
			}
			sourceRaw, err := json.Marshal(source)
			if err != nil {
				t.Fatalf("marshal source object: %v", err)
			}

			t.Log("Submit the partial object to the conversion webhook")
			review := apiextensionsv1.ConversionReview{
				TypeMeta: metav1.TypeMeta{
					APIVersion: apiextensionsv1.SchemeGroupVersion.String(),
					Kind:       "ConversionReview",
				},
				Request: &apiextensionsv1.ConversionRequest{
					UID:               types.UID("partial-" + test.kind),
					DesiredAPIVersion: test.destinationVersion,
					Objects: []runtime.RawExtension{
						{Raw: sourceRaw},
					},
				},
			}
			payload, err := json.Marshal(review)
			if err != nil {
				t.Fatalf("marshal conversion review: %v", err)
			}

			recorder := httptest.NewRecorder()
			request := httptest.NewRequest("POST", "/convert", bytes.NewReader(payload))
			handler.ServeHTTP(recorder, request)
			if recorder.Code != http.StatusOK {
				t.Fatalf("conversion webhook status = %d, body = %s", recorder.Code, recorder.Body)
			}

			t.Logf("Verify the converted %s object retains an empty metadata envelope", test.destinationVersion)
			var response apiextensionsv1.ConversionReview
			if err := json.Unmarshal(recorder.Body.Bytes(), &response); err != nil {
				t.Fatalf("unmarshal conversion response: %v", err)
			}
			if response.Response == nil {
				t.Fatalf("conversion response is nil: %s", recorder.Body)
			}
			if response.Response.Result.Status != metav1.StatusSuccess {
				t.Fatalf("conversion failed: %#v", response.Response.Result)
			}
			if len(response.Response.ConvertedObjects) != 1 {
				t.Fatalf("converted object count = %d, want 1", len(response.Response.ConvertedObjects))
			}

			var converted map[string]any
			if err := json.Unmarshal(response.Response.ConvertedObjects[0].Raw, &converted); err != nil {
				t.Fatalf("unmarshal converted object: %v", err)
			}
			if converted["apiVersion"] != test.destinationVersion || converted["kind"] != test.kind {
				t.Fatalf("converted type = %v %v, want %s %s",
					converted["apiVersion"], converted["kind"], test.destinationVersion, test.kind)
			}
			metadata, found := converted["metadata"]
			if !found {
				t.Fatalf("converted object is missing metadata: %s", response.Response.ConvertedObjects[0].Raw)
			}
			metadataMap, ok := metadata.(map[string]any)
			if !ok {
				t.Fatalf("converted metadata has type %T, want object", metadata)
			}
			if len(metadataMap) != 0 {
				t.Fatalf("converted metadata = %v, want empty object", metadataMap)
			}
		})
	}
}
