/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package controller

import (
	"testing"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/powerbudget"
)

func TestProductionQualificationProvider(t *testing.T) {
	t.Setenv(
		powerbudget.QualificationCatalogEnv,
		`{"NVIDIA-GB200":{"minWatts":200,"defaultWatts":1200,"maxWatts":1200}}`,
	)
	index, err := productionQualificationProvider()
	if err != nil {
		t.Fatal(err)
	}
	if index == nil || index["NVIDIA-GB200"].MaxWatts != 1200 {
		t.Fatalf("production qualification = %#v", index)
	}
}

func TestProductionQualificationProviderRejectsInvalidCatalog(t *testing.T) {
	for name, catalog := range map[string]string{
		"invalid bounds": `{"NVIDIA-GB200":{"minWatts":1200,"defaultWatts":200,"maxWatts":1200}}`,
		"null":           `null`,
	} {
		t.Run(name, func(t *testing.T) {
			t.Setenv(powerbudget.QualificationCatalogEnv, catalog)
			if _, err := productionQualificationProvider(); err == nil {
				t.Fatalf("productionQualificationProvider() succeeded for %s", name)
			}
		})
	}
}
