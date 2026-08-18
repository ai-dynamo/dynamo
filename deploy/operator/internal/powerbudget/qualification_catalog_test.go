/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import "testing"

func TestParseQualificationCatalogJSON(t *testing.T) {
	t.Run("empty is non-nil and fail-closed", func(t *testing.T) {
		index, err := ParseQualificationCatalogJSON("")
		if err != nil {
			t.Fatal(err)
		}
		if index == nil || len(index) != 0 {
			t.Fatalf("index = %#v, want non-nil empty map", index)
		}
	})

	t.Run("exact product bounds", func(t *testing.T) {
		index, err := ParseQualificationCatalogJSON(`{"NVIDIA-GB200":{"minWatts":200,"defaultWatts":1200,"maxWatts":1200}}`)
		if err != nil {
			t.Fatal(err)
		}
		got := index["NVIDIA-GB200"]
		if got.MinWatts != 200 || got.DefaultWatts != 1200 || got.MaxWatts != 1200 {
			t.Fatalf("constraints = %#v", got)
		}
	})

	for name, raw := range map[string]string{
		"malformed":       `{`,
		"null":            `null`,
		"unknown field":   `{"NVIDIA-GB200":{"minWatts":200,"defaultWatts":1200,"maxWatts":1200,"node":"x"}}`,
		"missing default": `{"NVIDIA-GB200":{"minWatts":200,"maxWatts":1200}}`,
		"reversed bounds": `{"NVIDIA-GB200":{"minWatts":1200,"defaultWatts":200,"maxWatts":1200}}`,
		"padded product":  `{" NVIDIA-GB200":{"minWatts":200,"defaultWatts":1200,"maxWatts":1200}}`,
		"trailing data":   `{} {}`,
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := ParseQualificationCatalogJSON(raw); err == nil {
				t.Fatalf("ParseQualificationCatalogJSON(%q) succeeded", raw)
			}
		})
	}
}
