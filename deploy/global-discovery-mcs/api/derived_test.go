// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"regexp"
	"testing"

	"github.com/stretchr/testify/assert"
)

// Golden values computed independently with Python
// (base64.b32hexencode(hashlib.sha256(b"ns/name").digest())) and with the
// derivedName function of sigs.k8s.io/mcs-api v0.5.x controllers/common.go.
var derivedGolden = []struct {
	namespace, name, want string
}{
	{"default", "nginx", "derived-iuq6ea9f4j"},
	{"llm", "llm-request-router-backend", "derived-b756q8b74k"},
	{"llm-stack", "backend-router", "derived-t1d4et30ia"},
	{"dynamo", "pool-frontend", "derived-apb6bcevn9"},
	{"a", "b", "derived-o56drg1jup"},
	{"kube-system", "kube-dns", "derived-bnmpvejc1s"},
	{"", "", "derived-h9fdlck2cc"},
}

func TestDerivedServiceNameGolden(t *testing.T) {
	for _, tt := range derivedGolden {
		assert.Equal(t, tt.want, DerivedServiceName(tt.namespace, tt.name), "%s/%s", tt.namespace, tt.name)
	}
}

func TestDerivedServiceNameIsDNSLabel(t *testing.T) {
	label := regexp.MustCompile(`^derived-[0-9a-v]{10}$`)
	for _, tt := range derivedGolden {
		got := DerivedServiceName(tt.namespace, tt.name)
		assert.Regexp(t, label, got)
		assert.Len(t, got, len(DerivedServicePrefix)+10)
	}
}

// The separator is part of the hashed input, so moving characters between the
// namespace and the name changes the result.
func TestDerivedServiceNameSeparator(t *testing.T) {
	assert.NotEqual(t, DerivedServiceName("ab", "c"), DerivedServiceName("a", "bc"))
	assert.Equal(t, DerivedServiceName("ns", "svc"), DerivedServiceName("ns", "svc"))
}
