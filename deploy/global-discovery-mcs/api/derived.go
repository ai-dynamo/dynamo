// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package api

import (
	"crypto/sha256"
	"encoding/base32"
	"strings"
)

// DerivedServicePrefix starts every derived Service name.
const DerivedServicePrefix = "derived-"

var base32HexNoPad = base32.HexEncoding.WithPadding(base32.NoPadding)

// DerivedServiceName returns the name of the derived Service that an
// importing cluster creates for the ServiceImport namespace/name:
// "derived-" followed by the first ten characters of the lowercase,
// unpadded base32hex encoding of the SHA-256 of "namespace/name".
//
// The algorithm is the derivedName function of sigs.k8s.io/mcs-api
// controllers, so the name is the same in every cluster and in every MCS
// implementation that follows mcs-api. Clients compute it locally to dial an
// import, for example derived-<hash>.<namespace>.svc.cluster.local.
func DerivedServiceName(namespace, name string) string {
	sum := sha256.Sum256([]byte(namespace + "/" + name))
	return DerivedServicePrefix + strings.ToLower(base32HexNoPad.EncodeToString(sum[:]))[:10]
}
