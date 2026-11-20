/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
)

const (
	workloadSetDigestVersion = "dynamo-lpx-compilation-set/v1"
)

// WorkloadDigest is the SHA-256 identity of a projected LPX workload.
type WorkloadDigest [32]byte

// String returns the digest in canonical sha256-prefixed hexadecimal form.
func (d WorkloadDigest) String() string {
	return fmt.Sprintf("sha256:%x", d[:])
}

func workloadSetDigest(projections []*ModelProjection) (WorkloadDigest, error) {
	first := projections[0]
	if len(projections) == 1 {
		return first.digest, nil
	}
	transcript := &digestTranscript{hash: sha256.New()}
	transcript.field("schema", []byte(workloadSetDigestVersion))
	for _, projection := range projections {
		if projection.configuredBuild.architecture != first.configuredBuild.architecture {
			return WorkloadDigest{}, fmt.Errorf(
				"LPX model projections have mixed target architectures %q and %q",
				first.configuredBuild.architecture.String(),
				projection.configuredBuild.architecture.String(),
			)
		}
		transcript.field("model", []byte(projection.model))
		// The field tags are digest inputs: a rename rolls every multi-model workload.
		transcript.field("compilation-digest", projection.digest[:])
	}
	return transcript.sum(), nil
}

type digestTranscript struct {
	hash hash.Hash
}

func (t *digestTranscript) field(tag string, value []byte) {
	writeLengthDelimited(t.hash, []byte(tag))
	writeLengthDelimited(t.hash, value)
}

func (t *digestTranscript) sum() WorkloadDigest {
	var digest WorkloadDigest
	copy(digest[:], t.hash.Sum(nil))
	return digest
}

func writeLengthDelimited(writer hash.Hash, value []byte) {
	var length [8]byte
	binary.BigEndian.PutUint64(length[:], uint64(len(value)))
	_, _ = writer.Write(length[:])
	_, _ = writer.Write(value)
}
