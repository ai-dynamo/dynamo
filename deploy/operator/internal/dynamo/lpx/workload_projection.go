/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"hash"
	"io"
)

const (
	// Digest domain strings and field tags are part of workload identity; changing
	// them changes every projected digest.
	modelProjectionDigestVersion = "dynamo-lpx-model-projection/v1"
	workloadSetDigestVersion     = "dynamo-lpx-workload/v1"
	maxLPXPartitions             = 256
)

// appendModelProjections appends one component's immutable model projections to
// the caller-owned destination, which may be nil. Existing elements are unchanged.
// intent.Build contains a normalized, non-nil build; Models is nonempty
// and Pipeline is selected by the validated resolver. Source inputs are not mutated;
// discard error results.
func appendModelProjections(dst []*ModelProjection, intent ModelProjectionInput) ([]*ModelProjection, error) {
	// Family projections record their model-independent digest fields once.
	var (
		fields            bytes.Buffer
		component         ModelProjection
		projectionVersion string
		err               error
	)
	switch intent.Build.family {
	case BuildFamilyXT:
		projectionVersion = xtProjectionVersion
		component, err = projectXTComponent(intent, digestTranscript{&fields})
	case BuildFamilyHX:
		projectionVersion = hxProjectionVersion
		component, err = projectHXComponent(intent, digestTranscript{&fields})
	default:
		return nil, fmt.Errorf("unsupported LPX target family %q", intent.Build.family)
	}
	if err != nil {
		return nil, err
	}
	// Bound the component's physical build before runtime expansion and request publication.
	boundedPartitions := len(component.partitions)
	if len(component.localPartitionIDs) > 0 {
		boundedPartitions = len(intent.Build.partitions)
	} else if boundedPartitions < 1 {
		return nil, fmt.Errorf("LPX projection has %d partitions, limit is 1..%d", boundedPartitions, maxLPXPartitions)
	}
	if boundedPartitions > maxLPXPartitions {
		return nil, fmt.Errorf("LPX projection has %d partitions, limit is 1..%d", boundedPartitions, maxLPXPartitions)
	}

	// Publish distinct logical identities backed by the component's immutable configuration.
	component.contractFormat = intent.Build.contractFormat
	component.compilerSnapshotDigest = intent.Build.contentID
	component.runtimeBuildRef = intent.RuntimeBuildRef
	component.pipeline = intent.Pipeline
	for _, model := range intent.Models {
		projection := component
		projection.model = model
		projection.digest = modelProjectionDigest(intent, projectionVersion, model, fields.Bytes())
		dst = append(dst, &projection)
	}
	return dst, nil
}

func workloadSetDigest(projections []*ModelProjection) (WorkloadDigest, error) {
	first := projections[0]
	if len(projections) == 1 {
		return first.digest, nil
	}
	hash := sha256.New()
	transcript := digestTranscript{hash}
	transcript.field("schema", []byte(workloadSetDigestVersion))
	for _, projection := range projections {
		if projection.configuredBuild.family != first.configuredBuild.family {
			return WorkloadDigest{}, fmt.Errorf(
				"LPX model projections have mixed target families %q and %q",
				first.configuredBuild.family,
				projection.configuredBuild.family,
			)
		}
		transcript.field("model", []byte(projection.model))
		transcript.field("model-projection-digest", projection.digest[:])
	}
	return sumDigest(hash), nil
}

// modelProjectionDigest hashes one logical model's identity header followed by
// its component's model-independent fields.
func modelProjectionDigest(intent ModelProjectionInput, projectionVersion, model string, fields []byte) WorkloadDigest {
	hash := sha256.New()
	transcript := digestTranscript{hash}
	transcript.field("schema", []byte(modelProjectionDigestVersion))
	transcript.field("projection-version", []byte(projectionVersion))
	// Ref is an acquisition locator, not build content. In particular, a
	// file-backed snapshot's ref contains its absolute checkout path.
	transcript.field("build-content-id", []byte(intent.Build.contentID))
	// Bind the family and pipeline directly; the pipeline already determines runtime mode.
	transcript.field("family", []byte(intent.Build.family))
	transcript.field("pipeline", []byte(intent.Pipeline))
	transcript.field("model", []byte(model))
	_, _ = hash.Write(fields)
	return sumDigest(hash)
}

// bindHybridRuntimeIO records the shared Cyborg I/O contract in a model projection digest.
func bindHybridRuntimeIO(
	transcript digestTranscript,
	pipeline Pipeline,
	ioFPGACount int32,
	ioFanoutFactor int32,
) {
	if pipeline != PipelineHybrid {
		return
	}

	transcript.intField("io-fpga-count", int64(ioFPGACount))
	if ioFanoutFactor > 1 {
		transcript.intField("io-fanout-factor", int64(ioFanoutFactor))
	}
}

// digestTranscript writes length-delimited tagged fields.
type digestTranscript struct {
	writer io.Writer
}

func (t digestTranscript) field(tag string, value []byte) {
	writeLengthDelimited(t.writer, []byte(tag))
	writeLengthDelimited(t.writer, value)
}

func (t digestTranscript) uint32Field(tag string, value uint32) {
	var encoded [4]byte
	binary.BigEndian.PutUint32(encoded[:], value)
	t.field(tag, encoded[:])
}

func (t digestTranscript) intField(tag string, value int64) {
	var encoded [8]byte
	binary.BigEndian.PutUint64(encoded[:], uint64(value))
	t.field(tag, encoded[:])
}

func sumDigest(hash hash.Hash) WorkloadDigest {
	var digest WorkloadDigest
	copy(digest[:], hash.Sum(nil))
	return digest
}

func writeLengthDelimited(writer io.Writer, value []byte) {
	var length [8]byte
	binary.BigEndian.PutUint64(length[:], uint64(len(value)))
	_, _ = writer.Write(length[:])
	_, _ = writer.Write(value)
}
