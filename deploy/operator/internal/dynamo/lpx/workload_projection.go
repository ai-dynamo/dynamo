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
	modelProjectionDigestVersion = "dynamo-node-local-compilation/v1"
	workloadSetDigestVersion     = "dynamo-lpx-compilation-set/v1"
	maxLPXPartitions             = 256
)

// projectComponent projects one DGD component's normalized, non-nil build into
// its models, in the order of the nonempty models list. The build is not mutated.
func projectComponent(name, runtimeBuildRef string, source *Build, pipeline Pipeline, models []string) ([]*Model, error) {
	// family projections record their model-independent digest fields once.
	var fields bytes.Buffer
	projected, err := source.family.project(source, pipeline, digestTranscript{&fields})
	if err != nil {
		return nil, err
	}
	// Bound the component's physical build before runtime expansion and request publication.
	if len(projected.partitionRequests) < 1 || len(projected.partitionRequests) > maxLPXPartitions {
		return nil, fmt.Errorf("LPX projection has %d partitions, limit is 1..%d", len(projected.partitionRequests), maxLPXPartitions)
	}

	// Publish distinct logical identities backed by the component's immutable configuration.
	projected.name, projected.runtimeBuildRef, projected.pipeline = name, runtimeBuildRef, pipeline
	result := make([]*Model, len(models))
	for index, model := range models {
		result[index] = &Model{name: model, digest: modelProjectionDigest(projected, model, fields.Bytes()), component: projected}
	}
	return result, nil
}

func workloadSetDigest(models []*Model) (WorkloadDigest, error) {
	first := models[0]
	if len(models) == 1 {
		return first.digest, nil
	}
	hash := sha256.New()
	transcript := digestTranscript{hash}
	transcript.field("schema", []byte(workloadSetDigestVersion))
	for _, model := range models {
		if model.component.configuredBuild.family != first.component.configuredBuild.family {
			return WorkloadDigest{}, fmt.Errorf(
				"LPX model projections have mixed target families %q and %q",
				first.component.configuredBuild.family.target,
				model.component.configuredBuild.family.target,
			)
		}
		transcript.field("model", []byte(model.name))
		transcript.field("compilation-digest", model.digest[:])
	}
	return sumDigest(hash), nil
}

// modelProjectionDigest hashes one logical model's identity header followed by
// its component's model-independent fields.
func modelProjectionDigest(projected *component, model string, fields []byte) WorkloadDigest {
	hash := sha256.New()
	transcript := digestTranscript{hash}
	transcript.field("schema", []byte(modelProjectionDigestVersion))
	transcript.field("lowerer", []byte(projected.configuredBuild.family.projectionVersion))
	// Ref is an acquisition locator, not build content. In particular, a
	// file-backed snapshot's ref contains its absolute checkout path.
	transcript.field("build-content-id", []byte(projected.configuredBuild.contentID))
	// Bind the family and pipeline directly; the pipeline already determines runtime mode.
	transcript.field("family", []byte(projected.configuredBuild.family.target))
	transcript.field("pipeline", []byte(projected.pipeline))
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
