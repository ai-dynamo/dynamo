/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package powerbudget

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strings"
)

// ErrUnqualifiedHardware classifies missing or drifted hardware qualification.
var ErrUnqualifiedHardware = errors.New("unqualified hardware")

// QualificationCatalogEnv carries the immutable process-start catalog from
// the Operator deployment into the DGD controller. The platform chart renders
// this value from operator-owned Helm values; an absent/empty catalog remains
// a non-nil fail-closed provider.
const QualificationCatalogEnv = "DYNAMO_POWER_QUALIFICATION_CATALOG"

// SKUConstraints are the qualified settable limits for one GPU class.
type SKUConstraints struct {
	MinWatts     int64 `json:"minWatts"`
	DefaultWatts int64 `json:"defaultWatts"`
	MaxWatts     int64 `json:"maxWatts"`
}

// QualificationIndex contains operator-owned constraints indexed by GPU class.
type QualificationIndex map[string]SKUConstraints

// ParseQualificationCatalogJSON parses the process-start production catalog.
// Empty input yields an empty, non-nil provider so transactional workloads fail
// closed as unqualified instead of falling back to caller-constructed data.
func ParseQualificationCatalogJSON(raw string) (QualificationIndex, error) {
	index := QualificationIndex{}
	if strings.TrimSpace(raw) == "" {
		return index, nil
	}

	decoder := json.NewDecoder(bytes.NewBufferString(raw))
	decoder.DisallowUnknownFields()
	if err := decoder.Decode(&index); err != nil {
		return nil, fmt.Errorf("decode qualification catalog: %w", err)
	}
	if index == nil {
		return nil, fmt.Errorf("decode qualification catalog: catalog must be a JSON object, not null")
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return nil, fmt.Errorf("decode qualification catalog: trailing JSON data")
	}

	for product, constraints := range index {
		if product == "" || strings.TrimSpace(product) != product {
			return nil, fmt.Errorf("qualification catalog product %q must be nonempty and whitespace-normalized", product)
		}
		if constraints.MinWatts < 1 ||
			constraints.DefaultWatts < constraints.MinWatts ||
			constraints.MaxWatts < constraints.DefaultWatts {
			return nil, fmt.Errorf(
				"qualification catalog product %q has invalid min/default/max [%d,%d,%d]",
				product,
				constraints.MinWatts,
				constraints.DefaultWatts,
				constraints.MaxWatts,
			)
		}
	}
	return index, nil
}

// ComponentBounds are the conservative per-GPU charges for one component.
type ComponentBounds struct {
	InGateWatts       int64
	UnenforcedWatts   int64
	QualifiedMinWatts int64
	QualifiedMaxWatts int64
}

// QualifiedBounds computes B_c and U_c for the one exact eligible GPU product.
func (index QualificationIndex) QualifiedBounds(requestedCapWatts int64, eligibleSKUs []string) (ComponentBounds, error) {
	if requestedCapWatts < 1 {
		return ComponentBounds{}, fmt.Errorf("requested cap must be at least 1 watt, got %d", requestedCapWatts)
	}
	if len(eligibleSKUs) != 1 || strings.TrimSpace(eligibleSKUs[0]) == "" {
		return ComponentBounds{}, fmt.Errorf(
			"%w: exactly one nonempty eligible GPU product is required, got %d",
			ErrUnqualifiedHardware,
			len(eligibleSKUs),
		)
	}

	sku := eligibleSKUs[0]
	constraints, found := index[sku]
	if !found {
		return ComponentBounds{}, fmt.Errorf("%w: GPU product %q has no qualification", ErrUnqualifiedHardware, sku)
	}
	if constraints.MinWatts < 1 || constraints.MaxWatts < constraints.MinWatts {
		return ComponentBounds{}, fmt.Errorf(
			"%w: GPU product %q has invalid constraints [%d,%d]",
			ErrUnqualifiedHardware,
			sku,
			constraints.MinWatts,
			constraints.MaxWatts,
		)
	}

	clamped := min(max(requestedCapWatts, constraints.MinWatts), constraints.MaxWatts)
	return ComponentBounds{
		InGateWatts:       clamped,
		UnenforcedWatts:   constraints.MaxWatts,
		QualifiedMinWatts: constraints.MinWatts,
		QualifiedMaxWatts: constraints.MaxWatts,
	}, nil
}

// ValidateLiveRange rejects any assigned live constraint pair that differs
// from the product-qualified catalog range.
func (bounds ComponentBounds) ValidateLiveRange(liveMinWatts, liveMaxWatts int64) error {
	if liveMinWatts != bounds.QualifiedMinWatts || liveMaxWatts != bounds.QualifiedMaxWatts {
		return fmt.Errorf(
			"%w: live range [%d,%d] differs from qualified range [%d,%d]",
			ErrUnqualifiedHardware,
			liveMinWatts,
			liveMaxWatts,
			bounds.QualifiedMinWatts,
			bounds.QualifiedMaxWatts,
		)
	}
	return nil
}
