// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamodb

import (
	"strconv"
	"strings"
	"time"

	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
)

// expr collects the attribute name and value placeholders of one request.
// DynamoDB rejects placeholders a request does not use, so every request
// builds its own.
type expr struct {
	names  map[string]string
	values map[string]ddbtypes.AttributeValue
}

// name returns the placeholder of an attribute name. Every attribute is
// referenced through a placeholder so that reserved words never matter.
func (e *expr) name(attr string) string {
	if e.names == nil {
		e.names = map[string]string{}
	}
	ph := "#" + attr
	e.names[ph] = attr
	return ph
}

// value returns the placeholder of a value. Placeholder names must be
// unique within one request.
func (e *expr) value(name string, v ddbtypes.AttributeValue) string {
	if e.values == nil {
		e.values = map[string]ddbtypes.AttributeValue{}
	}
	ph := ":" + name
	e.values[ph] = v
	return ph
}

// update builds an update expression from SET and REMOVE clauses.
type update struct {
	set    []string
	remove []string
}

func (u *update) setValue(e *expr, attr string, v ddbtypes.AttributeValue) {
	u.set = append(u.set, e.name(attr)+" = "+e.value(attr, v))
}

// setTime sets attr to t in Unix nanoseconds, or removes it for a zero t.
func (u *update) setTime(e *expr, attr string, t time.Time) {
	if n := unixNanos(t); n != nil {
		u.setValue(e, attr, numI(*n))
		return
	}
	u.remove = append(u.remove, e.name(attr))
}

// setExpiry sets expires_at, or leaves it alone when expiry is disabled.
func (u *update) setExpiry(e *expr, expiresAt int64) {
	if expiresAt != 0 {
		u.setValue(e, attrExpiresAt, numI(expiresAt))
	}
}

func (u *update) String() string {
	var b strings.Builder
	if len(u.set) > 0 {
		b.WriteString("SET ")
		b.WriteString(strings.Join(u.set, ", "))
	}
	if len(u.remove) > 0 {
		if b.Len() > 0 {
			b.WriteString(" ")
		}
		b.WriteString("REMOVE ")
		b.WriteString(strings.Join(u.remove, ", "))
	}
	return b.String()
}

// Conditions.

func notExists(e *expr, attr string) string {
	return "attribute_not_exists(" + e.name(attr) + ")"
}

// genIs requires the stored generation the caller read.
func genIs(e *expr, gen uint64) string {
	return e.name(attrGen) + " = " + e.value("seen_gen", numU(gen))
}

// Attribute values.

func str(s string) ddbtypes.AttributeValue {
	return &ddbtypes.AttributeValueMemberS{Value: s}
}

func numU(n uint64) ddbtypes.AttributeValue {
	return &ddbtypes.AttributeValueMemberN{Value: strconv.FormatUint(n, 10)}
}

func numI(n int64) ddbtypes.AttributeValue {
	return &ddbtypes.AttributeValueMemberN{Value: strconv.FormatInt(n, 10)}
}

func bin(b []byte) ddbtypes.AttributeValue {
	return &ddbtypes.AttributeValueMemberB{Value: b}
}

func (k key) attributes() map[string]ddbtypes.AttributeValue {
	return map[string]ddbtypes.AttributeValue{attrPK: str(k.pk), attrSK: str(k.sk)}
}

// itemSize estimates the DynamoDB size of an item: attribute names plus
// value lengths. It covers the S, N and B values this store writes.
func itemSize(av map[string]ddbtypes.AttributeValue) int {
	n := 0
	for name, v := range av {
		n += len(name)
		switch v := v.(type) {
		case *ddbtypes.AttributeValueMemberS:
			n += len(v.Value)
		case *ddbtypes.AttributeValueMemberN:
			n += len(v.Value)
		case *ddbtypes.AttributeValueMemberB:
			n += len(v.Value)
		default:
			n += 8
		}
	}
	return n
}
