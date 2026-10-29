// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package dynamodb

import (
	"bytes"
	"context"
	"fmt"
	"maps"
	"math/big"
	"slices"
	"strings"
	"sync"
	"unicode"

	"github.com/aws/aws-sdk-go-v2/aws"
	ddb "github.com/aws/aws-sdk-go-v2/service/dynamodb"
	ddbtypes "github.com/aws/aws-sdk-go-v2/service/dynamodb/types"
	"github.com/aws/smithy-go"
)

// fakeDB is an in-memory DynamoDB with the request semantics the store
// relies on: condition, update, key condition and projection expressions
// with placeholders (unused placeholders are rejected as DynamoDB does),
// ALL_OLD on failed conditions, ALL_NEW updates, paginated queries and
// BatchWriteItem limits. The dynamodb-local integration test checks the same
// store against the real semantics.
type fakeDB struct {
	mu    sync.Mutex
	items map[string]map[string]ddbtypes.AttributeValue
	// pageSize is the number of items per query page.
	pageSize int
	// unprocessed makes BatchWriteItem leave this many requests of each
	// call unprocessed while it is positive; every call decrements it.
	unprocessed int
	// before runs before every operation with the operation name and may
	// fail it.
	before func(op string) error
	calls  map[string]int
}

var _ API = (*fakeDB)(nil)

func newFakeDB() *fakeDB {
	return &fakeDB{items: map[string]map[string]ddbtypes.AttributeValue{}, pageSize: 2, calls: map[string]int{}}
}

func apiError(code, msg string) error {
	return &smithy.GenericAPIError{Code: code, Message: msg}
}

func validation(format string, args ...any) error {
	return apiError("ValidationException", fmt.Sprintf(format, args...))
}

func itemID(av map[string]ddbtypes.AttributeValue) string {
	return avString(av[attrPK]) + "\x00" + avString(av[attrSK])
}

func avString(v ddbtypes.AttributeValue) string {
	if s, ok := v.(*ddbtypes.AttributeValueMemberS); ok {
		return s.Value
	}
	return ""
}

func (f *fakeDB) start(op string) error {
	f.calls[op]++
	if f.before != nil {
		return f.before(op)
	}
	return nil
}

func (f *fakeDB) get(k map[string]ddbtypes.AttributeValue) map[string]ddbtypes.AttributeValue {
	if it, ok := f.items[itemID(k)]; ok {
		return maps.Clone(it)
	}
	return nil
}

func (f *fakeDB) put(it map[string]ddbtypes.AttributeValue) error {
	if n := itemSize(it); n > maxItemBytes {
		return validation("Item size has exceeded the maximum allowed size")
	}
	f.items[itemID(it)] = maps.Clone(it)
	return nil
}

func (f *fakeDB) GetItem(_ context.Context, in *ddb.GetItemInput, _ ...func(*ddb.Options)) (*ddb.GetItemOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.start("GetItem"); err != nil {
		return nil, err
	}
	return &ddb.GetItemOutput{Item: f.get(in.Key)}, nil
}

func (f *fakeDB) PutItem(_ context.Context, in *ddb.PutItemInput, _ ...func(*ddb.Options)) (*ddb.PutItemOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.start("PutItem"); err != nil {
		return nil, err
	}
	old := f.get(in.Item)
	if err := f.check(old, in.ConditionExpression, in.ExpressionAttributeNames, in.ExpressionAttributeValues, in.ReturnValuesOnConditionCheckFailure); err != nil {
		return nil, err
	}
	return &ddb.PutItemOutput{}, f.put(in.Item)
}

func (f *fakeDB) UpdateItem(_ context.Context, in *ddb.UpdateItemInput, _ ...func(*ddb.Options)) (*ddb.UpdateItemOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.start("UpdateItem"); err != nil {
		return nil, err
	}
	old := f.get(in.Key)
	p := placeholders{names: in.ExpressionAttributeNames, values: in.ExpressionAttributeValues}
	set, remove, err := p.parseUpdate(aws.ToString(in.UpdateExpression))
	if err != nil {
		return nil, err
	}
	if err := f.check(old, in.ConditionExpression, in.ExpressionAttributeNames, in.ExpressionAttributeValues, in.ReturnValuesOnConditionCheckFailure, &p); err != nil {
		return nil, err
	}
	next := old
	if next == nil {
		next = maps.Clone(in.Key)
	}
	maps.Copy(next, set)
	for _, name := range remove {
		delete(next, name)
	}
	if err := f.put(next); err != nil {
		return nil, err
	}
	out := &ddb.UpdateItemOutput{}
	if in.ReturnValues == ddbtypes.ReturnValueAllNew {
		out.Attributes = maps.Clone(next)
	}
	return out, nil
}

func (f *fakeDB) DeleteItem(_ context.Context, in *ddb.DeleteItemInput, _ ...func(*ddb.Options)) (*ddb.DeleteItemOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.start("DeleteItem"); err != nil {
		return nil, err
	}
	delete(f.items, itemID(in.Key))
	return &ddb.DeleteItemOutput{}, nil
}

func (f *fakeDB) Query(_ context.Context, in *ddb.QueryInput, _ ...func(*ddb.Options)) (*ddb.QueryOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.start("Query"); err != nil {
		return nil, err
	}
	p := placeholders{names: in.ExpressionAttributeNames, values: in.ExpressionAttributeValues}
	cond, err := p.parseCondition(aws.ToString(in.KeyConditionExpression))
	if err != nil {
		return nil, err
	}
	var projection []string
	if in.ProjectionExpression != nil {
		for ph := range strings.SplitSeq(*in.ProjectionExpression, ",") {
			name, err := p.name(strings.TrimSpace(ph))
			if err != nil {
				return nil, err
			}
			projection = append(projection, name)
		}
	}
	if err := p.checkUsed(); err != nil {
		return nil, err
	}
	var ids []string
	for id, it := range f.items {
		if cond.eval(it) {
			ids = append(ids, id)
		}
	}
	slices.Sort(ids)
	if in.ExclusiveStartKey != nil {
		start := itemID(in.ExclusiveStartKey)
		ids = slices.DeleteFunc(ids, func(id string) bool { return id <= start })
	}
	out := &ddb.QueryOutput{}
	if len(ids) > f.pageSize {
		ids = ids[:f.pageSize]
		last := f.items[ids[len(ids)-1]]
		out.LastEvaluatedKey = map[string]ddbtypes.AttributeValue{attrPK: last[attrPK], attrSK: last[attrSK]}
	}
	for _, id := range ids {
		it := maps.Clone(f.items[id])
		if projection != nil {
			for name := range it {
				if !slices.Contains(projection, name) {
					delete(it, name)
				}
			}
		}
		out.Items = append(out.Items, it)
	}
	return out, nil
}

func (f *fakeDB) BatchWriteItem(_ context.Context, in *ddb.BatchWriteItemInput, _ ...func(*ddb.Options)) (*ddb.BatchWriteItemOutput, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if err := f.start("BatchWriteItem"); err != nil {
		return nil, err
	}
	out := &ddb.BatchWriteItemOutput{UnprocessedItems: map[string][]ddbtypes.WriteRequest{}}
	for table, reqs := range in.RequestItems {
		if len(reqs) > batchSize {
			return nil, validation("Too many items requested for the BatchWriteItem call")
		}
		seen := map[string]bool{}
		for _, r := range reqs {
			id := itemID(r.DeleteRequest.Key)
			if seen[id] {
				return nil, validation("Provided list of item keys contains duplicates")
			}
			seen[id] = true
		}
		skip := 0
		if f.unprocessed > 0 {
			skip = min(f.unprocessed, len(reqs))
			out.UnprocessedItems[table] = reqs[:skip]
		}
		for _, r := range reqs[skip:] {
			delete(f.items, itemID(r.DeleteRequest.Key))
		}
	}
	if f.unprocessed > 0 {
		f.unprocessed--
	}
	return out, nil
}

// check evaluates a condition against the stored item (nil when absent)
// and returns ConditionalCheckFailedException, with the item for ALL_OLD,
// when it does not hold. p, when given, already holds placeholders used by
// other expressions of the request.
func (f *fakeDB) check(old map[string]ddbtypes.AttributeValue, cond *string, names map[string]string, values map[string]ddbtypes.AttributeValue, rv ddbtypes.ReturnValuesOnConditionCheckFailure, p ...*placeholders) error {
	ph := &placeholders{names: names, values: values}
	if len(p) > 0 {
		ph = p[0]
	}
	if cond != nil {
		c, err := ph.parseCondition(*cond)
		if err != nil {
			return err
		}
		if err := ph.checkUsed(); err != nil {
			return err
		}
		if !c.eval(old) {
			e := &ddbtypes.ConditionalCheckFailedException{Message: aws.String("The conditional request failed")}
			if rv == ddbtypes.ReturnValuesOnConditionCheckFailureAllOld {
				e.Item = old
			}
			return e
		}
		return nil
	}
	return ph.checkUsed()
}

// placeholders resolves #name and :value placeholders and records which
// were used.
type placeholders struct {
	names      map[string]string
	values     map[string]ddbtypes.AttributeValue
	usedNames  map[string]bool
	usedValues map[string]bool
}

func (p *placeholders) name(ph string) (string, error) {
	n, ok := p.names[ph]
	if !ok {
		return "", validation("An expression attribute name used in the document path is not defined; attribute name: %s", ph)
	}
	if p.usedNames == nil {
		p.usedNames = map[string]bool{}
	}
	p.usedNames[ph] = true
	return n, nil
}

func (p *placeholders) value(ph string) (ddbtypes.AttributeValue, error) {
	v, ok := p.values[ph]
	if !ok {
		return nil, validation("An expression attribute value used in expression is not defined; attribute value: %s", ph)
	}
	if p.usedValues == nil {
		p.usedValues = map[string]bool{}
	}
	p.usedValues[ph] = true
	return v, nil
}

func (p *placeholders) checkUsed() error {
	for ph := range p.names {
		if !p.usedNames[ph] {
			return validation("Value provided in ExpressionAttributeNames unused in expressions: keys: {%s}", ph)
		}
	}
	for ph := range p.values {
		if !p.usedValues[ph] {
			return validation("Value provided in ExpressionAttributeValues unused in expressions: keys: {%s}", ph)
		}
	}
	return nil
}

// parseUpdate parses "SET #a = :v, ... REMOVE #b, ...".
func (p *placeholders) parseUpdate(s string) (map[string]ddbtypes.AttributeValue, []string, error) {
	set := map[string]ddbtypes.AttributeValue{}
	var remove []string
	toks := tokenize(s)
	section := ""
	for i := 0; i < len(toks); i++ {
		switch t := toks[i]; {
		case t == "SET" || t == "REMOVE":
			section = t
		case t == ",":
		case section == "SET" && i+2 < len(toks) && toks[i+1] == "=":
			name, err := p.name(t)
			if err != nil {
				return nil, nil, err
			}
			v, err := p.value(toks[i+2])
			if err != nil {
				return nil, nil, err
			}
			set[name] = v
			i += 2
		case section == "REMOVE":
			name, err := p.name(t)
			if err != nil {
				return nil, nil, err
			}
			remove = append(remove, name)
		default:
			return nil, nil, validation("Invalid UpdateExpression: %q", s)
		}
	}
	return set, remove, nil
}

func tokenize(s string) []string {
	var toks []string
	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == ' ':
			i++
		case strings.ContainsRune("(),=", rune(c)):
			toks = append(toks, string(c))
			i++
		case c == '<' || c == '>':
			if i+1 < len(s) && (s[i+1] == '=' || s[i+1] == '>') {
				toks = append(toks, s[i:i+2])
				i += 2
			} else {
				toks = append(toks, string(c))
				i++
			}
		default:
			j := i
			for j < len(s) && (s[j] == '#' || s[j] == ':' || s[j] == '_' || unicode.IsLetter(rune(s[j])) || unicode.IsDigit(rune(s[j]))) {
				j++
			}
			if j == i {
				j++
			}
			toks = append(toks, s[i:j])
			i = j
		}
	}
	return toks
}

// cond is a parsed condition.
type cond interface {
	eval(it map[string]ddbtypes.AttributeValue) bool
}

type condFunc func(it map[string]ddbtypes.AttributeValue) bool

func (f condFunc) eval(it map[string]ddbtypes.AttributeValue) bool { return f(it) }

type parser struct {
	p    *placeholders
	toks []string
	pos  int
}

func (p *placeholders) parseCondition(s string) (cond, error) {
	ps := &parser{p: p, toks: tokenize(s)}
	c, err := ps.or()
	if err != nil {
		return nil, err
	}
	if ps.pos != len(ps.toks) {
		return nil, validation("Invalid ConditionExpression: %q", s)
	}
	return c, nil
}

func (ps *parser) peek() string {
	if ps.pos < len(ps.toks) {
		return ps.toks[ps.pos]
	}
	return ""
}

func (ps *parser) expect(tok string) error {
	if ps.peek() != tok {
		return validation("Syntax error: want %q at token %d of %v", tok, ps.pos, ps.toks)
	}
	ps.pos++
	return nil
}

func (ps *parser) or() (cond, error) {
	left, err := ps.and()
	if err != nil {
		return nil, err
	}
	for ps.peek() == "OR" {
		ps.pos++
		right, err := ps.and()
		if err != nil {
			return nil, err
		}
		l := left
		left = condFunc(func(it map[string]ddbtypes.AttributeValue) bool { return l.eval(it) || right.eval(it) })
	}
	return left, nil
}

func (ps *parser) and() (cond, error) {
	left, err := ps.unary()
	if err != nil {
		return nil, err
	}
	for ps.peek() == "AND" {
		ps.pos++
		right, err := ps.unary()
		if err != nil {
			return nil, err
		}
		l := left
		left = condFunc(func(it map[string]ddbtypes.AttributeValue) bool { return l.eval(it) && right.eval(it) })
	}
	return left, nil
}

func (ps *parser) unary() (cond, error) {
	switch tok := ps.peek(); tok {
	case "(":
		ps.pos++
		c, err := ps.or()
		if err != nil {
			return nil, err
		}
		return c, ps.expect(")")
	case "attribute_exists", "attribute_not_exists":
		ps.pos++
		if err := ps.expect("("); err != nil {
			return nil, err
		}
		name, err := ps.p.name(ps.peek())
		if err != nil {
			return nil, err
		}
		ps.pos++
		want := tok == "attribute_exists"
		return condFunc(func(it map[string]ddbtypes.AttributeValue) bool {
			_, ok := it[name]
			return ok == want
		}), ps.expect(")")
	case "begins_with":
		ps.pos++
		if err := ps.expect("("); err != nil {
			return nil, err
		}
		left, err := ps.operand()
		if err != nil {
			return nil, err
		}
		if err := ps.expect(","); err != nil {
			return nil, err
		}
		right, err := ps.operand()
		if err != nil {
			return nil, err
		}
		return condFunc(func(it map[string]ddbtypes.AttributeValue) bool {
			l, r := left(it), right(it)
			return avString(l) != "" && strings.HasPrefix(avString(l), avString(r))
		}), ps.expect(")")
	}
	left, err := ps.operand()
	if err != nil {
		return nil, err
	}
	op := ps.peek()
	ps.pos++
	right, err := ps.operand()
	if err != nil {
		return nil, err
	}
	return condFunc(func(it map[string]ddbtypes.AttributeValue) bool {
		c, ok := compare(left(it), right(it))
		if !ok {
			return false
		}
		switch op {
		case "=":
			return c == 0
		case "<>":
			return c != 0
		case "<":
			return c < 0
		case "<=":
			return c <= 0
		case ">":
			return c > 0
		case ">=":
			return c >= 0
		}
		return false
	}), nil
}

type operand func(it map[string]ddbtypes.AttributeValue) ddbtypes.AttributeValue

func (ps *parser) operand() (operand, error) {
	tok := ps.peek()
	ps.pos++
	switch {
	case strings.HasPrefix(tok, "#"):
		name, err := ps.p.name(tok)
		if err != nil {
			return nil, err
		}
		return func(it map[string]ddbtypes.AttributeValue) ddbtypes.AttributeValue { return it[name] }, nil
	case strings.HasPrefix(tok, ":"):
		v, err := ps.p.value(tok)
		if err != nil {
			return nil, err
		}
		return func(map[string]ddbtypes.AttributeValue) ddbtypes.AttributeValue { return v }, nil
	}
	return nil, validation("Syntax error: unexpected %q", tok)
}

// compare orders two values of the same type; ok is false for missing
// values and type mismatches.
func compare(a, b ddbtypes.AttributeValue) (int, bool) {
	switch a := a.(type) {
	case *ddbtypes.AttributeValueMemberS:
		if b, ok := b.(*ddbtypes.AttributeValueMemberS); ok {
			return strings.Compare(a.Value, b.Value), true
		}
	case *ddbtypes.AttributeValueMemberB:
		if b, ok := b.(*ddbtypes.AttributeValueMemberB); ok {
			return bytes.Compare(a.Value, b.Value), true
		}
	case *ddbtypes.AttributeValueMemberN:
		if b, ok := b.(*ddbtypes.AttributeValueMemberN); ok {
			x, okx := new(big.Int).SetString(a.Value, 10)
			y, oky := new(big.Int).SetString(b.Value, 10)
			if okx && oky {
				return x.Cmp(y), true
			}
		}
	}
	return 0, false
}
