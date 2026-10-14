# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explain retained oasdiff evidence without inferring accepted sets or runtime policy."""

import copy
import hashlib
import json
from collections import Counter
from functools import partial

import jsonpointer

from ..assessment.openapi import pointer


def digest(value) -> str:
    """Hash canonical JSON content; array order remains significant."""
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def identifier(kind: str, endpoint: str, field: str) -> str:
    """Reuse the historical inventory's location ID; it does not identify content or a defect."""
    key = json.dumps([kind, endpoint, field], separators=(",", ":"))
    return kind.upper() + "-" + hashlib.sha256(key.encode()).hexdigest()[:12]


def terminals(value, location: str) -> set[str]:
    """Enumerate JSON terminals, including empty containers, for lossless delta accounting."""
    if isinstance(value, (dict, list)) and value:
        pairs = value.items() if isinstance(value, dict) else enumerate(value)
        return set().union(
            *(terminals(child, location + pointer(key)[1:]) for key, child in pairs)
        )
    return {location}


def contains(parent: str, child: str) -> bool:
    """Return whether one JSON Pointer contains another, respecting segment boundaries."""
    return child == parent or child.startswith(parent + "/")


def candidate_groups(report: dict) -> list[dict]:
    """Partition raw candidates using the historical request-field/response-status IDs.

    Preserve complete nested deltas and exact report pointers. Unknown request
    body changes get an additional operation group; no historical count is assumed.
    Repeated legacy locations get side/layer-qualified IDs. No input mutation.
    """
    groups = []

    def add(kind, endpoint, field, location, delta, raw):
        """Retain one original location group and its separately hashed content."""
        groups.append(
            dict(
                id=identifier(kind, endpoint, field),
                kind=kind,
                endpoint=endpoint,
                field=field,
                location=location,
                delta=copy.deepcopy(delta),
                delta_sha256=digest(delta),
                raw_report_pointers=raw,
            )
        )

    for index, finding in enumerate(report.get("differences", [])):
        base = pointer("differences", index, "delta")
        schema_path = base + "/content/modified/application~1json/schema"
        schema = (
            finding["delta"]
            .get("content", {})
            .get("modified", {})
            .get("application/json", {})
            .get("schema", {})
        )
        for change, fields in schema.get("properties", {}).items():
            if change not in {"added", "deleted", "modified"} or not isinstance(
                fields, (list, dict)
            ):
                continue
            for key, value in (
                fields.items() if isinstance(fields, dict) else enumerate(fields)
            ):
                field = key if isinstance(fields, dict) else value
                if not isinstance(field, str):
                    continue
                raw = schema_path + pointer("properties", change, key)[1:]
                add(
                    "request",
                    finding["endpoint"],
                    field,
                    finding["location"] + pointer("properties", field)[1:],
                    {change: value if isinstance(fields, dict) else True},
                    [raw],
                )
        remaining = {key: value for key, value in schema.items() if key != "properties"}
        if remaining:
            add(
                "request",
                finding["endpoint"],
                "$root",
                finding["location"],
                remaining,
                [schema_path + pointer(key)[1:] for key in remaining],
            )
        covered = set().union(
            *(
                terminals(jsonpointer.resolve_pointer(report, p[1:]), p)
                for group in groups
                for p in group["raw_report_pointers"]
                if contains(base, p)
            )
        )
        leftover = terminals(finding["delta"], base) - covered
        if leftover:
            add(
                "request",
                finding["endpoint"],
                "$operation",
                finding["location"],
                {
                    p: jsonpointer.resolve_pointer(report, p[1:])
                    for p in sorted(leftover)
                },
                sorted(leftover),
            )
    for index, finding in enumerate(report.get("response_differences", [])):
        for change, statuses in finding["delta"].items():
            if (
                change not in {"added", "deleted", "modified"}
                or not isinstance(statuses, (list, dict))
                or not statuses
            ):
                add(
                    "response",
                    finding["endpoint"],
                    "$" + change,
                    finding["location"],
                    statuses,
                    [pointer("response_differences", index, "delta", change)],
                )
                continue
            for key, value in (
                statuses.items() if isinstance(statuses, dict) else enumerate(statuses)
            ):
                status = key if isinstance(statuses, dict) else value
                add(
                    "response",
                    finding["endpoint"],
                    str(status),
                    finding["location"] + pointer(status)[1:],
                    {change: value if isinstance(statuses, dict) else True},
                    [pointer("response_differences", index, "delta", change, key)],
                )
        if not finding["delta"]:
            add(
                "response",
                finding["endpoint"],
                "$root",
                finding["location"],
                {},
                [pointer("response_differences", index, "delta")],
            )
    for index, gap in enumerate(report.get("coverage_gaps", [])):
        field = gap["kind"] + ":" + gap["location"] + ":" + gap.get("field", "*")
        add(
            "gap",
            gap["endpoint"],
            field,
            gap["location"],
            gap,
            [pointer("coverage_gaps", index)],
        )
    duplicates = Counter(group["id"] for group in groups)
    for group in groups:
        if duplicates[group["id"]] > 1:
            group["id"] += (
                "-"
                + digest([group["delta"].get("side"), group["delta"].get("layer")])[:8]
            )
    return sorted(
        groups,
        key=lambda group: (
            group["kind"],
            group["endpoint"],
            group["field"],
            group["id"],
        ),
    )


def locate(document: dict, root: str, steps: tuple) -> tuple[list, list]:
    """Locate declarations through local refs and explicit composition branches only.

    `steps` contains schema-key segments, such as properties/name/items. Return
    literal declaration records and traversal limitations. Do not merge allOf,
    align union alternatives, infer absent-field rejection, or fetch references.
    Cycles and external refs become limitations. A missing root returns no declaration
    at that location, not proof that a runtime field or response cannot occur.
    """
    found, notes = {}, set()

    def visit(node, location, pending, seen):
        """Visit one literal branch; bound recursive references without flattening it."""
        key = (location, pending)
        if key in seen:
            notes.add("Recursive reference at " + location)
            return
        seen = seen | {key}
        if isinstance(node, dict) and "$ref" in node:
            ref = node["$ref"]
            if not isinstance(ref, str) or not ref.startswith("#/components/schemas/"):
                notes.add("Unsupported reference at " + location)
            else:
                try:
                    visit(
                        jsonpointer.resolve_pointer(document, ref[1:]),
                        ref,
                        pending,
                        seen,
                    )
                except jsonpointer.JsonPointerException:
                    notes.add("Unresolved reference " + ref)
            if len(node) == 1:
                return
            notes.add("Reference siblings remain separate at " + location)
        if not pending:
            found[location] = {"pointer": location, "declaration": copy.deepcopy(node)}
            return
        if not isinstance(node, dict):
            return
        token = pending[0]
        if token in node:
            child = node[token]
            if token == "properties" and len(pending) > 1 and isinstance(child, dict):
                if pending[1] in child:
                    visit(
                        child[pending[1]],
                        location + pointer(token, pending[1])[1:],
                        pending[2:],
                        seen,
                    )
            elif token != "properties":
                visit(child, location + pointer(token)[1:], pending[1:], seen)
        for keyword in ("allOf", "anyOf", "oneOf"):
            for index, branch in enumerate(node.get(keyword, [])):
                visit(branch, location + pointer(keyword, index)[1:], pending, seen)

    try:
        visit(jsonpointer.resolve_pointer(document, root[1:]), root, steps, set())
    except jsonpointer.JsonPointerException:
        pass  # The retained document has no declaration at this exact location.
    return [found[key] for key in sorted(found)], sorted(notes)


def describe(schema, document: dict, depth: int = 0) -> str:
    """Summarize literal schema constraints, without interpreting their combined accepted set."""
    if isinstance(schema, bool):
        return "unconstrained" if schema else "no value allowed"
    if not isinstance(schema, dict):
        return "non-schema declaration"
    if depth >= 4:
        return "nested declaration (see source)"
    parts = []
    if "$ref" in schema:
        ref = schema["$ref"]
        try:
            target = (
                jsonpointer.resolve_pointer(document, ref[1:])
                if ref.startswith("#/components/schemas/")
                else None
            )
        except jsonpointer.JsonPointerException:
            target = None
        parts.append(
            describe(target, document, depth + 1)
            if target is not None
            else "reference " + str(ref)
        )
    if "schema" in schema:
        parts.append(describe(schema["schema"], document, depth + 1))
    if "content" in schema:
        parts.append("media: " + ", ".join(sorted(schema["content"])))
    if "description" in schema and not (
        "content" in schema or "type" in schema or "$ref" in schema
    ):
        parts.append("description present; inspect source for constraints")
    if "type" in schema:
        value = schema["type"]
        parts.append(
            "type " + (" or ".join(value) if isinstance(value, list) else str(value))
        )
    for keyword in ("allOf", "anyOf", "oneOf"):
        if keyword in schema:
            branches = schema[keyword]
            parts.append(
                keyword
                + "("
                + "; ".join(
                    describe(branch, document, depth + 1) for branch in branches[:4]
                )
                + (f"; +{len(branches) - 4} branches" if len(branches) > 4 else "")
                + ")"
            )
    if "properties" in schema:
        names = sorted(schema["properties"])
        parts.append(
            "fields: "
            + ", ".join(names[:10])
            + (f", +{len(names)-10} more" if len(names) > 10 else "")
        )
    if "items" in schema:
        parts.append("items " + describe(schema["items"], document, depth + 1))
    for keyword in (
        "const",
        "enum",
        "required",
        "default",
        "nullable",
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "pattern",
        "format",
        "additionalProperties",
    ):
        if keyword in schema:
            value = schema[keyword]
            parts.append(
                keyword
                + "="
                + (
                    "schema"
                    if keyword == "additionalProperties" and isinstance(value, dict)
                    else json.dumps(value, sort_keys=True)
                )
            )
    return "; ".join(parts) or "no directly stated constraints"


def declaration(
    document: dict | None, filename: str, root: str, steps: tuple, keyword: str | None
) -> dict:
    """Return explicit presence/value summaries and pointers; never convert absent annotations to null."""
    if document is None:
        return {
            "summary": "projection unavailable",
            "evidence": [],
            "limitations": ["Projection was not supplied"],
        }
    records, notes = locate(document, root, steps)
    summaries = []
    for record in records:
        node = record["declaration"]
        if keyword and isinstance(node, dict):
            text = (
                keyword + "=" + json.dumps(node[keyword], sort_keys=True)
                if keyword in node
                else keyword + " not directly declared"
            )
            if any(key in node for key in ("allOf", "anyOf", "oneOf")):
                text += "; composition branches remain separate"
        else:
            text = describe(node, document)
        record.update(file=filename, summary=text)
        summaries.append(text)
    summary = (
        " / ".join(dict.fromkeys(summaries))
        if summaries
        else "not declared at inspected location"
    )
    if notes:
        summary += "; source traversal incomplete"
    return {"summary": summary, "evidence": records, "limitations": notes}


def related_aliases(
    matches: list, endpoint: str, steps: tuple, exact: bool = False
) -> list:
    """Return alias-match indexes for this request instance path, not matching leaf names.

    Schema steps must contain only properties/name and items pairs. With exact=False,
    also link aliases beneath a non-root parent. Unsupported paths return no matches.
    This locates retained evidence; it does not establish value compatibility.
    """
    path, index = [], 0
    while index < len(steps):
        if steps[index] == "properties" and index + 1 < len(steps):
            path.append(steps[index + 1])
            index += 2
        elif steps[index] == "items":
            path.append(None)
            index += 1
        else:
            return []
    if not path:
        return []
    result = []
    for index, match in enumerate(matches):
        target_path = match.get("path")
        if match.get("endpoint") != endpoint or not target_path:
            continue
        dynamo_path = [*target_path[:-1], match["dynamo_name"]]
        if any(
            path == candidate or (not exact and path == candidate[: len(path)])
            for candidate in (target_path, dynamo_path)
        ):
            result.append(index)
    return result


def build_findings(report: dict, projections: dict | None = None) -> dict:
    """Build additive, fully accounted findings from saved report/projection evidence.

    `projections` maps retained filenames to parsed JSON; omitted files are marked
    unavailable. This explains oasdiff's output, not a second diff/validation engine.
    Declaration categories use presence at the inspected location on each side;
    absent annotations do not make an otherwise declared field absent.
    Exact declared aliases classify spelling-only deltas as alias evidence. Value
    comparisons, parent constraints and ambiguous structures remain separate.
    Historical location IDs and raw deltas remain in groups; finding IDs exclude
    content, which has separate hashes. Every raw candidate terminal must belong
    to exactly one row or ValueError is raised. No input mutation or I/O occurs.
    """
    projections = projections or {}
    groups, rows, expected = candidate_groups(report), [], set()
    keywords = {
        "min": "minimum",
        "max": "maximum",
        "listOfTypes": "type",
        "additionalPropertiesAllowed": "additionalProperties",
    }
    scalar = {
        "type",
        "default",
        "nullable",
        "enum",
        "const",
        "required",
        "format",
        "minimum",
        "maximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "pattern",
        "uniqueItems",
        "additionalProperties",
    }

    def emit(
        kind,
        endpoint,
        status,
        media,
        root,
        steps,
        change,
        raw,
        category=None,
        interpretation=None,
        action=None,
    ):
        """Append one explained subtree with exact raw ownership and source declarations."""
        delta = jsonpointer.resolve_pointer(report, raw[1:])
        # Unknown comparator keys at one schema location need distinct identities.
        # Drop the endpoint-group array index, not the unsupported key's path.
        identity_change = (
            change if change != "unknown" else "unknown:" + raw.split("/delta", 1)[-1]
        )
        field = "$body"
        for index, step in enumerate(steps):
            if index and steps[index - 1] == "properties":
                field = ("" if field == "$body" else field + ".") + step
            elif step == "items":
                field += "[]"
            elif step == "additionalProperties":
                field += ".*"
        keyword = (
            keywords.get(change, change)
            if change not in {"added", "deleted", "structure", "unknown"}
            else None
        )
        declaration_keyword = keyword if keyword in scalar else None
        source = {}
        for side in ("framework", "dynamo"):
            filename = f"{side}.{'requests' if kind == 'request' else 'responses'}.normalized.json"
            source[side] = declaration(
                projections.get(filename), filename, root, steps, declaration_keyword
            )
        if change == "structure" and isinstance(delta, dict) and "default" in delta:
            for side in source:
                filename = f"{side}.{'requests' if kind == 'request' else 'responses'}.normalized.json"
                annotation = declaration(
                    projections.get(filename), filename, root, steps, "default"
                )
                source[side]["summary"] += "; " + annotation["summary"]
        if category is None:
            if change in {"structure", "unknown"}:
                category = "representation_ambiguity"
            else:
                presence = tuple(
                    bool(source[side]["evidence"]) for side in ("framework", "dynamo")
                )
                category = {
                    (True, False): "target_only",
                    (True, True): "both_declared_mismatch",
                    (False, True): "dynamo_only",
                    (False, False): "missing_coverage",
                }[presence]
        if any(source[side]["limitations"] for side in source):
            category = "missing_coverage"
        matches = report.get("alias_matches", [])
        aliases = related_aliases(matches, endpoint, steps) if kind == "request" else []
        exact_aliases = (
            related_aliases(matches, endpoint, steps, exact=True)
            if kind == "request"
            else []
        )
        if (
            change in {"added", "deleted"}
            and category in {"target_only", "dynamo_only"}
            and len(exact_aliases) == 1
        ):
            match = matches[exact_aliases[0]]
            if match.get("name_coverage") == "matched_via_declared_input_alias":
                aligned = {
                    side: declaration(
                        projections.get(f"{side}.requests.normalized.json"),
                        f"{side}.requests.normalized.json",
                        match[
                            "backend_location"
                            if side == "framework"
                            else "dynamo_location"
                        ],
                        (),
                        None,
                    )
                    for side in ("framework", "dynamo")
                }
                if all(
                    item["evidence"] and not item["limitations"]
                    for item in aligned.values()
                ):
                    source = aligned
                    category = "alias_evidence"
                    interpretation = (
                        f"The target's {match['backend_name']} maps to Dynamo's "
                        f"{match['dynamo_name']} through a declared input alias. "
                        "This spelling difference is not a missing field. "
                        "Value compatibility is not established by the name match."
                    )
                    action = "Review the linked alias value comparison, parent requirements and duplicate-name handling."
                else:
                    category = "missing_coverage"
                    interpretation = "A declared alias match is recorded, but its source declarations are incomplete."
                    action = "Recover the linked alias declarations before assessing the field."
        direction = (
            "reference requests → Dynamo admission/effects"
            if kind == "request"
            else "Dynamo outputs → reference-client expectations"
        )
        if interpretation is None:
            if change == "structure":
                interpretation = "Schema layout or alternatives changed; field deletion and narrowing are not established."
            elif keyword == "default":
                interpretation = "Default annotations differ; absent annotation is not a null default or proven runtime behavior."
            elif keyword == "required":
                interpretation = "Required-member declarations differ; branch conditions and serializer/admission rules still apply."
            elif change in {"added", "deleted"}:
                interpretation = (
                    "Dynamo-only" if change == "added" else "Reference-only"
                ) + " declaration at this location. "
                interpretation += (
                    "Extra accepted input is not itself a client incompatibility; missing input declarations do not prove rejection."
                    if kind == "request"
                    else "Extra possible output can affect clients; missing declarations do not prove the output never occurs."
                )
            elif change == "unknown":
                interpretation = "Unsupported comparator fragment; retained without a semantic verdict."
            else:
                interpretation = (
                    "Declared "
                    + str(keyword)
                    + " differs; no runtime incompatibility is confirmed."
                )
        if action is None:
            subject = (
                field
                if field != "$body"
                else (
                    f"HTTP {status} {media or 'response'}"
                    if kind == "response"
                    else "request body"
                )
            )
            if change in {"structure", "unknown"}:
                action = f"Resolve {subject}'s referenced/composed branches and align matching alternatives before classifying removal or restriction."
            elif keyword == "default":
                action = f"Check pinned-source handling of omitted {subject}; compare omission with an explicit value before proposing a default change."
            elif keyword == "required":
                action = f"Check omission of changed required members in {subject}, including conditional branches; verify admission or serialization separately."
            else:
                action = (
                    f"Check {subject} against pinned validation/translation code, then probe its declared boundary and backend effect."
                    if kind == "request"
                    else f"Check {subject} against the serializer and client decoder; include JSON/SSE error cases before changing the response contract."
                )
        if aliases and category != "alias_evidence":
            interpretation += " Related declared aliases exist; a spelling delta is not a missing-input verdict."
            action = (
                "Inspect the linked alias value probes and parent/duplicate-name constraints. "
                + action
            )
        location = root + pointer(*steps)[1:] if steps else root
        transport = (
            "SSE"
            if media == "text/event-stream"
            else "JSON"
            if media == "application/json"
            else "HTTP"
        )
        row = dict(
            id=identifier(
                "finding",
                endpoint,
                json.dumps(
                    [kind, status, media, location, identity_change],
                    separators=(",", ":"),
                ),
            ),
            kind=kind,
            endpoint=endpoint,
            status=status,
            media=media,
            transport=transport,
            direction=direction,
            field=field,
            location=location,
            change=change,
            category=category,
            declarations=source,
            interpretation=interpretation,
            suggested_action=action,
            raw_report_pointers=[raw],
            delta=copy.deepcopy(delta),
            alias_report_pointers=[
                pointer("alias_matches", index) for index in aliases
            ],
        )
        rows.append(row)

    def schema_walk(delta, raw, root, emit_at, steps=()):
        """Explain supported nested fields; keep union/allOf changes as indivisible ambiguous groups."""
        if (
            not isinstance(delta, dict)
            or not delta
            or any(key in delta for key in ("anyOf", "oneOf", "allOf"))
        ):
            emit_at(root, steps, "structure", raw)
            return
        for key, value in sorted(delta.items()):
            location = raw + pointer(key)[1:]
            if key == "properties" and isinstance(value, dict):
                for change, fields in sorted(value.items()):
                    part = location + pointer(change)[1:]
                    if change == "modified" and isinstance(fields, dict) and fields:
                        for name, body in sorted(fields.items()):
                            schema_walk(
                                body,
                                part + pointer(name)[1:],
                                root,
                                emit_at,
                                (*steps, "properties", name),
                            )
                    elif (
                        change in {"added", "deleted"}
                        and isinstance(fields, list)
                        and fields
                        and all(isinstance(name, str) for name in fields)
                    ):
                        for index, name in enumerate(fields):
                            emit_at(
                                root,
                                (*steps, "properties", name),
                                change,
                                part + pointer(index)[1:],
                            )
                    else:
                        emit_at(root, steps, "unknown", part)
                if not value:
                    emit_at(root, steps, "unknown", location)
            elif (
                key in {"items", "additionalProperties", "propertyNames"}
                and isinstance(value, dict)
                and not {"schemaAdded", "schemaDeleted"}.intersection(value)
            ):
                schema_walk(value, location, root, emit_at, (*steps, key))
            else:
                change = key if keywords.get(key, key) in scalar else "unknown"
                emit_at(root, steps, change, location)

    def content_walk(delta, raw, kind, endpoint, status, parent):
        """Walk oasdiff media maps while retaining unsupported fragments verbatim."""
        for change, media_values in (
            sorted(delta.items()) if isinstance(delta, dict) else []
        ):
            if change == "modified" and isinstance(media_values, dict) and media_values:
                for media, body in sorted(media_values.items()):
                    emit_at = partial(emit, kind, endpoint, status, media)
                    if not isinstance(body, dict) or not body:
                        emit_at(
                            parent + pointer("content", media)[1:],
                            (),
                            "unknown",
                            raw + pointer(change, media)[1:],
                        )
                        continue
                    for key, value in sorted(body.items()):
                        location = raw + pointer(change, media, key)[1:]
                        root = parent + pointer("content", media, key)[1:]
                        if key == "schema":
                            schema_walk(value, location, root, emit_at)
                        else:
                            emit_at(root, (), "unknown", location)
            elif (
                change in {"added", "deleted"}
                and isinstance(media_values, list)
                and media_values
            ):
                for index, media in enumerate(media_values):
                    emit(
                        kind,
                        endpoint,
                        status,
                        str(media),
                        parent + pointer("content", media)[1:],
                        (),
                        change,
                        raw + pointer(change, index)[1:],
                    )
            else:
                emit(
                    kind,
                    endpoint,
                    status,
                    None,
                    parent,
                    (),
                    "unknown",
                    raw + pointer(change)[1:],
                )
        if not isinstance(delta, dict) or not delta:
            emit(kind, endpoint, status, None, parent, (), "unknown", raw)

    def body_walk(delta, raw, parent, kind, endpoint, status=None):
        """Explain request/response content; retain empty bodies and other keys as unknown."""
        emit_at = partial(emit, kind, endpoint, status, None)
        for key, value in sorted(delta.items()):
            location = raw + pointer(key)[1:]
            if key == "content":
                content_walk(value, location, kind, endpoint, status, parent)
            else:
                emit_at(parent, (), "unknown", location)
        if not delta:
            emit_at(parent, (), "unknown", raw)

    for index, finding in enumerate(report.get("differences", [])):
        raw = pointer("differences", index, "delta")
        expected.update(terminals(finding["delta"], raw))
        parent = pointer("paths", finding["endpoint"], "post", "requestBody")
        body_walk(finding["delta"], raw, parent, "request", finding["endpoint"])
    for index, finding in enumerate(report.get("response_differences", [])):
        raw = pointer("response_differences", index, "delta")
        expected.update(terminals(finding["delta"], raw))
        for change, statuses in sorted(finding["delta"].items()):
            if change == "modified" and isinstance(statuses, dict) and statuses:
                for status, body in sorted(statuses.items()):
                    parent = finding["location"] + pointer(status)[1:]
                    body_walk(
                        body,
                        raw + pointer(change, status)[1:],
                        parent,
                        "response",
                        finding["endpoint"],
                        status,
                    )
            elif (
                change in {"added", "deleted"}
                and isinstance(statuses, list)
                and statuses
            ):
                for number, status in enumerate(statuses):
                    emit(
                        "response",
                        finding["endpoint"],
                        str(status),
                        None,
                        finding["location"] + pointer(status)[1:],
                        (),
                        change,
                        raw + pointer(change, number)[1:],
                    )
            else:
                emit(
                    "response",
                    finding["endpoint"],
                    None,
                    None,
                    finding["location"],
                    (),
                    "unknown",
                    raw + pointer(change)[1:],
                )
        if not finding["delta"]:
            emit(
                "response",
                finding["endpoint"],
                None,
                None,
                finding["location"],
                (),
                "unknown",
                raw,
            )
    for collection, category in (
        ("coverage_gaps", "missing_coverage"),
        ("exclusions", "intentional_exclusion"),
        ("alias_matches", "alias_evidence"),
        ("response_chunk_probes", "representation_ambiguity"),
    ):
        for index, item in enumerate(report.get(collection, [])):
            raw = pointer(collection, index)
            expected.update(terminals(item, raw))
            endpoint = item.get("endpoint", "*")
            location = item.get(
                "location", item.get("dynamo_location", item.get("field", "$scope"))
            )
            reason = item.get(
                "reason",
                "Declared alias names match; value and parent constraints require separate review."
                if collection == "alias_matches"
                else "Selected chunk comparison: "
                + item.get("status", "unknown")
                + "; no full-event equivalence.",
            )
            action = (
                "Inspect the named source/exporter limitation at "
                + location
                + "; add the missing declaration or scoped evidence before closing this gap."
                if collection == "coverage_gaps"
                else "Confirm this exclusion remains intended; do not claim compatibility for its contents."
                if collection == "exclusions"
                else "Review the alias value diff plus parent requiredness and simultaneous-name rejection; test the pinned admission path."
                if collection == "alias_matches"
                else "Review the selected chunk diff and its gaps; compare envelopes, errors, terminators and extensions separately."
            )
            rows.append(
                dict(
                    id=identifier(
                        "finding",
                        endpoint,
                        json.dumps(
                            [
                                collection,
                                item.get("kind"),
                                item.get("side"),
                                item.get("layer"),
                                location,
                                item.get("field"),
                            ],
                            separators=(",", ":"),
                        ),
                    ),
                    kind=item.get("kind", category),
                    endpoint=endpoint,
                    status=None,
                    media=None,
                    transport="not specified",
                    direction="not a runtime verdict",
                    field=item.get("field", item.get("backend_name", "$coverage")),
                    location=location,
                    change=category,
                    category=category,
                    declarations={
                        side: {
                            "summary": "not established by this evidence",
                            "evidence": [],
                            "limitations": [],
                        }
                        for side in ("framework", "dynamo")
                    },
                    interpretation=reason,
                    suggested_action=action,
                    raw_report_pointers=[raw],
                    delta=copy.deepcopy(item),
                    alias_report_pointers=[],
                )
            )
            row = rows[-1]
            if collection in {"coverage_gaps", "alias_matches"}:
                kind = item.get("kind", "request")
                row["transport"] = "JSON" if kind == "request" else "HTTP"
                for side in ("framework", "dynamo"):
                    source_location = item.get(
                        "backend_location"
                        if side == "framework"
                        else "dynamo_location",
                        item.get("projection_location", location),
                    )
                    filename = f"{side}.{'requests' if kind == 'request' else 'responses'}.normalized.json"
                    if source_location.startswith("#/"):
                        row["declarations"][side] = declaration(
                            projections.get(filename),
                            filename,
                            source_location,
                            (),
                            None,
                        )
                        parts = jsonpointer.JsonPointer(source_location[1:]).parts
                        if "responses" in parts:
                            row["status"] = parts[parts.index("responses") + 1]
                        if "content" in parts:
                            row["media"] = parts[parts.index("content") + 1]
                            row["transport"] = (
                                "SSE"
                                if row["media"] == "text/event-stream"
                                else "JSON"
                                if row["media"] == "application/json"
                                else "HTTP"
                            )
                if collection == "alias_matches":
                    row["interpretation"] += (
                        " Value comparison: "
                        + item.get("value_comparison", "not supplied")
                        + "."
                    )
    ownership = Counter()
    for row in rows:
        row["candidate_ids"] = sorted(
            group["id"]
            for group in groups
            if any(
                contains(a, b) or contains(b, a)
                for a in group["raw_report_pointers"]
                for b in row["raw_report_pointers"]
            )
        )
        row["delta_sha256"] = digest(row["delta"])
        row["content_sha256"] = digest([row["delta"], row["declarations"]])
        for raw in row["raw_report_pointers"]:
            ownership.update(
                terminals(jsonpointer.resolve_pointer(report, raw[1:]), raw)
            )
    if set(ownership) != expected or any(count != 1 for count in ownership.values()):
        raise ValueError(
            "Generated findings do not partition all raw candidate terminals exactly once"
        )
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Ambiguous duplicate finding locations")
    for group in groups:
        members = sorted(
            (row for row in rows if group["id"] in row["candidate_ids"]),
            key=lambda row: row["id"],
        )
        group["finding_ids"] = [row["id"] for row in members]
        group["declaration_sha256"] = digest([row["declarations"] for row in members])
        group["content_sha256"] = digest([group["delta"], group["declaration_sha256"]])
        if not group["finding_ids"]:
            raise ValueError("Candidate group has no generated finding: " + group["id"])
    return {
        "schema": "protocol-actionable-findings/v1",
        "behavioral_conformance": "not_assessed",
        "identity_policy": "IDs identify locations, not unchanged content, independent defects, or approved fixes.",
        "groups": groups,
        "findings": sorted(
            rows,
            key=lambda row: (
                row["kind"],
                row["endpoint"],
                row["status"] or "",
                row["media"] or "",
                row["location"],
                row["change"],
            ),
        ),
        "accounting": {
            "raw_terminal_count": len(expected),
            "owned_terminal_count": len(ownership),
            "unaccounted": [],
            "multiply_owned": [],
        },
        "counts": {
            "location_groups": len(groups),
            "readable_findings": len(rows),
            "categories": dict(
                sorted(Counter(row["category"] for row in rows).items())
            ),
        },
    }
