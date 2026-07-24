/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

import { Fragment } from "react";
import {
    type KubernetesField,
    type KubernetesType,
} from "./KubernetesApiTypes";

export function SafeText({ text }: { text: string }) {
    return (
        <>
            {text.split("<br />").map((part, index) => (
                <Fragment key={`${index}-${part}`}>
                    {index > 0 ? <br /> : null}
                    {part}
                </Fragment>
            ))}
        </>
    );
}

export function FieldGrid({
    fields,
    validAnchors,
}: {
    fields: KubernetesField[];
    validAnchors: Set<string>;
}) {
    if (fields.length === 0) return null;
    return (
        <div className="dynref-k8s-fields-wrap">
            <table className="dynref-k8s-fields">
                <thead>
                    <tr>
                        <th scope="col">Field</th>
                        <th scope="col">Type</th>
                        <th scope="col">Description</th>
                    </tr>
                </thead>
                <tbody>
                    {fields.map((field) => (
                        <FieldRow
                            key={field.name}
                            field={field}
                            validAnchors={validAnchors}
                        />
                    ))}
                </tbody>
            </table>
        </div>
    );
}

const FIELD_TYPE_LINK = /^\[([^\]]+)\]\(([^)]+)\)(.*)$/;

function FieldType({
    value,
    validAnchors,
}: {
    value: string;
    validAnchors: Set<string>;
}) {
    const match = FIELD_TYPE_LINK.exec(value);
    if (!match) return <>{value || "-"}</>;
    const [, label, href, suffix] = match;
    const isSafe = href.startsWith("#") || href.startsWith("https://");
    const isResolved = !href.startsWith("#") || validAnchors.has(href.slice(1));
    if (!isSafe || !isResolved) return <>{label}{suffix}</>;
    return (
        <>
            <a href={href}>{label}</a>
            {suffix}
        </>
    );
}

function FieldRow({
    field,
    validAnchors,
}: {
    field: KubernetesField;
    validAnchors: Set<string>;
}) {
    return (
        <tr>
            <th scope="row" className="dynref-k8s-field-name">
                {field.name}
                {field.required ? (
                    <>
                        {" "}
                        <span className="dynref-badge dynref-badge--red">required</span>
                    </>
                ) : null}
            </th>
            <td className="dynref-k8s-field-type">
                <FieldType value={field.type} validAnchors={validAnchors} />
            </td>
            <td className="dynref-k8s-field-desc">
                {field.description ? <SafeText text={field.description} /> : "—"}
                <FieldMetadata field={field} />
            </td>
        </tr>
    );
}

function FieldMetadata({ field }: { field: KubernetesField }) {
    if (!field.default && !field.validation) return null;
    return (
        <small className="dynref-k8s-field-meta">
            {field.default ? <>default <span className="dynref-mono">{field.default}</span></> : null}
            {field.default && field.validation ? " · " : null}
            {field.validation ? <>validation <SafeText text={field.validation} /></> : null}
        </small>
    );
}

export function EnumValues({
    values,
}: {
    values: KubernetesType["enumValues"];
}) {
    if (values.length === 0) return null;
    return (
        <dl className="dynref-k8s-enum-values">
            {values.map((value) => (
                <div key={value.name}>
                    <dt><span className="dynref-badge dynref-badge--gray">{value.name}</span></dt>
                    <dd>{value.description || "No description."}</dd>
                </div>
            ))}
        </dl>
    );
}
