/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ApiKubernetesReference -- compact grouped index of the Dynamo Kubernetes
 * CRD + operator-config API surface. Reads the typed data written by
 * docs/fern/scripts/gen_kubernetes_api.py.
 *
 * Layout (compact index / Option A of the plan):
 *   * Package summary hero: three CRD/config packages, total typed
 *     section count, and jump anchors matching the legacy deep-link
 *     contract (v1beta1 dedup preserved).
 *   * CSS-only, keyboard-accessible filter rail: four radio inputs
 *     hide the packages that don't match, exactly the same
 *     :focus-visible pill pattern as ArtifactBrowser / ApiSurfaceBrowser.
 *   * Per-package group: prose intro, Resource Types jump list, then a
 *     dense type index. Each type is a native <details> accordion whose
 *     <summary> carries the anchor id, so /page#dgd-reference resolves
 *     without JavaScript.
 *
 * Server component; shared vocabulary (panel, eyebrow, badges, chips,
 * copy affordance, focus ring) comes from ReferenceStyles. Only the
 * narrowly-scoped ``.dynref-k8s-*`` layout classes live here.
 */

import {
    type KubernetesPackage,
    type KubernetesReference,
    type KubernetesType,
    type KubernetesTypeRef,
} from "./KubernetesApiTypes";
import {
    EnumValues,
    FieldGrid,
    SafeText,
} from "./KubernetesSchemaDetails";

const K8S_CSS = `
.dynref-k8s-hero {
    display: grid;
    grid-template-columns: minmax(0, 1fr) max-content;
    gap: 10px 20px;
    margin-bottom: 12px;
    align-items: baseline;
}
.dynref-k8s-hero-title {
    display: flex; flex-wrap: wrap; align-items: baseline; gap: 10px;
    margin: 0;
    color: var(--pst-color-text-base);
    font-size: 20px; font-weight: 600; line-height: 1.2;
}
.dynref-k8s-hero-meta {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
}
.dynref-k8s-hero-meta a { color: inherit; text-decoration: underline; }
.dynref-k8s-jumps {
    display: flex; flex-wrap: wrap; gap: 6px; margin: 0;
    justify-content: flex-end;
}
.dynref-k8s-rail {
    display: flex; flex-wrap: wrap; gap: 8px; margin: 12px 0 6px;
}
.dynref-k8s-pill {
    display: inline-flex; align-items: center;
    min-height: 28px; padding: 4px 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: var(--rounded, 6px);
    background: transparent;
    color: var(--pst-color-text-base);
    font-size: 12.5px; line-height: 1; cursor: pointer;
}
.dynref-k8s-pill:hover { border-color: var(--nv-color-green, #76B900); }
.dynref-k8s-group {
    margin: 18px 0 6px; padding: 12px 14px 6px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    background: var(--pst-color-surface);
}
.dark .dynref-k8s-group { background: #161616; border-color: #2b2b2b; }
.dynref-k8s-group-header {
    display: flex; flex-wrap: wrap; align-items: baseline;
    gap: 8px 16px; margin-bottom: 8px;
}
.dynref-k8s-group-title {
    margin: 0;
    color: var(--pst-color-text-base);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 15px; font-weight: 600; overflow-wrap: anywhere;
}
.dynref-k8s-group-meta { margin: 0; color: var(--pst-color-text-muted); font-size: 12px; }
.dynref-k8s-group-desc {
    margin: 0 0 8px;
    color: var(--pst-color-text-muted);
    font-size: 12.5px; line-height: 1.5; white-space: pre-line;
}
.dynref-k8s-resources {
    display: flex; flex-wrap: wrap; gap: 6px;
    margin: 0 0 10px; padding: 0; list-style: none;
}
.dynref-k8s-resources li { margin: 0; }
.dynref-k8s-resources a {
    display: inline-flex; align-items: center; padding: 2px 8px;
    border: 1px solid var(--dynref-blue-border);
    border-radius: 6px;
    background: var(--dynref-blue-bg);
    color: var(--dynref-blue-fg);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 12px; text-decoration: none;
}
.dynref-k8s-resources a:hover { text-decoration: underline; }
.dynref-k8s-list { display: flex; flex-direction: column; gap: 0; margin-top: 6px; }
.dynref-k8s-row {
    padding: 0;
    border-top: 1px solid var(--border, var(--grayscale-a5));
    scroll-margin-top: 96px;
}
.dynref-k8s-row:last-of-type { border-bottom: 1px solid var(--border, var(--grayscale-a5)); }
.dynref-k8s-head {
    display: grid;
    grid-template-columns: max-content minmax(0, 1fr) minmax(0, 1fr);
    gap: 6px 14px; align-items: baseline;
    padding: 8px 4px; cursor: pointer; list-style: none;
}
.dynref-k8s-head::-webkit-details-marker { display: none; }
.dynref-k8s-head:focus-visible {
    outline: 2px solid var(--nv-color-green, #76B900);
    outline-offset: 2px; border-radius: 4px;
}
.dynref-k8s-caret {
    color: var(--pst-color-text-muted); font-size: 10px;
    transition: transform 0.15s ease; flex: 0 0 auto;
}
.dynref-k8s-row[open] > .dynref-k8s-head .dynref-k8s-caret { transform: rotate(90deg); }
.dynref-k8s-name {
    color: var(--pst-color-text-base);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 13px; font-weight: 600; overflow-wrap: anywhere;
    display: flex; align-items: center; gap: 6px;
}
.dynref-k8s-summary {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 12.5px; line-height: 1.4;
}
.dynref-k8s-body {
    padding: 6px 12px 14px 12px;
    display: flex; flex-direction: column; gap: 10px;
    border-left: 2px solid var(--border, var(--grayscale-a5));
    margin-left: 4px;
}
.dynref-k8s-desc {
    margin: 0;
    color: var(--pst-color-text-base);
    font-size: 12.5px; line-height: 1.5; white-space: pre-line;
}
.dynref-k8s-appears { margin: 0; color: var(--pst-color-text-muted); font-size: 12px; }
.dynref-k8s-appears a {
    color: inherit; text-decoration: underline;
    text-decoration-color: var(--nv-color-green, #76B900);
}
.dynref-k8s-fields-wrap {
    overflow-x: auto;
    padding-top: 6px;
    border-top: 1px dashed var(--border, var(--grayscale-a5));
}
.dynref-k8s-fields {
    width: 100%;
    min-width: 620px;
    border-collapse: collapse;
    table-layout: fixed;
}
.dynref-k8s-fields th,
.dynref-k8s-fields td {
    padding: 6px 8px;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    text-align: left;
    vertical-align: top;
}
.dynref-k8s-fields thead th {
    color: var(--pst-color-text-muted);
    font-size: 11px;
    font-weight: 600;
}
.dynref-k8s-fields thead th:nth-child(1) { width: 26%; }
.dynref-k8s-fields thead th:nth-child(2) { width: 25%; }
.dynref-k8s-field-name {
    color: var(--pst-color-text-base);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 12.5px; font-weight: 600; overflow-wrap: anywhere;
}
.dynref-k8s-field-type {
    color: var(--pst-color-text-muted);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 12px; overflow-wrap: anywhere;
}
.dynref-k8s-field-desc {
    color: var(--pst-color-text-base);
    font-size: 12.5px; line-height: 1.4; min-width: 0;
}
.dynref-k8s-field-meta {
    display: block; margin-top: 4px;
    color: var(--pst-color-text-muted); font-size: 11px;
}
.dynref-k8s-enum-values { display: grid; gap: 6px; margin: 0; }
.dynref-k8s-enum-values > div {
    display: grid; grid-template-columns: minmax(120px, max-content) minmax(0, 1fr);
    gap: 10px; align-items: baseline;
}
.dynref-k8s-enum-values dt,
.dynref-k8s-enum-values dd { margin: 0; }
.dynref-k8s-enum-values dt {
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
}
.dynref-k8s-enum-values dd {
    color: var(--pst-color-text-muted); font-size: 12.5px; line-height: 1.4;
}
.dynref-k8s-note { margin-top: 12px; color: var(--pst-color-text-muted); font-size: 12px; }
.dynref-k8s-underlying {
    color: var(--pst-color-text-muted);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 12px;
}
/* Filter show/hide -- default (k8s-all checked) shows every group. */
#k8s-v1alpha1:checked ~ .dynref-k8s-list-groups > .dynref-k8s-group[data-package="nvidia.com/v1beta1"],
#k8s-v1alpha1:checked ~ .dynref-k8s-list-groups > .dynref-k8s-group[data-package="operator.config.dynamo.nvidia.com/v1alpha1"],
#k8s-v1beta1:checked ~ .dynref-k8s-list-groups > .dynref-k8s-group[data-package="nvidia.com/v1alpha1"],
#k8s-v1beta1:checked ~ .dynref-k8s-list-groups > .dynref-k8s-group[data-package="operator.config.dynamo.nvidia.com/v1alpha1"],
#k8s-operator:checked ~ .dynref-k8s-list-groups > .dynref-k8s-group[data-package="nvidia.com/v1alpha1"],
#k8s-operator:checked ~ .dynref-k8s-list-groups > .dynref-k8s-group[data-package="nvidia.com/v1beta1"] {
    display: none;
}
#k8s-all:checked ~ .dynref-k8s-rail label[for="k8s-all"],
#k8s-v1alpha1:checked ~ .dynref-k8s-rail label[for="k8s-v1alpha1"],
#k8s-v1beta1:checked ~ .dynref-k8s-rail label[for="k8s-v1beta1"],
#k8s-operator:checked ~ .dynref-k8s-rail label[for="k8s-operator"] {
    border-color: var(--nv-color-green, #76B900);
    box-shadow: 0 0 0 1px var(--nv-color-green, #76B900);
    background: rgba(118, 185, 0, 0.08);
    font-weight: 700;
}
#k8s-all:focus-visible ~ .dynref-k8s-rail label[for="k8s-all"],
#k8s-v1alpha1:focus-visible ~ .dynref-k8s-rail label[for="k8s-v1alpha1"],
#k8s-v1beta1:focus-visible ~ .dynref-k8s-rail label[for="k8s-v1beta1"],
#k8s-operator:focus-visible ~ .dynref-k8s-rail label[for="k8s-operator"] {
    outline: 2px solid var(--nv-color-green, #76B900); outline-offset: 1px;
}
@media (max-width: 720px) {
    .dynref-k8s-hero,
    .dynref-k8s-head { grid-template-columns: minmax(0, 1fr); }
    .dynref-k8s-jumps { justify-content: flex-start; }
    .dynref-k8s-enum-values > div { grid-template-columns: minmax(0, 1fr); gap: 3px; }
}
`;

const PACKAGE_FILTER_IDS: Record<string, string> = {
    "nvidia.com/v1alpha1": "k8s-v1alpha1",
    "nvidia.com/v1beta1": "k8s-v1beta1",
    "operator.config.dynamo.nvidia.com/v1alpha1": "k8s-operator",
};

const PACKAGE_SHORT_LABELS: Record<string, string> = {
    "nvidia.com/v1alpha1": "v1alpha1",
    "nvidia.com/v1beta1": "v1beta1",
    "operator.config.dynamo.nvidia.com/v1alpha1": "operator config",
};

function totalTypeCount(reference: KubernetesReference): number {
    return reference.packages.reduce(
        (n, pkg) => n + pkg.types.length,
        0,
    );
}

function kindBadgeClass(kind: KubernetesType["kind"]): string {
    switch (kind) {
        case "resource":
            return "dynref-badge dynref-badge--green";
        case "enum":
            return "dynref-badge dynref-badge--amber";
        case "type":
            return "dynref-badge dynref-badge--gray";
        default: {
            const exhaustive: never = kind;
            return exhaustive;
        }
    }
}

function firstDescriptionLine(text: string): string {
    for (const line of text.split("\n")) {
        const trimmed = line.trim();
        if (trimmed) return trimmed;
    }
    return "";
}

function ReferenceHero({ reference }: { reference: KubernetesReference }) {
    const packages = reference.packages;
    return (
        <div className="dynref-k8s-hero">
            <div>
                <p className="dynref-eyebrow">Kubernetes API</p>
                <div className="dynref-k8s-hero-title">
                    {packages.length} packages · {totalTypeCount(reference)} typed sections
                </div>
                <p className="dynref-k8s-hero-meta">
                    Auto-generated from{" "}
                    <a href={reference.sourceHref}>ai-dynamo/dynamo</a>
                    {" · "}
                    v1beta1 same-name types keep their <span className="dynref-mono">v1beta1-</span>
                    prefixed anchors so every legacy deep link resolves.
                </p>
            </div>
            <ul className="dynref-k8s-jumps">
                {packages.map((pkg) => (
                    <li key={pkg.anchor}>
                        <a
                            className="dynref-badge dynref-badge--blue dynref-badge--outline"
                            href={`#${pkg.anchor}`}
                        >
                            {PACKAGE_SHORT_LABELS[pkg.name] ?? pkg.name}
                        </a>
                    </li>
                ))}
            </ul>
        </div>
    );
}

function FilterInputs() {
    return (
        <>
            <input
                className="dynref-vh"
                type="radio"
                id="k8s-all"
                name="dynref-k8s-filter"
                defaultChecked
            />
            <input
                className="dynref-vh"
                type="radio"
                id="k8s-v1alpha1"
                name="dynref-k8s-filter"
            />
            <input
                className="dynref-vh"
                type="radio"
                id="k8s-v1beta1"
                name="dynref-k8s-filter"
            />
            <input
                className="dynref-vh"
                type="radio"
                id="k8s-operator"
                name="dynref-k8s-filter"
            />
        </>
    );
}

function FilterRail({ reference }: { reference: KubernetesReference }) {
    return (
        <div className="dynref-k8s-rail">
            <label className="dynref-k8s-pill" htmlFor="k8s-all">
                All · {totalTypeCount(reference)}
            </label>
            {reference.packages.map((pkg) => (
                <label
                    key={pkg.anchor}
                    className="dynref-k8s-pill"
                    htmlFor={PACKAGE_FILTER_IDS[pkg.name]}
                >
                    {PACKAGE_SHORT_LABELS[pkg.name] ?? pkg.name} · {pkg.types.length}
                </label>
            ))}
        </div>
    );
}

function ResourceLinks({ refs }: { refs: KubernetesTypeRef[] }) {
    if (refs.length === 0) return null;
    return (
        <ul className="dynref-k8s-resources">
            {refs.map((ref) => (
                <li key={ref.anchor}>
                    <a href={`#${ref.anchor}`}>{ref.name}</a>
                </li>
            ))}
        </ul>
    );
}

function AppearsIn({ refs }: { refs: KubernetesTypeRef[] }) {
    if (refs.length === 0) return null;
    return (
        <p className="dynref-k8s-appears">
            <span className="dynref-label">Appears in:</span>{" "}
            {refs.map((ref, index) => (
                <span key={ref.anchor}>
                    {index > 0 ? " · " : ""}
                    <a href={`#${ref.anchor}`}>{ref.name}</a>
                </span>
            ))}
        </p>
    );
}

function TypeRow({
    type_,
    validAnchors,
}: {
    type_: KubernetesType;
    validAnchors: Set<string>;
}) {
    return (
        <details className="dynref-k8s-row" id={type_.anchor} data-kind={type_.kind}>
            <TypeSummary type_={type_} />
            <TypeBody type_={type_} validAnchors={validAnchors} />
        </details>
    );
}

function TypeSummary({ type_ }: { type_: KubernetesType }) {
    return (
        <summary className="dynref-k8s-head">
            <span className="dynref-k8s-name">
                <span className="dynref-k8s-caret" aria-hidden="true">▶</span>
                <span>{type_.displayName}</span>
                <span className={kindBadgeClass(type_.kind)}>{type_.kind}</span>
            </span>
            <p className="dynref-k8s-summary">{firstDescriptionLine(type_.description) || "—"}</p>
            <p className="dynref-k8s-underlying">
                {type_.underlyingType ? <>underlying <span className="dynref-mono">{type_.underlyingType}</span></> : null}
            </p>
        </summary>
    );
}

function TypeBody({
    type_,
    validAnchors,
}: {
    type_: KubernetesType;
    validAnchors: Set<string>;
}) {
    return (
        <div className="dynref-k8s-body">
            {type_.description ? <p className="dynref-k8s-desc"><SafeText text={type_.description} /></p> : null}
            {type_.validation ? <p className="dynref-k8s-appears">Validation: <SafeText text={type_.validation} /></p> : null}
            <AppearsIn refs={type_.appearsIn} />
            <FieldGrid fields={type_.fields} validAnchors={validAnchors} />
            <EnumValues values={type_.enumValues} />
        </div>
    );
}

function PackageGroup({ pkg }: { pkg: KubernetesPackage }) {
    const validAnchors = new Set(pkg.types.map((type_) => type_.anchor));
    return (
        <section
            className="dynref-k8s-group"
            data-package={pkg.name}
            id={pkg.anchor}
        >
            <div className="dynref-k8s-group-header">
                <h3 className="dynref-k8s-group-title">{pkg.name}</h3>
                <p className="dynref-k8s-group-meta">{pkg.types.length} typed sections</p>
            </div>
            {pkg.description ? (
                <p className="dynref-k8s-group-desc">{pkg.description}</p>
            ) : null}
            <ResourceLinks refs={pkg.resourceTypes} />
            <div className="dynref-k8s-list">
                {pkg.types.map((t) => (
                    <TypeRow
                        key={t.anchor}
                        type_={t}
                        validAnchors={validAnchors}
                    />
                ))}
            </div>
        </section>
    );
}

export function ApiKubernetesReference({
    reference,
}: {
    reference: KubernetesReference;
}) {
    return (
        <>
            <style>{K8S_CSS}</style>
            <section className="dynref-panel">
                <FilterInputs />
                <ReferenceHero reference={reference} />
                <FilterRail reference={reference} />
                <div className="dynref-k8s-list-groups">
                    {reference.packages.map((pkg) => (
                        <PackageGroup key={pkg.anchor} pkg={pkg} />
                    ))}
                </div>
                <p className="dynref-k8s-note">
                    Expand a type for its description, appears-in references, fields, and
                    enum values. The plain-Markdown source is linked in the hero for agent
                    exports and search indexing.
                </p>
            </section>
        </>
    );
}
