/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ApiSurfaceBrowser — compact grouped index of one Python module's public
 * surface. Reads the typed data written by
 * docs/fern/scripts/gen_python_api.py.
 *
 * Layout (mockup "A"):
 *   * Grouped by kind: Classes section, then Functions section — the
 *     per-row kind badge from the earlier draft is dropped because the
 *     group heading carries that information.
 *   * Two columns: fixed 240px symbol-name column + flexible detail column
 *     with the one-line summary. A small source-file:line link and a
 *     compact "Copy import" affordance sit inside the name column.
 *   * Each row is a native ``<details>`` element whose ``<summary>`` is the
 *     compact name row. Expanding reveals the signature and, for classes,
 *     every public method with its own signature and source link. No
 *     JavaScript is required for expand/collapse — Fern's markdown pipeline
 *     preserves ``<details>`` semantics for keyboard, screen readers, and
 *     agent Markdown exports.
 *
 * Filtering rail is CSS-only: three hidden radio inputs sit first inside
 * the panel and ``:checked`` general-sibling selectors hide the rows and
 * group headings that do not match — same pattern as ArtifactBrowser.tsx.
 * Inputs stay keyboard-operable via the shared ``.dynref-vh`` class from
 * ReferenceStyles, with a paired ``:focus-visible`` ring painted on each pill.
 *
 * Server component; shared vocabulary (panel, eyebrow, badges, chips,
 * copy buttons, focus ring) comes from ReferenceStyles — place
 * ``<ReferenceStyles />`` on the page alongside this component. Only the
 * ``.dynref-asb-*`` layout classes are defined here.
 */

import { API_MODULES, type ApiMethod, type ApiModule, type ApiSymbol } from "./api-reference.data";

const ASB_CSS = `
.dynref-asb-summary-line {
    display: flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 4px 12px;
    margin: 0 0 8px;
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
}

.dynref-asb-import {
    margin: 0 0 10px;
    padding: 8px 10px;
    background: rgba(120, 120, 120, 0.06);
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 6px;
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 12.5px;
    color: var(--pst-color-text-base);
    overflow-x: auto;
    white-space: pre;
}

.dark .dynref-asb-import {
    background: #161616;
    border-color: #333;
}

.dynref-asb-rail {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 0 0 12px;
}

.dynref-asb-pill {
    display: inline-flex;
    align-items: center;
    min-height: 28px;
    padding: 4px 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: var(--rounded, 6px);
    background: transparent;
    color: var(--pst-color-text-base);
    font-size: 12.5px;
    line-height: 1;
    cursor: pointer;
}

.dynref-asb-pill:hover {
    border-color: var(--nv-color-green, #76B900);
}

.dynref-asb-group {
    margin: 14px 0 4px;
    color: var(--pst-color-text-muted);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.dynref-asb-row {
    padding: 0;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
    scroll-margin-top: 96px;
}

.dynref-asb-row:last-of-type {
    border-bottom: 0;
}

.dynref-asb-head {
    display: grid;
    grid-template-columns: 240px minmax(0, 1fr);
    gap: 12px;
    align-items: baseline;
    padding: 8px 0;
    cursor: pointer;
    list-style: none;
}

.dynref-asb-head::-webkit-details-marker {
    display: none;
}

.dynref-asb-head:focus-visible {
    outline: 2px solid var(--nv-color-green, #76B900);
    outline-offset: 2px;
    border-radius: 4px;
}

.dynref-asb-namecol {
    display: flex;
    flex-direction: column;
    gap: 2px;
    min-width: 0;
}

.dynref-asb-nameline {
    display: flex;
    align-items: center;
    gap: 6px;
    min-width: 0;
    flex-wrap: wrap;
}

.dynref-asb-caret {
    flex: 0 0 auto;
    color: var(--pst-color-text-muted);
    font-size: 10px;
    transition: transform 0.15s ease;
}

.dynref-asb-row[open] > .dynref-asb-head .dynref-asb-caret {
    transform: rotate(90deg);
}

.dynref-asb-name {
    color: var(--pst-color-text-base);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
    overflow-wrap: anywhere;
}

.dynref-asb-name:hover {
    text-decoration: underline;
    text-decoration-color: var(--nv-color-green, #76B900);
}

.dynref-asb-src {
    color: var(--pst-color-text-muted);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 11px;
    text-decoration: none;
}

.dynref-asb-src:hover {
    color: var(--pst-color-text-base);
    text-decoration: underline;
}

.dynref-asb-copy {
    padding: 1px 6px;
    font-size: 11px;
    line-height: 1.3;
}

.dynref-asb-copy::before {
    font-size: 10px;
}

.dynref-asb-summarytext {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
    line-height: 1.4;
}

.dynref-asb-body {
    padding: 4px 12px 12px 12px;
    display: flex;
    flex-direction: column;
    gap: 10px;
    border-left: 2px solid var(--border, var(--grayscale-a5));
    margin-left: 4px;
}

.dynref-asb-sig {
    margin: 0;
    padding: 6px 8px;
    background: rgba(120, 120, 120, 0.06);
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 4px;
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 12px;
    color: var(--pst-color-text-base);
    overflow-x: auto;
    white-space: pre;
}

.dark .dynref-asb-sig {
    background: #161616;
    border-color: #333;
}

.dynref-asb-methods-heading {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.dynref-asb-methods {
    display: flex;
    flex-direction: column;
    gap: 6px;
    margin: 0;
    padding: 0;
    list-style: none;
}

.dynref-asb-method {
    display: grid;
    grid-template-columns: 200px minmax(0, 1fr);
    gap: 10px;
    padding: 4px 0;
    border-top: 1px dashed var(--border, var(--grayscale-a5));
}

.dynref-asb-method:first-child {
    border-top: 0;
}

.dynref-asb-method-name {
    color: var(--pst-color-text-base);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 12.5px;
    font-weight: 600;
    text-decoration: none;
    overflow-wrap: anywhere;
}

.dynref-asb-method-name:hover {
    text-decoration: underline;
}

.dynref-asb-method-body {
    display: flex;
    flex-direction: column;
    gap: 4px;
    min-width: 0;
}

.dynref-asb-method-summary {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 12px;
    line-height: 1.4;
}

.dynref-asb-method-sig {
    margin: 0;
    padding: 4px 6px;
    background: rgba(120, 120, 120, 0.05);
    border-radius: 3px;
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 11.5px;
    color: var(--pst-color-text-base);
    overflow-x: auto;
    white-space: pre;
}

.dark .dynref-asb-method-sig {
    background: #161616;
}

.dynref-asb-note {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 11.5px;
}

/* Filter show/hide — default (asb-all checked) leaves every row visible. */
#asb-class:checked ~ .dynref-asb-list > .dynref-asb-group[data-group="functions"],
#asb-class:checked ~ .dynref-asb-list > .dynref-asb-row[data-kind="function"],
#asb-function:checked ~ .dynref-asb-list > .dynref-asb-group[data-group="classes"],
#asb-function:checked ~ .dynref-asb-list > .dynref-asb-row[data-kind="class"] {
    display: none;
}

#asb-all:checked ~ .dynref-asb-rail label[for="asb-all"],
#asb-class:checked ~ .dynref-asb-rail label[for="asb-class"],
#asb-function:checked ~ .dynref-asb-rail label[for="asb-function"] {
    border-color: var(--nv-color-green, #76B900);
    box-shadow: 0 0 0 1px var(--nv-color-green, #76B900);
    background: rgba(118, 185, 0, 0.08);
    font-weight: 700;
}

#asb-all:focus-visible ~ .dynref-asb-rail label[for="asb-all"],
#asb-class:focus-visible ~ .dynref-asb-rail label[for="asb-class"],
#asb-function:focus-visible ~ .dynref-asb-rail label[for="asb-function"] {
    outline: 2px solid var(--nv-color-green, #76B900);
    outline-offset: 1px;
}

@media (max-width: 640px) {
    .dynref-asb-head,
    .dynref-asb-method {
        grid-template-columns: minmax(0, 1fr);
        gap: 4px;
    }
}
`;

function symbolAnchorId(symbol: ApiSymbol): string {
    const identity = symbol.qualname.toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
    return `dynref-asb-${identity}`;
}

function methodAnchorId(symbol: ApiSymbol, method: ApiMethod): string {
    return `${symbolAnchorId(symbol)}-${method.name.toLowerCase()}`;
}

function shortSource(path: string, line: number): string {
    const base = path.split("/").pop() ?? path;
    return line > 0 ? `${base}:${line}` : base;
}

function symbolImportStatement(symbol: ApiSymbol): string {
    const splitAt = symbol.importPath.lastIndexOf(".");
    if (splitAt < 1) return symbol.importPath;
    const moduleName = symbol.importPath.slice(0, splitAt);
    const importName = symbol.importPath.slice(splitAt + 1);
    return `from ${moduleName} import ${importName}`;
}

function CopyImport({ symbol }: { symbol: ApiSymbol }) {
    const statement = symbolImportStatement(symbol);
    return (
        <button
            className="dynref-copy dynref-badge dynref-badge--blue dynref-asb-copy"
            type="button"
            data-dynref-copy={symbolImportStatement(symbol)}
            aria-label={`Copy import statement for ${symbol.name}`}
            title={statement}
        >
            import
        </button>
    );
}

function SymbolHead({ symbol }: { symbol: ApiSymbol }) {
    return (
        <summary className="dynref-asb-head">
            <div className="dynref-asb-namecol">
                <span className="dynref-asb-nameline">
                    <span className="dynref-asb-caret" aria-hidden="true">▶</span>
                    <a className="dynref-asb-name" href={symbol.sourceHref}>{symbol.name}</a>
                    <CopyImport symbol={symbol} />
                </span>
                <a className="dynref-asb-src" href={symbol.sourceHref}>
                    {shortSource(symbol.sourcePath, symbol.sourceLine)}
                </a>
            </div>
            <p className="dynref-asb-summarytext">{symbol.summary || "\u2014"}</p>
        </summary>
    );
}

function MethodRow({ symbol, method }: { symbol: ApiSymbol; method: ApiMethod }) {
    return (
        <li className="dynref-asb-method" id={methodAnchorId(symbol, method)}>
            <a className="dynref-asb-method-name" href={method.sourceHref}>{method.name}</a>
            <div className="dynref-asb-method-body">
                {method.summary ? (
                    <p className="dynref-asb-method-summary">{method.summary}</p>
                ) : null}
                {method.signature ? (
                    <pre className="dynref-asb-method-sig">{method.signature}</pre>
                ) : null}
            </div>
        </li>
    );
}

function SymbolBody({ symbol }: { symbol: ApiSymbol }) {
    const hasMethods = symbol.methods.length > 0;
    return (
        <div className="dynref-asb-body">
            {symbol.signature ? (
                <pre className="dynref-asb-sig">{symbol.signature}</pre>
            ) : (
                <p className="dynref-asb-note">
                    No constructor or explicit signature captured; see source for detail.
                </p>
            )}
            {hasMethods ? (
                <>
                    <p className="dynref-asb-methods-heading">Public methods</p>
                    <ul className="dynref-asb-methods">
                        {symbol.methods.map((m) => (
                            <MethodRow key={`${symbol.qualname}.${m.name}`} symbol={symbol} method={m} />
                        ))}
                    </ul>
                </>
            ) : null}
        </div>
    );
}

function SymbolRow({ symbol }: { symbol: ApiSymbol }) {
    return (
        <details className="dynref-asb-row" data-kind={symbol.kind} id={symbolAnchorId(symbol)}>
            <SymbolHead symbol={symbol} />
            <SymbolBody symbol={symbol} />
        </details>
    );
}

function GroupHeading({ label, kind }: { label: string; kind: "classes" | "functions" }) {
    return <p className="dynref-asb-group" data-group={kind}>{label}</p>;
}

function FilterInputs() {
    return (
        <>
            <input className="dynref-vh" type="radio" id="asb-all" name="dynref-asb-filter" defaultChecked />
            <input className="dynref-vh" type="radio" id="asb-class" name="dynref-asb-filter" />
            <input className="dynref-vh" type="radio" id="asb-function" name="dynref-asb-filter" />
        </>
    );
}

function SurfaceHeader({ mod }: { mod: ApiModule }) {
    return (
        <div className="dynref-panel-header">
            <div>
                <p className="dynref-eyebrow">Public surface</p>
                <h3 className="dynref-h">
                    <span className="dynref-mono">{mod.name}</span>
                </h3>
            </div>
            <p className="dynref-muted">
                Source: <a className="dynref-mono" href={mod.sourceHref}>{mod.sourcePath}</a>
            </p>
        </div>
    );
}

interface Counts {
    classes: number;
    functions: number;
    total: number;
}

function moduleCounts(mod: ApiModule): Counts {
    const classes = mod.symbols.filter((s) => s.kind === "class").length;
    const functions = mod.symbols.filter((s) => s.kind === "function").length;
    return { classes, functions, total: mod.symbols.length };
}

function SurfaceSummary({ mod, counts }: { mod: ApiModule; counts: Counts }) {
    return (
        <p className="dynref-asb-summary-line">
            <span>{counts.classes} classes · {counts.functions} functions</span>
            <span>{mod.summary}</span>
        </p>
    );
}

function importSnippet(mod: ApiModule): string {
    const examples = mod.symbols.slice(0, 3).map(symbolImportStatement);
    const remaining = mod.symbols.length - examples.length;
    if (remaining > 0) examples.push(`# ... ${remaining} more public symbols`);
    return examples.join("\n");
}

function FilterRail({ counts }: { counts: Counts }) {
    return (
        <div className="dynref-asb-rail">
            <label className="dynref-asb-pill" htmlFor="asb-all">
                All · {counts.total}
            </label>
            <label className="dynref-asb-pill" htmlFor="asb-class">
                Classes · {counts.classes}
            </label>
            <label className="dynref-asb-pill" htmlFor="asb-function">
                Functions · {counts.functions}
            </label>
        </div>
    );
}

function SymbolList({ mod }: { mod: ApiModule }) {
    const classes = mod.symbols.filter((s) => s.kind === "class");
    const functions = mod.symbols.filter((s) => s.kind === "function");
    return (
        <div className="dynref-asb-list">
            {classes.length > 0 ? <GroupHeading label="Classes" kind="classes" /> : null}
            {classes.map((s) => (
                <SymbolRow key={s.qualname} symbol={s} />
            ))}
            {functions.length > 0 ? <GroupHeading label="Functions" kind="functions" /> : null}
            {functions.map((s) => (
                <SymbolRow key={s.qualname} symbol={s} />
            ))}
        </div>
    );
}

function UnknownModuleNotice() {
    return (
        <section className="dynref-panel">
            <p className="dynref-muted">
                This module's API surface is not available in this docs build.
            </p>
        </section>
    );
}

function findModule(name: string): ApiModule | undefined {
    return API_MODULES.find((m) => m.name === name);
}

interface ApiSurfaceBrowserProps {
    module: string;
}

export function ApiSurfaceBrowser({ module }: ApiSurfaceBrowserProps) {
    const mod = findModule(module);
    if (!mod) {
        return <UnknownModuleNotice />;
    }
    const counts = moduleCounts(mod);
    return (
        <>
            <style>{ASB_CSS}</style>
            <section className="dynref-panel">
                <FilterInputs />
                <SurfaceHeader mod={mod} />
                <SurfaceSummary mod={mod} counts={counts} />
                <pre className="dynref-asb-import">{importSnippet(mod)}</pre>
                <FilterRail counts={counts} />
                <SymbolList mod={mod} />
                <p className="dynref-asb-note">
                    Expand a row for the signature and, for classes, every public method.
                    The <span className="dynref-mono">import</span> pill copies the
                    complete import statement.
                </p>
            </section>
        </>
    );
}
