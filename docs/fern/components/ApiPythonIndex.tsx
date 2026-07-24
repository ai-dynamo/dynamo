/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ApiPythonIndex — the Python API landing page component.
 *
 * Compact grouped index of every curated Dynamo Python module. Each row
 * links to the per-module page (relative MDX path so Fern rewrites the
 * URL per snapshot at build time) and reports the module's class + function
 * counts. Reads the typed data written by
 * docs/fern/scripts/gen_python_api.py.
 *
 * Server component; shared vocabulary (panel, eyebrow, badges, mono) comes
 * from ReferenceStyles. Only the ``.dynref-api-*`` layout classes are
 * defined here.
 */

import { API_MODULES, type ApiModule } from "./api-reference.data";

const API_CSS = `
.dynref-api-list {
    display: grid;
    grid-template-columns: 220px minmax(0, 1fr) max-content;
    align-items: baseline;
    row-gap: 0;
    column-gap: 12px;
}

.dynref-api-row {
    display: contents;
}

.dynref-api-cell {
    padding: 8px 0;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynref-api-row:last-of-type .dynref-api-cell {
    border-bottom: 0;
}

.dynref-api-name {
    color: var(--pst-color-text-base);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 13.5px;
    font-weight: 600;
    text-decoration: none;
    overflow-wrap: anywhere;
}

.dynref-api-name:hover {
    text-decoration: underline;
    text-decoration-color: var(--nv-color-green, #76B900);
}

.dynref-api-summary {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
    line-height: 1.4;
}

.dynref-api-counts {
    display: flex;
    gap: 6px;
    justify-content: flex-end;
    white-space: nowrap;
}

.dynref-api-note {
    margin-top: 14px;
    color: var(--pst-color-text-muted);
    font-size: 12px;
}

@media (max-width: 640px) {
    .dynref-api-list {
        grid-template-columns: minmax(0, 1fr);
    }

    .dynref-api-counts {
        justify-content: flex-start;
    }
}
`;

function modulePageHref(mod: ApiModule): string {
    // React href values bypass Fern's MDX link rewriter. Prefix the landing
    // route segment so the browser resolves this against the current version.
    return `python/${mod.slug}`;
}

function moduleCounts(mod: ApiModule): { classes: number; functions: number } {
    const classes = mod.symbols.filter((s) => s.kind === "class").length;
    const functions = mod.symbols.filter((s) => s.kind === "function").length;
    return { classes, functions };
}

function totalSymbols(): number {
    return API_MODULES.reduce((n, m) => n + m.symbols.length, 0);
}

function IndexHeader() {
    return (
        <div className="dynref-index-header">
            <div>
                <p className="dynref-eyebrow">Python API</p>
                <div className="dynref-index-title">
                    {API_MODULES.length} curated Python modules
                </div>
            </div>
            <p className="dynref-index-meta">
                {totalSymbols()} public class + function symbols · statically
                discovered from{" "}
                <a href="https://github.com/ai-dynamo/dynamo">
                    ai-dynamo/dynamo
                </a>
            </p>
        </div>
    );
}

function CountBadges({ mod }: { mod: ApiModule }) {
    const { classes, functions } = moduleCounts(mod);
    return (
        <div className="dynref-api-cell dynref-api-counts">
            <span className="dynref-badge dynref-badge--blue">{classes} classes</span>
            <span className="dynref-badge dynref-badge--gray">{functions} fns</span>
        </div>
    );
}

function IndexRow({ mod }: { mod: ApiModule }) {
    return (
        <div className="dynref-api-row">
            <div className="dynref-api-cell">
                <a className="dynref-api-name" href={modulePageHref(mod)}>{mod.name}</a>
            </div>
            <div className="dynref-api-cell">
                <p className="dynref-api-summary">{mod.summary}</p>
            </div>
            <CountBadges mod={mod} />
        </div>
    );
}

export function ApiPythonIndex() {
    return (
        <>
            <style>{API_CSS}</style>
            <section className="dynref-panel">
                <IndexHeader />
                <div className="dynref-api-list">
                    {API_MODULES.map((mod) => (
                        <IndexRow mod={mod} key={mod.name} />
                    ))}
                </div>
                <p className="dynref-api-note">
                    Every module page groups public classes and functions, expands each
                    row into signatures and public methods, and deep-links to source on{" "}
                    <span className="dynref-mono">main</span>.
                </p>
            </section>
        </>
    );
}
