/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Compact categorized browser for the generated Rust crate inventory.
 */

import {
    RUST_BINDINGS,
    RUST_CRATES,
    RUST_WORKSPACE_VERSION,
    type RustBinding,
    type RustCrate,
    type RustCrateGroup,
} from "./rust-api-reference.data";

const GROUPS: { id: RustCrateGroup; label: string }[] = [
    { id: "core", label: "Core" },
    { id: "supporting", label: "Supporting" },
    { id: "development", label: "Development" },
    { id: "deprecated", label: "Deprecated" },
];

const RUST_CSS = `
.dynref-ari-group {
    margin: 13px 0 3px;
    color: var(--pst-color-text-muted);
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
}

.dynref-ari-row {
    display: grid;
    grid-template-columns: 190px minmax(0, 1fr) max-content;
    gap: 12px;
    align-items: center;
    padding: 8px 0;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

.dynref-ari-name {
    color: var(--pst-color-text-base);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 13px;
    font-weight: 600;
    text-decoration: none;
    overflow-wrap: anywhere;
}

.dynref-ari-name:hover,
.dynref-ari-source:hover {
    text-decoration: underline;
}

.dynref-ari-summary {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
    line-height: 1.35;
}

.dynref-ari-actions {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    gap: 6px;
    white-space: nowrap;
}

.dynref-ari-source {
    color: var(--pst-color-text-muted);
    font-size: 11px;
    text-decoration: none;
}

.dynref-ari-bindings {
    margin-top: 18px;
}

.dynref-ari-binding {
    display: grid;
    grid-template-columns: 190px 60px minmax(0, 1fr);
    gap: 12px;
    padding: 7px 0;
    border-bottom: 1px solid var(--border, var(--grayscale-a5));
}

#ari-all:checked ~ .dynref-filter-rail label[for="ari-all"],
#ari-core:checked ~ .dynref-filter-rail label[for="ari-core"],
#ari-supporting:checked ~ .dynref-filter-rail label[for="ari-supporting"],
#ari-development:checked ~ .dynref-filter-rail label[for="ari-development"],
#ari-deprecated:checked ~ .dynref-filter-rail label[for="ari-deprecated"] {
    border-color: var(--nv-color-green, #76B900);
    box-shadow: 0 0 0 1px var(--nv-color-green, #76B900);
    background: var(--dynref-green-bg);
    font-weight: 700;
}

#ari-core:checked ~ .dynref-ari-list > :not([data-group="core"]),
#ari-supporting:checked ~ .dynref-ari-list > :not([data-group="supporting"]),
#ari-development:checked ~ .dynref-ari-list > :not([data-group="development"]),
#ari-deprecated:checked ~ .dynref-ari-list > :not([data-group="deprecated"]) {
    display: none;
}

#ari-all:focus-visible ~ .dynref-filter-rail label[for="ari-all"],
#ari-core:focus-visible ~ .dynref-filter-rail label[for="ari-core"],
#ari-supporting:focus-visible ~ .dynref-filter-rail label[for="ari-supporting"],
#ari-development:focus-visible ~ .dynref-filter-rail label[for="ari-development"],
#ari-deprecated:focus-visible ~ .dynref-filter-rail label[for="ari-deprecated"] {
    outline: 2px solid var(--nv-color-green, #76B900);
    outline-offset: 2px;
}

@media (max-width: 640px) {
    .dynref-ari-row,
    .dynref-ari-binding {
        grid-template-columns: minmax(0, 1fr);
        gap: 4px;
    }

    .dynref-ari-actions {
        justify-content: flex-start;
    }
}
`;

function FilterInputs() {
    return (
        <>
            <input className="dynref-vh" type="radio" id="ari-all" name="ari-filter" defaultChecked />
            {GROUPS.map((group) => (
                <input className="dynref-vh" type="radio" id={`ari-${group.id}`} name="ari-filter" key={group.id} />
            ))}
        </>
    );
}

function FilterRail() {
    return (
        <div className="dynref-filter-rail">
            <label className="dynref-filter-pill" htmlFor="ari-all">All · {RUST_CRATES.length}</label>
            {GROUPS.map((group) => (
                <label className="dynref-filter-pill" htmlFor={`ari-${group.id}`} key={group.id}>
                    {group.label} · {RUST_CRATES.filter((crate) => crate.group === group.id).length}
                </label>
            ))}
        </div>
    );
}

function InstallButton({ crate }: { crate: RustCrate }) {
    return (
        <button
            className="dynref-copy dynref-badge dynref-badge--blue"
            type="button"
            data-dynref-copy={crate.installCommand}
            title={crate.installCommand}
        >
            cargo add
        </button>
    );
}

function CrateRow({ crate }: { crate: RustCrate }) {
    return (
        <div className="dynref-ari-row" data-group={crate.group}>
            <a className="dynref-ari-name" href={crate.docsHref}>{crate.name}</a>
            <p className="dynref-ari-summary">{crate.summary}</p>
            <div className="dynref-ari-actions">
                <span className="dynref-badge dynref-badge--gray">{crate.version}</span>
                {crate.badge ? <span className="dynref-badge dynref-badge--amber">{crate.badge}</span> : null}
                {crate.sourceHref ? <a className="dynref-ari-source" href={crate.sourceHref}>source</a> : null}
                <InstallButton crate={crate} />
            </div>
        </div>
    );
}

function CrateList() {
    return (
        <div className="dynref-ari-list">
            {GROUPS.map((group) => (
                <CrateGroup group={group} key={group.id} />
            ))}
        </div>
    );
}

function CrateGroup({ group }: { group: (typeof GROUPS)[number] }) {
    const crates = RUST_CRATES.filter((crate) => crate.group === group.id);
    return (
        <>
            <p className="dynref-ari-group" data-group={group.id}>{group.label}</p>
            {crates.map((crate) => <CrateRow crate={crate} key={crate.name} />)}
        </>
    );
}

function BindingRow({ binding }: { binding: RustBinding }) {
    return (
        <div className="dynref-ari-binding">
            <a className="dynref-ari-name" href={binding.sourceHref}>{binding.name}</a>
            <span className="dynref-badge dynref-badge--gray">{binding.language}</span>
            <p className="dynref-ari-summary">{binding.summary}</p>
        </div>
    );
}

function Bindings() {
    return (
        <div className="dynref-ari-bindings">
            <p className="dynref-ari-group">Language bindings</p>
            {RUST_BINDINGS.map((binding) => <BindingRow binding={binding} key={binding.name} />)}
        </div>
    );
}

function Header() {
    return (
        <div className="dynref-index-header">
            <div>
                <p className="dynref-eyebrow">Rust API</p>
                <p className="dynref-index-title">{RUST_CRATES.length} published crates</p>
            </div>
            <span className="dynref-badge dynref-badge--green">release {RUST_WORKSPACE_VERSION}</span>
        </div>
    );
}

export function ApiRustIndex() {
    return (
        <>
            <style>{RUST_CSS}</style>
            <section className="dynref-panel">
                <FilterInputs />
                <Header />
                <FilterRail />
                <CrateList />
                <Bindings />
            </section>
        </>
    );
}
