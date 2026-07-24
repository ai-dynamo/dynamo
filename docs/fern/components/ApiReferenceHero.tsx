/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * ApiReferenceHero — landing summary panel for the API Reference tab.
 *
 * Renders the "what is on this page" panel for the API Reference landing:
 * the language surfaces we cover, the source of truth (griffe over the
 * repository), and a per-language link rail keyed off API_MODULES. Server
 * component; shared vocabulary (panel, eyebrow, label, mono, badges) comes
 * from ReferenceStyles — place <ReferenceStyles /> on the page alongside
 * this component. Only the .dynref-arh-* layout classes are defined here,
 * following the same local-style pattern as ArtifactBrowser.tsx.
 *
 * Every link is emitted as a site route relative to the API landing page.
 * React href values bypass Fern's MDX link rewriter, so route-relative links
 * keep released snapshots on their own version rather than jumping to /dev.
 */

import { API_MODULES, type ApiModule } from "./api-reference.data";

const ARH_CSS = `
.dynref-arh-header {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: space-between;
    gap: 8px 16px;
    margin-bottom: 16px;
}

.dynref-arh-title {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 10px;
    margin: 0;
    color: var(--pst-color-text-base);
    font-size: 22px;
    font-weight: 600;
    line-height: 1.2;
}

.dynref-arh-meta {
    margin: 0;
}

.dynref-arh-meta a {
    color: inherit;
    text-decoration: underline;
}

.dynref-arh-langs {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 10px;
    margin-bottom: 14px;
}

.dynref-arh-lang {
    display: flex;
    flex-direction: column;
    gap: 6px;
    padding: 14px 16px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 10px;
    background: transparent;
}

.dark .dynref-arh-lang {
    background: #1d1d1d;
    border-color: #2e2e2e;
}

.dynref-arh-lang-name {
    display: flex;
    align-items: center;
    justify-content: space-between;
    color: var(--pst-color-text-base);
    font-size: 14px;
    font-weight: 600;
    gap: 6px;
}

.dynref-arh-lang-link {
    color: inherit;
    text-decoration: none;
}

.dynref-arh-lang-link:hover {
    text-decoration: underline;
    text-decoration-color: var(--nv-color-green, #76B900);
}

.dynref-arh-lang-desc {
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
    line-height: 1.4;
    margin: 0;
}

.dynref-arh-modules {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 2px;
}

.dynref-arh-module {
    display: inline-flex;
    align-items: center;
    padding: 3px 8px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 6px;
    color: var(--pst-color-text-base);
    font-family: var(--pst-font-family-monospace, ui-monospace, SFMono-Regular, Menlo, monospace);
    font-size: 12px;
    text-decoration: none;
    line-height: 1;
}

.dynref-arh-module:hover {
    border-color: var(--nv-color-green, #76B900);
    text-decoration: none;
}

.dark .dynref-arh-module {
    background: #161616;
    border-color: #333;
}

.dynref-arh-note {
    margin: 0;
    color: var(--pst-color-text-muted);
    font-size: 12.5px;
}
`;

interface LanguageCard {
    label: string;
    description: string;
    /** Site route relative to the rendered /reference/api landing URL. */
    landingHref: string;
    /** Optional API_MODULES for this language, when a typed data module
     *  ships. Empty for Rust / Kubernetes today. */
    modules: ApiModule[];
}

const LANGUAGE_CARDS: LanguageCard[] = [
    {
        label: "Python",
        description:
            "Worker decorators, HTTP frontend, planner / router / mocker configuration, and the Rust-backed runtime bindings.",
        landingHref: "api/python",
        modules: API_MODULES,
    },
    {
        label: "Rust",
        description:
            "Distributed runtime crates published on crates.io; see the release artifact inventory for currently shipped versions.",
        landingHref: "api/rust",
        modules: [],
    },
    {
        label: "Kubernetes",
        description:
            "DynamoGraphDeployment, DynamoGraphDeploymentRequest, and DynamoComponentDeployment custom-resource fields.",
        landingHref: "../kubernetes-api/full-api-reference",
        modules: [],
    },
];

function ModuleChips({ card }: { card: LanguageCard }) {
    if (card.modules.length === 0) {
        return null;
    }
    return (
        <div className="dynref-arh-modules">
            {card.modules.map((mod) => (
                <a
                    className="dynref-arh-module"
                    key={mod.name}
                    href={`api/python/${mod.slug}`}
                >
                    {mod.name}
                </a>
            ))}
        </div>
    );
}

function LangCard({ card }: { card: LanguageCard }) {
    return (
        <div className="dynref-arh-lang">
            <span className="dynref-arh-lang-name">
                <a className="dynref-arh-lang-link" href={card.landingHref}>{card.label}</a>
            </span>
            <p className="dynref-arh-lang-desc">{card.description}</p>
            <ModuleChips card={card} />
        </div>
    );
}

function HeroHeader({ symbolCount }: { symbolCount: number }) {
    const modLabel = API_MODULES.length === 1 ? "module" : "modules";
    return (
        <div className="dynref-arh-header">
            <div>
                <p className="dynref-eyebrow">API Reference</p>
                <div className="dynref-arh-title">
                    Programmatic surfaces across {LANGUAGE_CARDS.length} languages
                </div>
            </div>
            <p className="dynref-muted dynref-arh-meta">
                {API_MODULES.length} Python {modLabel} · {symbolCount} public symbols · statically discovered from{" "}
                <a href="https://github.com/ai-dynamo/dynamo">ai-dynamo/dynamo</a>
            </p>
        </div>
    );
}

function countSymbols(): number {
    return API_MODULES.reduce((n, m) => n + m.symbols.length, 0);
}

export function ApiReferenceHero() {
    return (
        <>
            <style>{ARH_CSS}</style>
            <section className="dynref-panel">
                <HeroHeader symbolCount={countSymbols()} />
                <div className="dynref-arh-langs">
                    {LANGUAGE_CARDS.map((card) => (
                        <LangCard card={card} key={card.label} />
                    ))}
                </div>
                <p className="dynref-arh-note">
                    Every symbol row on the module pages deep-links to its exact source file
                    and line on <span className="dynref-mono">main</span>. Types and signatures
                    come from the code, not from hand-maintained tables.
                </p>
            </section>
        </>
    );
}
