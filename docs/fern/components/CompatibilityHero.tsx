/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CompatibilityHero — current-release summary panel for the Compatibility page.
 *
 * Renders the current stable release (version, date, release-notes link), the
 * per-backend engine + NIXL pins, and the platform requirement rows (GPU, OS,
 * arch, CUDA) from releases.data.ts. Server component; shared vocabulary
 * (panel, eyebrow, label, mono, chips, badges) comes from ReferenceStyles —
 * place <ReferenceStyles /> on the page alongside this component. Only the
 * .dynref-hero-* layout classes are defined here.
 */

import {
  RELEASES,
  CURRENT_VERSION,
  CURRENT_DATE,
  PLATFORM,
} from "./releases.data";

const HERO_CSS = `
.dynref-hero-header {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: space-between;
    gap: 8px 16px;
    margin-bottom: 16px;
}

.dynref-hero-title {
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

.dynref-hero-meta {
    margin: 0;
}

.dynref-hero-meta a {
    color: inherit;
    text-decoration: underline;
}

.dynref-hero-backends {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 10px;
}

.dynref-hero-backend {
    padding: 12px 14px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 10px;
}

.dark .dynref-hero-backend {
    background: #1d1d1d;
    border-color: #2e2e2e;
}

.dynref-hero-pin {
    display: block;
    margin: 4px 0 2px;
    color: var(--pst-color-text-base);
    font-size: 15px;
}

.dynref-hero-reqs {
    display: grid;
    grid-template-columns: 88px 1fr;
    gap: 8px 12px;
    align-items: baseline;
    margin-top: 16px;
    padding-top: 14px;
    border-top: 1px solid var(--border, var(--grayscale-a5));
}

.dynref-hero-req-values {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 0 4px;
}
`;

interface BackendCard {
  label: string;
  pin?: string;
  nixl?: string;
}

export function CompatibilityHero() {
  const current = RELEASES.find((r) => r.version === CURRENT_VERSION);
  const pins = current?.pins ?? {};

  const backends: BackendCard[] = [
    { label: "SGLang", pin: pins.sglang, nixl: pins.nixlSglang },
    { label: "TensorRT-LLM", pin: pins.trtllm, nixl: pins.nixlTrtllm },
    { label: "vLLM", pin: pins.vllm, nixl: pins.nixlVllm },
  ];

  return (
    <>
      <style>{HERO_CSS}</style>
      <section className="dynref-panel">
        <div className="dynref-hero-header">
          <div>
            <p className="dynref-eyebrow">Current release</p>
            <div className="dynref-hero-title">
              Dynamo {CURRENT_VERSION}
              <span className="dynref-badge dynref-badge--green">Stable</span>
            </div>
          </div>
          <p className="dynref-muted dynref-hero-meta">
            Released {current?.date ?? CURRENT_DATE} ·{" "}
            <a href={current?.github}>Release notes</a>
          </p>
        </div>

        <div className="dynref-hero-backends">
          {backends.map((backend) => (
            <div className="dynref-hero-backend" key={backend.label}>
              <span className="dynref-label">{backend.label}</span>
              <span className="dynref-mono dynref-hero-pin">{backend.pin}</span>
              <span className="dynref-muted">
                NIXL <span className="dynref-mono">{backend.nixl}</span>
              </span>
            </div>
          ))}
        </div>

        <div className="dynref-hero-reqs">
          <span className="dynref-label">GPU</span>
          <div className="dynref-hero-req-values">
            {PLATFORM.gpus.map((gpu) => (
              <span className="dynref-chip dynref-chip--gpu" key={gpu}>
                {gpu}
              </span>
            ))}
          </div>

          <span className="dynref-label">OS</span>
          <div className="dynref-hero-req-values">
            {PLATFORM.os.map((row) => (
              <span
                className={
                  row.status === "Experimental"
                    ? "dynref-chip dynref-chip--amber dynref-chip--exp"
                    : `dynref-chip dynref-chip--${row.chip}`
                }
                key={`${row.name} ${row.version}`}
              >
                {row.name} {row.version}
                {row.status === "Experimental" ? " · experimental" : ""}
              </span>
            ))}
          </div>

          <span className="dynref-label">Arch</span>
          <div className="dynref-hero-req-values">
            {PLATFORM.arch.map((arch) => (
              <span className="dynref-chip dynref-chip--arch" key={arch}>
                {arch}
              </span>
            ))}
          </div>

          <span className="dynref-label">CUDA</span>
          <div className="dynref-hero-req-values">
            <span className="dynref-chip dynref-chip--cuda">
              13.0 · SGLang, vLLM
            </span>
            <span className="dynref-chip dynref-chip--cuda">
              13.1 · TensorRT-LLM
            </span>
            <span className="dynref-muted">
              CUDA 12 discontinued as of {CURRENT_VERSION}
            </span>
          </div>
        </div>

        <p className="dynref-muted dynref-grid-note">
          Early access: model builds are tracked in{" "}
          <a href="/dynamo/dev/reference/model-early-access-builds">
            Model Early Access Builds
          </a>
          ; platform previews under{" "}
          <a href="/dynamo/dev/reference/release-artifacts#early-access-artifacts">
            Early Access Artifacts
          </a>
          .
        </p>
      </section>
    </>
  );
}
