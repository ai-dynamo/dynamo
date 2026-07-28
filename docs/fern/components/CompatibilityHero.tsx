/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * CompatibilityHero — version-selectable compatibility summary for the
 * Compatibility page. Combines backend, NIXL, UCX, CUDA toolkit, minimum
 * driver, and current platform requirements in one card.
 */

"use client";

import { useEffect, useMemo, useState } from "react";

import {
  RELEASES,
  INTEL_RELEASES,
  MAIN_TOT,
  INTEL_MAIN_TOT,
  CURRENT_VERSION,
  CUDA_HISTORY,
  XPU_HISTORY,
  PLATFORM,
  INTEL_PLATFORM,
  type BackendPins,
  type Release,
} from "./releases.data";

const VERSION_PARAM = "compat-version";
const HARDWARE_PARAM = "compat-hardware";
const MAIN_VALUE = "main";
type Hardware = "nvidia" | "intel";

const HERO_CSS = `
.dynref-hero-header {
    display: flex;
    flex-wrap: wrap;
    align-items: flex-end;
    justify-content: space-between;
    gap: 12px 20px;
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

.dynref-hero-selector-wrap {
    display: grid;
    gap: 5px;
    min-width: min(100%, 220px);
}

.dynref-hero-selectors {
    display: flex;
    flex-wrap: nowrap;
    align-items: flex-end;
    justify-content: flex-end;
    gap: 10px;
}

.dynref-hero-hardware {
    display: grid;
    gap: 5px;
}

.dynref-hero-hardware-rail {
    display: flex;
    flex-wrap: nowrap;
    gap: 8px;
}

.dynref-hero-hardware-pill {
    display: inline-flex;
    align-items: center;
    min-height: 30px;
    padding: 6px 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 999px;
    background: transparent;
    color: var(--pst-color-text-base);
    font: inherit;
    font-size: 12.5px;
    line-height: 1;
    cursor: pointer;
}

.dynref-hero-hardware-pill:hover {
    border-color: var(--nv-color-green, #76B900);
}

.dynref-hero-hardware-pill[aria-pressed="true"] {
    border-color: var(--nv-color-green, #76B900);
    box-shadow: 0 0 0 1px var(--nv-color-green, #76B900);
    background: rgba(118, 185, 0, 0.08);
    font-weight: 700;
}

.dynref-hero-hardware-pill:focus-visible {
    outline: 2px solid var(--nv-color-green, #76B900);
    outline-offset: 1px;
}

.dynref-hero-selector {
    min-height: 38px;
    padding: 6px 34px 6px 10px;
    border: 1px solid var(--border, var(--grayscale-a5));
    border-radius: 8px;
    background: var(--pst-color-surface);
    color: var(--pst-color-text-base);
    font: inherit;
    font-size: 14px;
}

.dark .dynref-hero-selector {
    background: #1d1d1d;
    border-color: #3a3a3a;
}

.dynref-hero-meta {
    margin: -4px 0 16px;
}

.dynref-hero-meta a {
    color: inherit;
    text-decoration: underline;
}

.dynref-hero-backends {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
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

.dynref-hero-backend-name {
    display: block;
    color: var(--pst-color-text-base);
    font-size: 14px;
    font-weight: 600;
}

.dynref-hero-pin {
    display: block;
    margin: 4px 0 2px;
    color: var(--pst-color-text-base);
    font-size: 15px;
}

.dynref-hero-dependency {
    display: grid;
    grid-template-columns: auto 1fr;
    gap: 3px 8px;
    align-items: baseline;
    margin-top: 6px;
}

.dynref-hero-cuda-row {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 7px;
    margin-top: 8px;
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

.dynref-hero-empty {
    margin: 8px 0 0;
    color: var(--pst-color-text-muted);
    font-size: 12px;
}

@media (max-width: 520px) {
    .dynref-hero-selectors { flex-wrap: wrap; width: 100%; }
    .dynref-hero-hardware-rail { flex-wrap: wrap; }
    .dynref-hero-selector-wrap { width: 100%; }
    .dynref-hero-reqs { grid-template-columns: 1fr; gap: 4px; }
}
`;

type BackendKey = "sglang" | "trtllm" | "vllm";
type NixlKey = "nixlSglang" | "nixlTrtllm" | "nixlVllm";

interface BackendDefinition {
  key: BackendKey;
  nixlKey: NixlKey;
  label: "SGLang" | "TensorRT-LLM" | "vLLM";
}

const BACKENDS: BackendDefinition[] = [
  { key: "sglang", nixlKey: "nixlSglang", label: "SGLang" },
  { key: "trtllm", nixlKey: "nixlTrtllm", label: "TensorRT-LLM" },
  { key: "vllm", nixlKey: "nixlVllm", label: "vLLM" },
];

function releaseType(release?: Release): { variant: "green" | "gray" | "amber"; label: string } {
  if (!release) return { variant: "gray", label: "development" };
  if (release.kind === "stable") return { variant: "green", label: "GA release" };
  if (release.kind === "patch") return { variant: "gray", label: "Patch" };
  if (release.kind === "model-build") return { variant: "amber", label: "Model build" };
  return { variant: "amber", label: "Early access" };
}

function optionLabel(release: Release): string {
  const suffix = release.kind === "stable" ? "GA" : release.kind === "patch" ? "patch" : "preview";
  return `${release.version} — ${suffix}`;
}

function HardwareSelector({
  hardware,
  onChange,
}: {
  hardware: Hardware;
  onChange: (hardware: Hardware) => void;
}) {
  return (
    <div className="dynref-hero-hardware">
      <span className="dynref-label">Accelerator</span>
      <div className="dynref-hero-hardware-rail" role="group" aria-label="Accelerator">
        <button
          className="dynref-hero-hardware-pill"
          type="button"
          aria-pressed={hardware === "nvidia"}
          onClick={() => onChange("nvidia")}
        >
          NVIDIA GPU
        </button>
        <button
          className="dynref-hero-hardware-pill"
          type="button"
          aria-pressed={hardware === "intel"}
          onClick={() => onChange("intel")}
        >
          Intel GPU
        </button>
      </div>
    </div>
  );
}

export function CompatibilityHero() {
  const [hardware, setHardware] = useState<Hardware>("nvidia");
  const [selectedVersion, setSelectedVersion] = useState(CURRENT_VERSION);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const requestedHardware = params.get(HARDWARE_PARAM);
    const requested = params.get(VERSION_PARAM);
    if (requestedHardware === "intel" || requestedHardware === "nvidia") {
      setHardware(requestedHardware);
    }
    if (requested === MAIN_VALUE || RELEASES.some((release) => release.version === requested)) {
      setSelectedVersion(requested);
    }
  }, []);

  const isMain = selectedVersion === MAIN_VALUE;
  const selectedRelease = isMain
    ? undefined
    : RELEASES.find((release) => release.version === selectedVersion);
  const pins: BackendPins = isMain ? MAIN_TOT : (selectedRelease?.pins ?? {});
  const badge = releaseType(selectedRelease);
  const cudaVersion = selectedVersion.replace(/^v/, "");
  const cudaRows = useMemo(
    () => (isMain ? [] : CUDA_HISTORY.filter((row) => row.version === cudaVersion)),
    [cudaVersion, isMain],
  );

  function selectVersion(version: string) {
    setSelectedVersion(version);
    const url = new URL(window.location.href);
    if (version === CURRENT_VERSION) url.searchParams.delete(VERSION_PARAM);
    else url.searchParams.set(VERSION_PARAM, version);
    window.history.replaceState({}, "", url);
  }

  function selectHardware(nextHardware: Hardware) {
    const nextVersion =
      nextHardware === "intel" &&
      selectedVersion !== MAIN_VALUE &&
      !INTEL_RELEASES.some((release) => release.version === selectedVersion)
        ? CURRENT_VERSION
        : selectedVersion;
    setHardware(nextHardware);
    setSelectedVersion(nextVersion);
    const url = new URL(window.location.href);
    if (nextHardware === "nvidia") url.searchParams.delete(HARDWARE_PARAM);
    else url.searchParams.set(HARDWARE_PARAM, nextHardware);
    if (nextVersion === CURRENT_VERSION) url.searchParams.delete(VERSION_PARAM);
    else url.searchParams.set(VERSION_PARAM, nextVersion);
    window.history.replaceState({}, "", url);
  }

  if (hardware === "intel") {
    return (
      <IntelCompatibilityHero
        selectedVersion={selectedVersion}
        onVersionChange={selectVersion}
        onHardwareChange={selectHardware}
      />
    );
  }

  return (
    <>
      <style>{HERO_CSS}</style>
      <section className="dynref-panel" aria-labelledby="compatibility-selection-title">
        <div className="dynref-hero-header">
          <div>
            <p className="dynref-eyebrow">Compatibility by version</p>
            <div className="dynref-hero-title" id="compatibility-selection-title">
              {isMain ? "Dynamo main branch" : `Dynamo ${selectedRelease?.version ?? selectedVersion}`}
              <span className={`dynref-badge dynref-badge--${badge.variant}`}>{badge.label}</span>
            </div>
          </div>

          <div className="dynref-hero-selectors">
            <HardwareSelector hardware={hardware} onChange={selectHardware} />
            <label className="dynref-hero-selector-wrap">
              <span className="dynref-label">Dynamo version</span>
              <select
                className="dynref-hero-selector"
                value={selectedVersion}
                onChange={(event) => selectVersion(event.target.value)}
              >
                <option value={MAIN_VALUE}>main branch — development</option>
                {RELEASES.map((release) => (
                  <option value={release.version} key={release.version}>
                    {optionLabel(release)}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>

        <p className="dynref-muted dynref-hero-meta">
          {isMain ? (
            "Unreleased dependency pins from the tip of the main branch."
          ) : (
            <>
              Released {selectedRelease?.date ?? "date unavailable"}
              {selectedRelease?.github && (
                <>
                  {" · "}
                  <a href={selectedRelease.github}>Release notes</a>
                </>
              )}
              {selectedRelease?.ucx && (
                <>
                  {" · "}UCX <span className="dynref-mono">{selectedRelease.ucx}</span>
                </>
              )}
            </>
          )}
        </p>

        <div className="dynref-hero-backends">
          {BACKENDS.map((backend) => {
            const backendCuda = cudaRows.filter((row) => row.backend === backend.label);
            return (
              <div className="dynref-hero-backend" key={backend.label}>
                <span className="dynref-hero-backend-name">{backend.label}</span>
                <span className="dynref-mono dynref-hero-pin">
                  {pins[backend.key] ?? "Not included"}
                </span>
                <div className="dynref-hero-dependency">
                  <span className="dynref-label">NIXL</span>
                  <span className="dynref-mono">{pins[backend.nixlKey] ?? "—"}</span>
                </div>
                {backendCuda.map((row) => (
                  <div className="dynref-hero-cuda-row" key={row.toolkit}>
                    <span
                      className={`dynref-chip dynref-chip--cuda${
                        row.note === "Experimental" ? " dynref-chip--exp" : ""
                      }`}
                    >
                      CUDA {row.toolkit}
                    </span>
                    <span className="dynref-muted">
                      Driver <span className="dynref-mono">{row.minDriver}</span>
                    </span>
                  </div>
                ))}
                {backendCuda.length === 0 && (
                  <p className="dynref-hero-empty">
                    {isMain ? "CUDA and driver requirements are published at release." : "No CUDA requirement recorded for this build."}
                  </p>
                )}
              </div>
            );
          })}
        </div>

        {selectedVersion === CURRENT_VERSION && (
          <div className="dynref-hero-reqs">
            <span className="dynref-label">GPU</span>
            <div className="dynref-hero-req-values">
              {PLATFORM.gpus.map((gpu) => (
                <span className="dynref-chip dynref-chip--gpu" key={gpu}>{gpu}</span>
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
                <span className="dynref-chip dynref-chip--arch" key={arch}>{arch}</span>
              ))}
            </div>
          </div>
        )}

        {selectedRelease?.note && <p className="dynref-muted dynref-grid-note">{selectedRelease.note}</p>}
        {selectedRelease?.delta && <p className="dynref-muted dynref-grid-note">{selectedRelease.delta}</p>}

        <p className="dynref-muted dynref-grid-note">
          Backend versions listed are the versions tested and supported for the selected release.
          TensorRT-LLM does not support Python 3.11.
        </p>
      </section>
    </>
  );
}

function IntelCompatibilityHero({
  selectedVersion,
  onVersionChange,
  onHardwareChange,
}: {
  selectedVersion: string;
  onVersionChange: (version: string) => void;
  onHardwareChange: (hardware: Hardware) => void;
}) {
  const isMain = selectedVersion === MAIN_VALUE;
  const selectedRelease = isMain
    ? undefined
    : INTEL_RELEASES.find((candidate) => candidate.version === selectedVersion);
  const pins: BackendPins = isMain ? INTEL_MAIN_TOT : (selectedRelease?.pins ?? {});
  const badge = releaseType(selectedRelease);
  const xpuVersion = selectedVersion.replace(/^v/, "");

  return (
    <>
      <style>{HERO_CSS}</style>
      <section className="dynref-panel" aria-labelledby="intel-compatibility-selection-title">
        <div className="dynref-hero-header">
          <div>
            <p className="dynref-eyebrow">Compatibility by version</p>
            <div className="dynref-hero-title" id="intel-compatibility-selection-title">
              {isMain
                ? "Dynamo main branch"
                : `Dynamo ${selectedRelease?.version ?? selectedVersion}`}
              <span className={`dynref-badge dynref-badge--${badge.variant}`}>{badge.label}</span>
            </div>
          </div>

          <div className="dynref-hero-selectors">
            <HardwareSelector hardware="intel" onChange={onHardwareChange} />
            <label className="dynref-hero-selector-wrap">
              <span className="dynref-label">Dynamo version</span>
              <select
                className="dynref-hero-selector"
                value={selectedVersion}
                onChange={(event) => onVersionChange(event.target.value)}
              >
                <option value={MAIN_VALUE}>main branch — development</option>
                {INTEL_RELEASES.map((candidate) => (
                  <option value={candidate.version} key={candidate.version}>
                    {optionLabel(candidate)}
                  </option>
                ))}
              </select>
            </label>
          </div>
        </div>

        <p className="dynref-muted dynref-hero-meta">
          {isMain ? (
            "Unreleased dependency pins from the tip of the main branch."
          ) : selectedRelease ? (
            <>
              Released {selectedRelease.date ?? "date unavailable"}
              {selectedRelease.github && (
                <>
                  {" · "}
                  <a href={selectedRelease.github}>Release notes</a>
                </>
              )}
              {selectedRelease.ucx && (
                <>
                  {" · "}UCX <span className="dynref-mono">{selectedRelease.ucx}</span>
                </>
              )}
            </>
          ) : (
            "Release data unavailable."
          )}
        </p>

        <div className="dynref-hero-backends">
          {BACKENDS.filter((backend) => pins[backend.key]).map((backend) => {
            const xpuRows = XPU_HISTORY.filter(
              (row) => row.version === xpuVersion && row.backend === backend.label,
            );
            return (
              <div className="dynref-hero-backend" key={backend.label}>
                <span className="dynref-hero-backend-name">{backend.label}</span>
                <span className="dynref-mono dynref-hero-pin">{pins[backend.key]}</span>
                <div className="dynref-hero-dependency">
                  <span className="dynref-label">NIXL</span>
                  <span className="dynref-mono">{pins[backend.nixlKey] ?? "—"}</span>
                </div>
                {xpuRows.map((row) => (
                  <div
                    className="dynref-hero-cuda-row"
                    key={`${row.version}-${row.backend}`}
                  >
                    <span className="dynref-chip dynref-chip--cuda">
                      oneAPI {row.oneapi}
                    </span>
                    <span className="dynref-muted">
                      Driver <span className="dynref-mono">{row.minDriver}</span>
                    </span>
                  </div>
                ))}
                {xpuRows.length === 0 && (
                  <p className="dynref-hero-empty">
                    oneAPI and driver requirements are published at release.
                  </p>
                )}
              </div>
            );
          })}
        </div>

        {!isMain && (
          <div className="dynref-hero-reqs">
          <span className="dynref-label">GPU</span>
          <div className="dynref-hero-req-values">
            {INTEL_PLATFORM.gpus.map((gpu) => (
              <span className="dynref-chip dynref-chip--gpu" key={gpu}>{gpu}</span>
            ))}
          </div>

          <span className="dynref-label">OS</span>
          <div className="dynref-hero-req-values">
            {INTEL_PLATFORM.os.map((row) => (
              <span className={`dynref-chip dynref-chip--${row.chip}`} key={`${row.name} ${row.version}`}>
                {row.name} {row.version}
              </span>
            ))}
          </div>

          <span className="dynref-label">Arch</span>
          <div className="dynref-hero-req-values">
            {INTEL_PLATFORM.arch.map((arch) => (
              <span className="dynref-chip dynref-chip--arch" key={arch}>{arch}</span>
            ))}
          </div>
          </div>
        )}

        <p className="dynref-muted dynref-grid-note">
          Backend versions listed are the versions tested and supported for the selected release.
        </p>
      </section>
    </>
  );
}
