/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Hardware-aware install selector. Unsupported Intel XPU combinations
 * stay visible but disabled so readers can understand the support boundary.
 */
"use client";

import { useState } from "react";

import {
  INSTALL_DATA,
  type InstallBackend,
  type InstallChannel,
  type InstallForm,
} from "./install-selector-data";
import { LOCAL_SELECTOR_CSS } from "./local-selector-styles";

type Hardware = "nvidia" | "intel";

type Option<T extends string> = {
  id: T;
  label: string;
  sub?: string;
};

const HARDWARES: Option<Hardware>[] = [
  { id: "nvidia", label: "NVIDIA GPU", sub: "CUDA" },
  { id: "intel", label: "Intel XPU", sub: "source build" },
];

const BACKENDS: Option<InstallBackend>[] = [
  { id: "sglang", label: "SGLang" },
  { id: "trtllm", label: "TensorRT-LLM" },
  { id: "vllm", label: "vLLM" },
];

const CHANNELS: Option<InstallChannel>[] = [
  { id: "stable", label: "Stable release", sub: "QA-validated" },
  { id: "nightly", label: "Nightly", sub: "latest features" },
  { id: "source", label: "Source build", sub: "main branch" },
];

const FORMS: Option<InstallForm>[] = [
  { id: "container", label: "Container" },
  { id: "wheel", label: "Wheel" },
];

function channelEnabled(hardware: Hardware, channel: InstallChannel): boolean {
  return hardware === "nvidia" ? channel !== "source" : channel === "source";
}

function backendEnabled(hardware: Hardware, backend: InstallBackend): boolean {
  return hardware === "nvidia" || INSTALL_DATA[backend].source.length > 0;
}

function disabledReason(
  kind: "backend" | "channel" | "form",
  hardware: Hardware,
  label: string,
): string {
  if (kind === "backend") return `${label} does not currently have an Intel XPU local runtime path.`;
  if (kind === "channel") {
    return hardware === "intel"
      ? "Intel XPU runtime images are built from source rather than published as stable or nightly artifacts."
      : "Source build is only used for the Intel XPU path in this quickstart.";
  }
  return "A wheel is not available for this hardware, backend, and build combination.";
}

function SelectorRow<T extends string>({
  label,
  options,
  selected,
  onSelect,
  isDisabled = () => false,
  disabledTitle,
}: {
  label: string;
  options: Option<T>[];
  selected: T;
  onSelect: (value: T) => void;
  isDisabled?: (value: T) => boolean;
  disabledTitle?: (option: Option<T>) => string;
}) {
  return (
    <div className="lqs-row">
      <span className="lqs-label">{label}</span>
      <div className="lqs-options" role="group" aria-label={label}>
        {options.map((option) => {
          const disabled = isDisabled(option.id);
          return (
            <button
              key={option.id}
              type="button"
              className="lqs-chip"
              aria-pressed={selected === option.id}
              disabled={disabled}
              title={disabled ? disabledTitle?.(option) : undefined}
              onClick={() => onSelect(option.id)}
            >
              {option.label}
              {option.sub && <span className="lqs-chip-sub">{option.sub}</span>}
            </button>
          );
        })}
      </div>
    </div>
  );
}

export function InstallSelector({ hardware = "all" }: { hardware?: "all" | "nvidia" }) {
  const [selectedHardware, setSelectedHardware] = useState<Hardware>("nvidia");
  const [backend, setBackend] = useState<InstallBackend>("sglang");
  const [channel, setChannel] = useState<InstallChannel>("stable");
  const [versionIndex, setVersionIndex] = useState(0);
  const [form, setForm] = useState<InstallForm>("container");
  const [copyLabel, setCopyLabel] = useState("Copy");

  const activeHardware = hardware === "nvidia" ? "nvidia" : selectedHardware;
  const entries = INSTALL_DATA[backend][channel];
  const entry = entries[versionIndex] ?? entries[0];
  const command = entry?.commands[form] ?? entry?.commands.container ?? "";

  function chooseHardware(next: Hardware) {
    setSelectedHardware(next);
    setVersionIndex(0);
    setForm("container");
    if (next === "intel") {
      if (!backendEnabled(next, backend)) setBackend("vllm");
      setChannel("source");
    } else {
      setChannel("stable");
    }
  }

  function chooseBackend(next: InstallBackend) {
    setBackend(next);
    setVersionIndex(0);
    setForm("container");
  }

  function chooseChannel(next: InstallChannel) {
    setChannel(next);
    setVersionIndex(0);
    setForm("container");
  }

  async function copyCommand() {
    if (!command || !navigator.clipboard) return;
    await navigator.clipboard.writeText(command);
    setCopyLabel("Copied!");
    window.setTimeout(() => setCopyLabel("Copy"), 1200);
  }

  const badge = channel === "stable" ? "Stable" : channel === "nightly" ? "Nightly" : "Source";
  const title = channel === "stable"
    ? `Dynamo ${entry?.dynamo}`
    : channel === "nightly"
      ? entry?.latest
        ? "Latest nightly"
        : `Nightly ${entry?.dynamo}`
      : "Build Dynamo from source";
  const role = channel === "stable"
    ? "Latest stable release that supports this version"
    : channel === "nightly"
      ? entry?.latest
        ? "Latest nightly build"
        : "Pinned nightly wheel build"
      : "Intel XPU local runtime";
  const hardwareLabel = activeHardware === "nvidia" ? "NVIDIA GPU" : "Intel XPU";
  const versionRowLabel = `${INSTALL_DATA[backend].label} version`;

  return (
    <>
      <style>{LOCAL_SELECTOR_CSS}</style>
      <section className="lqs-panel" aria-label="Dynamo install selector">
        <div className="lqs-head">
          <h3>Choose your build</h3>
          <p>Unavailable combinations remain visible to show the current support boundary.</p>
        </div>

        {hardware === "all" && (
          <SelectorRow
            label="Hardware"
            options={HARDWARES}
            selected={activeHardware}
            onSelect={chooseHardware}
          />
        )}

        <SelectorRow
          label="Backend"
          options={BACKENDS}
          selected={backend}
          onSelect={chooseBackend}
          isDisabled={(value) => !backendEnabled(activeHardware, value)}
          disabledTitle={(option) => disabledReason("backend", activeHardware, option.label)}
        />

        <SelectorRow
          label="Dynamo build"
          options={CHANNELS}
          selected={channel}
          onSelect={chooseChannel}
          isDisabled={(value) => !channelEnabled(activeHardware, value)}
          disabledTitle={(option) => disabledReason("channel", activeHardware, option.label)}
        />

        {entry && (
          <div className="lqs-row">
            <span className="lqs-label">{versionRowLabel}</span>
            <div className="lqs-options" role="group" aria-label={versionRowLabel}>
              {entries.map((version, index) => {
                const hasContainer = Boolean(version.commands.container);
                const hasWheel = Boolean(version.commands.wheel);
                // Engine version leads on every channel, Dynamo version beneath
                // it, so a nightly chip reads the same way as a stable one.
                const displayVersion = version.backend_version;
                const displayMeta = version.source
                  ? "from main"
                  : version.dynamo
                    ? `Dynamo ${version.dynamo}`
                    : version.latest
                      ? "latest nightly"
                      : version.date ?? "nightly wheel";
                return (
                  <button
                    key={`${version.backend_version}-${version.dynamo ?? index}`}
                    type="button"
                    className="lqs-chip"
                    aria-pressed={versionIndex === index}
                    onClick={() => {
                      setVersionIndex(index);
                      setForm(hasContainer ? "container" : hasWheel ? "wheel" : "container");
                    }}
                  >
                    {displayVersion}
                    <span className="lqs-chip-sub">{displayMeta}</span>
                  </button>
                );
              })}
            </div>
          </div>
        )}

        <SelectorRow
          label="Install form"
          options={FORMS}
          selected={form}
          onSelect={setForm}
          isDisabled={(value) => !entry?.commands[value]}
          disabledTitle={(option) => disabledReason("form", activeHardware, option.label)}
        />

        {entry && (
          <div className="lqs-output">
            <div className={`lqs-rec lqs-rec--${channel}`}>
              <div className="lqs-eyebrow">{role}</div>
              <div className="lqs-title">
                <span className="lqs-badge">{badge}</span>
                {title}
              </div>
              <div className="lqs-support">
                {hardwareLabel} · {INSTALL_DATA[backend].label} {entry.backend_version}
              </div>
            </div>
            <div className="lqs-command">
              <button type="button" className="lqs-copy" onClick={copyCommand}>
                {copyLabel}
              </button>
              <pre>{command}</pre>
              {entry.note && <p className="lqs-hint">{entry.note}</p>}
            </div>
          </div>
        )}
      </section>
    </>
  );
}

export default InstallSelector;
