/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Two-axis worker launcher for the local CLI quickstart. It replaces nested
 * Fern tabs with one cohesive hardware/backend selector panel.
 */
"use client";

import { useState } from "react";

import { LOCAL_SELECTOR_CSS } from "./local-selector-styles";

type Hardware = "nvidia" | "intel";
type Backend = "sglang" | "trtllm" | "vllm";

type Option<T extends string> = { id: T; label: string; sub?: string };

const HARDWARES: Option<Hardware>[] = [
  { id: "nvidia", label: "NVIDIA GPU", sub: "CUDA" },
  { id: "intel", label: "Intel XPU", sub: "XPU runtime" },
];

const BACKENDS: Option<Backend>[] = [
  { id: "sglang", label: "SGLang" },
  { id: "trtllm", label: "TensorRT-LLM" },
  { id: "vllm", label: "vLLM" },
];

const COMMANDS: Record<Hardware, Partial<Record<Backend, string>>> = {
  nvidia: {
    sglang: "python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file",
    trtllm: "python3 -m dynamo.trtllm --model-path Qwen/Qwen3-0.6B --discovery-backend file",
    vllm: "python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file",
  },
  intel: {
    sglang: "python3 -m dynamo.sglang --model-path Qwen/Qwen3-0.6B --discovery-backend file",
    vllm: "VLLM_TARGET_DEVICE=xpu \\\n  python3 -m dynamo.vllm --model Qwen/Qwen3-0.6B --discovery-backend file",
  },
};

function ChoiceRow<T extends string>({
  label,
  options,
  selected,
  onSelect,
  isDisabled = () => false,
}: {
  label: string;
  options: Option<T>[];
  selected: T;
  onSelect: (value: T) => void;
  isDisabled?: (value: T) => boolean;
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
              title={disabled ? `${option.label} does not currently support the Intel XPU local path.` : undefined}
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

export function WorkerSelector() {
  const [hardware, setHardware] = useState<Hardware>("nvidia");
  const [backend, setBackend] = useState<Backend>("sglang");
  const [copyLabel, setCopyLabel] = useState("Copy");

  const command = COMMANDS[hardware][backend] ?? "";
  const backendLabel = BACKENDS.find((option) => option.id === backend)?.label ?? backend;
  const hardwareLabel = hardware === "nvidia" ? "NVIDIA GPU" : "Intel XPU";

  function chooseHardware(next: Hardware) {
    setHardware(next);
    if (!COMMANDS[next][backend]) setBackend("vllm");
  }

  async function copyCommand() {
    if (!command || !navigator.clipboard) return;
    await navigator.clipboard.writeText(command);
    setCopyLabel("Copied!");
    window.setTimeout(() => setCopyLabel("Copy"), 1200);
  }

  return (
    <>
      <style>{LOCAL_SELECTOR_CSS}</style>
      <section className="lqs-panel" aria-label="Dynamo worker command selector">
        <div className="lqs-head">
          <h3>Choose your worker command</h3>
          <p>Select the same hardware and backend you used for the installation.</p>
        </div>

        <ChoiceRow label="Hardware" options={HARDWARES} selected={hardware} onSelect={chooseHardware} />
        <ChoiceRow
          label="Backend"
          options={BACKENDS}
          selected={backend}
          onSelect={setBackend}
          isDisabled={(value) => !COMMANDS[hardware][value]}
        />

        <div className="lqs-output">
          <div className="lqs-rec lqs-rec--stable">
            <div className="lqs-eyebrow">Local worker</div>
            <div className="lqs-title">
              <span className="lqs-badge">Run</span>
              {backendLabel} worker
            </div>
            <div className="lqs-support">{hardwareLabel} · Qwen/Qwen3-0.6B · file discovery</div>
          </div>
          <div className="lqs-command">
            <button type="button" className="lqs-copy" onClick={copyCommand}>
              {copyLabel}
            </button>
            <pre>{command}</pre>
          </div>
        </div>
      </section>
    </>
  );
}

export default WorkerSelector;
