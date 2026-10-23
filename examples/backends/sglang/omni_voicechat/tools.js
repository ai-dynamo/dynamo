// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Display events produced by the native server tool stage; no client execution.
const rows = new Map();
let panel;

function ensurePanel() {
  if (panel) return panel;
  panel = document.createElement("aside");
  panel.id = "toolPanel";
  panel.setAttribute("aria-label", "Live tool calls");
  panel.style.cssText = "position:fixed;left:20px;bottom:20px;max-width:330px;max-height:240px;overflow:auto;padding:14px;border:1px solid #aabbb5;border-radius:12px;background:var(--surface,#fff);color:var(--text,#1d2420);font:13px/1.5 system-ui;z-index:5";
  const title = document.createElement("strong");
  title.textContent = "Live tool calls";
  const hint = document.createElement("p");
  hint.textContent = "Try: Pick a random number between 1 and 50.";
  hint.style.margin = "6px 0";
  panel.append(title, hint);
  document.body.append(panel);
  return panel;
}

export function observeToolEvent(event) {
  const container = ensurePanel();
  if (event.type === "session.created") {
    rows.clear();
    container.querySelectorAll("[data-call-id]").forEach(row => row.remove());
  }
  if (event.type === "response.function_call_arguments.done") {
    const row = document.createElement("div");
    row.dataset.callId = event.call_id;
    row.style.cssText = "border-top:1px solid #aabbb5;padding-top:6px;margin-top:6px;overflow-wrap:anywhere";
    const call = document.createElement("div");
    call.textContent = `${event.name}(${event.arguments})`;
    const output = document.createElement("div");
    output.setAttribute("role", "status");
    output.textContent = "Running on the server...";
    row.append(call, output);
    container.append(row);
    rows.set(event.call_id, output);
  }
  if (event.type === "conversation.item.added" && event.item.type === "function_call_output") {
    const output = rows.get(event.item.call_id);
    if (output) {
      output.textContent = `Result: ${event.item.output}`;
      output.dataset.complete = "true";
    }
  }
}

ensurePanel();
