// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { spawn } from "node:child_process";
import { isAbsolute } from "node:path";
import type { Request } from "./adapter.js";

export function helperClient(command: string, args: string[], env: NodeJS.ProcessEnv = {}) {
  if (!isAbsolute(command)) throw Error("helper_requires_absolute_executable");
  return (request: Request, signal: AbortSignal): Promise<unknown> => new Promise((resolve, reject) => {
    if (signal.aborted) return reject(Error("aborted"));
    const payload = JSON.stringify(request);
    if (Buffer.byteLength(payload) > request.budget.max_input_bytes) return reject(Error("input_budget"));
    const child = spawn(command, args, { shell: false, env, stdio: ["pipe", "pipe", "ignore"] });
    const chunks: Buffer[] = [];
    let size = 0;
    let failure = false;
    const abort = () => { failure = true; child.kill("SIGKILL"); };
    signal.addEventListener("abort", abort, { once: true });
    child.stdin.on("error", () => { failure = true; });
    child.on("error", () => { signal.removeEventListener("abort", abort); reject(Error("helper_unavailable")); });
    child.stdout.on("data", (chunk: Buffer) => {
      size += chunk.length;
      if (size > request.budget.max_output_bytes) abort();
      else chunks.push(chunk);
    });
    child.on("close", (code) => {
      signal.removeEventListener("abort", abort);
      if (failure || signal.aborted || code !== 0) return reject(Error("helper_failed"));
      try { resolve(JSON.parse(Buffer.concat(chunks).toString("utf8"))); }
      catch { reject(Error("helper_invalid_json")); }
    });
    child.stdin.end(payload);
  });
}
