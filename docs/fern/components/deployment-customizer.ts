/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Manifest engine for DeploymentCustomizer. It holds a comment- and
 * format-preserving YAML line model, DynamoGraphDeployment (DGD) and
 * DynamoGraphDeploymentRequest (DGDR) analysis, the customization transform,
 * generated prerequisites and deploy steps, syntax highlighting, and a line
 * diff. Edits rewrite only the lines they touch, so comments, key order, and
 * quoting in the source manifest survive.
 *
 * Keep this module free of imports: it is plain TypeScript that Node can run
 * directly, which keeps the transform testable outside the docs build.
 */

// ---------------------------------------------------------------------------
// YAML line model
// ---------------------------------------------------------------------------

export type ScalarStyle = "plain" | "single" | "double" | "block" | "flow";

export interface YScalar {
  kind: "scalar";
  line: number;
  col: number;
  raw: string;
  value: string;
  style: ScalarStyle;
  endLine: number;
}

export interface YEntry {
  key: string;
  line: number;
  col: number;
  value: YNode | null;
  endLine: number;
}

export interface YMap {
  kind: "map";
  line: number;
  col: number;
  entries: YEntry[];
  endLine: number;
}

export interface YItem {
  line: number;
  col: number;
  value: YNode | null;
  endLine: number;
}

export interface YSeq {
  kind: "seq";
  line: number;
  col: number;
  items: YItem[];
  endLine: number;
}

export type YNode = YScalar | YMap | YSeq;

export interface YDoc {
  start: number;
  end: number;
  root: YNode | null;
}

interface Tok {
  line: number;
  col: number;
  dash: boolean;
  text: string;
  endLine: number;
  more: string[];
}

const BLOCK_INDICATOR = /^[|>][1-9+-]{0,2}(\s+#.*)?$/;

function leadingSpaces(text: string): number {
  let n = 0;
  while (n < text.length && text[n] === " ") n++;
  return n;
}

/** Index of the closing quote for the quoted scalar starting at `start`, or -1. */
function quotedEnd(text: string, start: number): number {
  const quote = text[start];
  for (let i = start + 1; i < text.length; i++) {
    const c = text[i];
    if (quote === '"' && c === "\\") {
      i++;
      continue;
    }
    if (c === quote) {
      if (quote === "'" && text[i + 1] === "'") {
        i++;
        continue;
      }
      return i;
    }
  }
  return -1;
}

/** Start of a trailing comment, or -1. Quotes are honored for quoted and flow scalars. */
function commentStart(text: string, quoteAware: boolean): number {
  let quote = "";
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quoteAware) {
      if (quote) {
        if (quote === '"' && c === "\\") {
          i++;
          continue;
        }
        if (c === quote) quote = "";
        continue;
      }
      if (c === '"' || c === "'") {
        quote = c;
        continue;
      }
    }
    if (c === "#" && (i === 0 || text[i - 1] === " " || text[i - 1] === "\t")) return i;
  }
  return -1;
}

function stripComment(text: string, quoteAware: boolean): string {
  const at = commentStart(text, quoteAware);
  return (at < 0 ? text : text.slice(0, at)).trimEnd();
}

function flowDepth(text: string): number {
  let depth = 0;
  let quote = "";
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (quote) {
      if (quote === '"' && c === "\\") {
        i++;
        continue;
      }
      if (c === quote) quote = "";
      continue;
    }
    if (c === '"' || c === "'") quote = c;
    else if (c === "#" && (i === 0 || text[i - 1] === " ")) break;
    else if (c === "[" || c === "{") depth++;
    else if (c === "]" || c === "}") depth--;
  }
  return depth;
}

export function decodeQuoted(raw: string): string {
  if (raw.startsWith("'")) return raw.slice(1, -1).replace(/''/g, "'");
  try {
    return JSON.parse(raw) as string;
  } catch {
    return raw
      .slice(1, -1)
      .replace(/\\n/g, "\n")
      .replace(/\\t/g, "\t")
      .replace(/\\(["\\/])/g, "$1");
  }
}

interface KeySplit {
  key: string;
  rest: string;
  restOffset: number;
}

/** Splits `key: value` into its key and the text after the separator. */
export function splitKey(text: string): KeySplit | null {
  let key: string;
  let colon: number;
  if (text[0] === '"' || text[0] === "'") {
    const close = quotedEnd(text, 0);
    if (close < 0) return null;
    key = decodeQuoted(text.slice(0, close + 1));
    colon = close + 1;
    while (text[colon] === " ") colon++;
    if (text[colon] !== ":") return null;
  } else {
    if (/^[[{&*!|>%@`#?,]/.test(text)) return null;
    colon = -1;
    for (let j = 0; j < text.length; j++) {
      const c = text[j];
      if (c === "#" && j > 0 && (text[j - 1] === " " || text[j - 1] === "\t")) break;
      if (c === ":" && (j + 1 === text.length || text[j + 1] === " " || text[j + 1] === "\t")) {
        colon = j;
        break;
      }
    }
    if (colon < 0) return null;
    key = text.slice(0, colon).trimEnd();
  }
  if (colon + 1 < text.length && text[colon + 1] !== " " && text[colon + 1] !== "\t") return null;
  let rest = colon + 1;
  while (rest < text.length && (text[rest] === " " || text[rest] === "\t")) rest++;
  return { key, rest: text.slice(rest), restOffset: rest };
}

/** YAML line folding for multi-line flow scalars: single breaks become spaces. */
function fold(lines: string[]): string {
  let out = "";
  let pendingBreaks = 0;
  for (const line of lines.map((l) => l.trim())) {
    if (line === "") {
      pendingBreaks++;
      continue;
    }
    if (out !== "") out += pendingBreaks > 0 ? "\n".repeat(pendingBreaks) : " ";
    out += line;
    pendingBreaks = 0;
  }
  return out;
}

function blockValue(indicator: string, more: string[]): string {
  const content = more.filter((line) => line.trim() !== "");
  const base = content.length ? Math.min(...content.map(leadingSpaces)) : 0;
  const lines = more.map((line) => line.slice(Math.min(base, leadingSpaces(line))));
  return indicator.startsWith(">") ? fold(lines) : lines.join("\n");
}

function scalarAt(text: string, line: number, col: number, endLine: number, more: string[] = []): YScalar {
  const first = text[0];
  let raw: string;
  let value: string;
  let style: ScalarStyle;
  if (first === '"' || first === "'") {
    let close = quotedEnd(text, 0);
    raw = close >= 0 ? text.slice(0, close + 1) : text.trimEnd();
    if (close >= 0) value = decodeQuoted(raw);
    else {
      const joined = [text, ...more].join("\n");
      close = quotedEnd(joined, 0);
      const body = close >= 0 ? joined.slice(1, close) : joined.slice(1);
      value = decodeQuoted(first + fold(body.split("\n")) + first);
    }
    style = first === '"' ? "double" : "single";
  } else if (first === "|" || first === ">") {
    raw = stripComment(text, false);
    value = blockValue(raw, more);
    style = "block";
  } else if (first === "[" || first === "{") {
    raw = stripComment(text, true);
    value = [raw, ...more.map((l) => l.trim())].join(" ");
    style = "flow";
  } else {
    raw = stripComment(text, false);
    value = more.length ? fold([raw, ...more.map((l) => stripComment(l.trim(), false))]) : raw;
    style = "plain";
  }
  return { kind: "scalar", line, col, raw, value, style, endLine };
}

/** Last line of a value that may continue past its first line. */
function continuationEnd(lines: string[], i: number, end: number, value: string, owner: number): number {
  const v = value.trimStart();
  if (BLOCK_INDICATOR.test(v.trim())) {
    let last = i;
    for (let j = i + 1; j < end; j++) {
      const text = lines[j];
      if (text.trim() === "") continue;
      if (leadingSpaces(text) <= owner) break;
      last = j;
    }
    return last;
  }
  if ((v[0] === '"' || v[0] === "'") && quotedEnd(v, 0) < 0) {
    let joined = v;
    for (let j = i + 1; j < end; j++) {
      joined += `\n${lines[j]}`;
      if (quotedEnd(joined, 0) >= 0) return j;
    }
    return i;
  }
  if (v[0] === "[" || v[0] === "{") {
    let depth = flowDepth(v);
    for (let j = i + 1; j < end && depth > 0; j++) {
      depth += flowDepth(lines[j]);
      if (depth <= 0) return j;
    }
    return i;
  }
  if (v === "" || v.startsWith("#")) return i;
  // A plain scalar continues on more-indented lines; nothing else may nest under it.
  let last = i;
  for (let j = i + 1; j < end; j++) {
    const text = lines[j];
    if (text.trim() === "") continue;
    if (leadingSpaces(text) <= owner || text.trimStart().startsWith("#")) break;
    last = j;
  }
  return last;
}

function tokenize(lines: string[], start: number, end: number): Tok[] {
  const toks: Tok[] = [];
  let i = start;
  while (i < end) {
    const raw = lines[i];
    const indent = leadingSpaces(raw);
    const body = raw.slice(indent);
    if (body.trim() === "" || body.startsWith("#")) {
      i++;
      continue;
    }
    let col = indent;
    let text = body;
    let dashCol = -1;
    while (text === "-" || text.startsWith("- ") || text.startsWith("-\t")) {
      toks.push({ line: i, col, dash: true, text: "-", endLine: i, more: [] });
      dashCol = col;
      let n = 1;
      while (n < text.length && (text[n] === " " || text[n] === "\t")) n++;
      col += n;
      text = text.slice(n);
    }
    let last = i;
    if (text !== "" && !text.startsWith("#")) {
      const key = splitKey(text);
      const owner = key ? col : dashCol >= 0 ? dashCol : col - 1;
      last = continuationEnd(lines, i, end, key ? key.rest : text, owner);
      toks.push({ line: i, col, dash: false, text, endLine: last, more: lines.slice(i + 1, last + 1) });
    }
    i = last + 1;
  }
  return toks;
}

class Parser {
  private p = 0;
  private readonly toks: Tok[];

  constructor(toks: Tok[]) {
    this.toks = toks;
  }

  private peek(): Tok | undefined {
    return this.toks[this.p];
  }

  private lastEnd(fallback: number): number {
    return this.toks[this.p - 1]?.endLine ?? fallback;
  }

  /** Consumes tokens nested deeper than `col` that no node claimed. */
  private swallow(col: number): void {
    for (let t = this.peek(); t && t.col > col; t = this.peek()) this.p++;
  }

  node(minCol: number): YNode | null {
    const t = this.peek();
    if (!t || t.col < minCol) return null;
    if (t.dash) return this.seq(t.col);
    if (splitKey(t.text)) return this.map(t.col);
    this.p++;
    return scalarAt(t.text, t.line, t.col, t.endLine, t.more);
  }

  private seq(col: number): YSeq {
    const first = this.peek()!;
    const seq: YSeq = { kind: "seq", line: first.line, col, items: [], endLine: first.line };
    for (let t = this.peek(); t && t.dash && t.col === col; t = this.peek()) {
      this.p++;
      const item: YItem = { line: t.line, col, value: null, endLine: t.endLine };
      const next = this.peek();
      if (next && next.col > col) item.value = this.node(next.col);
      this.swallow(col);
      item.endLine = this.lastEnd(t.line);
      seq.items.push(item);
    }
    seq.endLine = this.lastEnd(first.line);
    return seq;
  }

  private map(col: number): YMap {
    const first = this.peek()!;
    const map: YMap = { kind: "map", line: first.line, col, entries: [], endLine: first.line };
    for (let t = this.peek(); t && !t.dash && t.col === col; t = this.peek()) {
      const split = splitKey(t.text);
      if (!split) break;
      this.p++;
      const entry: YEntry = { key: split.key, line: t.line, col, value: null, endLine: t.endLine };
      if (split.rest !== "" && !split.rest.startsWith("#")) {
        entry.value = scalarAt(split.rest, t.line, t.col + split.restOffset, t.endLine, t.more);
      } else {
        const next = this.peek();
        if (next && (next.col > col || (next.dash && next.col === col))) entry.value = this.node(next.col);
      }
      this.swallow(col);
      entry.endLine = this.lastEnd(t.line);
      map.entries.push(entry);
    }
    map.endLine = this.lastEnd(first.line);
    return map;
  }
}

function isDocBoundary(line: string): boolean {
  return /^(---|\.\.\.)(\s|$)/.test(line);
}

export function parseDocs(lines: string[]): YDoc[] {
  const docs: YDoc[] = [];
  let start = 0;
  for (let i = 0; i <= lines.length; i++) {
    if (i < lines.length && !isDocBoundary(lines[i])) continue;
    const toks = tokenize(lines, start, i);
    if (toks.length > 0) docs.push({ start, end: i, root: new Parser(toks).node(toks[0].col) });
    start = i + 1;
  }
  return docs;
}

export interface YFile {
  lines: string[];
  trailingNewline: boolean;
}

export function splitLines(text: string): YFile {
  const normalized = text.replace(/\r\n?/g, "\n");
  const trailingNewline = normalized.endsWith("\n");
  return { lines: (trailingNewline ? normalized.slice(0, -1) : normalized).split("\n"), trailingNewline };
}

export function joinLines(file: YFile): string {
  return file.lines.join("\n") + (file.trailingNewline ? "\n" : "");
}

// Navigation -----------------------------------------------------------------

export function entryOf(node: YNode | null | undefined, key: string): YEntry | undefined {
  return node?.kind === "map" ? node.entries.find((entry) => entry.key === key) : undefined;
}

export function child(node: YNode | null | undefined, key: string): YNode | null {
  return entryOf(node, key)?.value ?? null;
}

export function at(node: YNode | null | undefined, path: (string | number)[]): YNode | null {
  let current: YNode | null = node ?? null;
  for (const step of path) {
    if (!current) return null;
    if (typeof step === "number") current = current.kind === "seq" ? current.items[step]?.value ?? null : null;
    else current = child(current, step);
  }
  return current;
}

export function itemsOf(node: YNode | null | undefined): YItem[] {
  return node?.kind === "seq" ? node.items : [];
}

export function text(node: YNode | null | undefined): string | undefined {
  return node?.kind === "scalar" && node.style !== "block" ? node.value : undefined;
}

function intOf(node: YNode | null | undefined): number | null {
  const value = text(node);
  return value !== undefined && /^\d+$/.test(value.trim()) ? Number(value) : null;
}

function walk(node: YNode | null, visit: (node: YNode, key: string | null) => void, key: string | null = null): void {
  if (!node) return;
  visit(node, key);
  if (node.kind === "map") for (const entry of node.entries) walk(entry.value, visit, entry.key);
  if (node.kind === "seq") for (const item of node.items) walk(item.value, visit, key);
}

/** Lines that hold block scalar content; values there are scripts, not YAML structure. */
function blockContentLines(docs: YDoc[]): Set<number> {
  const lines = new Set<number>();
  for (const doc of docs) {
    walk(doc.root, (node) => {
      if (node.kind === "scalar" && node.style === "block") {
        for (let i = node.line + 1; i <= node.endLine; i++) lines.add(i);
      }
    });
  }
  return lines;
}

// Scalar formatting -----------------------------------------------------------

const AMBIGUOUS = /^(?:true|false|yes|no|on|off|y|n|null|~|[-+]?(?:\.inf|\.nan)|[-+]?(?:\d[\d_]*)?(?:\.\d*)?(?:[eE][-+]?\d+)?|0x[0-9a-fA-F_]+|0o[0-7_]+)$/i;

export function needsQuotes(value: string): boolean {
  if (value === "" || value.trim() !== value) return true;
  if (AMBIGUOUS.test(value)) return true;
  if (/^[-?:,[\]{}#&*!|>'"%@`]/.test(value)) return true;
  if (/: |\s#|:$|[\n\t]/.test(value)) return true;
  return false;
}

export function formatString(value: string): string {
  return needsQuotes(value) ? JSON.stringify(value) : value;
}

/** Formats `value` in the quoting style of the scalar it replaces. */
function formatLike(scalar: YScalar, value: string, numeric = false): string {
  if (scalar.style === "double") return JSON.stringify(value);
  if (scalar.style === "single") return `'${value.replace(/'/g, "''")}'`;
  if (numeric && /^\d+$/.test(value)) return value;
  return formatString(value);
}

type YValue = string | number | boolean | YValue[] | { [key: string]: YValue };

function emitScalar(value: string | number | boolean): string {
  return typeof value === "string" ? formatString(value) : String(value);
}

function isScalarValue(value: YValue): value is string | number | boolean {
  return typeof value !== "object";
}

function emitEntry(key: string, value: YValue, col: number, compact: boolean): string[] {
  const pad = " ".repeat(col);
  const name = formatString(key);
  if (isScalarValue(value)) return [`${pad}${name}: ${emitScalar(value)}`];
  if (Array.isArray(value)) {
    if (value.length === 0) return [`${pad}${name}: []`];
    return [`${pad}${name}:`, ...value.flatMap((item) => emitItem(item, compact ? col : col + 2, compact))];
  }
  const keys = Object.keys(value);
  if (keys.length === 0) return [`${pad}${name}: {}`];
  return [`${pad}${name}:`, ...keys.flatMap((k) => emitEntry(k, value[k], col + 2, compact))];
}

function emitItem(value: YValue, dashCol: number, compact: boolean): string[] {
  const pad = " ".repeat(dashCol);
  if (isScalarValue(value)) return [`${pad}- ${emitScalar(value)}`];
  if (Array.isArray(value)) {
    const nested = value.flatMap((item) => emitItem(item, dashCol + 2, compact));
    return [`${pad}-`, ...nested];
  }
  const lines = Object.keys(value).flatMap((k) => emitEntry(k, value[k], dashCol + 2, compact));
  if (lines.length === 0) return [`${pad}- {}`];
  return [`${pad}- ${lines[0].slice(dashCol + 2)}`, ...lines.slice(1)];
}

// Editor -------------------------------------------------------------------------

class Editor {
  lines: string[];
  docs: YDoc[];
  readonly trailingNewline: boolean;
  readonly compact: boolean;

  constructor(source: string) {
    const file = splitLines(source);
    this.lines = file.lines;
    this.trailingNewline = file.trailingNewline;
    this.docs = parseDocs(this.lines);
    this.compact = this.detectCompactSequences();
  }

  private detectCompactSequences(): boolean {
    let compact = 0;
    let indented = 0;
    for (const doc of this.docs) {
      walk(doc.root, (node) => {
        if (node.kind !== "map") return;
        for (const entry of node.entries) {
          if (entry.value?.kind !== "seq" || entry.value.line === entry.line) continue;
          if (entry.value.col === entry.col) compact++;
          else indented++;
        }
      });
    }
    return compact >= indented;
  }

  root(doc: number): YNode | null {
    return this.docs[doc]?.root ?? null;
  }

  text(): string {
    return joinLines({ lines: this.lines, trailingNewline: this.trailingNewline });
  }

  private reparse(): void {
    this.docs = parseDocs(this.lines);
  }

  /** Replaces single-line scalars in one pass; every edit keeps the line count. */
  replaceScalars(edits: { scalar: YScalar; raw: string }[]): void {
    const usable = edits
      .filter(({ scalar, raw }) => scalar.style !== "block" && scalar.endLine === scalar.line && raw !== scalar.raw)
      .sort((a, b) => b.scalar.line - a.scalar.line || b.scalar.col - a.scalar.col);
    if (usable.length === 0) return;
    for (const { scalar, raw } of usable) {
      const line = this.lines[scalar.line];
      this.lines[scalar.line] = line.slice(0, scalar.col) + raw + line.slice(scalar.col + scalar.raw.length);
    }
    this.reparse();
  }

  setScalar(node: YNode | null | undefined, value: string, numeric = false): void {
    if (node?.kind === "scalar") this.replaceScalars([{ scalar: node, raw: formatLike(node, value, numeric) }]);
  }

  insert(at: number, lines: string[]): void {
    if (lines.length === 0) return;
    this.lines.splice(at, 0, ...lines);
    this.reparse();
  }

  /** Deletes inclusive line ranges, highest first so earlier ranges stay valid. */
  deleteRanges(ranges: [number, number][]): void {
    if (ranges.length === 0) return;
    for (const [start, end] of [...ranges].sort((a, b) => b[0] - a[0])) this.lines.splice(start, end - start + 1);
    this.reparse();
  }

  /**
   * Deletes a map entry. When the entry starts a sequence item (its key follows
   * the item's dash), the dash moves to the next entry so the item survives.
   */
  deleteEntry(map: YNode | null, key: string): void {
    if (map?.kind !== "map") return;
    const index = map.entries.findIndex((entry) => entry.key === key);
    if (index < 0) return;
    const entry = map.entries[index];
    const line = this.lines[entry.line];
    if (leadingSpaces(line) === entry.col) {
      this.deleteRanges([[entry.line, entry.endLine]]);
      return;
    }
    const prefix = line.slice(0, entry.col);
    const next = map.entries[index + 1];
    if (next) {
      this.lines[next.line] = prefix + this.lines[next.line].slice(entry.col);
      this.lines.splice(entry.line, entry.endLine - entry.line + 1);
    } else {
      this.lines.splice(entry.line, entry.endLine - entry.line + 1, `${prefix}{}`);
    }
    this.reparse();
  }

  private insertionLine(map: YMap, key: string): number {
    const keys = map.entries.map((entry) => entry.key);
    const sorted = keys.every((k, i) => i === 0 || keys[i - 1] <= k);
    let index = map.entries.length;
    if (sorted) {
      index = map.entries.findIndex((entry) => entry.key > key);
      if (index < 0) index = map.entries.length;
    }
    if (index === 0) {
      const first = map.entries[0];
      if (leadingSpaces(this.lines[first.line]) === first.col) return first.line;
      index = 1;
    }
    return map.entries[index - 1].endLine + 1;
  }

  /** Sets `key` in `map` to a scalar or structure, inserting the entry when missing. */
  setEntry(map: YNode | null, key: string, value: YValue, numeric = false): void {
    if (map?.kind !== "map") return;
    const existing = entryOf(map, key);
    if (existing?.value?.kind === "scalar" && isScalarValue(value)) {
      this.setScalar(existing.value, String(value), numeric);
      return;
    }
    if (existing) return;
    this.insert(this.insertionLine(map, key), emitEntry(key, value, map.col, this.compact));
  }

  /** Appends items to the sequence under `key`, creating the key when missing. */
  appendItems(map: YNode | null, key: string, values: YValue[]): void {
    if (map?.kind !== "map" || values.length === 0) return;
    const existing = entryOf(map, key);
    if (!existing) {
      this.insert(this.insertionLine(map, key), emitEntry(key, values, map.col, this.compact));
      return;
    }
    if (existing.value?.kind === "seq") {
      const seq = existing.value;
      this.insert(seq.endLine + 1, values.flatMap((value) => emitItem(value, seq.col, this.compact)));
      return;
    }
    if (existing.value === null) {
      const dashCol = this.compact ? existing.col : existing.col + 2;
      this.insert(existing.endLine + 1, values.flatMap((value) => emitItem(value, dashCol, this.compact)));
    }
  }
}

// ---------------------------------------------------------------------------
// Manifest analysis
// ---------------------------------------------------------------------------

export interface ComponentInfo {
  key: string;
  doc: number;
  index: number;
  graph: string;
  name: string;
  type: string;
  replicas: number | null;
  nodes: number | null;
  gpus: number | null;
  dra: boolean;
  tp: { flag: string; value: number } | null;
  loadsModel: boolean;
  container: number | null;
}

export interface EnvInfo {
  scope: string;
  index: number;
  name: string;
  value: string;
  from: string | null;
}

export interface EnvScope {
  id: string;
  label: string;
}

export interface GraphInfo {
  doc: number;
  kind: "DynamoGraphDeployment" | "DynamoGraphDeploymentRequest";
  name: string;
  autoApply: boolean;
}

/** A model cache PVC the manifest already mounts, with the cache layout its workers read. */
export interface ModelPvc {
  claim: string;
  mountPath: string;
  subPath: string | null;
  env: [string, string][];
}

export interface ManifestInfo {
  graphs: GraphInfo[];
  components: ComponentInfo[];
  env: EnvInfo[];
  envScopes: EnvScope[];
  images: string[];
  models: string[];
  secret: string | null;
  modelPvc: ModelPvc | null;
  claims: string[];
  templateVars: string[];
  frontends: number;
}

const MODEL_FLAGS = new Set(["--model", "--model-path", "--served-model-name", "--model-name"]);
const MODEL_ENV = new Set(["MODEL", "MODEL_NAME", "MODEL_ID", "MODEL_PATH", "HF_MODEL_ID"]);
const TP_FLAGS = new Set(["--tensor-parallel-size", "--tp", "--tp-size", "-tp"]);
const MODEL_LOADERS = new Set(["worker", "prefill", "decode"]);
const MODEL_ID = /^[A-Za-z0-9][\w.-]*\/[\w.-]+$/;
const MODEL_IN_COMMAND = /(?:^|\s)--(?:model|model-path|served-model-name|model-name)[ =]+["']?([\w./-]+)/g;
const TEMPLATE_VAR = /\$\{([A-Za-z_][A-Za-z0-9_]*)\}/g;

export function isModelId(value: string): boolean {
  return MODEL_ID.test(value) && !value.includes("..");
}

function graphKind(root: YNode | null): GraphInfo["kind"] | null {
  const kind = text(child(root, "kind"));
  return kind === "DynamoGraphDeployment" || kind === "DynamoGraphDeploymentRequest" ? kind : null;
}

function mainContainerIndex(component: YNode | null): number | null {
  const containers = itemsOf(at(component, ["podTemplate", "spec", "containers"]));
  if (containers.length === 0) return null;
  const main = containers.findIndex((item) => text(child(item.value, "name")) === "main");
  return main >= 0 ? main : 0;
}

function containerOf(root: YNode | null, index: number, container: number | null): YNode | null {
  if (container === null) return null;
  return at(root, ["spec", "components", index, "podTemplate", "spec", "containers", container]);
}

function tpFlag(container: YNode | null): ComponentInfo["tp"] {
  const args = itemsOf(child(container, "args")).map((item) => text(item.value) ?? "");
  for (let i = 0; i < args.length; i++) {
    const [flag, inline] = args[i].split("=", 2);
    if (!TP_FLAGS.has(flag)) continue;
    const value = inline ?? args[i + 1];
    if (value !== undefined && /^\d+$/.test(value)) return { flag, value: Number(value) };
  }
  return null;
}

function envFrom(item: YNode | null): string | null {
  const from = child(item, "valueFrom");
  if (!from) return null;
  const secret = child(from, "secretKeyRef");
  if (secret) return `Secret ${text(child(secret, "name")) ?? "?"} / ${text(child(secret, "key")) ?? "?"}`;
  const config = child(from, "configMapKeyRef");
  if (config) return `ConfigMap ${text(child(config, "name")) ?? "?"} / ${text(child(config, "key")) ?? "?"}`;
  const field = child(from, "fieldRef");
  if (field) return `Field ${text(child(field, "fieldPath")) ?? "?"}`;
  const resource = child(from, "resourceFieldRef");
  if (resource) return `Resource ${text(child(resource, "resource")) ?? "?"}`;
  return "valueFrom";
}

function countInto(counts: Map<string, number>, value: string | undefined): void {
  if (value) counts.set(value, (counts.get(value) ?? 0) + 1);
}

function ranked(counts: Map<string, number>): string[] {
  return [...counts.entries()].sort((a, b) => b[1] - a[1]).map(([value]) => value);
}

export function analyzeManifest(source: string): ManifestInfo {
  const { lines } = splitLines(source);
  const docs = parseDocs(lines);
  const info: ManifestInfo = {
    graphs: [],
    components: [],
    env: [],
    envScopes: [],
    images: [],
    models: [],
    secret: null,
    modelPvc: null,
    claims: [],
    templateVars: [],
    frontends: 0,
  };
  const modelCounts = new Map<string, number>();
  const secretCounts = new Map<string, number>();
  const images = new Set<string>();

  docs.forEach((doc, docIndex) => {
    const root = doc.root;
    walk(root, (node, key) => {
      if (key === "image" && node.kind === "scalar") {
        const value = text(node);
        if (value) images.add(value);
      }
      if (node.kind === "map") {
        for (const ref of ["secretRef", "secretKeyRef"]) {
          const name = text(child(child(node, ref), "name"));
          if (name && (/hf|hugging/i.test(name) || text(child(child(node, ref), "key")) === "HF_TOKEN")) {
            countInto(secretCounts, name);
          }
        }
      }
      if (node.kind === "scalar") {
        for (const match of node.value.matchAll(MODEL_IN_COMMAND)) {
          if (isModelId(match[1])) countInto(modelCounts, match[1]);
        }
      }
      if (node.kind === "seq") {
        const values = node.items.map((item) => text(item.value) ?? "");
        values.forEach((value, i) => {
          const [flag, inline] = value.split("=", 2);
          if (!MODEL_FLAGS.has(flag)) return;
          const model = inline ?? values[i + 1];
          if (model && isModelId(model)) countInto(modelCounts, model);
        });
      }
    });
    if (text(child(root, "kind")) === "PersistentVolumeClaim") {
      const claim = text(at(root, ["metadata", "name"]));
      if (claim) info.claims.push(claim);
    }
    const kind = graphKind(root);
    if (!kind) return;
    const graphName = text(at(root, ["metadata", "name"])) ?? `document ${docIndex + 1}`;
    info.graphs.push({ doc: docIndex, kind, name: graphName, autoApply: text(at(root, ["spec", "autoApply"])) === "true" });
    if (kind === "DynamoGraphDeploymentRequest") {
      const model = text(at(root, ["spec", "model"]));
      if (model && isModelId(model)) countInto(modelCounts, model);
      const claim = text(at(root, ["spec", "modelCache", "pvcName"]));
      if (claim && !info.modelPvc) {
        const mountPath = text(at(root, ["spec", "modelCache", "pvcMountPath"])) ?? "/opt/model-cache";
        info.modelPvc = { claim, mountPath, subPath: null, env: [["HF_HOME", mountPath]] };
      }
      return;
    }
  });

  const multipleGraphs = info.graphs.filter((graph) => graph.kind === "DynamoGraphDeployment").length > 1;
  for (const graph of info.graphs) {
    if (graph.kind !== "DynamoGraphDeployment") continue;
    const root = docs[graph.doc].root;
    const prefix = multipleGraphs ? `${graph.name} · ` : "";
    const graphScope = `g:${graph.doc}`;
    info.envScopes.push({ id: graphScope, label: `${prefix}All components` });
    itemsOf(at(root, ["spec", "env"])).forEach((item, index) => {
      const name = text(child(item.value, "name"));
      if (name === undefined) return;
      info.env.push({ scope: graphScope, index, name, value: text(child(item.value, "value")) ?? "", from: envFrom(item.value) });
    });
    itemsOf(at(root, ["spec", "components"])).forEach((item, index) => {
      const component = item.value;
      const name = text(child(component, "name")) ?? `component-${index}`;
      const type = text(child(component, "type")) ?? "";
      const containerIndex = mainContainerIndex(component);
      const container = containerOf(root, index, containerIndex);
      const limits = at(container, ["resources", "limits"]);
      const requests = at(container, ["resources", "requests"]);
      const gpus = intOf(child(limits, "nvidia.com/gpu")) ?? intOf(child(requests, "nvidia.com/gpu"));
      const podSpec = at(component, ["podTemplate", "spec"]);
      const dra = itemsOf(child(podSpec, "resourceClaims")).length > 0 || itemsOf(at(container, ["resources", "claims"])).length > 0;
      if (type === "frontend") info.frontends++;
      info.components.push({
        key: `${graph.doc}:${index}`,
        doc: graph.doc,
        index,
        graph: graph.name,
        name,
        type,
        replicas: intOf(child(component, "replicas")),
        nodes: intOf(at(component, ["multinode", "nodeCount"])),
        gpus,
        dra,
        tp: tpFlag(container),
        loadsModel: MODEL_LOADERS.has(type) && container !== null,
        container: containerIndex,
      });
      if (container) {
        const scope = `c:${graph.doc}:${index}`;
        info.envScopes.push({ id: scope, label: `${prefix}${name}` });
        itemsOf(child(container, "env")).forEach((env, envIndex) => {
          const envName = text(child(env.value, "name"));
          if (envName === undefined) return;
          const value = text(child(env.value, "value")) ?? "";
          info.env.push({ scope, index: envIndex, name: envName, value, from: envFrom(env.value) });
          if (MODEL_ENV.has(envName) && isModelId(value)) countInto(modelCounts, value);
        });
        if (!info.modelPvc) info.modelPvc = modelPvcOf(container, podSpec, at(root, ["spec", "env"]));
      }
    });
  }

  const blocks = blockContentLines(docs);
  const vars = new Set<string>();
  lines.forEach((line, i) => {
    if (blocks.has(i) || line.trimStart().startsWith("#")) return;
    for (const match of line.matchAll(TEMPLATE_VAR)) vars.add(match[1]);
  });

  info.images = [...images];
  info.models = ranked(modelCounts);
  info.secret = ranked(secretCounts)[0] ?? null;
  info.templateVars = [...vars];
  return info;
}

const CACHE_ENV = ["HF_HOME", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"];

function cacheEnvOf(envList: YNode | null): [string, string][] {
  return itemsOf(envList)
    .map((env) => [text(child(env.value, "name")) ?? "", text(child(env.value, "value")) ?? ""] as [string, string])
    .filter(([name, value]) => CACHE_ENV.includes(name) && value !== "");
}

/** `graphEnv` is the DGD's spec.env, which the operator prepends to every container. */
function modelPvcOf(container: YNode | null, podSpec: YNode | null, graphEnv: YNode | null): ModelPvc | null {
  const claims = new Map<string, string>();
  for (const volume of itemsOf(child(podSpec, "volumes"))) {
    const name = text(child(volume.value, "name"));
    const claim = text(at(volume.value, ["persistentVolumeClaim", "claimName"]));
    if (name && claim) claims.set(name, claim);
  }
  const merged = new Map([...cacheEnvOf(graphEnv), ...cacheEnvOf(child(container, "env"))]);
  const cacheEnv = [...merged.entries()];
  for (const mount of itemsOf(child(container, "volumeMounts"))) {
    const claim = claims.get(text(child(mount.value, "name")) ?? "");
    const mountPath = text(child(mount.value, "mountPath")) ?? "";
    if (!claim || !mountPath) continue;
    const isCache =
      /huggingface|model/i.test(mountPath) || /model/i.test(claim) || cacheEnv.some(([, dir]) => dir.startsWith(mountPath));
    if (!isCache) continue;
    // Workers that set no cache variable read the runtime image default.
    const env: [string, string][] = cacheEnv.length > 0 ? cacheEnv : [["HF_HOME", DEFAULT_HF_HOME]];
    return { claim, mountPath, subPath: text(child(mount.value, "subPath")) ?? null, env };
  }
  return null;
}

// ---------------------------------------------------------------------------
// Container images
// ---------------------------------------------------------------------------

export interface ImageRef {
  registry: string;
  repo: string;
  tag: string | null;
  digest: string | null;
}

export function parseImage(ref: string): ImageRef | null {
  if (!ref || /\s|\$\{/.test(ref)) return null;
  const [name, digest] = ref.split("@", 2);
  const slash = name.lastIndexOf("/");
  const colon = name.lastIndexOf(":");
  const tag = colon > slash ? name.slice(colon + 1) : null;
  const path = colon > slash ? name.slice(0, colon) : name;
  const parts = path.split("/");
  const repo = parts.pop() ?? "";
  if (!repo) return null;
  return { registry: parts.join("/"), repo, tag, digest: digest ?? null };
}

export const PLACEHOLDER_REGISTRIES = ["my-registry", "your-registry"];
export const PLACEHOLDER_TAGS = ["my-tag", "your-tag", "latest"];

export interface ImageSettings {
  defaultRegistry: string;
  publishedRepos: string[];
  registry: string | null;
  tagMode: "release" | "keep" | "custom";
  tag: string;
}

function isDynamoImage(image: ImageRef, defaultRegistry: string): boolean {
  return image.registry === defaultRegistry || PLACEHOLDER_REGISTRIES.includes(image.registry);
}

/** Images the registry and tag settings apply to: Dynamo images that are not pinned by digest. */
export function isTaggableImage(ref: string, defaultRegistry: string): boolean {
  const image = parseImage(ref);
  return !!image && !image.digest && isDynamoImage(image, defaultRegistry);
}

export function isPublishedImage(ref: string, settings: Pick<ImageSettings, "publishedRepos">): boolean {
  const image = parseImage(ref);
  return !!image && settings.publishedRepos.includes(image.repo.replace(/-nightly$/, ""));
}

/** True when the manifest leaves an image registry or tag for the reader to fill in. */
export function hasPlaceholderImages(images: string[], defaultRegistry: string): boolean {
  return images.some((ref) => {
    const image = parseImage(ref);
    if (!image || image.digest || !isDynamoImage(image, defaultRegistry)) return false;
    return PLACEHOLDER_REGISTRIES.includes(image.registry) || PLACEHOLDER_TAGS.includes(image.tag ?? "latest");
  });
}

/** Rewrites Dynamo images to the selected registry and tag; digests and other images stay. */
export function rewriteImage(ref: string, settings: ImageSettings): string {
  const image = parseImage(ref);
  if (!image || image.digest || !isDynamoImage(image, settings.defaultRegistry)) return ref;
  const baseRepo = image.repo.replace(/-nightly$/, "");
  const published = settings.publishedRepos.includes(baseRepo);
  const custom = settings.registry && settings.registry !== settings.defaultRegistry ? settings.registry : null;
  if (!published && !custom) return ref;
  if (settings.tagMode === "keep") {
    return custom ? `${custom}/${image.repo}${image.tag ? `:${image.tag}` : ""}` : ref;
  }
  const variant = image.tag?.match(/^(?:\d+\.\d+\.\d+(?:\.post\d+)?|my-tag)(-[a-z][a-z0-9]*)$/)?.[1] ?? "";
  const tag = `${settings.tag}${settings.tag.endsWith(variant) ? "" : variant}`;
  return `${custom ?? settings.defaultRegistry}/${baseRepo}:${tag}`;
}

// ---------------------------------------------------------------------------
// Customization
// ---------------------------------------------------------------------------

export interface EnvRow {
  id: string;
  scope: string;
  name: string;
  value: string;
  from: string | null;
  orig: number | null;
}

export interface ComponentOverride {
  replicas?: number;
  gpus?: number;
  nodes?: number;
}

export type SecretMode = "kubectl" | "yaml" | "existing" | "none";
export type StorageMode = "download" | "pvc";

export interface StorageSettings {
  mode: StorageMode;
  pvcName: string;
  size: string;
  storageClass: string;
  accessMode: "ReadWriteMany" | "ReadWriteOnce";
  mountPath: string;
  createPvc: boolean;
  downloadJob: boolean;
}

export interface CustomizeSettings {
  fileName: string;
  model: string | null;
  image: ImageSettings;
  names: Record<number, string>;
  components: Record<string, ComponentOverride>;
  env: EnvRow[] | null;
  storage: StorageSettings;
  secret: { mode: SecretMode; name: string };
  templateVars: Record<string, string>;
  namespace: string;
  bundle: boolean;
  chatModel: boolean;
}

export interface OutputFile {
  id: "manifest" | "secret" | "model-cache" | "steps";
  name: string;
  language: "yaml" | "bash";
  content: string;
}

export interface CustomizeResult {
  manifest: string;
  files: OutputFile[];
  notes: string[];
}

export const HF_DOWNLOAD_IMAGE = "python:3.10-slim";
/** Hugging Face cache directory in the Dynamo runtime images. */
export const DEFAULT_HF_HOME = "/home/dynamo/.cache/huggingface";
export const HF_HUB_VERSION = "1.16.4";

/** One editable row per environment variable the manifest sets. */
export function initialEnvRows(info: ManifestInfo): EnvRow[] {
  return info.env.map((env) => ({
    id: `${env.scope}#${env.index}`,
    scope: env.scope,
    name: env.name,
    value: env.value,
    from: env.from,
    orig: env.index,
  }));
}

function envSeqParent(editor: Editor, scope: string, info: ManifestInfo): YNode | null {
  const [kind, docText, indexText] = scope.split(":");
  const doc = Number(docText);
  const root = editor.root(doc);
  if (kind === "g") return child(root, "spec");
  const component = info.components.find((c) => c.doc === doc && c.index === Number(indexText));
  return component ? containerOf(root, component.index, component.container) : null;
}

function applyEnv(editor: Editor, info: ManifestInfo, rows: EnvRow[]): void {
  for (const scope of info.envScopes) {
    const desired = rows.filter((row) => row.scope === scope.id && (row.orig !== null || row.name.trim() !== ""));
    const originals = info.env.filter((env) => env.scope === scope.id);
    const kept = new Map(desired.filter((row) => row.orig !== null).map((row) => [row.orig as number, row]));

    // Edits keep the line count, so original item indexes stay valid.
    const seq = () => child(envSeqParent(editor, scope.id, info), "env");
    const edits: { scalar: YScalar; raw: string }[] = [];
    itemsOf(seq()).forEach((item, index) => {
      const row = kept.get(index);
      const original = originals.find((env) => env.index === index);
      if (!row || !original) return;
      const nameNode = child(item.value, "name");
      const valueNode = child(item.value, "value");
      if (row.name.trim() && row.name !== original.name && nameNode?.kind === "scalar") {
        edits.push({ scalar: nameNode, raw: formatLike(nameNode, row.name.trim()) });
      }
      if (!original.from && row.value !== original.value && valueNode?.kind === "scalar") {
        edits.push({ scalar: valueNode, raw: valueNode.style === "plain" ? formatString(row.value) : formatLike(valueNode, row.value) });
      }
    });
    editor.replaceScalars(edits);

    const removed = itemsOf(seq())
      .map((item, index) => ({ item, index }))
      .filter(({ index }) => originals.some((env) => env.index === index) && !kept.has(index))
      .map(({ item }) => [item.line, item.endLine] as [number, number]);
    editor.deleteRanges(removed);

    const added = desired.filter((row) => row.orig === null).map((row) => ({ name: row.name.trim(), value: row.value }));
    const parent = envSeqParent(editor, scope.id, info);
    const remaining = itemsOf(child(parent, "env")).length;
    if (remaining === 0 && added.length === 0) {
      if (entryOf(parent, "env") && originals.length > 0) editor.deleteEntry(parent, "env");
      continue;
    }
    editor.appendItems(parent, "env", added);
  }
}

/** Sets an env var in a container, updating it in place when it already exists. */
function upsertEnv(editor: Editor, container: () => YNode | null, name: string, value: string): void {
  const existing = itemsOf(child(container(), "env")).find((item) => text(child(item.value, "name")) === name);
  if (existing) {
    const valueNode = child(existing.value, "value");
    if (valueNode?.kind === "scalar") editor.setScalar(valueNode, value);
    return;
  }
  editor.appendItems(container(), "env", [{ name, value }]);
}

function applyStorage(editor: Editor, info: ManifestInfo, storage: StorageSettings, notes: string[]): void {
  if (storage.mode !== "pvc") return;
  if (info.modelPvc) {
    const claim = info.modelPvc.claim;
    if (storage.pvcName && storage.pvcName !== claim) {
      renameScalars(editor, claim, storage.pvcName, ["claimName", "pvcName"]);
      const edits: { scalar: YScalar; raw: string }[] = [];
      for (const doc of editor.docs) {
        const name = at(doc.root, ["metadata", "name"]);
        if (text(child(doc.root, "kind")) === "PersistentVolumeClaim" && name?.kind === "scalar" && name.value === claim) {
          edits.push({ scalar: name, raw: formatLike(name, storage.pvcName) });
        }
      }
      editor.replaceScalars(edits);
    }
    return;
  }
  for (const graph of info.graphs) {
    if (graph.kind !== "DynamoGraphDeploymentRequest") continue;
    const spec = () => child(editor.root(graph.doc), "spec");
    const model = text(child(spec(), "model")) ?? "";
    const cache: { [key: string]: YValue } = {
      pvcName: storage.pvcName,
      pvcMountPath: storage.mountPath,
      pvcModelPath: `hub/models--${model.replace("/", "--")}/snapshots/<commit-hash>`,
    };
    editor.setEntry(spec(), "modelCache", cache);
  }
  const loaders = info.components.filter((component) => component.loadsModel);
  for (const component of loaders) {
    const root = () => editor.root(component.doc);
    const container = () => containerOf(root(), component.index, component.container);
    const podSpec = () => at(root(), ["spec", "components", component.index, "podTemplate", "spec"]);
    upsertEnv(editor, container, "HF_HOME", storage.mountPath);
    const mounted = itemsOf(child(container(), "volumeMounts")).some(
      (item) => text(child(item.value, "mountPath")) === storage.mountPath,
    );
    if (!mounted) editor.appendItems(container(), "volumeMounts", [{ mountPath: storage.mountPath, name: storage.pvcName }]);
    const hasVolume = itemsOf(child(podSpec(), "volumes")).some((item) => text(child(item.value, "name")) === storage.pvcName);
    if (!hasVolume) {
      editor.appendItems(podSpec(), "volumes", [{ name: storage.pvcName, persistentVolumeClaim: { claimName: storage.pvcName } }]);
    }
  }
  if (loaders.length === 0 && !info.graphs.some((graph) => graph.kind === "DynamoGraphDeploymentRequest")) {
    notes.push("No worker components were found to mount the model cache.");
  }
}

/** Renames every `<key>: <from>` scalar to `<to>`. */
function renameScalars(editor: Editor, from: string, to: string, keys: string[]): void {
  const edits: { scalar: YScalar; raw: string }[] = [];
  for (const doc of editor.docs) {
    walk(doc.root, (node) => {
      if (node.kind !== "map") return;
      for (const entry of node.entries) {
        if (keys.includes(entry.key) && entry.value?.kind === "scalar" && entry.value.value === from) {
          edits.push({ scalar: entry.value, raw: formatLike(entry.value, to) });
        }
      }
    });
  }
  editor.replaceScalars(edits);
}

function applySecret(editor: Editor, info: ManifestInfo, secret: CustomizeSettings["secret"]): void {
  const current = info.secret;
  if (!current) return;
  if (secret.mode !== "none") {
    if (secret.name && secret.name !== current) {
      const edits: { scalar: YScalar; raw: string }[] = [];
      for (const doc of editor.docs) {
        walk(doc.root, (node) => {
          for (const ref of ["secretRef", "secretKeyRef"]) {
            const name = child(child(node, ref), "name");
            if (name?.kind === "scalar" && name.value === current) edits.push({ scalar: name, raw: formatLike(name, secret.name) });
          }
        });
      }
      editor.replaceScalars(edits);
    }
    return;
  }
  const ranges: [number, number][] = [];
  for (const doc of editor.docs) {
    walk(doc.root, (node) => {
      if (node.kind !== "map") return;
      for (const item of itemsOf(child(node, "envFrom"))) {
        if (text(at(item.value, ["secretRef", "name"])) === current) ranges.push([item.line, item.endLine]);
      }
      for (const item of itemsOf(child(node, "env"))) {
        if (text(at(item.value, ["valueFrom", "secretKeyRef", "name"])) === current) ranges.push([item.line, item.endLine]);
      }
    });
  }
  editor.deleteRanges(ranges);
  // Drop the keys whose lists are now empty, one at a time because each edit reparses.
  for (let guard = 0; guard < 500; guard++) {
    let target: { map: YMap; key: string } | null = null;
    for (const doc of editor.docs) {
      walk(doc.root, (node) => {
        if (target || node.kind !== "map") return;
        const entry = node.entries.find((e) => (e.key === "envFrom" || e.key === "env") && e.value === null);
        if (entry) target = { map: node, key: entry.key };
      });
    }
    if (!target) return;
    const { map, key } = target as { map: YMap; key: string };
    editor.deleteEntry(map, key);
  }
}

function applyComponents(editor: Editor, info: ManifestInfo, overrides: Record<string, ComponentOverride>): void {
  for (const component of info.components) {
    const override = overrides[component.key];
    if (!override) continue;
    const node = () => at(editor.root(component.doc), ["spec", "components", component.index]);
    const container = () => containerOf(editor.root(component.doc), component.index, component.container);
    if (override.replicas !== undefined && override.replicas !== (component.replicas ?? 1)) {
      editor.setEntry(node(), "replicas", override.replicas, true);
    }
    if (override.nodes !== undefined && component.nodes !== null && override.nodes !== component.nodes) {
      editor.setScalar(at(node(), ["multinode", "nodeCount"]), String(override.nodes), true);
    }
    if (override.gpus !== undefined && component.gpus !== null && override.gpus !== component.gpus) {
      for (const section of ["limits", "requests"]) {
        editor.setScalar(at(container(), ["resources", section, "nvidia.com/gpu"]), String(override.gpus), true);
      }
    }
    const tp = component.tp;
    if (tp && component.gpus !== null && tp.value === component.gpus * (component.nodes ?? 1)) {
      const total = (override.gpus ?? component.gpus) * (override.nodes ?? component.nodes ?? 1);
      if (total !== tp.value) {
        const args = itemsOf(child(container(), "args"));
        const index = args.findIndex((item) => (text(item.value) ?? "").split("=", 2)[0] === tp.flag);
        const flagNode = args[index]?.value;
        const inline = (text(flagNode) ?? "").includes("=");
        const target = inline ? flagNode : args[index + 1]?.value;
        if (target?.kind === "scalar") editor.setScalar(target, inline ? `${tp.flag}=${total}` : String(total), true);
      }
    }
  }
}

function applyImages(editor: Editor, settings: ImageSettings): void {
  const edits: { scalar: YScalar; raw: string }[] = [];
  for (const doc of editor.docs) {
    walk(doc.root, (node, key) => {
      if (key !== "image" || node.kind !== "scalar" || node.style === "block") return;
      const next = rewriteImage(node.value, settings);
      if (next !== node.value) edits.push({ scalar: node, raw: formatLike(node, next) });
    });
  }
  editor.replaceScalars(edits);
}

function applyNames(editor: Editor, info: ManifestInfo, names: Record<number, string>): void {
  for (const graph of info.graphs) {
    const next = names[graph.doc]?.trim();
    if (!next || next === graph.name) continue;
    const edits: { scalar: YScalar; raw: string }[] = [];
    walk(editor.docs[graph.doc]?.root ?? null, (node) => {
      if (node.kind === "scalar" && node.style !== "block" && node.value === graph.name) {
        edits.push({ scalar: node, raw: formatLike(node, next) });
      }
    });
    editor.replaceScalars(edits);
  }
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function replaceText(lines: string[], pattern: RegExp, replacement: string): string[] {
  return lines.map((line) => line.replace(pattern, () => replacement));
}

export function replaceModel(lines: string[], from: string, to: string): string[] {
  if (!from || !to || from === to) return lines;
  const id = new RegExp(`(?<![\\w./-])${escapeRegExp(from)}(?![\\w./-])`, "g");
  const cacheDir = new RegExp(`models--${escapeRegExp(from.replace("/", "--"))}(?![\\w.-])`, "g");
  return replaceText(replaceText(lines, id, to), cacheDir, `models--${to.replace("/", "--")}`);
}

function substituteVars(lines: string[], vars: Record<string, string>): string[] {
  let out = lines;
  for (const [name, value] of Object.entries(vars)) {
    if (value.trim() === "") continue;
    out = replaceText(out, new RegExp(`\\$\\{${escapeRegExp(name)}\\}`, "g"), value.trim());
  }
  return out;
}

export function effectiveModel(info: ManifestInfo, settings: Pick<CustomizeSettings, "model" | "templateVars">): string | null {
  const chosen = settings.model?.trim();
  if (info.models[0]) return chosen || info.models[0];
  for (const name of ["MODEL_NAME", "MODEL", "MODEL_ID"]) {
    const value = settings.templateVars[name]?.trim();
    if (value) return value;
  }
  return null;
}

function secretYaml(name: string): string {
  return [
    "apiVersion: v1",
    "kind: Secret",
    "metadata:",
    `  name: ${formatString(name)}`,
    "type: Opaque",
    "stringData:",
    "  # Required for gated or private models; public models work with an empty token.",
    '  HF_TOKEN: ""',
    "",
  ].join("\n");
}

interface CacheLayout {
  mountPath: string;
  subPath: string | null;
  env: [string, string][];
}

function modelCacheYaml(
  model: string | null,
  storage: StorageSettings,
  secret: CustomizeSettings["secret"],
  layout: CacheLayout,
): string {
  const docs: string[][] = [];
  const claim = formatString(storage.pvcName);
  if (storage.createPvc) {
    docs.push([
      "apiVersion: v1",
      "kind: PersistentVolumeClaim",
      "metadata:",
      `  name: ${claim}`,
      "spec:",
      "  accessModes:",
      `  - ${storage.accessMode}`,
      "  resources:",
      "    requests:",
      `      storage: ${formatString(storage.size)}`,
      ...(storage.storageClass.trim() ? [`  storageClassName: ${formatString(storage.storageClass.trim())}`] : []),
    ]);
  }
  if (storage.downloadJob && model) {
    docs.push([
      "apiVersion: batch/v1",
      "kind: Job",
      "metadata:",
      "  name: model-download",
      "spec:",
      "  backoffLimit: 3",
      "  template:",
      "    metadata:",
      "      labels:",
      "        app: model-download",
      "    spec:",
      "      restartPolicy: Never",
      "      containers:",
      "      - name: model-download",
      `        image: ${HF_DOWNLOAD_IMAGE}`,
      '        command: ["sh", "-c"]',
      "        args:",
      "        - |",
      "          set -eux",
      `          pip install --no-cache-dir huggingface_hub==${HF_HUB_VERSION}`,
      '          hf download "$MODEL_NAME"',
      "        env:",
      "        - name: MODEL_NAME",
      `          value: ${formatString(model)}`,
      ...layout.env.flatMap(([name, value]) => [`        - name: ${name}`, `          value: ${formatString(value)}`]),
      ...(secret.mode === "none"
        ? []
        : ["        envFrom:", "        - secretRef:", `            name: ${formatString(secret.name)}`]),
      "        volumeMounts:",
      `        - name: ${claim}`,
      `          mountPath: ${formatString(layout.mountPath)}`,
      ...(layout.subPath ? [`          subPath: ${formatString(layout.subPath)}`] : []),
      "      volumes:",
      `      - name: ${claim}`,
      "        persistentVolumeClaim:",
      `          claimName: ${claim}`,
    ]);
  }
  return docs.map((doc) => doc.join("\n")).join("\n---\n") + "\n";
}

function shellQuote(value: string): string {
  return /^[\w@%+=:,./-]+$/.test(value) ? value : `'${value.replace(/'/g, "'\\''")}'`;
}

function deploySteps(
  info: ManifestInfo,
  settings: CustomizeSettings,
  files: OutputFile[],
  manifest: string,
  model: string | null,
): string {
  const ns = '--namespace "$NAMESPACE"';
  const out: string[] = [`export NAMESPACE=${shellQuote(settings.namespace.trim() || "dynamo-system")}`, ""];
  const secretFile = files.find((file) => file.id === "secret");
  const cacheFile = files.find((file) => file.id === "model-cache");
  const manifestFile = files.find((file) => file.id === "manifest")!;
  const downloads = settings.storage.mode === "pvc" && settings.storage.downloadJob;
  const needsSecret = settings.secret.mode === "kubectl" && (info.secret !== null || downloads);

  if (needsSecret) {
    out.push(
      "# Gated or private models need a token: export HF_TOKEN=hf_...",
      `kubectl create secret generic ${shellQuote(settings.secret.name)} ${ns} \\`,
      '  --from-literal=HF_TOKEN="${HF_TOKEN:-}" \\',
      "  --dry-run=client -o yaml | kubectl apply -f -",
      "",
    );
  }
  if (secretFile && !settings.bundle) {
    out.push("# Add a token to the Secret first if the model is gated or private.", `kubectl apply ${ns} -f ${secretFile.name}`, "");
  }
  const waitForDownload = downloads && !!cacheFile;
  if (cacheFile && !settings.bundle) {
    out.push("# Create the model cache and download the weights once.", `kubectl apply ${ns} -f ${cacheFile.name}`);
  }
  if (waitForDownload) {
    if (settings.bundle) out.push("# The bundle also creates the model cache and download Job.");
    else out.push(`kubectl wait --for=condition=complete job/model-download ${ns} --timeout=60m`);
    if (!info.modelPvc && info.graphs.some((graph) => graph.kind === "DynamoGraphDeploymentRequest")) {
      out.push(
        "# The last log line is the snapshot path; put its hash in spec.modelCache.pvcModelPath.",
        `kubectl logs job/model-download ${ns} | tail -n 1`,
      );
    }
    out.push("");
  } else if (cacheFile && !settings.bundle) {
    out.push("");
  }

  const unresolved = new Set([...manifest.matchAll(TEMPLATE_VAR)].map((match) => match[1]));
  const blockLines = blockContentLines(parseDocs(splitLines(manifest).lines));
  const templateVars = [...unresolved].filter((name) =>
    splitLines(manifest).lines.some((line, i) => !blockLines.has(i) && !line.trimStart().startsWith("#") && line.includes(`\${${name}}`)),
  );
  if (templateVars.length > 0) {
    out.push(
      "# Fill in the template variables, then substitute them with envsubst (gettext).",
      ...templateVars.map((name) => `export ${name}=${/NAMESPACE$/.test(name) ? '"$NAMESPACE"' : `"<${name.toLowerCase().replace(/_/g, "-")}>"`}`),
      `envsubst < ${manifestFile.name} | kubectl apply ${ns} -f -`,
    );
  } else {
    out.push(`kubectl apply ${ns} -f ${manifestFile.name}`);
  }
  if (settings.bundle && waitForDownload) {
    out.push(`kubectl wait --for=condition=complete job/model-download ${ns} --timeout=60m`);
  }

  const dgds = info.graphs.filter((graph) => graph.kind === "DynamoGraphDeployment");
  const dgdrs = info.graphs.filter((graph) => graph.kind === "DynamoGraphDeploymentRequest");
  const nameOf = (graph: GraphInfo) => settings.names[graph.doc]?.trim() || graph.name;
  let service: string | null = null;
  if (dgds.length === 1 && dgdrs.length === 0) {
    out.push(`kubectl wait --for=condition=Ready dgd/${nameOf(dgds[0])} ${ns} --timeout=30m`);
    if (info.frontends > 0) service = `${nameOf(dgds[0]).toLowerCase()}-frontend`;
  } else if (dgdrs.length === 1 && dgds.length === 0) {
    if (dgdrs[0].autoApply) {
      out.push(
        `kubectl wait --for=jsonpath='{.status.phase}'=Deployed dgdr/${nameOf(dgdrs[0])} ${ns} --timeout=60m`,
        `export DGD_NAME=$(kubectl get dgdr/${nameOf(dgdrs[0])} ${ns} -o jsonpath='{.status.dgdName}')`,
      );
      service = "${DGD_NAME}-frontend";
    } else {
      out.push("# autoApply is false: review the generated DGD in the DGDR status before deploying it.", `kubectl get dgdr/${nameOf(dgdrs[0])} ${ns} -o yaml`);
    }
  } else if (dgds.length > 1) {
    out.push(`kubectl get dgd ${ns} --watch`);
  }

  if (service) {
    out.push(
      "",
      "# Forward the Frontend and send a request.",
      `kubectl port-forward ${ns} "service/${service}" 8000:8000 \\`,
      "  >/tmp/dynamo-port-forward.log 2>&1 &",
      "until curl --silent --fail http://localhost:8000/health >/dev/null; do sleep 2; done",
    );
    if (settings.chatModel && model && !model.includes("${") && !model.includes("'")) {
      out.push(
        "curl --silent --show-error http://localhost:8000/v1/chat/completions \\",
        '  --header "Content-Type: application/json" \\',
        "  --data '{",
        `    "model": ${JSON.stringify(model)},`,
        '    "messages": [{"role": "user", "content": "What is NVIDIA Dynamo?"}],',
        '    "max_tokens": 200',
        "  }' | python3 -m json.tool",
      );
    } else {
      out.push("curl --silent http://localhost:8000/v1/models | python3 -m json.tool");
    }
  }
  return out.join("\n") + "\n";
}

export function customizeManifest(source: string, info: ManifestInfo, settings: CustomizeSettings): CustomizeResult {
  const notes: string[] = [];
  const editor = new Editor(source);

  applyComponents(editor, info, settings.components);
  if (settings.env) applyEnv(editor, info, settings.env);
  applyStorage(editor, info, settings.storage, notes);
  applySecret(editor, info, settings.secret);
  applyImages(editor, settings.image);
  applyNames(editor, info, settings.names);

  const model = effectiveModel(info, settings);
  let lines = substituteVars(editor.lines, settings.templateVars);
  if (settings.model?.trim() && info.models[0]) lines = replaceModel(lines, info.models[0], settings.model.trim());
  const manifestBody = joinLines({ lines, trailingNewline: editor.trailingNewline || true });

  const files: OutputFile[] = [];
  const extra: string[] = [];
  if (settings.secret.mode === "yaml") {
    const content = secretYaml(settings.secret.name);
    files.push({ id: "secret", name: `${settings.secret.name || "hf-token-secret"}.yaml`, language: "yaml", content });
    extra.push(content);
  }
  const existing = info.modelPvc;
  const storage: StorageSettings = {
    ...settings.storage,
    createPvc: settings.storage.createPvc && !info.claims.includes(existing?.claim ?? settings.storage.pvcName),
    downloadJob: settings.storage.downloadJob && !!model,
  };
  if (settings.storage.mode === "pvc" && settings.storage.downloadJob && !model) {
    notes.push("Set a model to generate the model download Job.");
  }
  if (storage.mode === "pvc" && (storage.createPvc || storage.downloadJob)) {
    const layout: CacheLayout = existing
      ? { mountPath: existing.mountPath, subPath: existing.subPath, env: existing.env }
      : { mountPath: storage.mountPath, subPath: null, env: [["HF_HOME", storage.mountPath]] };
    const content = modelCacheYaml(model, storage, settings.secret, layout);
    files.push({ id: "model-cache", name: "model-cache.yaml", language: "yaml", content });
    extra.push(content);
  }

  const manifest = settings.bundle && extra.length > 0 ? [...extra, manifestBody].map((doc) => doc.trimEnd()).join("\n---\n") + "\n" : manifestBody;
  const outputs: OutputFile[] = settings.bundle ? [] : files;
  const manifestFile: OutputFile = { id: "manifest", name: settings.fileName, language: "yaml", content: manifest };
  const all = [manifestFile, ...outputs];
  const stepSettings = { ...settings, storage };
  all.push({ id: "steps", name: "deploy-steps.sh", language: "bash", content: deploySteps(info, stepSettings, all, manifest, model) });

  for (const ref of info.images) {
    const image = parseImage(ref);
    if (!image || image.digest || !isDynamoImage(image, settings.image.defaultRegistry)) continue;
    const custom = settings.image.registry && settings.image.registry !== settings.image.defaultRegistry;
    if (!isPublishedImage(ref, settings.image) && !custom) {
      notes.push(`${image.repo} is not published on NGC. Build it, push it to your registry, and set Registry.`);
    }
  }
  if (info.images.some((ref) => parseImage(ref)?.digest)) notes.push("Digest-pinned images stay as written.");
  return { manifest, files: all, notes: [...new Set(notes)] };
}

// ---------------------------------------------------------------------------
// Syntax highlighting
// ---------------------------------------------------------------------------

export interface Token {
  t: string;
  c?: string;
}

const PLACEHOLDER = /\$\{[A-Za-z_][A-Za-z0-9_]*\}|<[a-z][\w-]*>|\bmy-(?:tag|registry)\b|\byour-[\w-]+/g;

/** Splits text into placeholder and non-placeholder runs with the given base class. */
function withPlaceholders(text: string, c: string | undefined, out: Token[]): void {
  let last = 0;
  for (const match of text.matchAll(PLACEHOLDER)) {
    const index = match.index ?? 0;
    if (index > last) out.push({ t: text.slice(last, index), c });
    out.push({ t: match[0], c: match[0].startsWith("$") ? "var" : "ph" });
    last = index + match[0].length;
  }
  if (last < text.length) out.push({ t: text.slice(last), c });
}

function plainClass(value: string, key: string | null, models: string[]): string | undefined {
  if (key === "kind") return "kind";
  if (/^(?:true|false|yes|no|on|off)$/i.test(value)) return "bool";
  if (/^(?:null|~)$/i.test(value)) return "null";
  if (/^[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$/.test(value) || /^\d+(?:\.\d+)?(?:Ki|Mi|Gi|Ti|Pi|Ei|m|k|M|G|T)$/.test(value)) return "num";
  if (/^--?[A-Za-z][\w.-]*(?:=.*)?$/.test(value)) return "flag";
  if (models.includes(value)) return "model";
  if (/^[a-z][\w+.-]*:\/\//i.test(value)) return "url";
  return "str";
}

function imageTokens(value: string, quote: string, out: Token[]): void {
  const image = parseImage(value);
  if (!image) {
    withPlaceholders(quote + value + quote, "str", out);
    return;
  }
  if (quote) out.push({ t: quote, c: "str" });
  if (image.registry) withPlaceholders(`${image.registry}/`, "img-reg", out);
  out.push({ t: image.repo, c: "img-repo" });
  if (image.tag !== null) {
    out.push({ t: ":", c: "punct" });
    withPlaceholders(image.tag, "img-tag", out);
  }
  if (image.digest) out.push({ t: `@${image.digest}`, c: "img-digest" });
  if (quote) out.push({ t: quote, c: "str" });
}

function flowTokens(text: string, models: string[], out: Token[]): void {
  const parts = text.split(/([[\]{},]|"(?:[^"\\]|\\.)*"|'(?:[^']|'')*')/);
  for (const part of parts) {
    if (!part) continue;
    if (/^[[\]{},]$/.test(part)) out.push({ t: part, c: "punct" });
    else if (part.startsWith('"') || part.startsWith("'")) withPlaceholders(part, "str", out);
    else {
      const match = part.match(/^(\s*)(.*?)(\s*)$/)!;
      if (match[1]) out.push({ t: match[1] });
      const split = splitKey(match[2]);
      if (split && !split.key.includes(" ")) {
        out.push({ t: match[2].slice(0, match[2].length - split.rest.length).trimEnd().replace(/:$/, ""), c: "key" });
        out.push({ t: ": ", c: "punct" });
        if (split.rest) withPlaceholders(split.rest, plainClass(split.rest, null, models), out);
      } else if (match[2]) {
        withPlaceholders(match[2], plainClass(match[2], null, models), out);
      }
      if (match[3]) out.push({ t: match[3] });
    }
  }
}

function valueTokens(value: string, key: string | null, models: string[], out: Token[]): void {
  const at = commentStart(value, true);
  const body = at < 0 ? value : value.slice(0, at);
  const comment = at < 0 ? "" : value.slice(at);
  const trimmed = body.trimEnd();
  const trailing = body.slice(trimmed.length);
  const anchor = trimmed.match(/^([&*][\w-]+|!{1,2}[\w-]*)(\s+|$)/);
  let rest = trimmed;
  if (anchor) {
    out.push({ t: anchor[1], c: "anchor" });
    if (anchor[2]) out.push({ t: anchor[2] });
    rest = trimmed.slice(anchor[0].length);
  }
  if (rest) {
    const first = rest[0];
    if (first === '"' || first === "'") {
      const inner = decodeQuoted(rest);
      if (key === "image" && quotedEnd(rest, 0) === rest.length - 1) imageTokens(inner, first, out);
      else withPlaceholders(rest, models.includes(inner) ? "model" : "str", out);
    } else if (first === "|" || first === ">") {
      out.push({ t: rest, c: "punct" });
    } else if (first === "[" || first === "{") {
      flowTokens(rest, models, out);
    } else if (key === "image") {
      imageTokens(rest, "", out);
    } else {
      withPlaceholders(rest, plainClass(rest, key, models), out);
    }
  }
  if (trailing) out.push({ t: trailing });
  if (comment) out.push({ t: comment, c: "cmt" });
}

/** Replaces `out` from `start` with one plain token if the tokens do not spell `line`. */
function ensureCovers(line: string, out: Token[], start: number): Token[] {
  const produced = out
    .slice(start)
    .map((token) => token.t)
    .join("");
  if (produced !== line) out.splice(start, out.length - start, { t: line });
  return out;
}

/** Shell-aware tokens for scripts in block scalars and generated deploy steps. */
export function shellTokens(line: string, out: Token[] = [], startOfCommand = true): Token[] {
  const start = out.length;
  const pattern =
    /(\s+)|(#.*$)|("(?:[^"\\]|\\.)*"?)|('[^']*'?)|(\$\{[^}]*\}|\$\([^)]*\)?|\$[A-Za-z_]\w*|\$[?!#@*0-9])|(\|\||&&|[|;&<>]|\\$)|(--?[A-Za-z][\w.-]*(?:=\S*)?)|([^\s|;&<>"'$]+)|(.)/g;
  let command = startOfCommand;
  for (const match of line.matchAll(pattern)) {
    const [token, space, comment, double, single, variable, operator, flag, word] = match;
    if (space) out.push({ t: token });
    else if (comment) out.push({ t: comment, c: "cmt" });
    else if (double) {
      let last = 0;
      for (const v of double.matchAll(/\$\{[^}]*\}|\$[A-Za-z_]\w*/g)) {
        const index = v.index ?? 0;
        if (index > last) out.push({ t: double.slice(last, index), c: "str" });
        out.push({ t: v[0], c: "var" });
        last = index + v[0].length;
      }
      if (last < double.length) out.push({ t: double.slice(last), c: "str" });
      command = false;
    } else if (single) {
      out.push({ t: single, c: "str" });
      command = false;
    } else if (variable) {
      out.push({ t: variable, c: "var" });
      command = false;
    } else if (operator) {
      out.push({ t: operator, c: "punct" });
      command = !["\\", ">", "<"].includes(operator);
    } else if (flag) {
      out.push({ t: flag, c: "flag" });
    } else if (word) {
      if (command && /^[A-Za-z_]\w*=/.test(word)) {
        const eq = word.indexOf("=");
        out.push({ t: word.slice(0, eq), c: "var" }, { t: "=", c: "punct" });
        withPlaceholders(word.slice(eq + 1), "str", out);
      } else if (command) {
        out.push({ t: word, c: "cmd" });
        command = false;
      } else {
        withPlaceholders(word, /^\d+$/.test(word) ? "num" : undefined, out);
      }
    } else {
      out.push({ t: token });
    }
  }
  return ensureCovers(line, out, start);
}

function jsonTokens(line: string, out: Token[]): void {
  const start = out.length;
  const pattern = /(\s+)|("(?:[^"\\]|\\.)*")(\s*:)?|(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)|(true|false|null)|([[\]{},:])|(.)/g;
  for (const match of line.matchAll(pattern)) {
    const [token, space, string, colon, number, literal, punct] = match;
    if (space) out.push({ t: token });
    else if (string) {
      out.push({ t: string, c: colon ? "key" : "str" });
      if (colon) out.push({ t: colon, c: "punct" });
    } else if (number) out.push({ t: number, c: "num" });
    else if (literal) out.push({ t: literal, c: literal === "null" ? "null" : "bool" });
    else if (punct) out.push({ t: punct, c: "punct" });
    else out.push({ t: token });
  }
  ensureCovers(line, out, start);
}

/** True when `line` leaves a single-quoted string open. */
function opensSingleQuote(line: string): boolean {
  let quote = "";
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quote === "'") {
      if (c === "'") quote = "";
    } else if (quote === '"') {
      if (c === "\\") i++;
      else if (c === '"') quote = "";
    } else if (c === "\\") i++;
    else if (c === "#" && (i === 0 || /\s/.test(line[i - 1]))) break;
    else if (c === "'" || c === '"') quote = c;
  }
  return quote === "'";
}

export function highlightBash(source: string): Token[][] {
  let continued = false;
  let inQuote = false;
  return splitLines(source).lines.map((line) => {
    const out: Token[] = [];
    let rest = line;
    if (inQuote) {
      const close = line.indexOf("'");
      if (close < 0) return [{ t: line, c: "str" }];
      out.push({ t: line.slice(0, close + 1), c: "str" });
      rest = line.slice(close + 1);
      inQuote = false;
    }
    shellTokens(rest, out, !continued && out.length === 0);
    inQuote = opensSingleQuote(rest);
    continued = /\\$/.test(line);
    return ensureCovers(line, out, 0);
  });
}

type BlockLanguage = "shell" | "yaml" | "json" | "text";

function blockLanguage(key: string | null): BlockLanguage {
  if (!key) return "text";
  if (/\.ya?ml$/i.test(key)) return "yaml";
  if (/\.json$/i.test(key)) return "json";
  if (/^(?:command|args|script|run|entrypoint)$/i.test(key) || /\.sh$/i.test(key)) return "shell";
  return "text";
}

function highlightBlock(lines: string[], language: BlockLanguage, models: string[]): Token[][] {
  const blank = (line: string): Token[] => (line ? [{ t: line }] : []);
  if (language === "yaml") {
    const content = lines.filter((line) => line.trim() !== "");
    const base = content.length ? Math.min(...content.map(leadingSpaces)) : 0;
    const nested = highlightYamlLines(
      lines.map((line) => (line.trim() === "" ? "" : line.slice(base))),
      models,
    );
    return lines.map((line, i) => (line.trim() === "" ? blank(line) : ensureCovers(line, [{ t: line.slice(0, base) }, ...nested[i]], 0)));
  }
  let continued = false;
  return lines.map((line) => {
    if (line.trim() === "") return blank(line);
    const indent = leadingSpaces(line);
    const out: Token[] = [{ t: line.slice(0, indent) }];
    if (language === "shell") {
      shellTokens(line.slice(indent), out, !continued);
      continued = /\\$/.test(line);
    } else if (language === "json") {
      jsonTokens(line.slice(indent), out);
    } else {
      withPlaceholders(line.slice(indent), "str", out);
    }
    return ensureCovers(line, out, 0);
  });
}

function highlightYamlLines(lines: string[], models: string[]): Token[][] {
  const result: Token[][] = [];
  const parents: { col: number; key: string }[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const indent = leadingSpaces(line);
    const body = line.slice(indent);
    i++;
    if (body.trim() === "") {
      result.push(line ? [{ t: line }] : []);
      continue;
    }
    if (isDocBoundary(line)) {
      parents.length = 0;
      result.push([{ t: line, c: "punct" }]);
      continue;
    }
    const out: Token[] = indent > 0 ? [{ t: line.slice(0, indent) }] : [];
    if (body.startsWith("#")) {
      out.push({ t: body, c: "cmt" });
      result.push(out);
      continue;
    }
    let rest = body;
    let col = indent;
    let dashCol = -1;
    while (rest === "-" || rest.startsWith("- ")) {
      const dash = rest.match(/^-\s*/)![0];
      out.push({ t: dash, c: "punct" });
      dashCol = col;
      col += dash.length;
      rest = rest.slice(dash.length);
    }
    if (dashCol >= 0) while (parents.length && parents[parents.length - 1].col > dashCol) parents.pop();
    const split = rest && !rest.startsWith("#") ? splitKey(rest) : null;
    let key: string | null = null;
    let owner = dashCol >= 0 ? dashCol : indent - 1;
    let valueText = rest;
    if (split) {
      while (parents.length && parents[parents.length - 1].col >= col) parents.pop();
      key = split.key;
      owner = col;
      const keyText = rest.slice(0, rest.length - split.rest.length);
      const colon = keyText.lastIndexOf(":");
      out.push({ t: keyText.slice(0, colon), c: "key" }, { t: keyText.slice(colon), c: "punct" });
      valueText = split.rest;
      if (valueText === "" || valueText.startsWith("#")) parents.push({ col, key });
    }
    if (valueText.startsWith("#")) out.push({ t: valueText, c: "cmt" });
    else valueTokens(valueText, key, models, out);
    result.push(ensureCovers(line, out, 0));

    if (BLOCK_INDICATOR.test(valueText.trim())) {
      let last = i - 1;
      for (let j = i; j < lines.length; j++) {
        if (lines[j].trim() === "") continue;
        if (leadingSpaces(lines[j]) <= owner) break;
        last = j;
      }
      const language = blockLanguage(key ?? parents[parents.length - 1]?.key ?? null);
      result.push(...highlightBlock(lines.slice(i, last + 1), language, models));
      i = last + 1;
    }
  }
  return result;
}

export function highlightYaml(source: string, models: string[] = []): Token[][] {
  return highlightYamlLines(splitLines(source).lines, models);
}

// ---------------------------------------------------------------------------
// Line diff
// ---------------------------------------------------------------------------

export interface DiffLine {
  type: "same" | "add" | "del";
  text: string;
  line: number | null;
  old: number | null;
}

/** Longest-common-subsequence line diff; `line` indexes the new text and `old` the original. */
export function diffLines(before: string[], after: string[]): DiffLine[] {
  let start = 0;
  while (start < before.length && start < after.length && before[start] === after[start]) start++;
  let endA = before.length;
  let endB = after.length;
  while (endA > start && endB > start && before[endA - 1] === after[endB - 1]) {
    endA--;
    endB--;
  }
  const a = before.slice(start, endA);
  const b = after.slice(start, endB);
  const width = b.length + 1;
  const table = new Uint32Array((a.length + 1) * width);
  for (let i = a.length - 1; i >= 0; i--) {
    for (let j = b.length - 1; j >= 0; j--) {
      table[i * width + j] = a[i] === b[j] ? table[(i + 1) * width + j + 1] + 1 : Math.max(table[(i + 1) * width + j], table[i * width + j + 1]);
    }
  }
  const out: DiffLine[] = [];
  for (let k = 0; k < start; k++) out.push({ type: "same", text: after[k], line: k, old: k });
  let i = 0;
  let j = 0;
  while (i < a.length || j < b.length) {
    if (i < a.length && j < b.length && a[i] === b[j]) {
      out.push({ type: "same", text: b[j], line: start + j, old: start + i });
      i++;
      j++;
    } else if (i < a.length && (j >= b.length || table[(i + 1) * width + j] >= table[i * width + j + 1])) {
      out.push({ type: "del", text: a[i], line: null, old: start + i });
      i++;
    } else {
      out.push({ type: "add", text: b[j], line: start + j, old: null });
      j++;
    }
  }
  for (let k = endB; k < after.length; k++) out.push({ type: "same", text: after[k], line: k, old: endA + (k - endB) });
  return out;
}
