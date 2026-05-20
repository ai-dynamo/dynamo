import fs from "node:fs";
import path from "node:path";

const figuresDir = path.dirname(new URL(import.meta.url).pathname);
const localRawRoot = path.join(figuresDir, "data", "raw_root");
const benchDir = process.env.BENCH_DIR || localRawRoot;
const outputDir = path.join(figuresDir, "data");

if (!fs.existsSync(path.join(benchDir, "results"))) {
  throw new Error(`expected raw benchmark TSVs under ${path.join(benchDir, "results")}; set BENCH_DIR to a workspace containing results/`);
}

const sourceFiles = {
  cold: "results/cold-dgd-full-d6dn5-offlinecache-20260501T002339Z/cold_dgd_vllm_intervals.tsv",
  optimizedRestore: "results/compare-targetready-samenode-s2877-20260430T015100Z-vs-080837Z/regular_restore_gantt.tsv",
  criuDevRestore: "results/regular-slow-criu-blog-tx5tk-20260503T222310Z/regular_restore_gantt.tsv",
  gmsPvcRestore: "results/gms-defapi-preload2-full-s2877-05061745/gms_restore_gantt.tsv",
  gmsLocalSsdRestore: "results/gms-local-ssd-pinned-manual-trigger-api-full-s2877-20260503T173107Z/gms_restore_gantt.tsv",
};

const modelOrder = [
  "qwen3-06b",
  "qwen3-8b",
  "gpt-oss-120b",
];

const modelLabels = new Map([
  ["qwen3-06b", "Qwen3 0.6B"],
  ["qwen3-8b", "Qwen3 8B"],
  ["gpt-oss-120b", "GPT-OSS 120B"],
]);

const phaseColors = new Map([
  ["Cold Start", "#9C755F"],
  ["Python/Dynamo boot", "#F28E2B"],
  ["vLLM config + inspect", "#B07AA1"],
  ["EngineCore setup", "#D4A6C8"],
  ["Model load", "#9C755F"],
  ["torch.compile", "#CC3311"],
  ["Profile/KV sizing", "#A0CBE8"],
  ["Kernel warmup", "#7F3C8D"],
  ["CUDA graph capture", "#0072B2"],
  ["agent setup", "#111827"],
  ["CRIU restore", "#377eb8"],
  ["CUDA restore", "#4daf4a"],
  ["wait for GMS", "#e41a1c"],
  ["GMS load setup", "#ffd92f"],
  ["GMS load weights", "#ff7f00"],
  ["GMS commit", "#a65628"],
  ["wake / remap", "#6a3d9a"],
]);

function readTsv(file) {
  const text = fs.readFileSync(file, "utf8").trim();
  if (!text) {
    return [];
  }
  const [headerLine, ...body] = text.split(/\n/);
  const header = headerLine.split("\t");
  return body.map((line) => {
    const row = {};
    const cells = line.split("\t");
    header.forEach((key, index) => {
      row[key] = cells[index] ?? "";
    });
    for (const key of [
      "start_s",
      "duration_s",
      "start_s_after_container",
      "end_s_after_container",
      "duration_s",
    ]) {
      if (row[key] !== undefined) {
        row[key] = Number(row[key]);
      }
    }
    return row;
  });
}

function validInterval(row) {
  return Number.isFinite(row?.start_s_after_container)
    && Number.isFinite(row?.end_s_after_container)
    && row.end_s_after_container > row.start_s_after_container;
}

function findByLabel(rows, label) {
  const row = rows.find((entry) => entry.label === label && validInterval(entry));
  return row || null;
}

function addSegment(segments, source, phase, start, end, rawPhase, sourceFile, lane = "Critical path") {
  if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
    return;
  }
  segments.push({
    figure: "",
    source_file: sourceFile,
    model_key: source.model_key,
    model_label: source.model_label || modelLabels.get(source.model_key) || source.model_key,
    lane,
    phase,
    start_s: start,
    duration_s: end - start,
    color: phaseColors.get(phase) || "#94A3B8",
    raw_phase: rawPhase,
  });
}

function compileColdStart() {
  const sourcePath = path.join(benchDir, sourceFiles.cold);
  const rows = readTsv(sourcePath).filter((row) => modelOrder.includes(row.model_key) && validInterval(row));
  const rowsByModel = new Map();
  for (const row of rows) {
    if (!rowsByModel.has(row.model_key)) {
      rowsByModel.set(row.model_key, []);
    }
    rowsByModel.get(row.model_key).push(row);
  }

  const segments = [];
  for (const modelKey of modelOrder) {
    const modelRows = rowsByModel.get(modelKey) || [];
    if (modelRows.length === 0) {
      continue;
    }
    const source = modelRows[0];
    const container = findByLabel(modelRows, "Container start -> first Python/vLLM log");
    const platform = findByLabel(modelRows, "Main-process vLLM platform detection");
    const runtime = findByLabel(modelRows, "Runtime/discovery/status-server bootstrap");
    const hfCache = findByLabel(modelRows, "HF cache local path check");
    const modelInspect = findByLabel(modelRows, "vLLM model-class inspection");
    const engineConfig = findByLabel(modelRows, "vLLM engine config, tokenizer, renderer, child spawn");
    const engineBoot = findByLabel(modelRows, "EngineCore child Python/bootstrap");
    const enginePreload = findByLabel(modelRows, "EngineCore pre-load setup");
    const modelLoad = findByLabel(modelRows, "Model load total");
    const determineMemory = findByLabel(modelRows, "Determine available memory total");
    const torchCompile = findByLabel(modelRows, "torch.compile");
    const kernelWarmup = findByLabel(modelRows, "Kernel warmup");
    const cudaGraph = findByLabel(modelRows, "CUDA graph capture");

    const baseOffset = container?.end_s_after_container ?? 0;
    addSegment(
      segments,
      source,
      "Python/Dynamo boot",
      (container?.end_s_after_container ?? platform?.start_s_after_container) - baseOffset,
      (runtime?.end_s_after_container ?? hfCache?.end_s_after_container ?? modelInspect?.start_s_after_container) - baseOffset,
      "Container startup excluded; Python/Dynamo boot starts at first Python/vLLM log",
      sourceFiles.cold,
    );
    addSegment(
      segments,
      source,
      "vLLM config + inspect",
      modelInspect?.start_s_after_container - baseOffset,
      engineConfig?.end_s_after_container - baseOffset,
      "vLLM model-class inspection; engine config/tokenizer/renderer/child spawn",
      sourceFiles.cold,
    );
    addSegment(
      segments,
      source,
      "EngineCore setup",
      engineBoot?.start_s_after_container - baseOffset,
      enginePreload?.end_s_after_container - baseOffset,
      "EngineCore child Python/bootstrap; EngineCore pre-load setup",
      sourceFiles.cold,
    );
    if (modelLoad) {
      addSegment(
        segments,
        source,
        "Model load",
        modelLoad.start_s_after_container - baseOffset,
        modelLoad.end_s_after_container - baseOffset,
        "Model load total",
        sourceFiles.cold,
      );
    }
    if (determineMemory && torchCompile) {
      addSegment(
        segments,
        source,
        "Profile/KV sizing",
        determineMemory.start_s_after_container - baseOffset,
        torchCompile.start_s_after_container - baseOffset,
        "Determine available memory before torch.compile",
        sourceFiles.cold,
      );
      addSegment(
        segments,
        source,
        "torch.compile",
        torchCompile.start_s_after_container - baseOffset,
        torchCompile.end_s_after_container - baseOffset,
        "torch.compile",
        sourceFiles.cold,
      );
      addSegment(
        segments,
        source,
        "Profile/KV sizing",
        torchCompile.end_s_after_container - baseOffset,
        determineMemory.end_s_after_container - baseOffset,
        "Determine available memory after torch.compile",
        sourceFiles.cold,
      );
    }
    if (kernelWarmup) {
      addSegment(
        segments,
        source,
        "Kernel warmup",
        kernelWarmup.start_s_after_container - baseOffset,
        kernelWarmup.end_s_after_container - baseOffset,
        "Kernel warmup",
        sourceFiles.cold,
      );
    }
    if (cudaGraph) {
      addSegment(
        segments,
        source,
        "CUDA graph capture",
        cudaGraph.start_s_after_container - baseOffset,
        cudaGraph.end_s_after_container - baseOffset,
        "CUDA graph capture",
        sourceFiles.cold,
      );
    }
  }
  return coalesce(segments);
}

function restorePhase(row) {
  if (["agent_detect_to_restore_start", "host_inspect", "agent_nsrestore_launch", "nsrestore_setup", "restore_agent_to_nsrestore"].includes(row.phase)) {
    return "agent setup";
  }
  if (row.phase === "criu_restore") {
    return "CRIU restore";
  }
  if (row.phase === "cuda_restore") {
    return "CUDA restore";
  }
  if (row.phase === "worker_wake_up" || row.phase === "main_gms_remap") {
    return "wake / remap";
  }
  if (row.phase === "agent_wait_after_nsrestore_for_gms") {
    return "wait for GMS";
  }
  if (row.phase === "gms_load_wait_for_rw" || row.phase === "gms_load_phase_a_allocate_va") {
    return "GMS load setup";
  }
  if (row.phase === "gms_load_phase_b_stream_to_gpu") {
    return "GMS load weights";
  }
  if (row.phase === "gms_load_metadata_commit" || row.phase === "gms_load_sentinel") {
    return "GMS commit";
  }
  return null;
}

function restoreLane(row, phase, comparisonMode) {
  if (comparisonMode) {
    return "Snapshot Restore";
  }
  if (phase.startsWith("GMS")) {
    return "GMS loader";
  }
  if (phase === "wake / remap") {
    return "worker main";
  }
  return "snapshot agent";
}

function compileRestoreSource(sourceKey, comparisonMode) {
  const sourcePath = path.join(benchDir, sourceFiles[sourceKey]);
  const rows = readTsv(sourcePath)
    .filter((row) => modelOrder.includes(row.model_key))
    .filter((row) => Number.isFinite(row.start_s) && Number.isFinite(row.duration_s) && row.duration_s > 0);

  const segments = [];
  for (const row of rows) {
    const phase = restorePhase(row);
    if (!phase) {
      continue;
    }
    const lane = restoreLane(row, phase, comparisonMode);
    addSegment(
      segments,
      {
        model_key: row.model_key,
        model_label: modelLabels.get(row.model_key),
      },
      phase,
      row.start_s,
      row.start_s + row.duration_s,
      row.phase,
      sourceFiles[sourceKey],
      lane,
    );
  }
  return coalesce(segments);
}

function coalesce(rows) {
  const sorted = [...rows].sort((a, b) => {
    const modelDiff = modelOrder.indexOf(a.model_key) - modelOrder.indexOf(b.model_key);
    if (modelDiff !== 0) {
      return modelDiff;
    }
    const laneDiff = a.lane.localeCompare(b.lane);
    if (laneDiff !== 0) {
      return laneDiff;
    }
    const phaseDiff = a.phase.localeCompare(b.phase);
    if (phaseDiff !== 0) {
      return phaseDiff;
    }
    return a.start_s - b.start_s;
  });

  const merged = [];
  for (const row of sorted) {
    const last = merged[merged.length - 1];
    if (
      last
      && last.model_key === row.model_key
      && last.lane === row.lane
      && last.phase === row.phase
      && row.start_s - (last.start_s + last.duration_s) <= 0.025
    ) {
      const end = Math.max(last.start_s + last.duration_s, row.start_s + row.duration_s);
      last.duration_s = end - last.start_s;
      last.raw_phase = `${last.raw_phase}, ${row.raw_phase}`;
      continue;
    }
    merged.push({ ...row });
  }
  return merged;
}

function coldTotals(coldRows) {
  const totals = new Map();
  for (const modelKey of modelOrder) {
    const rows = coldRows.filter((row) => row.model_key === modelKey);
    const total = rows.reduce((max, row) => Math.max(max, row.start_s + row.duration_s), 0);
    totals.set(modelKey, total);
  }
  return totals;
}

function attachFigure(rows, figure) {
  return rows.map((row) => ({
    ...row,
    figure,
  }));
}

const coldRows = compileColdStart();
const totals = coldTotals(coldRows);
const rows = [
  ...attachFigure(coldRows, "cold_start_bench"),
];

for (const [figure, sourceKey] of [
  ["regular_restore", "optimizedRestore"],
  ["regular_restore_criudev", "criuDevRestore"],
]) {
  for (const modelKey of modelOrder) {
    rows.push({
      figure,
      source_file: sourceFiles.cold,
      model_key: modelKey,
      model_label: modelLabels.get(modelKey),
      lane: "Cold Start",
      phase: "Cold Start",
      start_s: 0,
      duration_s: totals.get(modelKey),
      color: phaseColors.get("Cold Start"),
      raw_phase: "cold start total, excluding base container startup",
    });
  }
  rows.push(...attachFigure(compileRestoreSource(sourceKey, true), figure));
}

rows.push(...attachFigure(compileRestoreSource("gmsPvcRestore", false), "gms_pvc_restore_bench"));
rows.push(...attachFigure(compileRestoreSource("gmsLocalSsdRestore", false), "gms_sharded_ssd_restore_bench"));

fs.mkdirSync(outputDir, { recursive: true });
const header = [
  "figure",
  "source_file",
  "model_key",
  "model_label",
  "lane",
  "phase",
  "start_s",
  "duration_s",
  "color",
  "raw_phase",
];
const lines = [header.join("\t")];
for (const row of rows) {
  lines.push(header.map((key) => {
    const value = row[key] ?? "";
    if (typeof value === "number") {
      return value.toFixed(3);
    }
    return String(value).replaceAll("\t", " ").replaceAll("\n", " ");
  }).join("\t"));
}
fs.writeFileSync(path.join(outputDir, "benchmark_segments.tsv"), `${lines.join("\n")}\n`);

const sourceHeader = ["key", "path", "used_by"];
const sourceLines = [sourceHeader.join("\t")];
for (const [key, sourceFile] of Object.entries(sourceFiles)) {
  const usedBy = rows
    .filter((row) => row.source_file === sourceFile)
    .map((row) => row.figure)
    .filter((value, index, all) => all.indexOf(value) === index)
    .join(",");
  sourceLines.push([key, sourceFile, usedBy].join("\t"));
}
fs.writeFileSync(path.join(outputDir, "source_files.tsv"), `${sourceLines.join("\n")}\n`);
console.log(`wrote ${path.join(outputDir, "benchmark_segments.tsv")}`);
console.log(`wrote ${path.join(outputDir, "source_files.tsv")}`);
