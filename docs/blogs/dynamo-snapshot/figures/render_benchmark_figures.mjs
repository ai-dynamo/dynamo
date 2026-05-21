import fs from "node:fs";
import path from "node:path";

const figuresDir = path.dirname(new URL(import.meta.url).pathname);
const dataFile = path.join(figuresDir, "data", "benchmark_segments.tsv");
const outputDir = process.argv.includes("--out")
  ? path.resolve(process.argv[process.argv.indexOf("--out") + 1])
  : figuresDir;

const chartDefs = new Map([
  ["cold_start_bench", {
    output: "cold_start_bench.svg",
    title: "Dynamo vLLM Cold Start",
    subtitle: "Single-GPU (1xB200); excludes base container startup",
    type: "cold",
    laneOrder: ["Critical path"],
    phaseOrder: [
      "Python/Dynamo boot",
      "vLLM config + inspect",
      "EngineCore setup",
      "Model load",
      "torch.compile",
      "Profile/KV sizing",
      "Kernel warmup",
      "CUDA graph capture",
    ],
  }],
  ["regular_restore", {
    output: "regular_restore.svg",
    title: "Dynamo vLLM Snapshot Restore (Optimized)",
    subtitle: "Single-GPU (1xB200), with AIO and parallel memfd optimizations",
    type: "restore",
    laneOrder: ["Cold Start", "Snapshot Restore"],
    phaseOrder: ["Cold Start", "agent setup", "CRIU restore", "CUDA restore", "wake / remap"],
  }],
  ["regular_restore_criudev", {
    output: "regular_restore_criudev.svg",
    title: "Dynamo vLLM Snapshot Restore (Unoptimized)",
    subtitle: "Single-GPU (1xB200)",
    type: "restore",
    laneOrder: ["Cold Start", "Snapshot Restore"],
    phaseOrder: ["Cold Start", "agent setup", "CRIU restore", "CUDA restore", "wake / remap"],
  }],
  ["gms_pvc_restore_bench", {
    output: "gms_pvc_restore_bench.svg",
    title: "Dynamo vLLM Snapshot Restore with GMS",
    subtitle: "Single-GPU (1xB200), weights on NFS",
    type: "gms",
    laneOrder: ["GMS loader", "snapshot agent", "worker main"],
    phaseOrder: ["agent setup", "CRIU restore", "CUDA restore", "wait for GMS", "GMS load setup", "GMS load weights", "GMS commit", "wake / remap"],
  }],
  ["gms_sharded_ssd_restore_bench", {
    output: "gms_sharded_ssd_restore_bench.svg",
    title: "Dynamo vLLM Snapshot Restore with GMS",
    subtitle: "Single-GPU (1xB200), weights sharded across 8 NVMe SSDs",
    type: "gms",
    laneOrder: ["GMS loader", "snapshot agent", "worker main"],
    phaseOrder: ["agent setup", "CRIU restore", "CUDA restore", "GMS load setup", "GMS load weights", "GMS commit", "wake / remap"],
  }],
]);

const modelOrder = ["qwen3-06b", "qwen3-8b", "gpt-oss-120b"];
const modelRank = new Map(modelOrder.map((model, index) => [model, index]));

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function readTsv(file) {
  const text = fs.readFileSync(file, "utf8").trim();
  const [headerLine, ...body] = text.split(/\n/);
  const header = headerLine.split("\t");
  return body.map((line) => {
    const row = {};
    const cells = line.split("\t");
    header.forEach((key, index) => {
      row[key] = cells[index] ?? "";
    });
    row.start_s = Number(row.start_s);
    row.duration_s = Number(row.duration_s);
    row.end_s = row.start_s + row.duration_s;
    return row;
  });
}

function niceStep(maxValue, targetTicks) {
  if (!Number.isFinite(maxValue) || maxValue <= 0) {
    return 1;
  }
  const rough = maxValue / targetTicks;
  const power = 10 ** Math.floor(Math.log10(rough));
  for (const multiple of [1, 2, 5, 10]) {
    if (rough <= multiple * power) {
      return multiple * power;
    }
  }
  return 10 * power;
}

function tickLabel(value) {
  if (value >= 60) {
    const minutes = Math.floor(value / 60);
    const seconds = Math.round(value - minutes * 60);
    return seconds === 0 ? `${minutes}m` : `${minutes}m${String(seconds).padStart(2, "0")}s`;
  }
  return value >= 10 ? `${Math.round(value)}s` : `${value.toFixed(1)}s`;
}

function durationLabel(duration) {
  return duration >= 10 ? `${duration.toFixed(0)}s` : `${duration.toFixed(1)}s`;
}

function renderCold(def, rows) {
  const models = groupByModel(rows);
  const width = 1900;
  const left = 220;
  const right = 70;
  const top = 250;
  const rowHeight = 74;
  const barHeight = 44;
  const plotWidth = width - left - right;
  const axisBottom = top + models.length * rowHeight + 32;
  const noteTop = axisBottom + 72;
  const height = noteTop + 98;
  const maxEnd = Math.max(...rows.map((row) => row.end_s));
  const maxScale = maxEnd * 1.04;
  const x = (value) => left + (value / maxScale) * plotWidth;
  const tickStep = niceStep(maxEnd, 6);

  const svg = baseSvg(width, height, "Dynamo vLLM cold start serial critical path Gantt chart");
  svg.push(`<text x="${left}" y="64" class="title">Dynamo vLLM Cold Start</text>`);
  svg.push(`<text x="${left}" y="110" class="subtitle">${esc(def.subtitle)}</text>`);
  renderLegend(svg, def.phaseOrder, rows, left, 153, width - right);
  renderColdAxis(svg, left, right, top, axisBottom, width, tickStep, maxScale, x);

  models.forEach((model, index) => {
    const y = top + index * rowHeight;
    const centerY = y + barHeight / 2;
    if (index % 2 === 1) {
      svg.push(`<rect x="${left}" y="${y - 7}" width="${plotWidth}" height="${rowHeight}" fill="#F8FAFC"/>`);
    }
    svg.push(`<line x1="${left}" y1="${y + rowHeight - 9}" x2="${width - right}" y2="${y + rowHeight - 9}" stroke="#F1F5F9" stroke-width="1"/>`);
    svg.push(`<text x="${left - 14}" y="${centerY + 8}" class="model" text-anchor="end">${esc(model.label)}</text>`);
    for (const row of model.rows) {
      renderBar(svg, row, x, y, barHeight);
    }
  });

  renderColdNote(svg, left, noteTop, plotWidth);
  svg.push(`</svg>`);
  return `${svg.join("\n")}\n`;
}

function renderRestore(def, rows) {
  const width = 1900;
  const left = 240;
  const right = 90;
  const top = 190;
  const rowHeight = 50;
  const laneGap = 14;
  const groupGap = 42;
  const models = groupByModel(rows);
  const modelHeight = def.laneOrder.length * (rowHeight + laneGap) + groupGap;
  const laneAreaHeight = models.length * modelHeight - groupGap + 34;
  const legendTop = top + laneAreaHeight + 58;
  const legendColors = colorsByPhase(rows);
  const visibleLegendPhases = def.phaseOrder.filter((phase) => legendColors.has(phase));
  const legendColumns = def.type === "restore"
    ? visibleLegendPhases.length
    : Math.min(4, visibleLegendPhases.length);
  const legendRows = Math.ceil(visibleLegendPhases.length / legendColumns);
  const height = legendTop + legendRows * 34 + 50;
  const maxEnd = Math.max(...rows.map((row) => row.end_s));
  const maxScale = Math.ceil(maxEnd / 5) * 5 || 5;
  const chartWidth = width - left - right;
  const x = (value) => left + (value / maxScale) * chartWidth;
  const tickStep = niceStep(maxScale, def.type === "gms" ? 4 : 10);

  const svg = [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" style="background:#f8fafc;color-scheme:light">`,
    styleBlock(),
    `<rect width="100%" height="100%" fill="#f8fafc"/>`,
    `<rect x="${left - 12}" y="${top - 22}" width="${chartWidth + 24}" height="${laneAreaHeight}" rx="6" fill="#ffffff" stroke="#e2e8f0"/>`,
    `<text class="title" x="${left}" y="64">${esc(def.title)}</text>`,
    `<text class="subtitle" x="${left}" y="110">${esc(def.subtitle)}</text>`,
  ];

  for (let tick = 0; tick <= maxScale + 0.001; tick += tickStep) {
    const tx = x(tick);
    svg.push(`<line x1="${tx.toFixed(1)}" y1="${top}" x2="${tx.toFixed(1)}" y2="${top + laneAreaHeight - 10}" stroke="#e5e7eb"/>`);
    svg.push(`<line x1="${tx.toFixed(1)}" y1="${top - 6}" x2="${tx.toFixed(1)}" y2="${top}" stroke="#64748b"/>`);
    svg.push(`<text class="axis-label" x="${tx.toFixed(1)}" y="${top - 10}" text-anchor="middle">${tickLabel(tick)}</text>`);
    svg.push(`<text class="axis-label" x="${tx.toFixed(1)}" y="${top + laneAreaHeight + 16}" text-anchor="middle">${tickLabel(tick)}</text>`);
  }
  svg.push(`<line x1="${left}" y1="${top}" x2="${width - right}" y2="${top}" stroke="#64748b" stroke-width="1"/>`);
  svg.push(`<line x1="${left}" y1="${top - 22}" x2="${left}" y2="${top + laneAreaHeight - 10}" stroke="#0f172a" stroke-width="1.2"/>`);

  let yCursor = top + 28;
  for (const model of models) {
    svg.push(`<text class="model-label" x="${left - 10}" y="${yCursor - 12}" text-anchor="end">${esc(model.label)}</text>`);
    for (const lane of def.laneOrder) {
      const laneRows = model.rows.filter((row) => row.lane === lane);
      if (laneRows.length === 0) {
        yCursor += rowHeight + laneGap;
        continue;
      }
      svg.push(`<text class="lane-label" x="${left - 10}" y="${yCursor + 31}" text-anchor="end">${esc(lane)}</text>`);
      svg.push(`<line x1="${left}" y1="${yCursor + rowHeight + 4}" x2="${width - right}" y2="${yCursor + rowHeight + 4}" stroke="#f1f5f9"/>`);
      for (const row of laneRows) {
        renderBar(svg, row, x, yCursor, rowHeight);
      }
      yCursor += rowHeight + laneGap;
    }
    yCursor += groupGap;
  }

  renderRestoreLegend(svg, def.phaseOrder, rows, left, legendTop, width - right, legendColumns);
  svg.push(`</svg>`);
  return `${svg.join("\n")}\n`;
}

function groupByModel(rows) {
  const groups = [];
  for (const modelKey of modelOrder) {
    const modelRows = rows
      .filter((row) => row.model_key === modelKey)
      .sort((a, b) => a.start_s - b.start_s || a.phase.localeCompare(b.phase));
    if (modelRows.length === 0) {
      continue;
    }
    groups.push({
      key: modelKey,
      label: modelRows[0].model_label,
      rows: modelRows,
    });
  }
  return groups;
}

function baseSvg(width, height, ariaLabel) {
  return [
    `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img" aria-label="${esc(ariaLabel)}">`,
    `<rect width="${width}" height="${height}" fill="#FFFFFF"/>`,
    styleBlock(),
  ];
}

function styleBlock() {
  return `<style>
  text { font-family: Inter, Arial, Helvetica, sans-serif; letter-spacing: 0; fill: #0F172A; }
  .title { font-size: 50px; font-weight: 800; }
  .subtitle { font-size: 26px; fill: #475569; }
  .axis, .axis-label { font-size: 21px; font-weight: 700; fill: #334155; }
  .model, .model-label { font-size: 24px; font-weight: 800; fill: #0F172A; }
  .lane-label { font-size: 20px; font-weight: 700; fill: #475569; }
  .legend, .legend-label { font-size: 22px; font-weight: 700; fill: #1E293B; }
  .bar-label { font-size: 19px; font-weight: 800; fill: #0F172A; }
</style>`;
}

function renderLegend(svg, phaseOrder, rows, left, y, maxX) {
  const colors = colorsByPhase(rows);
  const columns = 4;
  const columnWidth = (maxX - left) / columns;
  let visibleIndex = 0;
  for (const phase of phaseOrder) {
    if (!colors.has(phase)) {
      continue;
    }
    const legendX = left + (visibleIndex % columns) * columnWidth;
    const legendY = y + Math.floor(visibleIndex / columns) * 42;
    svg.push(`<rect x="${legendX}" y="${legendY - 20}" width="23" height="23" rx="3" fill="${colors.get(phase)}"/>`);
    svg.push(`<text x="${legendX + 33}" y="${legendY + 1}" class="legend">${esc(phase)}</text>`);
    visibleIndex += 1;
  }
}

function renderRestoreLegend(svg, phaseOrder, rows, left, y, maxX, columns) {
  const colors = colorsByPhase(rows);
  const columnWidth = (maxX - left) / columns;
  let visibleIndex = 0;
  for (const phase of phaseOrder) {
    if (!colors.has(phase)) {
      continue;
    }
    const legendX = left + (visibleIndex % columns) * columnWidth;
    const legendY = y + Math.floor(visibleIndex / columns) * 34;
    svg.push(`<rect x="${legendX}" y="${legendY - 17}" width="22" height="22" rx="3" fill="${colors.get(phase)}" stroke="#0f172a" stroke-opacity="0.20"/>`);
    svg.push(`<text class="legend-label" x="${legendX + 32}" y="${legendY + 1}">${esc(phase)}</text>`);
    visibleIndex += 1;
  }
}

function colorsByPhase(rows) {
  const colors = new Map();
  for (const row of rows) {
    if (!colors.has(row.phase)) {
      colors.set(row.phase, row.color);
    }
  }
  return colors;
}

function renderColdAxis(svg, left, right, top, axisBottom, width, tickStep, maxScale, x) {
  for (let tick = 0; tick <= maxScale + 0.001; tick += tickStep) {
    const tx = x(tick);
    svg.push(`<line x1="${tx.toFixed(1)}" y1="${top}" x2="${tx.toFixed(1)}" y2="${axisBottom}" stroke="#E2E8F0" stroke-width="1"/>`);
    svg.push(`<text x="${tx.toFixed(1)}" y="${top - 10}" class="axis" text-anchor="middle">${tickLabel(tick)}</text>`);
    svg.push(`<text x="${tx.toFixed(1)}" y="${axisBottom + 24}" class="axis" text-anchor="middle">${tickLabel(tick)}</text>`);
  }
  svg.push(`<line x1="${left}" y1="${top}" x2="${width - right}" y2="${top}" stroke="#CBD5E1" stroke-width="1.2"/>`);
  svg.push(`<line x1="${left}" y1="${axisBottom}" x2="${width - right}" y2="${axisBottom}" stroke="#CBD5E1" stroke-width="1.2"/>`);
  svg.push(`<text x="${width - right}" y="${axisBottom + 58}" class="axis" text-anchor="end">seconds</text>`);
}

function renderColdNote(svg, left, top, width) {
  svg.push(`<rect x="${left}" y="${top}" width="${width}" height="74" rx="6" fill="#F8FAFC" stroke="#E2E8F0"/>`);
  svg.push(`<text x="${left + 24}" y="${top + 30}" style="font-size:20px;font-weight:800;fill:#0F172A">Takeaway</text>`);
  svg.push(`<text x="${left + 24}" y="${top + 57}" style="font-size:20px;font-weight:600;fill:#334155">With high-bandwidth network storage, small-model startup is mostly engine initialization, not weight load; cached compile/warmup artifacts trim only part of the path.</text>`);
}

function renderBar(svg, row, x, y, height) {
  const barX = x(row.start_s);
  const barWidth = Math.max(1.2, x(row.end_s) - barX);
  svg.push(`<rect x="${barX.toFixed(1)}" y="${y}" width="${barWidth.toFixed(1)}" height="${height}" rx="3" fill="${row.color}" stroke="#0f172a" stroke-opacity="0.22" stroke-width="1">`);
  svg.push(`<title>${esc(`${row.model_label} ${row.phase}: ${row.duration_s.toFixed(3)}s (${row.start_s.toFixed(3)}s -> ${row.end_s.toFixed(3)}s); raw=${row.raw_phase}`)}</title>`);
  svg.push(`</rect>`);
  const label = durationLabel(row.duration_s);
  const labelWidth = Math.max(label.length * 11 + 18, 44);
  if (barWidth > labelWidth + 8 && row.duration_s >= 1) {
    const labelX = barX + barWidth / 2;
    const labelY = y + height / 2;
    svg.push(`<rect x="${(labelX - labelWidth / 2).toFixed(1)}" y="${(labelY - 16).toFixed(1)}" width="${labelWidth.toFixed(1)}" height="32" rx="5" fill="#FFFFFF" fill-opacity="0.96" stroke="#0F172A" stroke-opacity="0.35"/>`);
    svg.push(`<text x="${labelX.toFixed(1)}" y="${(labelY + 7).toFixed(1)}" class="bar-label" text-anchor="middle">${esc(label)}</text>`);
  }
}

const rows = readTsv(dataFile)
  .filter((row) => Number.isFinite(row.start_s) && Number.isFinite(row.duration_s) && row.duration_s > 0)
  .sort((a, b) => {
    const figureDiff = a.figure.localeCompare(b.figure);
    if (figureDiff !== 0) {
      return figureDiff;
    }
    const modelDiff = (modelRank.get(a.model_key) ?? 99) - (modelRank.get(b.model_key) ?? 99);
    if (modelDiff !== 0) {
      return modelDiff;
    }
    return a.start_s - b.start_s;
  });

fs.mkdirSync(outputDir, { recursive: true });
for (const [figure, def] of chartDefs) {
  const figureRows = rows.filter((row) => row.figure === figure);
  const svg = def.type === "cold"
    ? renderCold(def, figureRows)
    : renderRestore(def, figureRows);
  const output = path.join(outputDir, def.output);
  fs.writeFileSync(output, svg);
  console.log(`wrote ${output}`);
}
