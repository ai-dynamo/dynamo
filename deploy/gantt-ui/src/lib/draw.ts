import type { TraceRow, Span } from "../types";
import { spanColor } from "./colors";
import { fmtTime } from "./time";

export interface HitRect {
  x: number;
  y: number;
  w: number;
  h: number;
  rowIndex: number;
  spanIndex: number;
}

export interface HoverInfo {
  rowIndex: number;
  spanIndex: number;
}

export interface BrushState {
  x1: number;
  x2: number;
}

export interface DrawParams {
  ctx: CanvasRenderingContext2D;
  dpr: number;
  width: number;
  height: number;
  rows: TraceRow[];
  viewMin: bigint;
  viewMax: bigint;
  hover: HoverInfo | null;
  brush: BrushState | null;
  selectedRowIndex: number | null;
}

const COLORS = {
  bg: "#0d1117",
  surface: "#161b22",
  rowEven: "#161b22",
  rowOdd: "#1c2128",
  gutterBg: "#0d1117",
  border: "#30363d",
  gridLine: "#21262d",
  textPrimary: "#e6edf3",
  textSecondary: "#7d8590",
  textLabel: "#8b949e",
  axisText: "#7d8590",
  brushFill: "rgba(56,139,253,0.15)",
  brushStroke: "#58a6ff",
  hoverOutline: "#58a6ff",
  selectedRow: "rgba(56,139,253,0.12)",
  selectedBorder: "#58a6ff",
};

const LABEL_W = 200;
const ROW_H = 36;
const AXIS_H = 32;
const HEADER_H = 0;
const BAR_PAD_Y = 6;
const BAR_RADIUS = 3;
const N_TICKS = 8;

export function drawGantt(params: DrawParams): HitRect[] {
  const { ctx, dpr, width, height, rows, viewMin, viewMax, hover, brush, selectedRowIndex } = params;
  const hitRects: HitRect[] = [];

  ctx.save();
  ctx.scale(dpr, dpr);

  const chartW = width - LABEL_W;
  const viewDur = Number(viewMax - viewMin);
  const rowCount = rows.length;
  const contentH = rowCount * ROW_H;

  // background
  ctx.fillStyle = COLORS.bg;
  ctx.fillRect(0, 0, width, height);

  // gutter background
  ctx.fillStyle = COLORS.gutterBg;
  ctx.fillRect(0, 0, LABEL_W, height);

  // gutter separator
  ctx.strokeStyle = COLORS.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(LABEL_W, 0);
  ctx.lineTo(LABEL_W, height);
  ctx.stroke();

  if (viewDur <= 0 || rowCount === 0) {
    ctx.fillStyle = COLORS.textSecondary;
    ctx.font = "13px -apple-system, BlinkMacSystemFont, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("No data to display", width / 2, height / 2);
    ctx.restore();
    return hitRects;
  }

  // row backgrounds + labels
  ctx.font = "11px ui-monospace, SFMono-Regular, Menlo, monospace";
  ctx.textBaseline = "middle";

  for (let i = 0; i < rowCount; i++) {
    const y = HEADER_H + i * ROW_H;
    const isSelected = selectedRowIndex === i;

    // row bg in chart area
    ctx.fillStyle = isSelected ? COLORS.selectedRow : i % 2 === 0 ? COLORS.rowEven : COLORS.rowOdd;
    ctx.fillRect(LABEL_W, y, chartW, ROW_H);

    if (isSelected) {
      ctx.fillStyle = COLORS.selectedRow;
      ctx.fillRect(0, y, LABEL_W, ROW_H);
      ctx.strokeStyle = COLORS.selectedBorder;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(0, y + 0.5);
      ctx.lineTo(width, y + 0.5);
      ctx.moveTo(0, y + ROW_H - 0.5);
      ctx.lineTo(width, y + ROW_H - 0.5);
      ctx.stroke();
    }

    // label
    const row = rows[i];
    let label = row.label;
    if (label.length > 26) {
      label = label.substring(0, 12) + ".." + label.substring(label.length - 12);
    }
    ctx.fillStyle = isSelected ? COLORS.textPrimary : COLORS.textLabel;
    ctx.textAlign = "right";
    ctx.fillText(label, LABEL_W - 12, y + ROW_H / 2);

    // row bottom border
    ctx.strokeStyle = COLORS.gridLine;
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(LABEL_W, y + ROW_H);
    ctx.lineTo(width, y + ROW_H);
    ctx.stroke();
  }

  // time axis background
  const axisY = HEADER_H + contentH;
  ctx.fillStyle = COLORS.gutterBg;
  ctx.fillRect(0, axisY, width, AXIS_H);
  ctx.strokeStyle = COLORS.border;
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(LABEL_W, axisY);
  ctx.lineTo(width, axisY);
  ctx.stroke();

  // grid lines + tick labels
  ctx.font = "10px -apple-system, BlinkMacSystemFont, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  ctx.fillStyle = COLORS.axisText;

  for (let i = 0; i <= N_TICKS; i++) {
    const frac = i / N_TICKS;
    const x = LABEL_W + frac * chartW;
    const tNs = viewMin + BigInt(Math.round(viewDur * frac));

    // grid line
    ctx.strokeStyle = COLORS.gridLine;
    ctx.lineWidth = 0.5;
    ctx.beginPath();
    ctx.moveTo(x, HEADER_H);
    ctx.lineTo(x, axisY);
    ctx.stroke();

    // tick mark
    ctx.strokeStyle = COLORS.border;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, axisY);
    ctx.lineTo(x, axisY + 4);
    ctx.stroke();

    // label
    ctx.fillStyle = COLORS.axisText;
    ctx.fillText(fmtTime(tNs), x, axisY + 8);
  }

  // span bars
  for (let ri = 0; ri < rowCount; ri++) {
    const row = rows[ri];
    const rowY = HEADER_H + ri * ROW_H;

    for (let si = 0; si < row.spans.length; si++) {
      const span = row.spans[si];
      const relS = Number(span.startNs - viewMin);
      const relE = Number(span.endNs - viewMin);

      if (relE < 0 || relS > viewDur) continue;

      const clampS = Math.max(relS, 0);
      const clampE = Math.min(relE, viewDur);
      const xS = LABEL_W + (clampS / viewDur) * chartW;
      const xE = LABEL_W + (clampE / viewDur) * chartW;

      if (isNaN(xS) || isNaN(xE)) continue;

      const w = Math.max(xE - xS, 2);
      const barY = rowY + BAR_PAD_Y;
      const barH = ROW_H - BAR_PAD_Y * 2;

      const isHovered = hover?.rowIndex === ri && hover?.spanIndex === si;

      // bar fill
      ctx.fillStyle = spanColor(span.name);
      ctx.globalAlpha = isHovered ? 1.0 : 0.85;
      roundRect(ctx, xS, barY, w, barH, BAR_RADIUS);
      ctx.fill();
      ctx.globalAlpha = 1.0;

      // hover outline
      if (isHovered) {
        ctx.strokeStyle = COLORS.hoverOutline;
        ctx.lineWidth = 2;
        roundRect(ctx, xS, barY, w, barH, BAR_RADIUS);
        ctx.stroke();
      }

      // inline label
      if (w > 50) {
        const shortName = span.name
          .replace("kvbm.", "")
          .replace("vllm.", "")
          .substring(0, 16);
        ctx.fillStyle = "#ffffff";
        ctx.globalAlpha = 0.95;
        ctx.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
        ctx.textAlign = "left";
        ctx.textBaseline = "middle";
        ctx.fillText(shortName, xS + 4, rowY + ROW_H / 2, w - 8);
        ctx.globalAlpha = 1.0;
      }

      hitRects.push({ x: xS, y: barY, w, h: barH, rowIndex: ri, spanIndex: si });
    }
  }

  // brush overlay
  if (brush) {
    const bx = Math.min(brush.x1, brush.x2);
    const bw = Math.abs(brush.x2 - brush.x1);
    ctx.fillStyle = COLORS.brushFill;
    ctx.fillRect(bx, HEADER_H, bw, contentH);
    ctx.strokeStyle = COLORS.brushStroke;
    ctx.lineWidth = 1;
    ctx.strokeRect(bx, HEADER_H, bw, contentH);
  }

  ctx.restore();
  return hitRects;
}

function roundRect(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  if (w < r * 2) r = w / 2;
  if (h < r * 2) r = h / 2;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.lineTo(x + w - r, y);
  ctx.quadraticCurveTo(x + w, y, x + w, y + r);
  ctx.lineTo(x + w, y + h - r);
  ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
  ctx.lineTo(x + r, y + h);
  ctx.quadraticCurveTo(x, y + h, x, y + h - r);
  ctx.lineTo(x, y + r);
  ctx.quadraticCurveTo(x, y, x + r, y);
  ctx.closePath();
}

export const LAYOUT = { LABEL_W, ROW_H, AXIS_H, HEADER_H, BAR_PAD_Y };
