import { useRef, useEffect, useCallback, useState } from "react";
import { useGanttStore } from "../store";
import { drawGantt, LAYOUT, type HitRect, type HoverInfo, type BrushState } from "../lib/draw";

export function GanttCanvas() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const hitRectsRef = useRef<HitRect[]>([]);

  const traceRows = useGanttStore((s) => s.traceRows);
  const viewMin = useGanttStore((s) => s.viewMin);
  const viewMax = useGanttStore((s) => s.viewMax);
  const zoomAt = useGanttStore((s) => s.zoomAt);
  const pan = useGanttStore((s) => s.pan);
  const resetZoom = useGanttStore((s) => s.resetZoom);
  const brushZoom = useGanttStore((s) => s.brushZoom);
  const setHover = useGanttStore((s) => s.setHover);
  const selectRow = useGanttStore((s) => s.selectRow);
  const selectedRowIndex = useGanttStore((s) => s.selectedRowIndex);

  const [size, setSize] = useState({ w: 800, h: 400 });
  const [hoverInfo, setHoverInfo] = useState<HoverInfo | null>(null);
  const [brush, setBrush] = useState<BrushState | null>(null);
  const [panning, setPanning] = useState<{ startX: number } | null>(null);
  const wasDragging = useRef(false);
  const mouseXRef = useRef(0);

  // resize observer
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      for (const e of entries) {
        const { width, height } = e.contentRect;
        if (width > 0 && height > 0) setSize({ w: width, h: height });
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  // draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size.w * dpr;
    canvas.height = size.h * dpr;
    canvas.style.width = `${size.w}px`;
    canvas.style.height = `${size.h}px`;

    hitRectsRef.current = drawGantt({
      ctx,
      dpr,
      width: size.w,
      height: size.h,
      rows: traceRows,
      viewMin,
      viewMax,
      hover: hoverInfo,
      brush,
      selectedRowIndex,
    });
  }, [size, traceRows, viewMin, viewMax, hoverInfo, brush, selectedRowIndex]);

  const getCanvasXY = useCallback(
    (e: React.MouseEvent | MouseEvent) => {
      const rect = canvasRef.current?.getBoundingClientRect();
      if (!rect) return { x: 0, y: 0 };
      return { x: e.clientX - rect.left, y: e.clientY - rect.top };
    },
    [],
  );

  const getChartFrac = useCallback(
    (canvasX: number) => {
      return Math.max(0, Math.min(1, (canvasX - LAYOUT.LABEL_W) / (size.w - LAYOUT.LABEL_W)));
    },
    [size.w],
  );

  const hitTest = useCallback(
    (cx: number, cy: number): HoverInfo | null => {
      for (const r of hitRectsRef.current) {
        if (cx >= r.x && cx <= r.x + r.w && cy >= r.y && cy <= r.y + r.h) {
          return { rowIndex: r.rowIndex, spanIndex: r.spanIndex };
        }
      }
      return null;
    },
    [],
  );

  const onMouseMove = useCallback(
    (e: React.MouseEvent) => {
      const { x, y } = getCanvasXY(e);
      mouseXRef.current = x;
      const hit = hitTest(x, y);
      setHoverInfo(hit);

      if (hit) {
        const row = traceRows[hit.rowIndex];
        const span = row?.spans[hit.spanIndex];
        if (span) {
          setHover({
            span,
            row,
            x: e.clientX,
            y: e.clientY,
          });
        }
      } else {
        setHover(null);
      }
    },
    [getCanvasXY, hitTest, traceRows, setHover],
  );

  const onWheel = useCallback(
    (e: React.WheelEvent) => {
      e.preventDefault();
      const { x } = getCanvasXY(e);
      const frac = getChartFrac(x);
      zoomAt(frac, e.deltaY > 0 ? 1.25 : 0.8);
    },
    [getCanvasXY, getChartFrac, zoomAt],
  );

  const onMouseDown = useCallback(
    (e: React.MouseEvent) => {
      const { x } = getCanvasXY(e);
      wasDragging.current = false;
      if (x < LAYOUT.LABEL_W) return;
      e.preventDefault();
      if (e.shiftKey) {
        setPanning({ startX: e.clientX });
      } else {
        setBrush({ x1: x, x2: x });
      }
    },
    [getCanvasXY],
  );

  const onClick = useCallback(
    (e: React.MouseEvent) => {
      if (wasDragging.current) { wasDragging.current = false; return; }
      const { x, y } = getCanvasXY(e);
      const rowIdx = Math.floor((y - LAYOUT.HEADER_H) / LAYOUT.ROW_H);
      if (rowIdx < 0 || rowIdx >= traceRows.length) return;

      const hit = hitTest(x, y);
      if (hit || x < LAYOUT.LABEL_W) {
        selectRow(selectedRowIndex === rowIdx ? null : rowIdx);
      }
    },
    [getCanvasXY, traceRows, hitTest, selectRow, selectedRowIndex],
  );

  // global mousemove/mouseup for brush/pan
  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      if (brush) {
        const { x } = getCanvasXY(e);
        setBrush((b) => (b ? { ...b, x2: x } : null));
      }
      if (panning) {
        const dx = e.clientX - panning.startX;
        const chartW = size.w - LAYOUT.LABEL_W;
        if (chartW > 0) pan(-dx / chartW);
        setPanning({ startX: e.clientX });
      }
    };
    const onUp = (e: MouseEvent) => {
      if (brush) {
        const { x } = getCanvasXY(e);
        const chartW = size.w - LAYOUT.LABEL_W;
        if (chartW > 0 && Math.abs(x - brush.x1) > 5) {
          const f1 = getChartFrac(brush.x1);
          const f2 = getChartFrac(x);
          brushZoom(Math.min(f1, f2), Math.max(f1, f2));
          wasDragging.current = true;
        }
        setBrush(null);
      }
      if (panning) {
        wasDragging.current = true;
        setPanning(null);
      }
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
    };
  }, [brush, panning, size.w, getCanvasXY, getChartFrac, pan, brushZoom]);

  const inGutter = mouseXRef.current < LAYOUT.LABEL_W;
  const cursor = panning
    ? "grabbing"
    : brush
      ? "col-resize"
      : inGutter
        ? "pointer"
        : hoverInfo
          ? "pointer"
          : "crosshair";

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden">
      <canvas
        ref={canvasRef}
        style={{ cursor, display: "block" }}
        onMouseMove={onMouseMove}
        onMouseDown={onMouseDown}
        onWheel={onWheel}
        onDoubleClick={resetZoom}
        onClick={onClick}
        onMouseLeave={() => {
          setHoverInfo(null);
          setHover(null);
        }}
      />
    </div>
  );
}
