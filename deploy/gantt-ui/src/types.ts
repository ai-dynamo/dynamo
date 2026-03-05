export interface Span {
  name: string;
  startNs: bigint;
  endNs: bigint;
  durMs: number;
  attributes: Record<string, string>;
}

export interface TraceRow {
  traceID: string;
  label: string;
  spans: Span[];
  minNs: bigint;
  maxNs: bigint;
  totalDurMs: number;
  attributes: Record<string, string>;
}

export interface ResponseEntry {
  scenario: string;
  isl: number;
  request: number;
  completion: string;
  prompt_key: string;
  traceparent: string;
  response_id: string;
  finish_reason: string;
  stop_reason: string | null;
  tool_calls: number;
  refusal: string | null;
  has_reasoning: boolean;
  prompt_tokens: number;
  completion_tokens: number;
  seed: number;
}

export interface TempoSearchResult {
  traces: Array<{
    traceID: string;
    startTimeUnixNano: string;
    durationMs: number;
  }>;
}

export interface TempoTraceData {
  batches?: Array<{
    scopeSpans?: Array<{
      spans?: Array<{
        name?: string;
        startTimeUnixNano?: string;
        endTimeUnixNano?: string;
        attributes?: Array<{
          key: string;
          value: { stringValue?: string; intValue?: string };
        }>;
      }>;
    }>;
  }>;
  resourceSpans?: Array<{
    scopeSpans?: Array<{
      spans?: Array<{
        name?: string;
        startTimeUnixNano?: string;
        endTimeUnixNano?: string;
        attributes?: Array<{
          key: string;
          value: { stringValue?: string; intValue?: string };
        }>;
      }>;
    }>;
  }>;
}
