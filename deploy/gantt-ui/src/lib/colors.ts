export const SPAN_COLORS: Record<string, string> = {
  "kvbm.get_matched_tokens": "#4dabf7",
  "kvbm.stage_local_matches": "#4dabf7",
  "kvbm.r2h": "#ff922b",
  "kvbm.worker_remote_transfer": "#ff922b",
  "kvbm.remote_transfer": "#ff922b",
  "kvbm.h2d": "#f06595",
  "kvbm.update_state_after_alloc": "#20c997",
  "kvbm.trigger_onboarding": "#20c997",
  "kvbm.onboard_from_g4": "#20c997",
  "vllm.prefill_scheduled": "#51cf66",
  "vllm.first_token": "#ffd43b",
  "vllm.finished": "#845ef7",
  "kvbm.request_finished": "#845ef7",
  "llm_request": "#339af0",
  "kvbm.flush_onboarding": "#868e96",
  "kvbm.build_connector_meta": "#868e96",
  "kvbm.schedule_new_request": "#868e96",
  "kvbm.start_load_kv": "#868e96",
  "kvbm.save_kv_layer": "#868e96",
};

export function spanColor(name: string): string {
  return SPAN_COLORS[name] ?? "#868e96";
}

export const LEGEND_ITEMS = [
  { label: "match", color: "#4dabf7" },
  { label: "R2H", color: "#ff922b" },
  { label: "H2D", color: "#f06595" },
  { label: "onboard", color: "#20c997" },
  { label: "prefill", color: "#51cf66" },
  { label: "first_tok", color: "#ffd43b" },
  { label: "finished", color: "#845ef7" },
  { label: "other", color: "#868e96" },
];
