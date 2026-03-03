#!/usr/bin/env bash
# Shared functions for parsing mx-target logs and generating startup timing tables.
#
# Usage: source this file, then call print_startup_table.
# Requires: LOCAL_ARTIFACTS set to the artifacts directory.
# Optional: P2P_SUBDIR and DISK_SUBDIR (default: "p2p" and "disk").

# Parse an mx-target log and print timing values as key=value pairs.
# Usage: parse_target_timings <logfile> <mode>
parse_target_timings() {
  local log="$1" mode="$2"
  if [[ ! -s "$log" ]]; then
    echo "MISSING=1"
    return
  fi

  # Strip ANSI codes for reliable parsing
  local clean
  clean=$(sed 's/\x1b\[[0-9;]*m//g' "$log")

  # Runtime start: first timestamp in log
  local start_ts
  start_ts=$(echo "$clean" | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' | head -1)

  # Ready for inference
  local ready_ts
  ready_ts=$(echo "$clean" | grep 'warmup complete' | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' | head -1)

  # Total startup (seconds)
  local total_s="N/A"
  if [[ -n "$start_ts" && -n "$ready_ts" ]]; then
    local s_epoch r_epoch
    s_epoch=$(date -d "$start_ts" +%s 2>/dev/null) || s_epoch=""
    r_epoch=$(date -d "$ready_ts" +%s 2>/dev/null) || r_epoch=""
    if [[ -n "$s_epoch" && -n "$r_epoch" ]]; then
      total_s=$(( r_epoch - s_epoch ))
    fi
  fi

  # Weight loading (disk mode)
  local disk_weight_s
  disk_weight_s=$(echo "$clean" | grep -oP 'Loading weights took \K[\d.]+' | head -1)

  # RDMA transfer (P2P mode): average across all workers
  local rdma_s="" rdma_bw=""
  local _rdma_times _rdma_bws
  _rdma_times=$(echo "$clean" | grep 'Transfer complete:' | grep -oP 'in \K[\d.]+(?=s)' || true)
  _rdma_bws=$(echo "$clean" | grep 'Transfer complete:' | grep -oP '\(\K[\d.]+(?= Gbps)' || true)
  if [[ -n "$_rdma_times" ]]; then
    rdma_s=$(echo "$_rdma_times" | awk '{s+=$1; n++} END {if(n>0) printf "%.2f", s/n}')
    rdma_bw=$(echo "$_rdma_bws" | awk '{s+=$1; n++} END {if(n>0) printf "%.1f Gbps", s/n}')
  else
    # Fallback to old TIMING format (single worker)
    rdma_s=$(echo "$clean" | grep 'TIMING.*Time:' | grep -oP '[\d.]+(?=s)' | head -1)
    rdma_bw=$(echo "$clean" | grep 'TIMING.*Bandwidth' | grep -oP '[\d.]+ Gbps' | head -1)
  fi

  # NIXL registration
  local nixl_s
  nixl_s=$(echo "$clean" | grep 'Target tensors registered' | grep -oP '[\d.]+(?=s)' | head -1)

  # Weight processing (P2P mode)
  local weight_proc_s
  weight_proc_s=$(echo "$clean" | grep 'Weight processing complete' | grep -oP '[\d.]+(?=s)' | head -1)

  # torch.compile
  local compile_s
  compile_s=$(echo "$clean" | grep 'torch.compile takes' | grep -oP '[\d.]+(?= s)' | head -1)

  # DeepGEMM warmup: progress bars lack timestamps, so measure the gap
  # between torch.compile end and CUDA graph capture start.
  # DeepGEMM warmup runs between these two events.
  local deepgemm="0"
  if echo "$clean" | grep -q 'DeepGEMM warmup'; then
    local compile_end_ts cuda_start_ts
    compile_end_ts=$(echo "$clean" | grep 'torch.compile takes' | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' | head -1)
    cuda_start_ts=$(echo "$clean" | grep -m1 'Capturing CUDA graphs' | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' | head -1)
    if [[ -n "$compile_end_ts" && -n "$cuda_start_ts" ]]; then
      local ce cs_dg
      ce=$(date -d "$compile_end_ts" +%s 2>/dev/null) || ce=""
      cs_dg=$(date -d "$cuda_start_ts" +%s 2>/dev/null) || cs_dg=""
      if [[ -n "$ce" && -n "$cs_dg" && "$cs_dg" -gt "$ce" ]]; then
        deepgemm=$(( cs_dg - ce ))
      fi
    fi
  fi

  # CUDA graph capture
  local cuda_graph_s="0"
  local cg_start cg_end
  cg_start=$(echo "$clean" | grep -m1 'Capturing CUDA graphs' | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' | head -1)
  cg_end=$(echo "$clean" | grep 'warmup complete' | grep -oP '\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}' | head -1)
  if [[ -n "$cg_start" && -n "$cg_end" ]]; then
    local cs ce
    cs=$(date -d "$cg_start" +%s 2>/dev/null) || cs=""
    ce=$(date -d "$cg_end" +%s 2>/dev/null) || ce=""
    if [[ -n "$cs" && -n "$ce" ]]; then
      cuda_graph_s=$(( ce - cs ))
    fi
  fi

  # Emit values
  if [[ "$mode" == *"p2p"* ]]; then
    echo "WEIGHT_LOAD=${rdma_s:-N/A}s RDMA (${rdma_bw:-N/A})"
    echo "NIXL_REG=${nixl_s:-N/A}s"
    echo "WEIGHT_PROC=${weight_proc_s:-N/A}s"
  else
    echo "WEIGHT_LOAD=${disk_weight_s:-N/A}s disk"
    echo "NIXL_REG=N/A"
    echo "WEIGHT_PROC=N/A"
  fi
  if [[ -n "$compile_s" ]]; then
    echo "TORCH_COMPILE=${compile_s}s"
  else
    echo "TORCH_COMPILE=0 (cached)"
  fi
  if [[ "$deepgemm" == "0" ]]; then
    echo "DEEPGEMM_WARMUP=0 (cached)"
  else
    echo "DEEPGEMM_WARMUP=${deepgemm}"
  fi
  echo "CUDA_GRAPH=${cuda_graph_s}"
  if [[ "$total_s" == "N/A" ]]; then
    echo "TOTAL=N/A (log truncated)"
  else
    echo "TOTAL=${total_s}"
  fi
}

# Format seconds as Xm Ys. Passes through non-numeric values unchanged.
fmt_time() {
  local val="$1"
  # Pass through non-numeric values (e.g. "N/A (log truncated)", "0 (cached)")
  local num
  num=$(echo "$val" | grep -oP '^[\d.]+' || true)
  if [[ -z "$num" ]]; then echo "$val"; return; fi
  local int_s
  int_s=$(printf '%.0f' "$num" 2>/dev/null) || { echo "$val"; return; }
  local suffix
  suffix=$(echo "$val" | sed "s/^${num}//")
  if (( int_s >= 60 )); then
    echo "$((int_s / 60))m $((int_s % 60))s${suffix}"
  else
    echo "${int_s}s${suffix}"
  fi
}

# Print the startup timing table and save to file.
# Uses LOCAL_ARTIFACTS, P2P_SUBDIR (default: "p2p"), DISK_SUBDIR (default: "disk").
print_startup_table() {
  local p2p_dir="${P2P_SUBDIR:-p2p}"
  local disk_dir="${DISK_SUBDIR:-disk}"

  local has_table=false
  for m in "$p2p_dir" "$disk_dir"; do
    local target_log="${LOCAL_ARTIFACTS}/${m}/logs/mx-target.log"
    [[ -s "$target_log" ]] && has_table=true
  done
  if ! $has_table; then
    return
  fi

  echo ""
  echo "===== mx-target Startup Timing ====="
  echo ""

  # Parse both logs
  local p2p_log="${LOCAL_ARTIFACTS}/${p2p_dir}/logs/mx-target.log"
  local disk_log="${LOCAL_ARTIFACTS}/${disk_dir}/logs/mx-target.log"
  local p2p_data="" disk_data=""
  [[ -s "$p2p_log" ]] && p2p_data=$(parse_target_timings "$p2p_log" "$p2p_dir")
  [[ -s "$disk_log" ]] && disk_data=$(parse_target_timings "$disk_log" "$disk_dir")

  # Helper to extract a value from parsed data
  get_val() { echo "$1" | grep "^$2=" | head -1 | sed "s/^$2=//"; }

  # Determine columns
  local has_p2p=false has_disk=false
  [[ -n "$p2p_data" && -z "$(echo "$p2p_data" | grep 'MISSING=1')" ]] && has_p2p=true
  [[ -n "$disk_data" && -z "$(echo "$disk_data" | grep 'MISSING=1')" ]] && has_disk=true

  local stages=("WEIGHT_LOAD:Weight loading" "NIXL_REG:NIXL registration" "WEIGHT_PROC:Weight processing" "TORCH_COMPILE:torch.compile" "DEEPGEMM_WARMUP:DeepGEMM warmup" "CUDA_GRAPH:CUDA graph capture" "TOTAL:Total startup")

  # Render table to stdout and file simultaneously
  _render_table() {
    if $has_p2p && $has_disk; then
      printf "%-30s  %-28s  %-28s\n" "Stage" "P2P" "Disk"
      printf "%-30s  %-28s  %-28s\n" "-----" "---" "----"
    elif $has_p2p; then
      printf "%-30s  %-28s\n" "Stage" "P2P"
      printf "%-30s  %-28s\n" "-----" "---"
    else
      printf "%-30s  %-28s\n" "Stage" "Disk"
      printf "%-30s  %-28s\n" "-----" "----"
    fi

    for entry in "${stages[@]}"; do
      local key="${entry%%:*}" label="${entry##*:}"
      local pval="" dval=""
      $has_p2p && pval=$(get_val "$p2p_data" "$key")
      $has_disk && dval=$(get_val "$disk_data" "$key")
      if [[ "$key" == "TOTAL" || "$key" == "CUDA_GRAPH" || "$key" == "DEEPGEMM_WARMUP" ]]; then
        [[ -n "$pval" ]] && pval=$(fmt_time "$pval")
        [[ -n "$dval" ]] && dval=$(fmt_time "$dval")
      fi
      if $has_p2p && $has_disk; then
        printf "%-30s  %-28s  %-28s\n" "$label" "${pval:---}" "${dval:---}"
      elif $has_p2p; then
        printf "%-30s  %-28s\n" "$label" "${pval:---}"
      else
        printf "%-30s  %-28s\n" "$label" "${dval:---}"
      fi
    done
  }

  _render_table

  # Save to file
  local table_file="${LOCAL_ARTIFACTS}/startup_timing.txt"
  {
    echo "mx-target Startup Timing"
    echo "========================"
    echo ""
    _render_table
  } > "$table_file"
  echo ""
  echo "Saved to: $table_file"
}
