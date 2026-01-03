-- Dynamo Body Injector Lua Filter
--
-- This filter reads routing headers set by the EPP Dynamo plugins
-- and injects the nvext field into the request body for model servers.
--
-- Headers consumed:
--   x-worker-instance-id     - Target worker ID (decode worker in disagg mode)
--   x-prefiller-host-port    - Prefill worker ID (disaggregated mode only)
--   x-dynamo-token-data      - JSON-encoded token IDs for KV cache routing
--   x-dynamo-routing-mode    - "aggregated" or "disaggregated"
--
-- Body modification:
--   Adds "nvext" field with backend_instance_id, prefill_worker_id,
--   decode_worker_id, and token_data as appropriate.

function envoy_on_request(request_handle)
  -- Read headers set by EPP/Dynamo scorer
  local worker_id = request_handle:headers():get("x-worker-instance-id")

  -- If no worker ID, nothing to inject - pass through unchanged
  if worker_id == nil or worker_id == "" then
    return
  end

  -- Get other routing headers
  local prefill_worker_id = request_handle:headers():get("x-prefiller-host-port")
  local token_data = request_handle:headers():get("x-dynamo-token-data")
  local routing_mode = request_handle:headers():get("x-dynamo-routing-mode")

  -- Get request body (must buffer entire body)
  local body = request_handle:body()
  if body == nil then
    request_handle:logWarn("[DBI] No request body found")
    return
  end

  local body_str = body:getBytes(0, body:length())
  if body_str == nil or body_str == "" then
    request_handle:logDebug("[DBI] Empty request body")
    return
  end

  -- Build nvext JSON based on routing mode
  local nvext_content = ""

  if routing_mode == "disaggregated" then
    -- Disaggregated serving: separate prefill and decode workers
    local parts = {}

    if prefill_worker_id and prefill_worker_id ~= "" then
      table.insert(parts, string.format('"prefill_worker_id":%s', prefill_worker_id))
    end

    if worker_id and worker_id ~= "" then
      table.insert(parts, string.format('"decode_worker_id":%s', worker_id))
    end

    nvext_content = table.concat(parts, ",")
  else
    -- Aggregated serving: single backend instance
    nvext_content = string.format('"backend_instance_id":%s', worker_id)
  end

  -- Add token_data if present (already JSON-encoded array from header)
  if token_data and token_data ~= "" and token_data ~= "[]" then
    if nvext_content ~= "" then
      nvext_content = nvext_content .. ","
    end
    nvext_content = nvext_content .. string.format('"token_data":%s', token_data)
  end

  -- If nothing to inject, skip
  if nvext_content == "" then
    request_handle:logDebug("[DBI] No nvext content to inject")
    return
  end

  -- Build the nvext field
  local nvext_json = '"nvext":{' .. nvext_content .. '}'

  -- Find position to insert (before closing brace)
  -- Handle both compact JSON and formatted JSON
  local insert_pos = nil
  for i = #body_str, 1, -1 do
    local char = body_str:sub(i, i)
    if char == "}" then
      insert_pos = i
      break
    end
  end

  if insert_pos == nil then
    request_handle:logWarn("[DBI] Could not find closing brace in JSON body")
    return
  end

  -- Check if we need a comma (body has content before closing brace)
  local prefix = body_str:sub(1, insert_pos - 1)
  local needs_comma = false
  for i = #prefix, 1, -1 do
    local char = prefix:sub(i, i)
    if char:match("%S") then  -- non-whitespace
      if char ~= "{" and char ~= "," then
        needs_comma = true
      end
      break
    end
  end

  -- Build new body
  local new_body
  if needs_comma then
    new_body = prefix .. "," .. nvext_json .. "}"
  else
    new_body = prefix .. nvext_json .. "}"
  end

  -- Replace body
  request_handle:body():setBytes(new_body)

  -- Update Content-Length header
  request_handle:headers():replace("content-length", tostring(#new_body))

  -- Log success
  request_handle:logInfo(string.format(
    "[DBI] Injected nvext for worker %s (mode: %s, body size: %d -> %d)",
    worker_id,
    routing_mode or "aggregated",
    #body_str,
    #new_body
  ))
end

-- Optional: Handle response (currently not needed)
function envoy_on_response(response_handle)
  -- No response processing needed
end

