/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Lazy-loading stubs for the NIXL C API.
//
// When nixl is not available at build time, these stubs are compiled in.
// At runtime, they attempt to dlopen("libnixl_capi.so") and forward all
// calls to the real implementation. If the real library is not found,
// they abort with a clear error message (same behavior as the old stubs).
//
// This allows building without nixl present while still using nixl at
// runtime when the shared library is installed.

#include "wrapper.h"

#include <cstdlib>
#include <dlfcn.h>
#include <iostream>

namespace {

// Result of the one-shot dlopen attempt, capturing both the handle and any
// error string so the diagnostic is not lost between get_nixl_handle() and
// resolve().
struct NixlHandle {
    void *handle;
    const char *error; // captured from dlerror() on failure
};

// Thread-safe lazy initialization of the nixl C API shared library handle.
// C++11 guarantees thread-safe initialization of function-local static variables.
const NixlHandle &
get_nixl_handle() {
    static NixlHandle h = []() -> NixlHandle {
        void *hdl = dlopen("libnixl_capi.so", RTLD_NOW | RTLD_LOCAL);
        const char *err = hdl ? nullptr : dlerror();
        return {hdl, err};
    }();
    return h;
}

// Resolve a symbol from the nixl C API shared library.
// Aborts if the library is not loaded or the symbol is not found.
void *
resolve(const char *name) {
    const auto &h = get_nixl_handle();
    if (!h.handle) {
        std::cerr << "nixl error: libnixl_capi.so not found. "
                  << "Install nixl or ensure the nixl library directory "
                  << "is in LD_LIBRARY_PATH.";
        if (h.error) {
            std::cerr << " dlopen error: " << h.error;
        }
        std::cerr << "\n";
        std::abort();
    }
    dlerror(); // clear any stale error
    void *sym = dlsym(h.handle, name);
    const char *err = dlerror();
    if (err) {
        std::cerr << "nixl error: symbol '" << name << "' not found in libnixl_capi.so: " << err
                  << "\n";
        std::abort();
    }
    return sym;
}

} // anonymous namespace

extern "C" {

// clang-format off
// Opaque struct definitions (never dereferenced by stubs; needed for type completeness)
struct nixl_capi_agent_s { /* empty */ };
struct nixl_capi_string_list_s { /* empty */ };
struct nixl_capi_params_s { /* empty */ };
struct nixl_capi_mem_list_s { /* empty */ };
struct nixl_capi_backend_s { /* empty */ };
struct nixl_capi_opt_args_s { /* empty */ };
struct nixl_capi_param_iter_s { /* empty */ };
struct nixl_capi_xfer_dlist_s { /* empty */ };
struct nixl_capi_reg_dlist_s { /* empty */ };
struct nixl_capi_xfer_req_s { /* empty */ };
struct nixl_capi_notif_map_s { /* empty */ };
struct nixl_capi_xfer_dlist_handle_s { /* empty */ };
struct nixl_capi_query_resp_list_s { /* empty */ };
// clang-format on

// ---- Core agent functions ----

nixl_capi_status_t
nixl_capi_create_agent(const char *name, nixl_capi_agent_t *agent) {
    using fn_t = nixl_capi_status_t (*)(const char *, nixl_capi_agent_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_create_agent");
    return real(name, agent);
}

nixl_capi_status_t
nixl_capi_create_configured_agent(const char *name,
                                  const nixl_capi_agent_config_t *cfg,
                                  nixl_capi_agent_t *agent) {
    using fn_t =
        nixl_capi_status_t (*)(const char *, const nixl_capi_agent_config_t *, nixl_capi_agent_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_create_configured_agent");
    return real(name, cfg, agent);
}

nixl_capi_status_t
nixl_capi_destroy_agent(nixl_capi_agent_t agent) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t);
    static fn_t real = (fn_t)resolve("nixl_capi_destroy_agent");
    return real(agent);
}

// ---- Metadata functions ----

nixl_capi_status_t
nixl_capi_get_local_md(nixl_capi_agent_t agent, void **data, size_t *len) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t, void **, size_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_get_local_md");
    return real(agent, data, len);
}

nixl_capi_status_t
nixl_capi_get_local_partial_md(nixl_capi_agent_t agent,
                               nixl_capi_reg_dlist_t descs,
                               void **data,
                               size_t *len,
                               nixl_capi_opt_args_t opt_args) {
    using fn_t = nixl_capi_status_t (*)(
        nixl_capi_agent_t, nixl_capi_reg_dlist_t, void **, size_t *, nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_get_local_partial_md");
    return real(agent, descs, data, len, opt_args);
}

nixl_capi_status_t
nixl_capi_load_remote_md(nixl_capi_agent_t agent, const void *data, size_t len, char **agent_name) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t, const void *, size_t, char **);
    static fn_t real = (fn_t)resolve("nixl_capi_load_remote_md");
    return real(agent, data, len, agent_name);
}

nixl_capi_status_t
nixl_capi_send_local_md(nixl_capi_agent_t agent, nixl_capi_opt_args_t opt_args) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_send_local_md");
    return real(agent, opt_args);
}

nixl_capi_status_t
nixl_capi_send_local_partial_md(nixl_capi_agent_t agent,
                                nixl_capi_reg_dlist_t descs,
                                nixl_capi_opt_args_t opt_args) {
    using fn_t =
        nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_reg_dlist_t, nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_send_local_partial_md");
    return real(agent, descs, opt_args);
}

nixl_capi_status_t
nixl_capi_invalidate_remote_md(nixl_capi_agent_t agent, const char *remote_agent) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t, const char *);
    static fn_t real = (fn_t)resolve("nixl_capi_invalidate_remote_md");
    return real(agent, remote_agent);
}

nixl_capi_status_t
nixl_capi_invalidate_local_md(nixl_capi_agent_t agent, nixl_capi_opt_args_t opt_args) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_invalidate_local_md");
    return real(agent, opt_args);
}

nixl_capi_status_t
nixl_capi_check_remote_md(nixl_capi_agent_t agent,
                          const char *remote_name,
                          nixl_capi_xfer_dlist_t descs) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t, const char *, nixl_capi_xfer_dlist_t);
    static fn_t real = (fn_t)resolve("nixl_capi_check_remote_md");
    return real(agent, remote_name, descs);
}

nixl_capi_status_t
nixl_capi_fetch_remote_md(nixl_capi_agent_t agent,
                          const char *remote_name,
                          nixl_capi_opt_args_t opt_args) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t, const char *, nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_fetch_remote_md");
    return real(agent, remote_name, opt_args);
}

// ---- Transfer descriptor list prep/handle functions ----

nixl_capi_status_t
nixl_capi_prep_xfer_dlist(nixl_capi_agent_t agent,
                          const char *agent_name,
                          nixl_capi_xfer_dlist_t descs,
                          nixl_capi_xfer_dlist_handle_t *dlist_handle,
                          nixl_capi_opt_args_t opt_args) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t,
                                        const char *,
                                        nixl_capi_xfer_dlist_t,
                                        nixl_capi_xfer_dlist_handle_t *,
                                        nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_prep_xfer_dlist");
    return real(agent, agent_name, descs, dlist_handle, opt_args);
}

nixl_capi_status_t
nixl_capi_release_xfer_dlist_handle(nixl_capi_agent_t agent,
                                    nixl_capi_xfer_dlist_handle_t dlist_handle) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_xfer_dlist_handle_t);
    static fn_t real = (fn_t)resolve("nixl_capi_release_xfer_dlist_handle");
    return real(agent, dlist_handle);
}

nixl_capi_status_t
nixl_capi_make_xfer_req(nixl_capi_agent_t agent,
                        nixl_capi_xfer_op_t operation,
                        nixl_capi_xfer_dlist_handle_t local_descs,
                        const int *local_indices,
                        size_t local_indices_count,
                        nixl_capi_xfer_dlist_handle_t remote_descs,
                        const int *remote_indices,
                        size_t remote_indices_count,
                        nixl_capi_xfer_req_t *req_hndl,
                        nixl_capi_opt_args_t opt_args) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t,
                                        nixl_capi_xfer_op_t,
                                        nixl_capi_xfer_dlist_handle_t,
                                        const int *,
                                        size_t,
                                        nixl_capi_xfer_dlist_handle_t,
                                        const int *,
                                        size_t,
                                        nixl_capi_xfer_req_t *,
                                        nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_make_xfer_req");
    return real(agent,
                operation,
                local_descs,
                local_indices,
                local_indices_count,
                remote_descs,
                remote_indices,
                remote_indices_count,
                req_hndl,
                opt_args);
}

// ---- Connection functions ----

nixl_capi_status_t
nixl_capi_agent_make_connection(nixl_capi_agent_t agent,
                                const char *remote_agent,
                                nixl_capi_opt_args_t opt_args) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t, const char *, nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_agent_make_connection");
    return real(agent, remote_agent, opt_args);
}

// ---- Plugin and parameter functions ----

nixl_capi_status_t
nixl_capi_get_available_plugins(nixl_capi_agent_t agent, nixl_capi_string_list_t *plugins) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_string_list_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_get_available_plugins");
    return real(agent, plugins);
}

nixl_capi_status_t
nixl_capi_destroy_string_list(nixl_capi_string_list_t list) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_string_list_t);
    static fn_t real = (fn_t)resolve("nixl_capi_destroy_string_list");
    return real(list);
}

nixl_capi_status_t
nixl_capi_string_list_size(nixl_capi_string_list_t list, size_t *size) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_string_list_t, size_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_string_list_size");
    return real(list, size);
}

nixl_capi_status_t
nixl_capi_string_list_get(nixl_capi_string_list_t list, size_t index, const char **str) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_string_list_t, size_t, const char **);
    static fn_t real = (fn_t)resolve("nixl_capi_string_list_get");
    return real(list, index, str);
}

nixl_capi_status_t
nixl_capi_get_plugin_params(nixl_capi_agent_t agent,
                            const char *plugin_name,
                            nixl_capi_mem_list_t *mems,
                            nixl_capi_params_t *params) {
    using fn_t = nixl_capi_status_t (*)(
        nixl_capi_agent_t, const char *, nixl_capi_mem_list_t *, nixl_capi_params_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_get_plugin_params");
    return real(agent, plugin_name, mems, params);
}

nixl_capi_status_t
nixl_capi_destroy_mem_list(nixl_capi_mem_list_t list) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_mem_list_t);
    static fn_t real = (fn_t)resolve("nixl_capi_destroy_mem_list");
    return real(list);
}

nixl_capi_status_t
nixl_capi_destroy_params(nixl_capi_params_t params) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_params_t);
    static fn_t real = (fn_t)resolve("nixl_capi_destroy_params");
    return real(params);
}

// ---- Backend functions ----

nixl_capi_status_t
nixl_capi_create_backend(nixl_capi_agent_t agent,
                         const char *plugin_name,
                         nixl_capi_params_t params,
                         nixl_capi_backend_t *backend) {
    using fn_t = nixl_capi_status_t (*)(
        nixl_capi_agent_t, const char *, nixl_capi_params_t, nixl_capi_backend_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_create_backend");
    return real(agent, plugin_name, params, backend);
}

nixl_capi_status_t
nixl_capi_destroy_backend(nixl_capi_backend_t backend) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_backend_t);
    static fn_t real = (fn_t)resolve("nixl_capi_destroy_backend");
    return real(backend);
}

nixl_capi_status_t
nixl_capi_get_backend_params(nixl_capi_agent_t agent,
                             nixl_capi_backend_t backend,
                             nixl_capi_mem_list_t *mems,
                             nixl_capi_params_t *params) {
    using fn_t = nixl_capi_status_t (*)(
        nixl_capi_agent_t, nixl_capi_backend_t, nixl_capi_mem_list_t *, nixl_capi_params_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_get_backend_params");
    return real(agent, backend, mems, params);
}

// ---- Optional arguments functions ----

nixl_capi_status_t
nixl_capi_create_opt_args(nixl_capi_opt_args_t *args) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_opt_args_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_create_opt_args");
    return real(args);
}

nixl_capi_status_t
nixl_capi_destroy_opt_args(nixl_capi_opt_args_t args) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_destroy_opt_args");
    return real(args);
}

nixl_capi_status_t
nixl_capi_opt_args_add_backend(nixl_capi_opt_args_t args, nixl_capi_backend_t backend) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_opt_args_t, nixl_capi_backend_t);
    static fn_t real = (fn_t)resolve("nixl_capi_opt_args_add_backend");
    return real(args, backend);
}

nixl_capi_status_t
nixl_capi_opt_args_set_notif_msg(nixl_capi_opt_args_t args, const void *data, size_t len) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_opt_args_t, const void *, size_t);
    static fn_t real = (fn_t)resolve("nixl_capi_opt_args_set_notif_msg");
    return real(args, data, len);
}

nixl_capi_status_t
nixl_capi_opt_args_get_notif_msg(nixl_capi_opt_args_t args, void **data, size_t *len) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_opt_args_t, void **, size_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_opt_args_get_notif_msg");
    return real(args, data, len);
}

nixl_capi_status_t
nixl_capi_opt_args_set_has_notif(nixl_capi_opt_args_t args, bool has_notif) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_opt_args_t, bool);
    static fn_t real = (fn_t)resolve("nixl_capi_opt_args_set_has_notif");
    return real(args, has_notif);
}

nixl_capi_status_t
nixl_capi_opt_args_get_has_notif(nixl_capi_opt_args_t args, bool *has_notif) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_opt_args_t, bool *);
    static fn_t real = (fn_t)resolve("nixl_capi_opt_args_get_has_notif");
    return real(args, has_notif);
}

nixl_capi_status_t
nixl_capi_opt_args_set_skip_desc_merge(nixl_capi_opt_args_t args, bool skip_merge) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_opt_args_t, bool);
    static fn_t real = (fn_t)resolve("nixl_capi_opt_args_set_skip_desc_merge");
    return real(args, skip_merge);
}

nixl_capi_status_t
nixl_capi_opt_args_get_skip_desc_merge(nixl_capi_opt_args_t args, bool *skip_merge) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_opt_args_t, bool *);
    static fn_t real = (fn_t)resolve("nixl_capi_opt_args_get_skip_desc_merge");
    return real(args, skip_merge);
}

nixl_capi_status_t
nixl_capi_opt_args_set_ip_addr(nixl_capi_opt_args_t args, const char *ip_addr) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_opt_args_t, const char *);
    static fn_t real = (fn_t)resolve("nixl_capi_opt_args_set_ip_addr");
    return real(args, ip_addr);
}

nixl_capi_status_t
nixl_capi_opt_args_set_port(nixl_capi_opt_args_t args, uint16_t port) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_opt_args_t, uint16_t);
    static fn_t real = (fn_t)resolve("nixl_capi_opt_args_set_port");
    return real(args, port);
}

// ---- Parameter functions ----

nixl_capi_status_t
nixl_capi_create_params(nixl_capi_params_t *params) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_params_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_create_params");
    return real(params);
}

nixl_capi_status_t
nixl_capi_params_add(nixl_capi_params_t params, const char *key, const char *value) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_params_t, const char *, const char *);
    static fn_t real = (fn_t)resolve("nixl_capi_params_add");
    return real(params, key, value);
}

nixl_capi_status_t
nixl_capi_params_is_empty(nixl_capi_params_t params, bool *is_empty) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_params_t, bool *);
    static fn_t real = (fn_t)resolve("nixl_capi_params_is_empty");
    return real(params, is_empty);
}

nixl_capi_status_t
nixl_capi_params_create_iterator(nixl_capi_params_t params, nixl_capi_param_iter_t *iter) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_params_t, nixl_capi_param_iter_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_params_create_iterator");
    return real(params, iter);
}

nixl_capi_status_t
nixl_capi_params_iterator_next(nixl_capi_param_iter_t iter,
                               const char **key,
                               const char **value,
                               bool *has_next) {
    using fn_t =
        nixl_capi_status_t (*)(nixl_capi_param_iter_t, const char **, const char **, bool *);
    static fn_t real = (fn_t)resolve("nixl_capi_params_iterator_next");
    return real(iter, key, value, has_next);
}

nixl_capi_status_t
nixl_capi_params_destroy_iterator(nixl_capi_param_iter_t iter) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_param_iter_t);
    static fn_t real = (fn_t)resolve("nixl_capi_params_destroy_iterator");
    return real(iter);
}

// ---- Memory list functions ----

nixl_capi_status_t
nixl_capi_mem_list_is_empty(nixl_capi_mem_list_t list, bool *is_empty) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_mem_list_t, bool *);
    static fn_t real = (fn_t)resolve("nixl_capi_mem_list_is_empty");
    return real(list, is_empty);
}

nixl_capi_status_t
nixl_capi_mem_list_size(nixl_capi_mem_list_t list, size_t *size) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_mem_list_t, size_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_mem_list_size");
    return real(list, size);
}

nixl_capi_status_t
nixl_capi_mem_list_get(nixl_capi_mem_list_t list, size_t index, nixl_capi_mem_type_t *mem_type) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_mem_list_t, size_t, nixl_capi_mem_type_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_mem_list_get");
    return real(list, index, mem_type);
}

nixl_capi_status_t
nixl_capi_mem_type_to_string(nixl_capi_mem_type_t mem_type, const char **str) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_mem_type_t, const char **);
    static fn_t real = (fn_t)resolve("nixl_capi_mem_type_to_string");
    return real(mem_type, str);
}

// ---- Transfer descriptor list functions ----

nixl_capi_status_t
nixl_capi_create_xfer_dlist(nixl_capi_mem_type_t mem_type, nixl_capi_xfer_dlist_t *dlist) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_mem_type_t, nixl_capi_xfer_dlist_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_create_xfer_dlist");
    return real(mem_type, dlist);
}

nixl_capi_status_t
nixl_capi_destroy_xfer_dlist(nixl_capi_xfer_dlist_t dlist) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_xfer_dlist_t);
    static fn_t real = (fn_t)resolve("nixl_capi_destroy_xfer_dlist");
    return real(dlist);
}

nixl_capi_status_t
nixl_capi_xfer_dlist_add_desc(nixl_capi_xfer_dlist_t dlist,
                              uintptr_t addr,
                              size_t len,
                              uint64_t dev_id) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_xfer_dlist_t, uintptr_t, size_t, uint64_t);
    static fn_t real = (fn_t)resolve("nixl_capi_xfer_dlist_add_desc");
    return real(dlist, addr, len, dev_id);
}

nixl_capi_status_t
nixl_capi_xfer_dlist_len(nixl_capi_xfer_dlist_t dlist, size_t *len) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_xfer_dlist_t, size_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_xfer_dlist_len");
    return real(dlist, len);
}

nixl_capi_status_t
nixl_capi_xfer_dlist_clear(nixl_capi_xfer_dlist_t dlist) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_xfer_dlist_t);
    static fn_t real = (fn_t)resolve("nixl_capi_xfer_dlist_clear");
    return real(dlist);
}

nixl_capi_status_t
nixl_capi_xfer_dlist_resize(nixl_capi_xfer_dlist_t dlist, size_t new_size) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_xfer_dlist_t, size_t);
    static fn_t real = (fn_t)resolve("nixl_capi_xfer_dlist_resize");
    return real(dlist, new_size);
}

nixl_capi_status_t
nixl_capi_xfer_dlist_get_type(nixl_capi_xfer_dlist_t dlist, nixl_capi_mem_type_t *mem_type) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_xfer_dlist_t, nixl_capi_mem_type_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_xfer_dlist_get_type");
    return real(dlist, mem_type);
}

nixl_capi_status_t
nixl_capi_xfer_dlist_desc_count(nixl_capi_xfer_dlist_t dlist, size_t *count) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_xfer_dlist_t, size_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_xfer_dlist_desc_count");
    return real(dlist, count);
}

nixl_capi_status_t
nixl_capi_xfer_dlist_is_empty(nixl_capi_xfer_dlist_t dlist, bool *is_empty) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_xfer_dlist_t, bool *);
    static fn_t real = (fn_t)resolve("nixl_capi_xfer_dlist_is_empty");
    return real(dlist, is_empty);
}

nixl_capi_status_t
nixl_capi_xfer_dlist_trim(nixl_capi_xfer_dlist_t dlist) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_xfer_dlist_t);
    static fn_t real = (fn_t)resolve("nixl_capi_xfer_dlist_trim");
    return real(dlist);
}

nixl_capi_status_t
nixl_capi_xfer_dlist_rem_desc(nixl_capi_xfer_dlist_t dlist, int index) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_xfer_dlist_t, int);
    static fn_t real = (fn_t)resolve("nixl_capi_xfer_dlist_rem_desc");
    return real(dlist, index);
}

nixl_capi_status_t
nixl_capi_xfer_dlist_print(nixl_capi_xfer_dlist_t dlist) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_xfer_dlist_t);
    static fn_t real = (fn_t)resolve("nixl_capi_xfer_dlist_print");
    return real(dlist);
}

// ---- Registration descriptor list functions ----

nixl_capi_status_t
nixl_capi_create_reg_dlist(nixl_capi_mem_type_t mem_type, nixl_capi_reg_dlist_t *dlist) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_mem_type_t, nixl_capi_reg_dlist_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_create_reg_dlist");
    return real(mem_type, dlist);
}

nixl_capi_status_t
nixl_capi_destroy_reg_dlist(nixl_capi_reg_dlist_t dlist) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_reg_dlist_t);
    static fn_t real = (fn_t)resolve("nixl_capi_destroy_reg_dlist");
    return real(dlist);
}

nixl_capi_status_t
nixl_capi_reg_dlist_add_desc(nixl_capi_reg_dlist_t dlist,
                             uintptr_t addr,
                             size_t len,
                             uint64_t dev_id,
                             const void *metadata,
                             size_t metadata_len) {
    using fn_t = nixl_capi_status_t (*)(
        nixl_capi_reg_dlist_t, uintptr_t, size_t, uint64_t, const void *, size_t);
    static fn_t real = (fn_t)resolve("nixl_capi_reg_dlist_add_desc");
    return real(dlist, addr, len, dev_id, metadata, metadata_len);
}

nixl_capi_status_t
nixl_capi_reg_dlist_clear(nixl_capi_reg_dlist_t dlist) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_reg_dlist_t);
    static fn_t real = (fn_t)resolve("nixl_capi_reg_dlist_clear");
    return real(dlist);
}

nixl_capi_status_t
nixl_capi_reg_dlist_resize(nixl_capi_reg_dlist_t dlist, size_t new_size) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_reg_dlist_t, size_t);
    static fn_t real = (fn_t)resolve("nixl_capi_reg_dlist_resize");
    return real(dlist, new_size);
}

nixl_capi_status_t
nixl_capi_reg_dlist_get_type(nixl_capi_reg_dlist_t dlist, nixl_capi_mem_type_t *mem_type) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_reg_dlist_t, nixl_capi_mem_type_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_reg_dlist_get_type");
    return real(dlist, mem_type);
}

nixl_capi_status_t
nixl_capi_reg_dlist_desc_count(nixl_capi_reg_dlist_t dlist, size_t *count) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_reg_dlist_t, size_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_reg_dlist_desc_count");
    return real(dlist, count);
}

nixl_capi_status_t
nixl_capi_reg_dlist_is_empty(nixl_capi_reg_dlist_t dlist, bool *is_empty) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_reg_dlist_t, bool *);
    static fn_t real = (fn_t)resolve("nixl_capi_reg_dlist_is_empty");
    return real(dlist, is_empty);
}

nixl_capi_status_t
nixl_capi_reg_dlist_trim(nixl_capi_reg_dlist_t dlist) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_reg_dlist_t);
    static fn_t real = (fn_t)resolve("nixl_capi_reg_dlist_trim");
    return real(dlist);
}

nixl_capi_status_t
nixl_capi_reg_dlist_rem_desc(nixl_capi_reg_dlist_t dlist, int index) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_reg_dlist_t, int);
    static fn_t real = (fn_t)resolve("nixl_capi_reg_dlist_rem_desc");
    return real(dlist, index);
}

nixl_capi_status_t
nixl_capi_reg_dlist_print(nixl_capi_reg_dlist_t dlist) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_reg_dlist_t);
    static fn_t real = (fn_t)resolve("nixl_capi_reg_dlist_print");
    return real(dlist);
}

// ---- Memory registration functions ----

nixl_capi_status_t
nixl_capi_register_mem(nixl_capi_agent_t agent,
                       nixl_capi_reg_dlist_t dlist,
                       nixl_capi_opt_args_t opt_args) {
    using fn_t =
        nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_reg_dlist_t, nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_register_mem");
    return real(agent, dlist, opt_args);
}

nixl_capi_status_t
nixl_capi_deregister_mem(nixl_capi_agent_t agent,
                         nixl_capi_reg_dlist_t dlist,
                         nixl_capi_opt_args_t opt_args) {
    using fn_t =
        nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_reg_dlist_t, nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_deregister_mem");
    return real(agent, dlist, opt_args);
}

// ---- Transfer request functions ----

nixl_capi_status_t
nixl_capi_create_xfer_req(nixl_capi_agent_t agent,
                          nixl_capi_xfer_op_t operation,
                          nixl_capi_xfer_dlist_t local_descs,
                          nixl_capi_xfer_dlist_t remote_descs,
                          const char *remote_agent,
                          nixl_capi_xfer_req_t *req_hndl,
                          nixl_capi_opt_args_t opt_args) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t,
                                        nixl_capi_xfer_op_t,
                                        nixl_capi_xfer_dlist_t,
                                        nixl_capi_xfer_dlist_t,
                                        const char *,
                                        nixl_capi_xfer_req_t *,
                                        nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_create_xfer_req");
    return real(agent, operation, local_descs, remote_descs, remote_agent, req_hndl, opt_args);
}

nixl_capi_status_t
nixl_capi_post_xfer_req(nixl_capi_agent_t agent,
                        nixl_capi_xfer_req_t req_hndl,
                        nixl_capi_opt_args_t opt_args) {
    using fn_t =
        nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_xfer_req_t, nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_post_xfer_req");
    return real(agent, req_hndl, opt_args);
}

nixl_capi_status_t
nixl_capi_get_xfer_status(nixl_capi_agent_t agent, nixl_capi_xfer_req_t req_hndl) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_xfer_req_t);
    static fn_t real = (fn_t)resolve("nixl_capi_get_xfer_status");
    return real(agent, req_hndl);
}

nixl_capi_status_t
nixl_capi_query_xfer_backend(nixl_capi_agent_t agent,
                             nixl_capi_xfer_req_t req_hndl,
                             nixl_capi_backend_t *backend) {
    using fn_t =
        nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_xfer_req_t, nixl_capi_backend_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_query_xfer_backend");
    return real(agent, req_hndl, backend);
}

nixl_capi_status_t
nixl_capi_estimate_xfer_cost(nixl_capi_agent_t agent,
                             nixl_capi_xfer_req_t req_hndl,
                             nixl_capi_opt_args_t opt_args,
                             int64_t *duration_us,
                             int64_t *err_margin_us,
                             nixl_capi_cost_t *method) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t,
                                        nixl_capi_xfer_req_t,
                                        nixl_capi_opt_args_t,
                                        int64_t *,
                                        int64_t *,
                                        nixl_capi_cost_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_estimate_xfer_cost");
    return real(agent, req_hndl, opt_args, duration_us, err_margin_us, method);
}

nixl_capi_status_t
nixl_capi_destroy_xfer_req(nixl_capi_xfer_req_t req) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_xfer_req_t);
    static fn_t real = (fn_t)resolve("nixl_capi_destroy_xfer_req");
    return real(req);
}

nixl_capi_status_t
nixl_capi_release_xfer_req(nixl_capi_agent_t agent, nixl_capi_xfer_req_t req) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_xfer_req_t);
    static fn_t real = (fn_t)resolve("nixl_capi_release_xfer_req");
    return real(agent, req);
}

// ---- Notification functions ----

nixl_capi_status_t
nixl_capi_get_notifs(nixl_capi_agent_t agent,
                     nixl_capi_notif_map_t notif_map,
                     nixl_capi_opt_args_t opt_args) {
    using fn_t =
        nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_notif_map_t, nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_get_notifs");
    return real(agent, notif_map, opt_args);
}

nixl_capi_status_t
nixl_capi_gen_notif(nixl_capi_agent_t agent,
                    const char *remote_agent,
                    const void *data,
                    size_t len,
                    nixl_capi_opt_args_t opt_args) {
    using fn_t = nixl_capi_status_t (*)(
        nixl_capi_agent_t, const char *, const void *, size_t, nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_gen_notif");
    return real(agent, remote_agent, data, len, opt_args);
}

nixl_capi_status_t
nixl_capi_create_notif_map(nixl_capi_notif_map_t *notif_map) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_notif_map_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_create_notif_map");
    return real(notif_map);
}

nixl_capi_status_t
nixl_capi_destroy_notif_map(nixl_capi_notif_map_t notif_map) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_notif_map_t);
    static fn_t real = (fn_t)resolve("nixl_capi_destroy_notif_map");
    return real(notif_map);
}

nixl_capi_status_t
nixl_capi_notif_map_size(nixl_capi_notif_map_t map, size_t *size) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_notif_map_t, size_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_notif_map_size");
    return real(map, size);
}

nixl_capi_status_t
nixl_capi_notif_map_get_agent_at(nixl_capi_notif_map_t map, size_t index, const char **agent_name) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_notif_map_t, size_t, const char **);
    static fn_t real = (fn_t)resolve("nixl_capi_notif_map_get_agent_at");
    return real(map, index, agent_name);
}

nixl_capi_status_t
nixl_capi_notif_map_get_notifs_size(nixl_capi_notif_map_t map,
                                    const char *agent_name,
                                    size_t *size) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_notif_map_t, const char *, size_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_notif_map_get_notifs_size");
    return real(map, agent_name, size);
}

nixl_capi_status_t
nixl_capi_notif_map_get_notif(nixl_capi_notif_map_t map,
                              const char *agent_name,
                              size_t index,
                              const void **data,
                              size_t *len) {
    using fn_t = nixl_capi_status_t (*)(
        nixl_capi_notif_map_t, const char *, size_t, const void **, size_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_notif_map_get_notif");
    return real(map, agent_name, index, data, len);
}

nixl_capi_status_t
nixl_capi_notif_map_clear(nixl_capi_notif_map_t map) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_notif_map_t);
    static fn_t real = (fn_t)resolve("nixl_capi_notif_map_clear");
    return real(map);
}

// ---- Query response list functions ----

nixl_capi_status_t
nixl_capi_create_query_resp_list(nixl_capi_query_resp_list_t *list) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_query_resp_list_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_create_query_resp_list");
    return real(list);
}

nixl_capi_status_t
nixl_capi_destroy_query_resp_list(nixl_capi_query_resp_list_t list) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_query_resp_list_t);
    static fn_t real = (fn_t)resolve("nixl_capi_destroy_query_resp_list");
    return real(list);
}

nixl_capi_status_t
nixl_capi_query_resp_list_size(nixl_capi_query_resp_list_t list, size_t *size) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_query_resp_list_t, size_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_query_resp_list_size");
    return real(list, size);
}

nixl_capi_status_t
nixl_capi_query_resp_list_has_value(nixl_capi_query_resp_list_t list,
                                    size_t index,
                                    bool *has_value) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_query_resp_list_t, size_t, bool *);
    static fn_t real = (fn_t)resolve("nixl_capi_query_resp_list_has_value");
    return real(list, index, has_value);
}

nixl_capi_status_t
nixl_capi_query_resp_list_get_params(nixl_capi_query_resp_list_t list,
                                     size_t index,
                                     nixl_capi_params_t *params) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_query_resp_list_t, size_t, nixl_capi_params_t *);
    static fn_t real = (fn_t)resolve("nixl_capi_query_resp_list_get_params");
    return real(list, index, params);
}

nixl_capi_status_t
nixl_capi_query_mem(nixl_capi_agent_t agent,
                    nixl_capi_reg_dlist_t descs,
                    nixl_capi_query_resp_list_t resp,
                    nixl_capi_opt_args_t opt_args) {
    using fn_t = nixl_capi_status_t (*)(nixl_capi_agent_t,
                                        nixl_capi_reg_dlist_t,
                                        nixl_capi_query_resp_list_t,
                                        nixl_capi_opt_args_t);
    static fn_t real = (fn_t)resolve("nixl_capi_query_mem");
    return real(agent, descs, resp, opt_args);
}

// ---- Telemetry functions ----

nixl_capi_status_t
nixl_capi_get_xfer_telemetry(nixl_capi_agent_t agent,
                             nixl_capi_xfer_req_t req_hndl,
                             nixl_capi_xfer_telemetry_t telemetry) {
    using fn_t =
        nixl_capi_status_t (*)(nixl_capi_agent_t, nixl_capi_xfer_req_t, nixl_capi_xfer_telemetry_t);
    static fn_t real = (fn_t)resolve("nixl_capi_get_xfer_telemetry");
    return real(agent, req_hndl, telemetry);
}

// ---- Stub detection ----
// Returns true if the real nixl library is NOT available at runtime.
// Unlike other functions, this does NOT abort when the library is missing.
bool
nixl_capi_is_stub() {
    return (get_nixl_handle().handle == nullptr);
}

} // extern "C"
