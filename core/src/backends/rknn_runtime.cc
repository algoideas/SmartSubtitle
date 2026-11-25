// RKNN runtime dynamic loader.
//
// The HAP's libentry.so must NOT link librknnrt.so directly: a Worker thread
// dlopen("libentry.so") with that DT_NEEDED crashes inside musl's symbol
// resolution on some devices (see faultlogger "WorkerThread __find_sym").
// Instead the RKNN runtime is loaded at runtime on first use — the same
// pattern the standalone CLI uses on the boards.
//
// These definitions replace the shared library's symbols for the whole
// module; npu_backend.cc keeps calling the plain rknn_* API.
#include <dlfcn.h>

#include <mutex>

#include "rknn_api.h"

namespace {

void* RknnRtHandle()
{
    static void* handle = nullptr;
    static std::once_flag once;
    std::call_once(once, []() {
        // dlopen 失败时保持 nullptr, 调用方返回 RKNN_ERR 并在上层报错
        handle = dlopen("librknnrt.so", RTLD_NOW | RTLD_LOCAL);
    });
    return handle;
}

void* Resolve(const char* name)
{
    void* h = RknnRtHandle();
    return h != nullptr ? dlsym(h, name) : nullptr;
}

}  // namespace

extern "C" {

int rknn_init(rknn_context* context, void* model, uint32_t size, uint32_t flag,
              rknn_init_extend* extend)
{
    using Fn = int (*)(rknn_context*, void*, uint32_t, uint32_t, rknn_init_extend*);
    static Fn fn = reinterpret_cast<Fn>(Resolve("rknn_init"));
    return fn != nullptr ? fn(context, model, size, flag, extend) : -1;
}

int rknn_query(rknn_context context, rknn_query_cmd cmd, void* info, uint32_t size)
{
    using Fn = int (*)(rknn_context, rknn_query_cmd, void*, uint32_t);
    static Fn fn = reinterpret_cast<Fn>(Resolve("rknn_query"));
    return fn != nullptr ? fn(context, cmd, info, size) : -1;
}

int rknn_run(rknn_context context, rknn_run_extend* extend)
{
    using Fn = int (*)(rknn_context, rknn_run_extend*);
    static Fn fn = reinterpret_cast<Fn>(Resolve("rknn_run"));
    return fn != nullptr ? fn(context, extend) : -1;
}

int rknn_outputs_release(rknn_context context, uint32_t n_outputs, rknn_output outputs[])
{
    using Fn = int (*)(rknn_context, uint32_t, rknn_output*);
    static Fn fn = reinterpret_cast<Fn>(Resolve("rknn_outputs_release"));
    return fn != nullptr ? fn(context, n_outputs, outputs) : -1;
}

int rknn_inputs_set(rknn_context context, uint32_t n_inputs, rknn_input inputs[])
{
    using Fn = int (*)(rknn_context, uint32_t, rknn_input*);
    static Fn fn = reinterpret_cast<Fn>(Resolve("rknn_inputs_set"));
    return fn != nullptr ? fn(context, n_inputs, inputs) : -1;
}

int rknn_outputs_get(rknn_context context, uint32_t n_outputs, rknn_output outputs[],
                     rknn_output_extend* extend)
{
    using Fn = int (*)(rknn_context, uint32_t, rknn_output*, rknn_output_extend*);
    static Fn fn = reinterpret_cast<Fn>(Resolve("rknn_outputs_get"));
    return fn != nullptr ? fn(context, n_outputs, outputs, extend) : -1;
}

int rknn_set_core_mask(rknn_context context, rknn_core_mask core_mask)
{
    using Fn = int (*)(rknn_context, rknn_core_mask);
    static Fn fn = reinterpret_cast<Fn>(Resolve("rknn_set_core_mask"));
    return fn != nullptr ? fn(context, core_mask) : -1;
}

int rknn_destroy(rknn_context context)
{
    using Fn = int (*)(rknn_context);
    static Fn fn = reinterpret_cast<Fn>(Resolve("rknn_destroy"));
    return fn != nullptr ? fn(context) : -1;
}

}  // extern "C"
