/*
 * SmartSubtitle - NAPI module registration.
 * Exposes the real-time subtitle engine to ArkTS under the module name
 * "entry" (libentry.so):
 *   startEngine(backend: string, modelDir: string): boolean
 *   stopEngine(): void
 *   feedAudio(pcm: ArrayBuffer): void          // 16 kHz mono int16 chunks
 *   setSubtitleListener(cb: (text: string, isFinal: boolean) => void): void
 *   getLastError(): string
 */
#include <napi/native_api.h>

#include <chrono>
#include <string>
#include <thread>

#include "asr_backend.h"

namespace {

using subtitle::AsrCallback;
using subtitle::AsrResult;
using subtitle::BackendConfig;
using subtitle::CreateBackend;

struct EngineState {
    std::unique_ptr<subtitle::AsrBackend> backend;
    napi_threadsafe_function tsfn = nullptr;
    std::string lastError = "ok";
};

EngineState g_engine;

// Runs on the JS thread: delivers a recognition result to the ArkTS listener.
void CallJsListener(napi_env env, napi_value jsCallback, void* /*context*/, void* data)
{
    auto* result = static_cast<AsrResult*>(data);
    if (jsCallback != nullptr) {
        napi_value text;
        napi_value isFinal;
        napi_create_string_utf8(env, result->text.c_str(), result->text.size(), &text);
        napi_get_boolean(env, result->isFinal, &isFinal);
        napi_value argv[2] = {text, isFinal};
        napi_value undefined;
        napi_get_undefined(env, &undefined);
        napi_call_function(env, undefined, jsCallback, 2, argv, nullptr);
    }
    delete result;
}

std::string NapiToString(napi_env env, napi_value value)
{
    size_t len = 0;
    napi_get_value_string_utf8(env, value, nullptr, 0, &len);
    std::string s(len, '\0');
    napi_get_value_string_utf8(env, value, s.data(), len + 1, &len);
    return s;
}

napi_value StartEngine(napi_env env, napi_callback_info info)
{
    size_t argc = 2;
    napi_value argv[2];
    napi_value thisArg;
    void* data = nullptr;
    napi_get_cb_info(env, info, &argc, argv, &thisArg, &data);

    std::string backendName = NapiToString(env, argv[0]);
    std::string modelDir = NapiToString(env, argv[1]);

    g_engine.backend = CreateBackend(backendName);

    // Resolve the JS listener registered via setSubtitleListener.
    napi_value jsListener = nullptr;
    napi_get_named_property(env, thisArg, "onSubtitle", &jsListener);

    if (g_engine.tsfn == nullptr) {
        napi_value resource;
        napi_create_string_utf8(env, "SubtitleCallback", NAPI_AUTO_LENGTH, &resource);
        napi_create_threadsafe_function(env, jsListener, nullptr, resource, 0, 1,
                                        nullptr, nullptr, nullptr, CallJsListener,
                                        &g_engine.tsfn);
    }

    BackendConfig cfg;
    cfg.modelDir = modelDir;
    cfg.backend = backendName;
    // Paraformer streaming model covers zh + latin; language only biases UI.
    cfg.language = "zh";
    cfg.sampleRate = 16000;

    AsrCallback cb = [tsfn = g_engine.tsfn](const AsrResult& r) {
        if (tsfn == nullptr) {
            return;
        }
        auto* copy = new AsrResult{r};
        // Non-blocking: subtitle frames may be dropped under overload.
        napi_call_threadsafe_function(tsfn, copy, napi_tsfn_nonblocking);
    };

    bool ok = g_engine.backend->Init(cfg, cb);
    if (!ok) {
        g_engine.lastError = g_engine.backend->LastError();
        g_engine.backend.reset();
    }
    napi_value result;
    napi_get_boolean(env, ok, &result);
    return result;
}

napi_value StopEngine(napi_env env, napi_callback_info /*info*/)
{
    g_engine.backend.reset();
    if (g_engine.tsfn != nullptr) {
        napi_release_threadsafe_function(g_engine.tsfn, napi_tsfn_release);
        g_engine.tsfn = nullptr;
    }
    napi_value undefined;
    napi_get_undefined(env, &undefined);
    return undefined;
}

napi_value FeedAudio(napi_env env, napi_callback_info info)
{
    size_t argc = 1;
    napi_value argv[1];
    napi_get_cb_info(env, info, &argc, argv, nullptr, nullptr);

    void* pcmData = nullptr;
    size_t pcmLen = 0;
    napi_get_arraybuffer_info(env, argv[0], &pcmData, &pcmLen);
    if (g_engine.backend != nullptr) {
        g_engine.backend->Feed(static_cast<int16_t*>(pcmData), pcmLen / sizeof(int16_t));
    }
    napi_value undefined;
    napi_get_undefined(env, &undefined);
    return undefined;
}

napi_value GetLastError(napi_env env, napi_callback_info /*info*/)
{
    napi_value result;
    napi_create_string_utf8(env, g_engine.lastError.c_str(), g_engine.lastError.size(), &result);
    return result;
}

napi_value Init(napi_env env, napi_value exports)
{
    napi_property_descriptor desc[] = {
        {"startEngine", nullptr, StartEngine, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"stopEngine", nullptr, StopEngine, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"feedAudio", nullptr, FeedAudio, nullptr, nullptr, nullptr, napi_default, nullptr},
        {"getLastError", nullptr, GetLastError, nullptr, nullptr, nullptr, napi_default, nullptr},
    };
    napi_define_properties(env, exports, sizeof(desc) / sizeof(desc[0]), desc);
    return exports;
}

}  // namespace

static napi_module smartSubtitleModule = {
    .nm_version = 1,
    .nm_flags = 0,
    .nm_filename = nullptr,
    .nm_register_func = Init,
    .nm_modname = "entry",
    .nm_priv = ((void *)0),
    .reserved = {0},
};

extern "C" __attribute__((constructor)) void RegisterEntryModule(void)
{
    napi_module_register(&smartSubtitleModule);
}
