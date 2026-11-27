/*
 * smartsubtitle_cli — standalone real-time subtitle binary (no OpenHarmony
 * dependency). Useful on any aarch64 Linux/OpenHarmony system to validate the
 * CPU (sherpa-onnx) or NPU (RK3588 RKNN) pipeline before integrating the
 * library into an application.
 *
 * Usage:
 *   smartsubtitle_cli --backend cpu --model-dir /data/models --wav test.wav
 *   cat speech.pcm | smartsubtitle_cli --backend cpu --model-dir /data/models --stdin
 *   smartsubtitle_cli --backend npu --model-dir /data/models --mic
 *
 * Input must be 16 kHz, mono, signed 16-bit PCM (WAV or raw).
 * Output: one line per result, "PARTIAL <text>" / "FINAL <text>".
 */
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <string>
#include <thread>
#include <vector>

#include "asr_backend.h"

#if SSUB_HAS_ALSA && !defined(__OHOS__)
// 仅非 OpenHarmony 的 Linux 平台使用 libasound; OHOS 走 OHAudio
#include <alsa/asoundlib.h>
#endif

#if defined(__OHOS__)
// OpenHarmony: 音频采集走 OHAudio(OH_AudioCapturer), 不使用 libasound
#include <ohaudio/native_audiostreambuilder.h>
#include <ohaudio/native_audiocapturer.h>
#endif

namespace {

volatile sig_atomic_t g_stop = 0;
void OnSigInt(int) { g_stop = 1; }

subtitle::AsrCallback MakePrinter()
{
    return [](const subtitle::AsrResult& r) {
        printf("%s %s\n", r.isFinal ? "FINAL  " : "PARTIAL", r.text.c_str());
        fflush(stdout);
    };
}

// Strip a RIFF header; leaves the rest of the stream as raw PCM.
bool LoadWav(const std::string& path, std::vector<int16_t>& pcm)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        return false;
    }
    char header[44];
    f.read(header, sizeof(header));
    if (!f || std::memcmp(header, "RIFF", 4) != 0) {
        return false;
    }
    // 数据块为 16-bit LE PCM: 必须两字节合并为一个 int16 样本。
    // (旧实现 istreambuf_iterator<char> 逐字节值扩展, 使样本数翻倍、
    // 值域变成 0..255 的字节流, 特征全错 —— 板卡 CLI 乱码的根因)
    std::vector<char> bytes(std::istreambuf_iterator<char>(f),
                            (std::istreambuf_iterator<char>()));
    pcm.resize(bytes.size() / 2);
    if (!pcm.empty()) {
        std::memcpy(pcm.data(), bytes.data(), pcm.size() * sizeof(int16_t));
    }
    return !pcm.empty();
}

void Run(subtitle::AsrBackend& backend, const std::vector<int16_t>& pcm)
{
    // Stream in 100 ms chunks to mimic a live microphone.
    const size_t chunk = 1600;
    for (size_t off = 0; off < pcm.size() && !g_stop; off += chunk) {
        size_t n = std::min(chunk, pcm.size() - off);
        backend.Feed(pcm.data() + off, n);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    std::this_thread::sleep_for(std::chrono::seconds(2));  // drain tail
}

#if SSUB_HAS_ALSA && !defined(__OHOS__)
bool RunMic(subtitle::AsrBackend& backend)
{
    snd_pcm_t* handle = nullptr;
    if (snd_pcm_open(&handle, "default", SND_PCM_STREAM_CAPTURE, 0) < 0) {
        fprintf(stderr, "cannot open ALSA capture device\n");
        return false;
    }
    snd_pcm_set_params(handle, SND_PCM_FORMAT_S16_LE, SND_PCM_ACCESS_RW_INTERLEAVED,
                       1, 16000, 1, 200000);
    std::vector<int16_t> chunk(1600);
    while (!g_stop) {
        int n = snd_pcm_readi(handle, chunk.data(), chunk.size());
        if (n > 0) {
            backend.Feed(chunk.data(), n);
        }
    }
    snd_pcm_close(handle);
    return true;
}
#endif

#if defined(__OHOS__)
// OpenHarmony: 使用 OHAudio(OH_AudioCapturer) 采集麦克风 PCM。
// ReadEvent 回调运行在音频线程, 直接将采样喂给识别后端。
int32_t OnCapturerRead(OH_AudioCapturer*, void* userData,
                       void* buffer, int32_t lruSize)
{
    auto* backend = static_cast<subtitle::AsrBackend*>(userData);
    if (backend != nullptr && lruSize > 0) {
        backend->Feed(static_cast<int16_t*>(buffer),
                      static_cast<size_t>(lruSize) / sizeof(int16_t));
    }
    return lruSize;
}

bool RunMicOHOS(subtitle::AsrBackend& backend)
{
    OH_AudioStreamBuilder* builder = nullptr;
    OH_AudioStreamBuilder_Create(&builder, AUDIOSTREAM_TYPE_CAPTURER);
    OH_AudioStreamBuilder_SetSamplingRate(builder, 16000);
    OH_AudioStreamBuilder_SetChannelCount(builder, 1);
    OH_AudioStreamBuilder_SetSampleFormat(builder, AUDIOSTREAM_SAMPLE_S16LE);

    OH_AudioCapturer_Callbacks callbacks{};
    callbacks.OH_AudioCapturer_OnReadData = OnCapturerRead;
    OH_AudioStreamBuilder_SetCapturerCallback(builder, callbacks, &backend);

    OH_AudioCapturer* capturer = nullptr;
    if (OH_AudioStreamBuilder_GenerateCapturer(builder, &capturer) !=
            AUDIOSTREAM_SUCCESS ||
        OH_AudioCapturer_Start(capturer) != AUDIOSTREAM_SUCCESS) {
        fprintf(stderr, "cannot start OHAudio capturer\n");
        OH_AudioStreamBuilder_Destroy(builder);
        return false;
    }

    // 采集回调在音频线程工作, 主线程等待退出信号
    while (!g_stop) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    OH_AudioCapturer_Release(capturer);
    OH_AudioStreamBuilder_Destroy(builder);
    return true;
}
#endif  // __OHOS__

}  // namespace

int main(int argc, char** argv)
{
    signal(SIGINT, OnSigInt);

    std::string backendName = "cpu";
    std::string modelDir;
    std::string wavPath;
    bool useStdin = false;
    bool useMic = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](std::string& dst) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a.c_str());
                exit(1);
            }
            dst = argv[++i];
        };
        if (a == "--backend") next(backendName);
        else if (a == "--model-dir") next(modelDir);
        else if (a == "--wav") next(wavPath);
        else if (a == "--stdin") useStdin = true;
        else if (a == "--mic") useMic = true;
        else {
            fprintf(stderr, "unknown option %s\n", a.c_str());
            return 1;
        }
    }

    auto backend = subtitle::CreateBackend(backendName);
    subtitle::BackendConfig cfg;
    cfg.modelDir = modelDir;
    cfg.backend = backendName;
    cfg.sampleRate = 16000;
    if (!backend->Init(cfg, MakePrinter())) {
        fprintf(stderr, "backend init failed: %s\n", backend->LastError().c_str());
        return 1;
    }

    if (useMic) {
#if defined(__OHOS__)
        // OpenHarmony: 走 OHAudio 采集
        if (!RunMicOHOS(*backend)) {
            return 1;
        }
        return 0;
#elif SSUB_HAS_ALSA && !defined(__OHOS__)
        RunMic(*backend);
        return 0;
#else
        fprintf(stderr, "built without audio capture support; use --wav or --stdin\n");
        return 1;
#endif
    }

    std::vector<int16_t> pcm;
    if (useStdin) {
        pcm.assign(std::istreambuf_iterator<char>(std::cin),
                   std::istreambuf_iterator<char>());
        // Re-interpret bytes: istreambuf gives signed chars on some platforms.
        std::vector<int16_t> raw(pcm.size() / 2);
        std::memcpy(raw.data(), pcm.data(), raw.size() * sizeof(int16_t));
        Run(*backend, raw);
        return 0;
    }

    if (!LoadWav(wavPath, pcm)) {
        fprintf(stderr, "cannot read WAV file: %s\n", wavPath.c_str());
        return 1;
    }
    Run(*backend, pcm);
    return 0;
}
