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

#if SSUB_HAS_ALSA
#include <alsa/asoundlib.h>
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
    pcm.assign(std::istreambuf_iterator<char>(f), {});
    return true;
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

#if SSUB_HAS_ALSA
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
#if SSUB_HAS_ALSA
        RunMic(*backend);
        return 0;
#else
        fprintf(stderr, "built without ALSA support; use --wav or --stdin\n");
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
