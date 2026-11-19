// NPU backend implementation based on the RKNN Runtime API (rknpu2).
//
// Pipeline (16 kHz mono int16):
//   mic chunks -> Silero VAD gate (sherpa-onnx, CPU) -> RKNN encoder
//   (converted Paraformer streaming encoder, .rknn) -> greedy CTC decode
//   on CPU (tokens.txt) -> partial/final results.
//
// The .rknn models are produced offline by rknn-toolkit2 on an x86 host, see
// README "NPU 推理（RK3588）" for the conversion commands.
#include "npu_backend.h"

#if __has_include("rknn_api.h")

#include <chrono>
#include <fstream>
#include <mutex>

#include <rknn_api.h>

#if __has_include("sherpa-onnx/c-api/c-api.h")
#include "sherpa-onnx/c-api/c-api.h"
#define SSUB_HAS_SHERPA_VAD 1
#endif

namespace subtitle {
namespace {

constexpr int kSampleRate = 16000;
// Encoder window consumed per RKNN invocation, 480 ms of 16-bit mono PCM.
constexpr int kWindowSamples = 7680;

std::vector<char> ReadFile(const std::string& path)
{
    std::ifstream f(path, std::ios::binary);
    return std::vector<char>(std::istreambuf_iterator<char>(f),
                             std::istreambuf_iterator<char>());
}

class NpuBackend : public AsrBackend {
public:
    bool Init(const BackendConfig& config, AsrCallback callback) override
    {
        callback_ = std::move(callback);

        if (LoadModel(config.modelDir + "/paraformer-encoder.rknn") != RKNN_SUCC) {
            lastError_ = "failed to load paraformer-encoder.rknn from " + config.modelDir;
            return false;
        }
        // Bind to NPU core 0 (RK3588 has three cores; one core is enough for
        // a streaming encoder at this size and keeps the others free).
        rknn_set_core_mask(&ctx_, RKNN_NPU_CORE_0);

        if (!LoadTokens(config.modelDir + "/tokens.txt")) {
            lastError_ = "failed to load tokens.txt from " + config.modelDir;
            return false;
        }

#ifdef SSUB_HAS_SHERPA_VAD
        SherpaOnnxVadModelConfig vadCfg{};
        std::string vadModel = config.modelDir + "/silero_vad.onnx";
        vadCfg.silero_vad.model = vadModel.c_str();
        vadCfg.silero_vad.threshold = 0.5f;
        vadCfg.sample_rate = kSampleRate;
        vadCfg.sample_rate_str = std::to_string(kSampleRate).c_str();
        vad_ = SherpaOnnxCreateVoiceActivityDetector(&vadCfg, 5.0f);
#endif

        worker_ = std::thread(&NpuBackend::DecodeLoop, this);
        running_ = true;
        return true;
    }

    void Feed(const int16_t* pcm, size_t samples) override
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < samples; i++) {
            pending_.push_back(pcm[i]);
        }
#ifdef SSUB_HAS_SHERPA_VAD
        if (vad_ != nullptr) {
            SherpaOnnxVoiceActivityDetectorAcceptWaveform(
                vad_, pcm, samples, kSampleRate);
        }
#endif
    }

    void Reset() override
    {
        std::lock_guard<std::mutex> lk(mu_);
        pending_.clear();
        context_.clear();
    }

    std::string LastError() const override
    {
        return lastError_;
    }

    ~NpuBackend()
    {
        running_ = false;
        if (worker_.joinable()) {
            worker_.join();
        }
        if (vad_ != nullptr) {
            SherpaOnnxDestroyVoiceActivityDetector(vad_);
        }
        rknn_destroy(&ctx_);
    }

private:
    rknn_status LoadModel(const std::string& path)
    {
        std::vector<char> model = ReadFile(path);
        if (model.empty()) {
            return RKNN_ERR_MODEL_INVALID;
        }
        memset(&ctx_, 0, sizeof(ctx_));
        return rknn_init(&ctx_, model.data(), model.size(), 0, nullptr);
    }

    bool LoadTokens(const std::string& path)
    {
        std::ifstream f(path);
        if (!f) {
            return false;
        }
        std::string line;
        while (std::getline(f, line)) {
            auto sp = line.find(' ');
            if (sp != std::string::npos) {
                tokens_[std::stoi(line.substr(0, sp))] = line.substr(sp + 1);
            }
        }
        return !tokens_.empty();
    }

    // Run one encoder window on the NPU and greedily decode the CTC output.
    void RunWindow(const int16_t* window, size_t n)
    {
        rknn_input in{};
        in.index = 0;
        in.type = RKNN_TENSOR_INT16;
        in.fmt = RKNN_TENSOR_NHWC;
        in.size = n * sizeof(int16_t);
        in.buf = const_cast<int16_t*>(window);
        in.pass_through = 0;
        if (rknn_inputs_set(&ctx_, 1, &in) != RKNN_SUCC) {
            lastError_ = "rknn_inputs_set failed";
            return;
        }
        if (rknn_run(&ctx_, nullptr) != RKNN_SUCC) {
            lastError_ = "rknn_run failed";
            return;
        }
        rknn_output out{};
        memset(&out, 0, sizeof(out));
        out.index = 0;
        out.want_float = 1;
        out.is_prealloc = 0;
        if (rknn_outputs_get(&ctx_, 1, &out, nullptr) != RKNN_SUCC) {
            lastError_ = "rknn_outputs_get failed";
            return;
        }

        // Greedy CTC: pick argmax per frame, dedupe consecutive blanks (id 0).
        const float* logits = reinterpret_cast<const float*>(out.buf);
        int frames = out.size / sizeof(float) / vocabSize_;
        std::string text;
        int prev = -1;
        for (int t = 0; t < frames; t++) {
            const float* row = logits + t * vocabSize_;
            int best = 0;
            for (int c = 1; c < vocabSize_; c++) {
                if (row[c] > row[best]) {
                    best = c;
                }
            }
            if (best != 0 && best != prev) {
                auto it = tokens_.find(best);
                if (it != tokens_.end()) {
                    text += it->second;
                }
            }
            prev = best;
        }
        rknn_outputs_release(&ctx_, 1, &out);

        if (!text.empty()) {
            AsrResult res;
            res.text = context_ + text;
            res.isFinal = false;
            callback_(res);
        }
    }

    void DecodeLoop()
    {
        std::vector<int16_t> window(kWindowSamples);
        while (running_) {
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (pending_.size() < kWindowSamples) {
                    window.clear();
                } else {
#ifdef SSUB_HAS_SHERPA_VAD
                    if (vad_ != nullptr &&
                        !SherpaOnnxVoiceActivityDetectorContainsSpeech(vad_)) {
                        pending_.clear();  // idle: drop silence, keep NPU cool
                        continue;
                    }
#endif
                    window.assign(pending_.begin(),
                                  pending_.begin() + kWindowSamples);
                    pending_.erase(pending_.begin(),
                                   pending_.begin() + kWindowSamples);
                }
            }
            if (window.empty()) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }
            RunWindow(window.data(), window.size());
        }
    }

    rknn_context ctx_ = 0;
    int vocabSize_ = 0;  // read from the model's output tensor shape at init
    AsrCallback callback_{};
    std::mutex mu_;
    std::vector<int16_t> pending_;
    std::string context_;   // committed text for the current utterance
    std::string lastError_ = "ok";
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::map<int, std::string> tokens_;
    SherpaOnnxVoiceActivityDetector* vad_ = nullptr;
};

}  // namespace

std::unique_ptr<AsrBackend> CreateNpuBackend()
{
    return std::make_unique<NpuBackend>();
}

#else  // RKNN not available

#include <atomic>
#include <map>
#include <memory>
#include <string>
#include <thread>

#include "asr_backend.h"

namespace subtitle {
namespace {
class NpuStub : public AsrBackend {
public:
    bool Init(const BackendConfig&, AsrCallback) override
    {
        lastError_ = "RKNN runtime was not linked into this build (see README NPU section)";
        return false;
    }
    void Feed(const int16_t*, size_t) override {}
    void Reset() override {}
    std::string LastError() const override { return lastError_; }
private:
    std::string lastError_;
};
}
std::unique_ptr<AsrBackend> CreateNpuBackend()
{
    return std::make_unique<NpuStub>();
}
}  // namespace subtitle

#endif
