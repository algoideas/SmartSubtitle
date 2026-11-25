// CPU backend implementation based on the sherpa-onnx C API.
//
// Pipeline (16 kHz mono int16):
//   mic chunks -> Silero VAD gate -> streaming Paraformer -> partial/final
// Endpointing uses sherpa-onnx's rule-based endpoint detector
// (rule1/2/3 in OnlineRecognizerEndpointConfig).
#include "cpu_backend.h"
#include "npu_backend.h"

#if __has_include("sherpa-onnx/c-api/c-api.h")

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "sherpa-onnx/c-api/c-api.h"

// core 保持与 OpenHarmony 无关, 错误走 stderr; 上层 NAPI 可再转发到 hilog
#define SSUB_LOGE(...)                    \
    do {                                  \
        fprintf(stderr, "[SSUB] ");      \
        fprintf(stderr, __VA_ARGS__);    \
        fprintf(stderr, "\n");           \
    } while (0)

namespace subtitle {
namespace {

// Ring buffer that decouples the audio thread from the decoding loop.
class PcmRing {
public:
    explicit PcmRing(size_t capacitySamples)
        : buf_(capacitySamples), head_(0), tail_(0) {}

    void Push(const int16_t* data, size_t n)
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (size_t i = 0; i < n; i++) {
            size_t next = (head_ + 1) % buf_.size();
            if (next == tail_) {           // overflow: drop oldest
                tail_ = (tail_ + 1) % buf_.size();
            }
            buf_[head_] = data[i];
            head_ = next;
        }
    }

    // Drain up to buf_.size()/2 samples into out; returns count drained.
    size_t Pop(std::vector<int16_t>& out)
    {
        std::lock_guard<std::mutex> lk(mu_);
        size_t avail = (head_ + buf_.size() - tail_) % buf_.size();
        size_t take = std::min(avail, buf_.size() / 2);
        out.clear();
        for (size_t i = 0; i < take; i++) {
            out.push_back(buf_[tail_]);
            tail_ = (tail_ + 1) % buf_.size();
        }
        return take;
    }

private:
    std::mutex mu_;
    std::vector<int16_t> buf_;
    size_t head_;
    size_t tail_;
};

class CpuBackend : public AsrBackend {
public:
    bool Init(const BackendConfig& config, AsrCallback callback) override
    {
        callback_ = std::move(callback);
        modelDir_ = config.modelDir;

        SherpaOnnxOnlineRecognizerConfig recCfg{};
        SherpaOnnxOnlineModelConfig& m = recCfg.model_config;
        // 流式 Paraformer 中英模型拆分为 encoder/decoder 两部分(见
        // third_party/models 下的发布产物)
        std::string encoder = modelDir_ + "/encoder.int8.onnx";
        std::string decoder = modelDir_ + "/decoder.int8.onnx";
        std::string tokens = modelDir_ + "/tokens.txt";

        m.paraformer.encoder = encoder.c_str();
        m.paraformer.decoder = decoder.c_str();
        m.tokens = tokens.c_str();
        m.num_threads = 4;
        m.provider = "cpu";
        m.debug = 0;
        // 注: 梅尔滤波截止高频在 vendored sherpa features.h 中已改为
        // 默认 0 (奈奎斯特频率), 与 paraformer 训练配置一致

        // Chinese + English (Latin) decoding; zh model already handles latin
        // characters via BPE tokens.
        recCfg.decoding_method = "greedy_search";
        recCfg.max_active_paths = 4;

        // Endpointing: finalise after 0.8 s trailing silence, or 2.4 s of
        // trailing silence containing speech at utterance start.
        recCfg.enable_endpoint = 1;
        recCfg.rule1_min_trailing_silence = 2.4f;
        recCfg.rule2_min_trailing_silence = 0.8f;
        recCfg.rule3_min_utterance_length = 20.0f;

        try {
            recognizer_ = SherpaOnnxCreateOnlineRecognizer(&recCfg);
        } catch (const std::exception& e) {
            SSUB_LOGE("create recognizer exception: %{public}s", e.what());
            lastError_ = std::string("recognizer init exception: ") + e.what();
            return false;
        }
        if (recognizer_ == nullptr) {
            lastError_ = "failed to create sherpa-onnx online recognizer, check model files in " + modelDir_;
            return false;
        }
        stream_ = SherpaOnnxCreateOnlineStream(recognizer_);
        if (stream_ == nullptr) {
            lastError_ = "failed to create recognizer stream";
            return false;
        }

        // Silero VAD (models/silero_vad.onnx) gates silence away from the
        // recognizer, saving CPU on long idle periods.
        SherpaOnnxVadModelConfig vadCfg{};
        std::string vadModel = modelDir_ + "/silero_vad.onnx";
        vadCfg.silero_vad.model = vadModel.c_str();
        vadCfg.silero_vad.threshold = 0.5f;
        vadCfg.silero_vad.min_speech_duration = 0.25f;
        vadCfg.silero_vad.min_silence_duration = 0.5f;
        // silero VAD v5 在 16kHz 下固定窗口为 512, 缺省为 0 会触发 exit(-1)
        vadCfg.silero_vad.window_size = 512;
        vadCfg.sample_rate = config.sampleRate;
        try {
            vad_ = SherpaOnnxCreateVoiceActivityDetector(&vadCfg, 5.0f);
        } catch (const std::exception& e) {
            SSUB_LOGE("create VAD exception: %{public}s", e.what());
            vad_ = nullptr;
        }
        // VAD is optional: continue without it when the model is missing.
        hasVad_ = (vad_ != nullptr);

        worker_ = std::thread(&CpuBackend::DecodeLoop, this);
        running_ = true;
        return true;
    }

    void Feed(const int16_t* pcm, size_t samples) override
    {
        ring_.Push(pcm, samples);
    }

    void Reset() override
    {
        if (stream_ != nullptr && recognizer_ != nullptr) {
            // Force-finalise by feeding the endpoint detector a reset.
            SherpaOnnxOnlineStreamReset(recognizer_, stream_);
        }
    }

    std::string LastError() const override
    {
        return lastError_;
    }

    ~CpuBackend()
    {
        running_ = false;
        if (worker_.joinable()) {
            worker_.join();
        }
        if (stream_ != nullptr) {
            SherpaOnnxDestroyOnlineStream(stream_);
        }
        if (recognizer_ != nullptr) {
            SherpaOnnxDestroyOnlineRecognizer(recognizer_);
        }
        if (vad_ != nullptr) {
            SherpaOnnxDestroyVoiceActivityDetector(vad_);
        }
    }

private:
    void DecodeLoop()
    {
        std::vector<int16_t> chunk;
        std::vector<float> floats;
        const int kSampleRate = 16000;
        while (running_) {
            if (ring_.Pop(chunk) == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            try {
                DecodeChunk(chunk, floats, kSampleRate);
            } catch (const std::exception& e) {
                // 捕获 ORT/解码异常, 避免未捕获异常导致整个进程 abort
                SSUB_LOGE("decode exception: %{public}s", e.what());
            }
        }
    }

    void DecodeChunk(const std::vector<int16_t>& chunk, std::vector<float>& floats,
                     int kSampleRate)
    {

            // sherpa-onnx v1.10.30 的 C API 接收 float 采样, 且 sample_rate
            // 位于第 2 个参数。paraformer 的 normalize_samples=false:
            // FeatureExtractor 内部会把输入乘回 32768, 因此这里喂
            // 归一化 [-1,1] 采样
            floats.resize(chunk.size());
            for (size_t i = 0; i < chunk.size(); ++i) {
                floats[i] = chunk[i] / 32768.0f;
            }

            if (getenv("SSUB_CPU_DEBUG")) {
                static int dbg_cnt = 0;
                if (dbg_cnt++ % 10 == 0) {
                    fprintf(stderr,
                            "cpu: chunk=%d hasVad=%d detected=%d\n",
                            dbg_cnt, hasVad_,
                            hasVad_
                                ? SherpaOnnxVoiceActivityDetectorDetected(vad_)
                                : -1);
                }
            }

            // 识别器持续运行, 断句交给端点检测(IsEndpoint)处理。
            // 注: 不以 VAD Detected() 做门控 — 其语义为"已完成的语音段是否
            // 待弹出", 持续语音期间不会置位, 且内部缓冲不弹出会在 5 秒后
            // 冻结, 曾导致 CPU 后端无输出。
            SherpaOnnxOnlineStreamAcceptWaveform(
                stream_, kSampleRate, floats.data(), floats.size());

            while (SherpaOnnxOnlineStreamIsEndpoint(recognizer_, stream_)) {
                // 音频尾部不足一个分块时帧数不够, 等下一块再触发,
                // 否则 GetFrames 会因帧数越界抛异常
                if (!SherpaOnnxIsOnlineStreamReady(recognizer_, stream_)) {
                    break;
                }
                SherpaOnnxDecodeOnlineStream(recognizer_, stream_);
                const SherpaOnnxOnlineRecognizerResult* r =
                    SherpaOnnxGetOnlineStreamResult(recognizer_, stream_);
                if (r != nullptr && r->text != nullptr && r->text[0] != '\0') {
                    AsrResult res{r->text, true};
                    callback_(res);
                }
                if (r != nullptr) {
                    SherpaOnnxDestroyOnlineRecognizerResult(r);
                }
                SherpaOnnxOnlineStreamReset(recognizer_, stream_);
            }

            if (SherpaOnnxIsOnlineStreamReady(recognizer_, stream_)) {
                SherpaOnnxDecodeOnlineStream(recognizer_, stream_);
                const SherpaOnnxOnlineRecognizerResult* r =
                    SherpaOnnxGetOnlineStreamResult(recognizer_, stream_);
                if (getenv("SSUB_CPU_DEBUG")) {
                    fprintf(stderr, "cpu decode: text_len=%zu\n",
                            r != nullptr && r->text != nullptr
                                ? strlen(r->text)
                                : -1);
                }
                if (r != nullptr && r->text != nullptr && r->text[0] != '\0') {
                    AsrResult res{r->text, false};
                    callback_(res);
                }
                if (r != nullptr) {
                    SherpaOnnxDestroyOnlineRecognizerResult(r);
                }
            }
        }

    std::string modelDir_;
    std::string lastError_ = "ok";
    PcmRing ring_{16000 * 2};  // 2 s of audio headroom
    std::thread worker_;
    std::atomic<bool> running_{false};
    AsrCallback callback_{};

    const SherpaOnnxOnlineRecognizer* recognizer_ = nullptr;
    const SherpaOnnxOnlineStream* stream_ = nullptr;
    SherpaOnnxVoiceActivityDetector* vad_ = nullptr;
    bool hasVad_ = false;
};

}  // namespace

std::unique_ptr<AsrBackend> CreateCpuBackend()
{
    // CPU 推理复用与 NPU 一致的已验证管线 (ManualFbank + ORT int8 编码器)
    return CreateNpuBackend(true);
}

}  // namespace subtitle

#else  // sherpa-onnx not available

#include <atomic>
#include <string>
#include <memory>

#include "asr_backend.h"

namespace subtitle {
namespace {
class CpuStub : public AsrBackend {
public:
    bool Init(const BackendConfig&, AsrCallback cb) override
    {
        lastError_ = "sherpa-onnx was not linked into this build (see README CPU section)";
        return false;
    }
    void Feed(const int16_t*, size_t) override {}
    void Reset() override {}
    std::string LastError() const override { return lastError_; }
private:
    std::string lastError_;
};
}
std::unique_ptr<AsrBackend> CreateCpuBackend()
{
    return std::make_unique<CpuStub>();
}
}  // namespace subtitle

#endif
