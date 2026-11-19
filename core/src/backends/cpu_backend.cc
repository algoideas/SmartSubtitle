// CPU backend implementation based on the sherpa-onnx C API.
//
// Pipeline (16 kHz mono int16):
//   mic chunks -> Silero VAD gate -> streaming Paraformer -> partial/final
// Endpointing uses sherpa-onnx's rule-based endpoint detector
// (rule1/2/3 in OnlineRecognizerEndpointConfig).
#include "cpu_backend.h"

#if __has_include("sherpa-onnx/c-api/c-api.h")

#include <algorithm>
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "sherpa-onnx/c-api/c-api.h"

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
        std::string paraformer = modelDir_ + "/paraformer-streaming-zh-en.int8.onnx";
        std::string tokens = modelDir_ + "/tokens.txt";

        m.paraformer = paraformer.c_str();
        m.tokens = tokens.c_str();
        m.num_threads = 4;
        m.provider = "cpu";
        m.debug = 0;

        // Chinese + English (Latin) decoding; zh model already handles latin
        // characters via BPE tokens.
        recCfg.decoding_method = "modified_beam_search";
        recCfg.max_active_paths = 4;

        // Endpointing: finalise after 0.8 s trailing silence, or 2.4 s of
        // trailing silence containing speech at utterance start.
        recCfg.endpoint_config.rule1_min_trailing_silence = 2.4f;
        recCfg.endpoint_config.rule2_min_trailing_silence = 0.8f;
        recCfg.endpoint_config.rule3_min_utterance_length = 20.0f;

        recognizer_ = SherpaOnnxCreateOnlineRecognizer(&recCfg);
        if (recognizer_ == nullptr) {
            lastError_ = "failed to create sherpa-onnx online recognizer, check model files in " + modelDir_;
            return false;
        }
        stream_ = SherpaOnnxOnlineRecognizerCreateStream(recognizer_);
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
        vadCfg.sample_rate = config.sampleRate;
        vadCfg.sample_rate_str = std::to_string(config.sampleRate).c_str();
        vad_ = SherpaOnnxCreateVoiceActivityDetector(&vadCfg, 5.0f);
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
            SherpaOnnxOnlineRecognizerReset(recognizer_, stream_);
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
        const int kSampleRate = 16000;
        while (running_) {
            if (ring_.Pop(chunk) == 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                continue;
            }

            if (hasVad_ && !SherpaOnnxVoiceActivityDetectorContainsSpeech(vad_)) {
                // Feed VAD so its state advances, but skip the recognizer.
                SherpaOnnxVoiceActivityDetectorAcceptWaveform(
                    vad_, chunk.data(), chunk.size(), kSampleRate);
                continue;
            }

            SherpaOnnxOnlineStreamAcceptWaveform(
                stream_, chunk.data(), chunk.size(), kSampleRate);

            while (SherpaOnnxOnlineRecognizerIsEndpoint(recognizer_, stream_)) {
                SherpaOnnxOnlineRecognizerDecode(recognizer_, stream_);
                const SherpaOnnxOnlineRecognizerResult* r =
                    SherpaOnnxOnlineRecognizerGetResult(recognizer_, stream_);
                if (r != nullptr && r->text != nullptr && r->text[0] != '\0') {
                    AsrResult res{r->text, true};
                    callback_(res);
                }
                if (r != nullptr) {
                    SherpaOnnxDestroyOnlineRecognizerResult(r);
                }
                SherpaOnnxOnlineStreamReset(recognizer_, stream_);
            }

            if (SherpaOnnxOnlineRecognizerIsReady(recognizer_, stream_)) {
                SherpaOnnxOnlineRecognizerDecode(recognizer_, stream_);
                const SherpaOnnxOnlineRecognizerResult* r =
                    SherpaOnnxOnlineRecognizerGetResult(recognizer_, stream_);
                if (r != nullptr && r->text != nullptr && r->text[0] != '\0') {
                    AsrResult res{r->text, false};
                    callback_(res);
                }
                if (r != nullptr) {
                    SherpaOnnxDestroyOnlineRecognizerResult(r);
                }
            }
        }
    }

    std::string modelDir_;
    std::string lastError_ = "ok";
    PcmRing ring_{16000 * 2};  // 2 s of audio headroom
    std::thread worker_;
    std::atomic<bool> running_{false};
    AsrCallback callback_{};

    SherpaOnnxOnlineRecognizer* recognizer_ = nullptr;
    SherpaOnnxOnlineStream* stream_ = nullptr;
    SherpaOnnxVoiceActivityDetector* vad_ = nullptr;
    bool hasVad_ = false;
};

}  // namespace

std::unique_ptr<AsrBackend> CreateCpuBackend()
{
    return std::make_unique<CpuBackend>();
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
