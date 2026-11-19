// SmartSubtitle - ASR backend interface.
//
// A backend consumes 16 kHz mono int16 PCM and emits recognition results
// (partial / final) through a callback. Two implementations are provided:
//   - CpuBackend: sherpa-onnx (streaming Paraformer + Silero VAD)
//   - NpuBackend: RKNN models on Rockchip RK3588 NPU
#ifndef SMARTSUBTITLE_ASR_BACKEND_H_
#define SMARTSUBTITLE_ASR_BACKEND_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace subtitle {

struct BackendConfig {
    std::string modelDir;        // directory holding models / tokens
    std::string language;        // "zh" / "en"
    std::string backend;         // "cpu" / "npu"
    int sampleRate = 16000;      // only 16 kHz is supported
};

struct AsrResult {
    std::string text;
    bool isFinal = false;
};

using AsrCallback = std::function<void(const AsrResult&)>;

class AsrBackend {
public:
    virtual ~AsrBackend() = default;

    // Returns false when the backend cannot be initialised (missing models,
    // unsupported device, ...). On success the callback stays installed for
    // the backend lifetime.
    virtual bool Init(const BackendConfig& config, AsrCallback callback) = 0;

    // Feed a chunk of 16 kHz mono int16 PCM.
    virtual void Feed(const int16_t* pcm, size_t samples) = 0;

    // Flush internal state (e.g. force-finalise the current utterance).
    virtual void Reset() = 0;

    // Human readable last-error string for diagnostics / UI display.
    virtual std::string LastError() const = 0;
};

// Factory: creates the requested backend. Never returns nullptr; a stub
// backend that only records "not supported" is returned when the optional
// third-party dependency (sherpa-onnx / RKNN) was not compiled in.
std::unique_ptr<AsrBackend> CreateBackend(const std::string& name);

}  // namespace subtitle

#endif  // SMARTSUBTITLE_ASR_BACKEND_H_
