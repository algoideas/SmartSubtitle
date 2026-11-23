// CPU backend: sherpa-onnx streaming Paraformer + Silero VAD.
// Compiled only when the sherpa-onnx headers/libs are present; otherwise the
// factory falls back to the stub backend (see asr_backend.cpp).
#ifndef SMARTSUBTITLE_CPU_BACKEND_H_
#define SMARTSUBTITLE_CPU_BACKEND_H_

#include "asr_backend.h"

#include <memory>

namespace subtitle {

std::unique_ptr<AsrBackend> CreateCpuBackend();

}  // namespace subtitle

#endif  // SMARTSUBTITLE_CPU_BACKEND_H_
