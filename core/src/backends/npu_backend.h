// NPU backend: RKNN models running on the Rockchip RK3588 NPU.
// Compiled only when the rknpu headers/libs are present; otherwise the
// factory falls back to the stub backend (see asr_backend.cpp).
#ifndef SMARTSUBTITLE_NPU_BACKEND_H_
#define SMARTSUBTITLE_NPU_BACKEND_H_

#include "asr_backend.h"

#include <memory>

namespace subtitle {

std::unique_ptr<AsrBackend> CreateNpuBackend(bool use_ort = false);

}  // namespace subtitle

#endif  // SMARTSUBTITLE_NPU_BACKEND_H_
