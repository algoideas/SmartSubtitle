// Backend factory. Maps the backend name from the UI ("cpu" / "npu") to the
// compiled-in implementation; falls back to a stub that reports why the
// backend is unavailable when the optional third-party runtime is missing.
#include <string>

#include "asr_backend.h"

namespace subtitle {

std::unique_ptr<AsrBackend> CreateCpuBackend();
std::unique_ptr<AsrBackend> CreateNpuBackend(bool use_ort = false);

std::unique_ptr<AsrBackend> CreateBackend(const std::string& name)
{
    if (name == "npu") {
        return CreateNpuBackend();
    }
    return CreateCpuBackend();
}

}  // namespace subtitle
