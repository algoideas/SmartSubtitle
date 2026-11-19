# SmartSubtitle - libentry.so build (OpenHarmony NAPI layer).
#
# The real-time subtitle engine lives in the standalone core/ directory at the
# repository root (no OpenHarmony dependency; it also builds the standalone
# smartsubtitle_cli binary). This CMakeLists only wraps the core with the
# NAPI bridge used by the ArkTS UI.
cmake_minimum_required(VERSION 3.5)
project(SmartSubtitle)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(CORE_ROOT ${NATIVERENDER_ROOT_PATH}/../../../../../core)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wno-int-conversion -Wno-writable-strings")

# sherpa-onnx (CPU backend) and rknpu2 (NPU backend, RK3588) prebuilt runtimes
# are vendored under third_party/ (see README for the exact layout):
#   third_party/sherpa-onnx/include + third_party/sherpa-onnx/libs/${OHOS_ARCH}
#   third_party/rknpu/include       + third_party/rknpu/libs/${OHOS_ARCH}
set(SHERPA_ONNX_DIR ${NATIVERENDER_ROOT_PATH}/third_party/sherpa-onnx)
set(RKNPU_DIR ${NATIVERENDER_ROOT_PATH}/third_party/rknpu)

include_directories(
    ${CORE_ROOT}/include/subtitle
    ${SHERPA_ONNX_DIR}/include
    ${RKNPU_DIR}/include
)

add_library(entry SHARED
    napi_init.cpp
    ${CORE_ROOT}/src/backends/asr_backend.cpp
    ${CORE_ROOT}/src/backends/cpu_backend.cc
    ${CORE_ROOT}/src/backends/npu_backend.cc
)
target_compile_definitions(entry PRIVATE SSUB_HAS_SHERPA=1 SSUB_HAS_RKNN=1)

find_library(hilog-lib hilog_ndk.z)

target_link_libraries(entry PUBLIC
    libace_napi.z.so
    ${hilog-lib}
)

# Prebuilt third-party runtimes (present once the model/runtime download step
# from the README has been executed).
target_link_libraries(entry PUBLIC
    -L${SHERPA_ONNX_DIR}/libs/${OHOS_ARCH} sherpa-onnx-c-api
    -L${RKNPU_DIR}/libs/${OHOS_ARCH} rknnrt
)
