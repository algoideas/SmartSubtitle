# SmartSubtitle - libentry.so build (OpenHarmony NAPI layer).
#
# The real-time subtitle engine lives in the standalone core/ directory at the
# repository root (no OpenHarmony dependency; it also builds the standalone
# smartsubtitle_cli binary). This CMakeLists only wraps the core with the
# NAPI bridge used by the ArkTS UI.
cmake_minimum_required(VERSION 3.5)
project(SmartSubtitle)

set(NATIVERENDER_ROOT_PATH ${CMAKE_CURRENT_SOURCE_DIR})
set(CORE_ROOT ${NATIVERENDER_ROOT_PATH}/../../../../core)

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

find_library(hilog-lib hilog_ndk.z)

target_link_libraries(entry PUBLIC
    libace_napi.z.so
    ${hilog-lib}
)

# 预编译第三方运行时(见 README 的下载/构建步骤), 缺失时对应后端自动退化为
# stub, 保证工程始终可编译。
if(EXISTS ${SHERPA_ONNX_DIR}/libs/arm64-v8a/libsherpa-onnx-c-api.a)
    target_compile_definitions(entry PRIVATE SSUB_HAS_SHERPA=1)
    target_include_directories(entry PRIVATE ${SHERPA_ONNX_DIR}/libs/arm64-v8a)
    # c-api 引用 sherpa_onnx::/knf::/kaldi_decoder:: 等核心符号, 需要整组库;
    # onnxruntime 组件库与依赖库(nsync/protobuf/onnx/abseil 等)用
    # --start-group/--end-group 包装, 规避静态库之间的循环依赖
    file(GLOB SHERPA_ALL_LIBS ${SHERPA_ONNX_DIR}/libs/arm64-v8a/lib*.a)
    list(REMOVE_ITEM SHERPA_ALL_LIBS ${SHERPA_ONNX_DIR}/libs/arm64-v8a/libsherpa-onnx-c-api.a)
    file(GLOB ORT_ALL_LIBS ${NATIVERENDER_ROOT_PATH}/third_party/onnxruntime/libs/arm64-v8a/lib*.a)
    target_link_libraries(entry PUBLIC
        ${SHERPA_ONNX_DIR}/libs/arm64-v8a/libsherpa-onnx-c-api.a
        -Wl,--start-group
        ${SHERPA_ALL_LIBS}
        ${ORT_ALL_LIBS}
        -Wl,--end-group
    )
    # onnxruntime 静态库内部使用 libc++
    target_link_libraries(entry PUBLIC -Wl,--no-as-needed -lc++)
endif()

if(EXISTS ${RKNPU_DIR}/libs/arm64-v8a/librknnrt.so)
    target_compile_definitions(entry PRIVATE SSUB_HAS_RKNN=1)
    target_link_libraries(entry PUBLIC
        ${RKNPU_DIR}/libs/arm64-v8a/librknnrt.so
    )
elseif(EXISTS ${NATIVERENDER_ROOT_PATH}/third_party/rknn/libs/arm64-v8a/librknnrt.so)
    # 回退: 仓库内 third_party/rknn (librknnrt.so 取自 rknn-toolkit2 v2.1.0)
    target_compile_definitions(entry PRIVATE SSUB_HAS_RKNN=1)
    target_link_libraries(entry PUBLIC
        ${NATIVERENDER_ROOT_PATH}/third_party/rknn/libs/arm64-v8a/librknnrt.so
    )
endif()
