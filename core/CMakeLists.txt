# SmartSubtitle core — standalone C/C++ real-time subtitle engine.
#
# Builds:
#   subtitle_core      static library (backend factory + CPU/NPU backends)
#   smartsubtitle_cli  standalone executable for headless aarch64 systems
#
# The core does NOT depend on OpenHarmony: it compiles with any CMake toolchain
# file. See README "CPU 推理" for building with the OpenHarmony clang wrapper:
#   cmake -DCMAKE_TOOLCHAIN_FILE=... (ohos-clang toolchain) -DCMAKE_BUILD_TYPE=Release
cmake_minimum_required(VERSION 3.5)
project(smartsubtitle_core CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CORE_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

# ===== sherpa-onnx (CPU backend) =====
# Build or download a sherpa-onnx OHOS aarch64 release first (see README) and
# point SHERPA_ONNX_ROOT at its install prefix:
#   $SHERPA_ONNX_ROOT/include/sherpa-onnx/c-api/c-api.h
#   $SHERPA_ONNX_ROOT/lib/libsherpa-onnx-c-api.so (+ onnxruntime, etc.)
set(SHERPA_ONNX_ROOT "$ENV{SHERPA_ONNX_ROOT}" CACHE PATH "sherpa-onnx install prefix")
if(EXISTS ${SHERPA_ONNX_ROOT}/include/sherpa-onnx/c-api/c-api.h)
    set(SHERPA_ONNX_FOUND ON)
    include_directories(${SHERPA_ONNX_ROOT}/include)
    link_directories(${SHERPA_ONNX_ROOT}/lib)
    add_compile_definitions(SSUB_HAS_SHERPA=1)
    message(STATUS "sherpa-onnx: ${SHERPA_ONNX_ROOT}")
else()
    message(WARNING "sherpa-onnx not found at ${SHERPA_ONNX_ROOT}; CPU backend will be a stub")
endif()

# ===== RKNN runtime (NPU backend, RK3588) =====
# 默认使用仓库内 third_party/rknn (头文件来自 RknnFacialRecognitionDemo,
# librknnrt.so 取自 rknn-toolkit2 v2.1.0), 也可用环境变量 RKNPU_ROOT
# 指向 rknpu2 运行时:
#   $RKNPU_ROOT/include/rknn_api.h
#   $RKNPU_ROOT/lib/librknnrt.so 或 $RKNPU_ROOT/lib/arm64-v8a/librknnrt.so
if(NOT RKNPU_ROOT)
    set(RKNPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/rknn")
endif()
set(RKNPU_ROOT "${RKNPU_ROOT}" CACHE PATH "rknpu2 runtime prefix")
if(EXISTS ${RKNPU_ROOT}/include/rknn_api.h)
    set(RKNN_FOUND ON)
    include_directories(${RKNPU_ROOT}/include)
    add_compile_definitions(SSUB_HAS_RKNN=1)
    # 库文件兼容两种布局: lib/ 与 lib/arm64-v8a/
    if(EXISTS ${RKNPU_ROOT}/lib/arm64-v8a/librknnrt.so)
        link_directories(${RKNPU_ROOT}/lib/arm64-v8a)
    else()
        link_directories(${RKNPU_ROOT}/lib)
    endif()
    message(STATUS "rknpu2: ${RKNPU_ROOT}")
else()
    message(WARNING "rknpu2 not found at ${RKNPU_ROOT}; NPU backend will be a stub")
endif()

# ===== subtitle core library =====
add_library(subtitle_core STATIC
    src/backends/asr_backend.cpp
    src/backends/cpu_backend.cc
    src/backends/npu_backend.cc
)
target_include_directories(subtitle_core
    PUBLIC ${CORE_ROOT}/include/subtitle
    PRIVATE ${CORE_ROOT}/src/backends
)
target_compile_options(subtitle_core PRIVATE -fPIC -O2)

if(SHERPA_ONNX_FOUND)
    # c-api.a 引用 sherpa_onnx:: 核心符号, 需要整组库; make install 会把
    # sherpa-onnx 组件库与 onnxruntime 组件库+依赖库(nsync/protobuf/onnx/
    # abseil 等)全部装到 install/lib, 此处 glob 全量链接, 并用
    # --start-group/--end-group 规避静态库之间的循环依赖
    file(GLOB SHERPA_LIBS "${SHERPA_ONNX_ROOT}/lib/lib*.a")
    target_link_libraries(subtitle_core PUBLIC
        -Wl,--start-group
        ${SHERPA_LIBS}
        -Wl,--end-group
    )
endif()
if(RKNN_FOUND)
    target_link_libraries(subtitle_core PUBLIC rknnrt)
endif()

# ===== standalone CLI =====
add_executable(smartsubtitle_cli cli/main.cc)
target_link_libraries(smartsubtitle_cli PRIVATE subtitle_core)

# OpenHarmony: 麦克风采集走 OHAudio(OH_AudioCapturer), 由 __OHOS__ 宏控制
if(CMAKE_SYSROOT AND EXISTS "${CMAKE_SYSROOT}/usr/lib/aarch64-linux-ohos/libohaudio.so")
    target_compile_definitions(smartsubtitle_cli PRIVATE SSUB_HAS_OHAUDIO=1)
    target_link_libraries(smartsubtitle_cli PRIVATE ohaudio)
endif()

# Optional live-microphone capture via ALSA (other Linux aarch64 boards, only
# for host-sysroot builds; the OHOS cross build uses OHAudio instead). The CLI
# also supports reading raw PCM from a file or stdin, which needs nothing.
if(EXISTS /usr/include/alsa/asoundlib.h AND NOT CMAKE_SYSROOT)
    target_compile_definitions(smartsubtitle_cli PRIVATE SSUB_HAS_ALSA=1)
    target_link_libraries(smartsubtitle_cli PRIVATE asound)
endif()
