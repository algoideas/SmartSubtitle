#!/bin/bash
# build-ohos.sh — 在 OpenHarmony (aarch64) 上独立交叉编译构建 sherpa-onnx。
#
# 不依赖 OpenHarmony 应用工程：任意装了 CMake 的 Linux 主机可直接执行。
# 用法：
#   OHOS_SDK_NATIVE=~/tools/openharmony/Sdk/18/native ./build-ohos.sh
# 产物：
#   third_party/sherpa-onnx/build-ohos/install/lib/libsherpa-onnx-*.so 等
#   third_party/sherpa-onnx/build-ohos/install/include
set -euo pipefail

# 防御: 输出管道被宿主/终端关闭时忽略 SIGPIPE, 避免长构建脚本被误杀
trap '' PIPE

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SHERPA_ROOT=${SCRIPT_DIR}/sherpa-onnx
BUILD_DIR=${SHERPA_ROOT}/build-ohos
INSTALL_DIR=${BUILD_DIR}/install

# 工具链：OpenHarmony SDK 自带 clang
export OHOS_LLVM=${OHOS_LLVM:-${OHOS_SDK_NATIVE:-$HOME/tools/openharmony/Sdk/18/native}/llvm/bin}
if [ ! -x "${OHOS_LLVM}/clang++" ]; then
    echo "错误: 未找到 OpenHarmony clang 工具链 (${OHOS_LLVM})" >&2
    echo "请安装 SDK 18 或指定 OHOS_LLVM=<llvm bin 目录>" >&2
    exit 1
fi
echo "== 使用工具链: ${OHOS_LLVM} =="
"${OHOS_LLVM}/clang++" --version | head -1

# aarch64 sysroot 随 SDK 提供
SYSROOT=${OHOS_LLVM}/../sysroot
TOOLCHAIN=${SCRIPT_DIR}/ohos-aarch64.toolchain.cmake

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# C++ runtime (libc++) lives in <llvm>/lib/aarch64-linux-ohos; the static
# onnxruntime archive needs it at link time. CXX_STANDARD_LIBRARIES is
# appended at the END of the link line, after all objects/archives.
CXX_LIB_FLAGS="-Wl,--no-as-needed -L${OHOS_LLVM}/../lib/aarch64-linux-ohos -lc++"

cmake -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX="${INSTALL_DIR}" \
      -DBUILD_SHARED_LIBS=OFF \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
      -DCMAKE_CXX_STANDARD_LIBRARIES="${CXX_LIB_FLAGS}" \
      -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
      -DSHERPA_ONNX_ENABLE_TESTS=OFF \
      -DSHERPA_ONNX_ENABLE_CHECK=OFF \
      -DSHERPA_ONNX_ENABLE_C_API=ON \
      -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
      -DSHERPA_ONNX_ENABLE_TTS=OFF \
      -DSHERPA_ONNX_ENABLE_BINARY=OFF \
      "$@" ../

make -j"$(nproc)" sherpa-onnx-c-api
make install

# make install 只安装 c-api 目标, 把其余核心静态库一并复制,
# 供下游(subtitle_core/HAP)链接 sherpa_onnx:: 符号
cp "${BUILD_DIR}"/lib*.a "${INSTALL_DIR}/lib/" 2>/dev/null || true

echo
echo "== 构建完成 =="
echo "  库与头文件: ${INSTALL_DIR}"
ls -1 "${INSTALL_DIR}"/lib/*.a 2>/dev/null || true
