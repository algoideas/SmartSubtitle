#!/bin/bash
# build-onnxruntime-ohos.sh — 用 OpenHarmony clang 从源码交叉编译
# onnxruntime v1.17.1 (aarch64, 静态库)。
#
# 背景: sherpa-onnx 自带的 Linux aarch64 预编译 onnxruntime 按 GNU libstdc++
# 构建, 与 OpenHarmony 的 libc++ (std::__n1) ABI 不兼容, 因此必须从源码构建。
#
# 用法: ./build-onnxruntime-ohos.sh
# 产物: third_party/onnxruntime/build-ohos/onnxruntime-linux-aarch64-static_lib-1.17.1.zip
#       (布局与 sherpa-onnx 期望的官方 zip 一致, 供 sherpa-onnx 离线使用)
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# onnxruntime 检出 v1.17.1 发布对应的提交 (54153c73, 2023-08-10)
ORT_SRC=${SCRIPT_DIR}/onnxruntime
BUILD_DIR=${ORT_SRC}/build-ohos
OUT_ZIP=${SCRIPT_DIR}/onnxruntime-linux-aarch64-static_lib-1.17.1.zip

export OHOS_LLVM=${OHOS_LLVM:-$HOME/tools/openharmony/Sdk/18/native/llvm/bin}
if [ ! -x "${OHOS_LLVM}/clang++" ]; then
    echo "错误: 未找到 OpenHarmony clang 工具链 (${OHOS_LLVM})" >&2
    exit 1
fi

SYSROOT=${OHOS_SYSROOT:-${OHOS_LLVM}/../../sysroot}
CC=${OHOS_LLVM}/aarch64-unknown-linux-ohos-clang
CXX=${OHOS_LLVM}/aarch64-unknown-linux-ohos-clang++

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${ORT_SRC}/cmake" \
    -DCMAKE_TOOLCHAIN_FILE=${SCRIPT_DIR}/ohos-aarch64.toolchain.cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="${CC}" \
    -DCMAKE_CXX_COMPILER="${CXX}" \
    -DCMAKE_SYSROOT="${SYSROOT}" \
    -Donnxruntime_BUILD_UNIT_TESTS=OFF \
    -Donnxruntime_ENABLE_PYTHON=OFF \
    -Donnxruntime_ENABLE_TRAINING=OFF \
    -Donnxruntime_BUILD_SHARED_LIB=OFF \
    -Donnxruntime_USE_FULL_PROTOBUF=OFF \
    -Donnxruntime_ENABLE_MICROSOFT_EMITTER=OFF \
    -Donnxruntime_DISABLE_RTTI=OFF \
    -DCMAKE_CXX_STANDARD_LIBRARIES="-L${OHOS_LLVM}/../lib/aarch64-linux-ohos -Wl,--no-as-needed -lc++"

make -j"$(nproc)" all
# re2 目标不在 all 里(contrib tokenizer 需要它)
make -j"$(nproc)" re2

# 打包成 sherpa-onnx 期望的目录布局:
#   include/onnxruntime/...  include/*.h(展平的 session 头)  lib/lib*.a
#
# lib/ 放入 ORT 构建树中的全部静态库(组件库 + nsync/protobuf/onnx/
# abseil/flatbuffers 依赖库)。sherpa-onnx 的 cmake 用
# file(GLOB .../lib/lib*.a) 全量链接, 因此不需要手动合并成单个 .a
# (合并反而会因 ar 按成员名去重丢失符号)。
STAGE=${BUILD_DIR}/stage
rm -rf "${STAGE}"
mkdir -p "${STAGE}/include" "${STAGE}/lib"
# 公开头文件在源码树 include/onnxruntime 下
cp -r "${ORT_SRC}/include/onnxruntime" "${STAGE}/include/"
# sherpa-onnx 只以 include/ 为头文件搜索路径, 展平 core/session 头文件
cp "${ORT_SRC}"/include/onnxruntime/core/session/*.h "${STAGE}/include/"

find "${BUILD_DIR}" -name 'lib*.a' \
    -not -path '*onnxruntime-src*' \
    -not -path "${STAGE}/*" \
    -not -path "${OBJS:-${BUILD_DIR}/objs}/*" \
    -exec cp {} "${STAGE}/lib/" \;
ls "${STAGE}/lib" | wc -l

cd "${STAGE}" && zip -qr "${OUT_ZIP}" include lib
echo "== 完成: ${OUT_ZIP} =="
