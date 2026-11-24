#!/bin/bash
# 在 OpenHarmony aarch64 目标上编译 rknn_model_demo
# 用法: ./build_ohos.sh [OHOS_LLVM路径]
# 默认工具链: ~/tools/openharmony/Sdk/18/native/llvm/bin
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
OHOS_LLVM="${1:-$HOME/tools/openharmony/Sdk/18/native/llvm/bin}"
TOOLCHAIN_FILE="$SCRIPT_DIR/ohos.toolchain.cmake"
BUILD_DIR="$SCRIPT_DIR/build"

export OHOS_LLVM

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
    -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
    -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j"$(nproc)"

echo
echo "build done:"
ls -la "$BUILD_DIR/rknn_model_demo"
echo
echo "部署到设备:"
echo "  hdc file send $BUILD_DIR/rknn_model_demo /data/local/tmp/"
echo "  hdc file send $RKNN_RT_LIB /data/local/tmp/lib/  # 或使用 HAP 内打包的库"
echo "  hdc shell \"export LD_LIBRARY_PATH=/data/local/tmp/lib:\$LD_LIBRARY_PATH; /data/local/tmp/rknn_model_demo <model.rknn>\""
