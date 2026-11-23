#!/usr/bin/env bash
# Description: Linux one-shot build script for SmartSubtitle.
#              Produces both the OpenHarmony HAP application and the
#              standalone smartsubtitle_cli binary.
#
# Usage:
#   ./build-hap.sh         # ohpm install + native cli + hvigorw assembleHap + sign
#   ./build-hap.sh -b      # build only (skip ohpm install)
#   ./build-hap.sh -c      # clean build outputs
#   ./build-hap.sh -r      # clean + rebuild
#   ./build-hap.sh -h      # help
set -euo pipefail

# 防御: 输出管道被宿主/终端关闭时忽略 SIGPIPE, 避免长构建脚本被误杀
trap '' PIPE

export OHOS_BASE_SDK_HOME=~/tools/openharmony/Sdk
export OHOS_SDK_HOME=$OHOS_BASE_SDK_HOME
export OHOS_LLVM=${OHOS_LLVM:-$OHOS_BASE_SDK_HOME/18/native/llvm/bin}

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Defaults
DO_CLEAN=0
DO_BUILD=1
SKIP_INSTALL=0
SIGN_HAP=1

print_help() {
    cat <<EOF
SmartSubtitle build script
Usage: $0 [options]
  -b         Build only (skip ohpm install)
  -c         Clean build outputs
  -r         Rebuild (clean + build)
  -h         Show this help
EOF
}

# Locate HarmonyOS command-line-tools
TOOLCHAIN_BIN=""
for candidate in \
    "$HOME/tools/toolchain/command-line-tools/bin" \
    "$HOME/tools/toolchain/command-line-tools/tool\node\bin" \
    "$HOME/tools/command-line-tools/bin" \
    "$HOME/command-line-tools/bin"; do
    if [[ -x "$candidate/ohpm" && -x "$candidate/hvigorw" ]]; then
        TOOLCHAIN_BIN="$candidate"
        break
    fi
done
if [[ -n "$TOOLCHAIN_BIN" ]]; then
    export PATH="$TOOLCHAIN_BIN:$PATH"
fi

while getopts ":bcrh" opt; do
    case "$opt" in
        b) SKIP_INSTALL=1 ;;
        c) DO_CLEAN=1; DO_BUILD=0 ;;
        r) DO_CLEAN=1; DO_BUILD=1 ;;
        h) print_help; exit 0 ;;
        *) echo "Unknown option: -$OPTARG" >&2; print_help; exit 1 ;;
    esac
done

# Clean
if [[ "$DO_CLEAN" -eq 1 ]]; then
    echo "[build-hap] cleaning..."
    rm -rf "$PROJECT_DIR/build" "$PROJECT_DIR/entry/build" "$PROJECT_DIR/.hvigor"
    rm -rf "$PROJECT_DIR/third_party/sherpa-onnx/build-ohos"
    rm -rf "$PROJECT_DIR/third_party/onnxruntime/build-ohos"
    rm -rf "$PROJECT_DIR/core/build-cli"
    rm -f "$PROJECT_DIR/SmartSubtitle.hap"
    rm -f "$PROJECT_DIR/dist/smartsubtitle_cli"
fi

# Install deps
if [[ "$DO_BUILD" -eq 1 ]]; then
    if [[ "$SKIP_INSTALL" -eq 0 ]]; then
        echo "[build-hap] ohpm install..."
        ohpm install 2>&1 | tail -8 || true
    fi

    # ============ Part 1: standalone smartsubtitle_cli ============
    # sherpa-onnx c-api 库(HAP 与 CLI 链接都需要); 始终执行构建脚本
    # (onnxruntime zip 变更后 cmake 会自动重新解包, 增量构建很快)
    SHERPA_INSTALL="$PROJECT_DIR/third_party/sherpa-onnx/build-ohos/install"
    echo "[build-hap] building sherpa-onnx-c-api (incremental)..."
    "$PROJECT_DIR/third_party/build-ohos.sh"

    echo "[build-hap] building smartsubtitle_cli (aarch64, OHOS clang)..."
    CLI_BUILD_DIR="$PROJECT_DIR/core/build-cli"
    mkdir -p "$CLI_BUILD_DIR"
    cmake -S "$PROJECT_DIR/core" -B "$CLI_BUILD_DIR" \
        -DCMAKE_TOOLCHAIN_FILE="$PROJECT_DIR/core/toolchain.cmake" \
        -DCMAKE_BUILD_TYPE=Release \
        -DSHERPA_ONNX_ROOT="$SHERPA_INSTALL" \
        -DORT_DEPS_DIR="$PROJECT_DIR/third_party/onnxruntime/build-ohos/_deps" \
        > "$PROJECT_DIR/cli-config.log" 2>&1 || { echo "[build-hap] CLI cmake failed, see cli-config.log"; tail -20 "$PROJECT_DIR/cli-config.log"; exit 1; }
    cmake --build "$CLI_BUILD_DIR" -j"$(nproc)"

    mkdir -p "$PROJECT_DIR/dist"
    cp "$CLI_BUILD_DIR/smartsubtitle_cli" "$PROJECT_DIR/dist/smartsubtitle_cli"
    echo "[build-hap] built: $PROJECT_DIR/dist/smartsubtitle_cli"

    # 分发产物到 entry 的预编译目录(entry CMakeLists 按此布局链接)
    ENTRY_TP="$PROJECT_DIR/entry/src/main/cpp/third_party"
    mkdir -p "$ENTRY_TP/sherpa-onnx/include/sherpa-onnx/c-api" \
             "$ENTRY_TP/sherpa-onnx/libs/arm64-v8a" \
             "$ENTRY_TP/onnxruntime/libs/arm64-v8a"
    cp "$SHERPA_INSTALL/include/sherpa-onnx/c-api/c-api.h" \
       "$ENTRY_TP/sherpa-onnx/include/sherpa-onnx/c-api/"
    cp "$SHERPA_INSTALL/lib/"lib*.a \
       "$ENTRY_TP/sherpa-onnx/libs/arm64-v8a/"
    # ORT 全部组件库与依赖库(nsync/protobuf/onnx/abseil/flatbuffers/re2)
    cp "$PROJECT_DIR/onnxruntime/build-ohos/stage/lib/"lib*.a \
       "$ENTRY_TP/onnxruntime/libs/arm64-v8a/" 2>/dev/null || true
    find "$PROJECT_DIR/third_party/onnxruntime/build-ohos/_deps" \
        -name 'lib*.a' -not -path '*onnxruntime-src*' \
        -exec cp {} "$ENTRY_TP/onnxruntime/libs/arm64-v8a/" \; 2>/dev/null || true
    # onnxruntime 的依赖库(nsync/protobuf/onnx/abseil 等)
    mkdir -p "$ENTRY_TP/onnxruntime/deps"
    find "$PROJECT_DIR/third_party/onnxruntime/build-ohos/_deps" \
        -name 'lib*.a' -not -path '*onnxruntime-src*' \
        -exec cp {} "$ENTRY_TP/onnxruntime/deps/" \; 2>/dev/null || true

    # ============ Part 1.7: 模型预置到 HAP rawfile ============
    RAWFILE_MODELS="$PROJECT_DIR/entry/src/main/resources/rawfile/models"
    mkdir -p "$RAWFILE_MODELS"
    MODEL_SRC="$PROJECT_DIR/third_party/models/sherpa-onnx-streaming-paraformer-bilingual-zh-en"
    for f in encoder.int8.onnx decoder.int8.onnx tokens.txt; do
        if [[ -f "$MODEL_SRC/$f" ]]; then
            cp "$MODEL_SRC/$f" "$RAWFILE_MODELS/"
        fi
    done
    if [[ -f "$PROJECT_DIR/third_party/models/vad/silero_vad.onnx" ]]; then
        cp "$PROJECT_DIR/third_party/models/vad/silero_vad.onnx" "$RAWFILE_MODELS/"
    fi
    echo "[build-hap] rawfile models:"; ls -la "$RAWFILE_MODELS"

    # ============ Part 2: HAP application ============
    echo "[build-hap] hvigorw assembleHap..."
    HVIGORW=""
    if command -v hvigorw >/dev/null 2>&1; then
        HVIGORW="$(command -v hvigorw)"
    elif [[ -x "$PROJECT_DIR/hvigorw" ]]; then
        HVIGORW="$PROJECT_DIR/hvigorw"
    fi
    if [[ -z "$HVIGORW" ]]; then
        echo "ERROR: hvigorw not found. Please install HarmonyOS Commandline Tools." >&2
        exit 1
    fi
    "$HVIGORW" assembleHap --mode module -p product=default -p buildMode=debug --no-daemon

    # Find produced HAP
    HAP_FILE="$(find "$PROJECT_DIR/entry/build" -name '*.hap' -print -quit 2>/dev/null || true)"
    if [[ -z "$HAP_FILE" || ! -f "$HAP_FILE" ]]; then
        echo "ERROR: no .hap produced in entry/build" >&2
        exit 1
    fi
    cp "$HAP_FILE" "$PROJECT_DIR/SmartSubtitle.hap"
    echo "[build-hap] built: $PROJECT_DIR/SmartSubtitle.hap"

    # Sign HAP using hap-sign-tool (OpenHarmony standard signing)
    if [[ "$SIGN_HAP" -eq 1 ]]; then
        SDK_PATH=$OHOS_BASE_SDK_HOME
        SIGN_TOOL_PATH="$SDK_PATH/18/toolchains/lib"
        NOSIGN_HAP="$PROJECT_DIR/entry/build/default/outputs/default/entry-default-unsigned.hap"
        SIGNED_HAP="$PROJECT_DIR/entry/build/default/outputs/default/entry-default-signed.hap"
        PROFILE_PATH="$PROJECT_DIR/signature"

        if [[ ! -d "$PROFILE_PATH" ]]; then
            echo "[build-hap] signature directory not found, skip signing"
        elif [[ ! -f "$SIGN_TOOL_PATH/hap-sign-tool.jar" ]]; then
            echo "[build-hap] hap-sign-tool.jar not found at $SIGN_TOOL_PATH, skip signing"
        elif [[ ! -f "$NOSIGN_HAP" ]]; then
            echo "[build-hap] unsigned HAP not found at $NOSIGN_HAP, skip signing"
        else
            echo "[build-hap] signing HAP..."

            # Generate p7b profile if not exists
            if [[ ! -f "$PROFILE_PATH/SmartSubtitle.p7b" ]]; then
                echo "[build-hap] generating p7b profile..."
                java -jar "$SIGN_TOOL_PATH/hap-sign-tool.jar" sign-profile \
                    -keyAlias "openharmony application profile release" \
                    -signAlg "SHA256withECDSA" \
                    -mode "localSign" \
                    -profileCertFile "$SIGN_TOOL_PATH/OpenHarmonyProfileRelease.pem" \
                    -inFile "$PROFILE_PATH/UnsgnedReleasedProfileTemplate.json" \
                    -keystoreFile "$SIGN_TOOL_PATH/OpenHarmony.p12" \
                    -outFile "$PROFILE_PATH/SmartSubtitle.p7b" \
                    -keyPwd "123456" \
                    -keystorePwd "123456"
            fi

            # Sign the unsigned HAP
            java -jar "$SIGN_TOOL_PATH/hap-sign-tool.jar" sign-app \
                -keyAlias "openharmony application release" \
                -signAlg "SHA256withECDSA" \
                -mode "localSign" \
                -appCertFile "$PROFILE_PATH/OpenHarmonyApplication.pem" \
                -profileFile "$PROFILE_PATH/SmartSubtitle.p7b" \
                -inFile "$NOSIGN_HAP" \
                -keystoreFile "$SIGN_TOOL_PATH/OpenHarmony.p12" \
                -outFile "$SIGNED_HAP" \
                -keyPwd "123456" \
                -keystorePwd "123456"

            # Copy signed HAP to project root
            if [[ -f "$SIGNED_HAP" ]]; then
                cp -rf "$SIGNED_HAP" "$PROJECT_DIR/SmartSubtitle.hap"
                echo "[build-hap] signed: $PROJECT_DIR/SmartSubtitle.hap"
            else
                echo "[build-hap] signing produced no output file"
            fi
        fi
    fi
fi

echo "[build-hap] done."
