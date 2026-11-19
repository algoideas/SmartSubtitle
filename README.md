# SmartSubtitle — OpenHarmony 实时 AI 字幕

面向 OpenHarmony aarch64 平台的**端侧实时中英文字幕**应用。核心推理全部在本地完成，不依赖任何云端服务：

- **CPU 推理**：[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) + 流式 Paraformer（中英双语）+ Silero VAD，可运行在**任意 aarch64 芯片**的 OpenHarmony 系统上；
- **NPU 推理**：模型经 [RKNN](https://github.com/airockchip/rknn-toolkit2) 转换后运行在 **Rockchip RK3588** NPU 上（rknpu2 运行时），优先适配 RK3588。

## 目录

- [整体架构](#整体架构)
- [仓库与版本要求](#仓库与版本要求)
- [环境准备](#环境准备)
- [CPU 推理环境搭建（任意 aarch64 OpenHarmony）](#cpu-推理环境搭建)
- [NPU 推理环境搭建（RK3588）](#npu-推理环境搭建rk3588)
- [独立二进制（CLI）构建与运行](#独立二进制构建与运行)
- [OpenHarmony 应用构建](#openharmony-应用构建)
- [应用实现说明](#应用实现说明)
- [常见问题](#常见问题)

## 整体架构

```
┌────────────────────────────────────────────────────────────┐
│                    OpenHarmony 应用 (HAP)                  │
│  ArkTS UI (entry/src/main/ets)                             │
│    pages/Index.ets      麦克风采集 + 字幕渲染 + 后端切换      │
│              │ feedAudio(ArrayBuffer) / startEngine()      │
│              ▼                                             │
│  NAPI 桥接 (entry/src/main/cpp/napi_init.cpp)              │
└──────────────┬─────────────────────────────────────────────┘
               │ 依赖
┌──────────────▼─────────────────────────────────────────────┐
│              subtitle core (core/，纯 C/C++，无 OHOS 依赖)  │
│  AsrBackend 接口                                            │
│   ├─ CpuBackend   sherpa-onnx 流式 Paraformer + Silero VAD  │
│   └─ NpuBackend   RKNN Runtime (RK3588 NPU) + CPU CTC 解码  │
│  smartsubtitle_cli  独立可执行程序（脱离应用直接运行）        │
└────────────────────────────────────────────────────────────┘
```

关键设计点：

1. **核心与解耦**：`core/` 目录不包含任何 OpenHarmony API，用任意 CMake + clang 工具链即可编译出静态库 `subtitle_core` 与独立二进制 `smartsubtitle_cli`，可直接在任何 aarch64 Linux/OpenHarmony 系统上以命令行方式运行验证；OpenHarmony 应用只是核心库之上的一层 NAPI 封装。
2. **双后端**：`AsrBackend` 统一接口（`Init/Feed/Reset`），CPU 与 NPU 后端可运行时切换；当某后端的第三方依赖未编译进来时自动退化为 stub 并给出明确的错误提示，工程始终可编译。
3. **实时管线**：麦克风 16 kHz/16bit/单声道 PCM → 100 ms 块 → Silero VAD 静音门控 → 流式识别 → partial/final 结果经 NAPI 线程安全函数回传 UI 渲染。

## 仓库与版本要求

| 组件 | 仓库 | 版本 | 用途 |
|---|---|---|---|
| sherpa-onnx | https://github.com/k2-fsa/sherpa-onnx | v1.10.30 | CPU 推理运行时（ONNX Runtime 封装） |
| onnxruntime | https://github.com/microsoft/onnxruntime | 1.17.1（sherpa-onnx 内部依赖） | CPU 推理引擎 |
| Paraformer（流式中英） | https://github.com/k2-fsa/sherpa-onnx/releases/tag/asr-models | sherpa-onnx-streaming-paraformer-bilingual-zh-en (2024-11-04) | 实时中英文字幕识别模型 |
| Silero VAD | https://github.com/snakers4/silero-vad（随 sherpa-onnx asr-models 发布 silero_vad.onnx） | v5 | 语音活动检测 |
| rknn-toolkit2 | https://github.com/airockchip/rknn-toolkit2 | v2.1.0 | x86 主机上把 ONNX 模型转换为 .rknn |
| rknpu2 | https://github.com/airockchip/rknn-model-zoo 与 https://github.com/airockchip/rknn-toolkit2 中的 runtime | v2.1.0 | RK3588 板端 NPU 运行时（librknnrt.so） |
| OpenHarmony SDK | 本地 SDK | API 18（兼容 API 15+） | 应用编译工具链（hvigor/clang） |

> 版本为当前验证组合；sherpa-onnx 向后兼容良好，更新版本亦可，但建议同步更新其附带的 onnxruntime。

## 环境准备

以下所有命令在 **x86_64 Linux 构建主机**（Ubuntu 20.04/22.04）上执行。

### 1. OpenHarmony SDK 与 clang 工具链

本工程统一使用 OpenHarmony SDK 自带的 clang 工具链编译原生代码：

```
~/tools/openharmony/Sdk/18/native/llvm/bin/clang++ --version
```

若该路径不存在，从 https://developer.openharmony.cn/ 下载 Linux 版 SDK（API 18）并解压到 `~/tools/openharmony/Sdk`。

### 2. 构建工具

```bash
# CMake >= 3.15、Python >= 3.10（rknn-toolkit2 需要）、hvigor 随 DevEco/SDK 提供
sudo apt install cmake python3.10 python3-pip
pip3 install --user numpy onnx  # rknn-toolkit2 的基础依赖
```

### 3. 下载第三方运行时与模型

```bash
mkdir -p ~/smartsubtitle-work && cd ~/smartsubtitle-work

# (1) sherpa-onnx 源码（用于交叉编译或获取预编译库）
git clone https://github.com/k2-fsa/sherpa-onnx.git
cd sherpa-onnx && git checkout v1.10.30 && cd ..

# (2) 流式 Paraformer 中英模型 + Silero VAD
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
tar xf sherpa-onnx-streaming-paraformer-bilingual-zh-en.tar.bz2
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx

# (3) rknpu2（RK3588 NPU 运行时）
git clone https://github.com/airockchip/rknn-toolkit2.git
cd rknn-toolkit2 && git checkout v2.1.0 && cd ..
```

## CPU 推理环境搭建

目标：让识别模型跑在**任意 aarch64 芯片**的 OpenHarmony 系统上（纯 CPU，ONNX Runtime）。

### 1. 模型准备

使用 sherpa-onnx 发布的流式 Paraformer 中英双语模型（int8 量化版体积小、适合端侧）：

```bash
cd ~/smartsubtitle-work/sherpa-onnx-streaming-paraformer-bilingual-zh-en
# int8 量化版（推荐端侧使用，约 30 MB）
ls sherpa-onnx-streaming-paraformer-bilingual-zh-en.int8.onnx tokens.txt
```

部署到设备（设备端通过 hdc shell 执行）：

```bash
hdc shell mkdir -p /data/local/tmp/smartsubtitle/models
hdc file send sherpa-onnx-streaming-paraformer-bilingual-zh-en.int8.onnx \
    /data/local/tmp/smartsubtitle/models/paraformer-streaming-zh-en.int8.onnx
hdc file send tokens.txt /data/local/tmp/smartsubtitle/models/tokens.txt
hdc file send ~/smartsubtitle-work/silero_vad.onnx /data/local/tmp/smartsubtitle/models/silero_vad.onnx
```

> 模型目录约定：`paraformer-streaming-zh-en.int8.onnx`、`tokens.txt`、`silero_vad.onnx`。应用（`pages/Index.ets`）默认从 `/data/local/tmp/smartsubtitle/models` 读取，可在代码中修改 `MODEL_DIR` 常量。

### 2. 交叉编译 sherpa-onnx（aarch64 / OpenHarmony）

使用 OpenHarmony clang 工具链静态编译 sherpa-onnx C API：

```bash
export OHOS_LLVM=~/tools/openharmony/Sdk/18/native/llvm/bin
export PATH=$OHOS_LLVM:$PATH

cd ~/smartsubtitle-work/sherpa-onnx
mkdir -p build-ohos && cd build-ohos

cat > toolchain.cmake <<'EOF'
set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)
set(OHOS_LLVM $ENV{OHOS_LLVM})
set(CMAKE_C_COMPILER   ${OHOS_LLVM}/clang)
set(CMAKE_CXX_COMPILER ${OHOS_LLVM}/clang++)
set(CMAKE_AR           ${OHOS_LLVM}/llvm-ar)
set(CMAKE_RANLIB       ${OHOS_LLVM}/llvm-ranlib)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
EOF

cmake -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_SHARED_LIBS=ON \
      -DSHERPA_ONNX_ENABLE_PYTHON=OFF \
      -DSHERPA_ONNX_ENABLE_TESTS=OFF \
      -DSHERPA_ONNX_ENABLE_CHECK=OFF \
      -DSHERPA_ONNX_ENABLE_C_API=ON \
      -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
      ..
make -j$(nproc) sherpa-onnx-c-api
```

> 若使用 sherpa-onnx 官方 OHOS 预编译包（其 release 页提供 `sherpa-onnx-v*-linux-openharmony-*` 类似产物），可跳过此步，直接取其中的 `include/` 与 `lib/`。

### 3. 安装到工程 third_party 目录

```bash
cd ~/smartsubtitle-work/sherpa-onnx/build-ohos
PREFIX=~/smartsubtitle-work/sherpa-onnx-ohos-aarch64
mkdir -p $PREFIX/lib
cp -r ../sherpa-onnx/c-api $PREFIX/include 2>/dev/null || mkdir -p $PREFIX/include/sherpa-onnx/c-api
cp -r ../sherpa-onnx/c-api/c-api.h $PREFIX/include/sherpa-onnx/c-api/
cp lib/libsherpa-onnx-*.so $PREFIX/lib/
cp -r _deps/onnxruntime-build-install/onnxruntime-linux-aarch64-1.17.1 $PREFIX/lib/  # 视实际路径
```

随后把上面 `$PREFIX` 中的 `include/` 与 `lib/arm64-v8a/`（把 `.so` 汇总到该目录）复制到本仓库 `entry/src/main/cpp/third_party/sherpa-onnx/` 下：

```
entry/src/main/cpp/third_party/sherpa-onnx/
├── include/sherpa-onnx/c-api/c-api.h
└── libs/arm64-v8a/libsherpa-onnx-c-api.so, libonnxruntime.so ...
```

## NPU 推理环境搭建（RK3588）

目标：把 Paraformer 编码器经 RKNN 转换后跑在 RK3588 的 NPU 上，降低 CPU 占用与功耗。

### 1. x86 主机安装 rknn-toolkit2（模型转换用）

```bash
# 建议使用 conda 独立环境
conda create -n rknn python=3.10 -y && conda activate rknn
cd ~/smartsubtitle-work/rknn-toolkit2/rknn-toolkit2
pip install -r packages/requirements_cp310-2.1.0.txt
pip install packages/rknn_toolkit2-2.1.0+*-cp310-cp310-linux_x86_64.whl
```

### 2. 模型转换（ONNX → RKNN）

RK3588 NPU 对动态 shape 与部分算子（如部分 Attention 变体）支持有限，转换策略为：

1. **编码器**：把流式 Paraformer 的 encoder 固定窗口长度（480 ms / 7680 采样）后导出为静态 shape ONNX，再转 `.rknn`；
2. **解码器 / CTC**：保留在 CPU 上以贪心 CTC 解码（见 `core/src/backends/npu_backend.cc`），避免在 NPU 上实现动态长度解码。

```python
# convert_paraformer.py —— 在 x86 主机执行
import onnx
from rknn.api import RKNN

ONNX_MODEL = 'paraformer-encoder-fp32.onnx'   # 固定窗口(7680采样)的编码器导出
OUT        = 'paraformer-encoder.rknn'

rknn = RKNN(verbose=True)
rknn.config(
    mean_values=[[0]], std_values=[[1]],        # 前端已归一化，此处恒等
    quantized_dtype='w8a16',                    # RK3588 支持 int16 激活，保精度
    target_platform='rk3588',
    optimization_level=3,
)
ret = rknn.load_onnx(model=ONNX_MODEL)
assert ret == 0, 'load_onnx failed'
ret = rknn.build(do_quantization=False)         # 流式识别对精度敏感，先不做量化
assert ret == 0, 'build failed'
ret = rknn.export_rknn(OUT)
assert ret == 0, 'export_rknn failed'
rknn.release()
```

```bash
python3 convert_paraformer.py
# 产物：paraformer-encoder.rknn
```

### 3. RK3588 板端部署（OpenHarmony）

```bash
# NPU 运行时（rknpu2 runtime 中取出 aarch64 版本）
hdc shell mkdir -p /data/local/tmp/smartsubtitle/models /system/lib64
hdc file send rknn-toolkit2/rknn-toolkit2/runtime/Linux/librknn_api/arm64/librknnrt.so /system/lib64/
hdc file send paraformer-encoder.rknn /data/local/tmp/smartsubtitle/models/
# Silero VAD 与 tokens.txt 与 CPU 后端共用
hdc file send ~/smartsubtitle-work/silero_vad.onnx /data/local/tmp/smartsubtitle/models/
```

工程内运行时头文件与库的目录约定：

```
entry/src/main/cpp/third_party/rknpu/
├── include/rknn_api.h
└── libs/arm64-v8a/librknnrt.so
```

> NPU 后端初始化时通过 `rknn_set_core_mask` 绑定到 NPU 核 0（RK3588 三核 NPU，流式编码器单核即可，其余核留给相机/渲染等场景）。若需多路并发字幕，可在 `npu_backend.cc` 中扩展多实例并分核绑定。

## 独立二进制构建与运行

`core/` 目录不依赖 OpenHarmony，可编译出独立可执行文件 `smartsubtitle_cli`，用于在任何 aarch64 Linux/OpenHarmony 系统（包括 RK3588 开发板的终端）上直接验证识别管线。

### 1. 构建（使用 OpenHarmony clang）

```bash
cd SmartSubtitle/core
export OHOS_LLVM=~/tools/openharmony/Sdk/18/native/llvm/bin
export PATH=$OHOS_LLVM:$PATH
export SHERPA_ONNX_ROOT=~/smartsubtitle-work/sherpa-onnx-ohos-aarch64   # CPU 后端
# export RKNPU_ROOT=~/smartsubtitle-work/rknn-toolkit2/rknn-toolkit2/runtime/Linux  # NPU 后端

mkdir -p build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain.cmake \    # 与 CPU 章节相同的 clang 工具链文件
      -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
# 产物：libsubtitle_core.a 与 smartsubtitle_cli
```

### 2. 运行

```bash
# 从 WAV 文件识别（16 kHz/16bit/单声道）
./smartsubtitle_cli --backend cpu --model-dir /data/local/tmp/smartsubtitle/models --wav test.wav

# 从管道读取原始 PCM
arecord -D hw:0,0 -f S16_LE -r 16000 -c 1 | ./smartsubtitle_cli --backend cpu \
    --model-dir /data/local/tmp/smartsubtitle/models --stdin

# RK3588 上使用 NPU 后端 + 板载麦克风（需 ALSA）
./smartsubtitle_cli --backend npu --model-dir /data/local/tmp/smartsubtitle/models --mic
```

输出格式：每行一个结果，`PARTIAL <text>` 为实时部分结果，`FINAL <text>` 为断句后的最终结果。

## OpenHarmony 应用构建

1. 用 DevEco Studio 打开本仓库根目录，或在工程根目录命令行构建：

```bash
# 依赖：hvigor + SDK 18；第三方运行时已按上文放入 entry/src/main/cpp/third_party/
hvigorw assembleHap --mode module -p module=entry -p buildMode=release
```

2. 产物位于 `entry/build/default/outputs/default/entry-default-signed.hap`（需在 `build-profile.json5` 中配置签名 material）。
3. 安装并运行：

```bash
hdc install entry-default-signed.hap
hdc shell aa start -a EntryAbility -b com.ohos.smartsubtitle
```

## 应用实现说明

| 路径 | 职责 |
|---|---|
| `core/include/subtitle/asr_backend.h` | 后端统一接口与结果模型（partial/final） |
| `core/src/backends/cpu_backend.cc` | CPU 后端：sherpa-onnx 流式 Paraformer + Silero VAD 门控 + 规则断句 |
| `core/src/backends/npu_backend.cc` | NPU 后端：RKNN 编码器推理 + CPU 贪心 CTC 解码 |
| `core/cli/main.cc` | 独立 CLI：WAV/stdin/ALSA 麦克风三种输入 |
| `entry/src/main/cpp/napi_init.cpp` | NAPI 桥接 + 线程安全回调（`napi_threadsafe_function`） |
| `entry/src/main/ets/pages/Index.ets` | UI：麦克风采集（`@ohos.multimedia.audio` AudioCapturer）、字幕滚动渲染、CPU/NPU 切换 |
| `entry/src/main/module.json5` | 权限声明：`MICROPHONE`、`INTERNET` |

实时管线时序：

```
AudioCapturer(16k/16bit/单声道)
  → readData 回调 → feedAudio(ArrayBuffer) → NAPI → PcmRing 缓冲
  → 解码线程：VAD 门控 → 流式识别 → partial/final
  → napi_threadsafe_function → ArkTS onSubtitle → 字幕 UI（partial 灰色、final 白色）
```

断句策略：sherpa-onnx 规则断句——句尾静音 ≥ 0.8 s 立即断句；起始静音场景放宽至 2.4 s；单句超过 20 s 强制断句（`OnlineRecognizerEndpointConfig`）。

## 常见问题

1. **`startEngine` 返回 false**：先用 `getLastError()` 查看原因。最常见是模型目录不对或文件名不匹配（必须包含 `paraformer-streaming-zh-en.int8.onnx` / `tokens.txt`）。
2. **工程编译通过但未链接 sherpa-onnx/RKNN**：说明 `third_party/` 下对应运行时未放置，CPU/NPU 后端此时为 stub，只有 NAPI 与 UI 可用。按上文目录约定补齐即可。
3. **NPU 后端 `rknn_run failed`**：确认 `.rknn` 模型转换时 `target_platform='rk3588'`，且板端 `librknnrt.so` 版本 ≥ 转换用 toolkit 版本。
4. **延迟偏高**：CPU 后端可将 `num_threads` 调至 4–6；确认使用的是 int8 量化模型；RK3588 上优先切换 NPU 后端。
5. **识别结果全为空**：检查采样率是否为 16 kHz、声道是否为单声道；VAD 门控在持续静音时会主动丢弃音频，属于预期行为。

## 许可

Apache-2.0。第三方组件许可见其上游仓库（sherpa-onnx: Apache-2.0，Silero VAD: MIT，rknn-toolkit2: Apache-2.0 / BSD-3）。
