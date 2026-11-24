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

# 说明: 以下文件因超过 GitHub 50M 建议上限未入库, 均来自上面的
# tar.bz2 压缩包内, 解压即得:
#   - decoder.int8.onnx (68 MB, 端侧推理必需)
#   - encoder.onnx / encoder.int8.onnx / decoder.onnx (>100M 原始模型)
# onnxruntime 交叉编译依赖 tensorboard.zip (73 MB) 会在配置阶段
# 由 onnxruntime 的 cmake 自动下载, 或手动放入 third_party/_deps-download/

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

sherpa-onnx v1.10.30 原始源码已按原样引入本仓库 `third_party/sherpa-onnx/`（未修改，便于后续对照修改源码）。它依赖的 onnxruntime 官方预编译包按 GNU libstdc++ 构建，与 OpenHarmony 的 libc++（`std::__n1` 版本空间）ABI 不兼容，因此必须用 OHOS clang 从源码重建 onnxruntime，再让 sherpa-onnx 使用。仓库已提供全流程脚本：

```bash
export OHOS_LLVM=~/tools/openharmony/Sdk/18/native/llvm/bin
export PATH=$OHOS_LLVM:$PATH

cd third_party

# (1) 交叉编译 onnxruntime 静态库（源码位于 third_party/onnxruntime，
#     检出 v1.17.1 发布对应提交 54153c73），产出
#     onnxruntime-linux-aarch64-static_lib-1.17.1.zip
./build-onnxruntime-ohos.sh

# (2) 交叉编译 sherpa-onnx C API（自动使用 (1) 的本地 zip），产出
#     sherpa-onnx/build-ohos/lib/libsherpa-onnx-c-api.a 与安装目录
#     sherpa-onnx/install/
./build-ohos.sh
```

两个脚本均为独立脚本，不依赖 OpenHarmony 应用工程。已对源码做过两处必要修改（均已入库）：

- `onnxruntime/cmake/deps.txt`：全部第三方依赖（abseil、protobuf、onnx、eigen、cutlass、tensorboard 等 45 项）改为指向 `third_party/_deps-download/` 本地预下载文件，其中 onnx（rel-1.14.1 分支）与 eigen 3.4 因归档下载被网络阻断，改为本地克隆后打包的 zip（`third_party/onnx-rel-1.14.1.zip`、`third_party/eigen-3.4.zip`），SHA1 已同步更新；
- `onnxruntime/onnxruntime/core/platform/posix/env.cc`：OpenHarmony musl 没有 `pthread_setaffinity_np`，该处以 `sched_setaffinity` 等价实现（带 `__OHOS__` 条件编译，不影响其他平台）；
- `sherpa-onnx/cmake/onnxruntime-linux-aarch64-static.cmake`：onnxruntime 下载 URL/HASH 改为指向本地 zip。

> 注意：`build-ohos.sh` 只构建并安装 `sherpa-onnx-c-api` 目标（本工程所需的 C API）。sherpa-onnx 自带的 `bin/sherpa-onnx` 命令行演示程序当前无法在 OHOS 上链接（其用例代码未适配），如需请自行处理。

### 3. 安装到工程 third_party 目录

`build-ohos.sh` 的安装产物位于 `third_party/sherpa-onnx/install/`，复制到应用 NDK 引用目录即可：

```
entry/src/main/cpp/third_party/sherpa-onnx/
├── include/sherpa-onnx/c-api/c-api.h
└── libs/arm64-v8a/libsherpa-onnx-c-api.a
```

`libsherpa-onnx-c-api.a` 为静态库，应用链接时还需同时链接 onnxruntime 的合并静态库（`onnxruntime/build-ohos/libonnxruntime.a`，或从 `onnxruntime-linux-aarch64-static_lib-1.17.1.zip` 解包），链接顺序：`libsherpa-onnx-c-api.a` 在前、`libonnxruntime.a` 在后，并追加 `-lc++`（见 `entry/src/main/cpp/CMakeLists.txt`）。

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

仓库内置转换脚本 `third_party/rknn/convert_paraformer_encoder.py`，支持 float 与可选 int8 量化：

```bash
pip3 install rknn-toolkit2==v2.1.0
python3 third_party/rknn/convert_paraformer_encoder.py \
    --onnx paraformer-encoder-fp32.onnx --output paraformer-encoder.rknn
```

RK3588 NPU 对动态 shape 与部分算子（如部分 Attention 变体）支持有限，转换策略为：

1. **编码器**：把流式 Paraformer 的 encoder 固定窗口长度（480 ms / 7680 采样）后导出为静态 shape ONNX，再转 `.rknn`；
2. **解码器 / CTC**：保留在 CPU 上以贪心 CTC 解码（见 `core/src/backends/npu_backend.cc`），避免在 NPU 上实现动态长度解码。

### 3. RK3588 板端部署（OpenHarmony）

RKNN 运行时已内置在仓库 `third_party/rknn/`（头文件取自 OpenHarmony 样例
`applications_ai_model_samples/RknnFacialRecognitionDemo_api12`，`librknnrt.so`
取自 rknn-toolkit2 v2.1.0），构建时自动链接并打包进 HAP：

```bash
third_party/rknn/include/           # rknn_api.h 等
third_party/rknn/lib/arm64-v8a/librknnrt.so
entry/src/main/cpp/third_party/rknn/   # 构建时 entry 侧使用的副本
entry/libs/arm64-v8a/librknnrt.so      # 打包进 HAP 的运行时库
```

模型文件部署到 RK3588 板端（OpenHarmony）：

```bash
hdc shell mkdir -p /data/local/tmp/smartsubtitle/models
hdc file send paraformer-encoder.rknn /data/local/tmp/smartsubtitle/models/
# Silero VAD 与 tokens.txt 与 CPU 后端共用
hdc file send ~/smartsubtitle-work/silero_vad.onnx /data/local/tmp/smartsubtitle/models/
```

> 注：`librknnrt.so` 由 NPU 后端在运行时自动加载；CPU 后端不依赖它，
> 在非 RK3588 设备上同样可以正常安装运行。

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
| `entry/src/main/ets/workers/SubtitleWorker.ets` | Worker 线程：承载耗时的引擎初始化与推理，结果经 `postMessage` 回传 |
| `entry/src/main/ets/pages/Index.ets` | UI：麦克风采集（`@ohos.multimedia.audio` AudioCapturer）、模型预置复制、字幕滚动渲染、CPU/NPU 切换 |
| `entry/src/main/module.json5` | 权限声明：`MICROPHONE`、`INTERNET` |

模型预置与加载：模型文件（`encoder.int8.onnx` / `decoder.int8.onnx` / `tokens.txt` / `silero_vad.onnx`）打包在 HAP 的
`resources/rawfile/models/` 下（`build-hap.sh` 自动从 `third_party/models/` 复制）；应用启动时经
`resourceManager.getRawFileContentSync` 复制到沙箱 `filesDir/models/`（已存在且大小一致则跳过），
再由 Worker 线程按路径加载，避免阻塞 UI 线程。

线程模型：UI 线程仅负责采集与渲染；Worker（`worker.ThreadWorker`）内完成 `startEngine`
（ONNX Runtime 初始化）与 `feedAudio` 推理喂入；识别结果经 `napi_threadsafe_function` 回调
Worker，再 `postMessage` 到 UI 线程。采集回调中的 ArrayBuffer 属于外部内存，需先 `slice(0)`
拷贝再发送（Worker 序列化不支持直接克隆）。

解码方式：CPU 后端使用 `greedy_search`（当前 sherpa-onnx v1.10.30 构建仅支持贪心解码，
`modified_beam_search` 会触发 "Unsupported decoding method" 并 abort）。

双语说明：流式 Paraformer 中英双语模型可同时识别中文与 English，无需语言切换；UI 中"字幕语言"显示固定的"中文 + English"。

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
