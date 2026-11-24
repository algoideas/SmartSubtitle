# RKNN 运行时与验证程序

本目录存放 RK3588 NPU 推理所需的 RKNN 运行时、模型转换脚本与验证程序。

## 目录结构

```
rknn/
├── include/                  # RKNN API 头文件 (rknn_api.h 等)
├── lib/arm64-v8a/            # librknnrt.so (RKNN 运行时, aarch64)
├── application/rknn_model_demo/  # OpenHarmony 上的验证程序
└── convert_paraformer_encoder.py # ONNX -> RKNN 模型转换脚本
```

## 版本与来源

| 文件 | 版本/来源 |
|---|---|
| librknnrt.so | rknn-toolkit2 v2.4.1 (`rknpu2/runtime/Linux/libarm64/librknnrt.so`) |
| rknn_api.h 等 | RknnFacialRecognitionDemo_api12 样例工程 |
| convert_paraformer_encoder.py | 本项目编写, 依赖 rknn-toolkit2 v2.1.0+ |

`librknnrt.so` 的 ELF 为 aarch64 (OpenHarmony musl 构建), 依赖
`libc++_shared.so` 与 `libc.so`，均已随 HAP 打包或系统自带。

## 验证程序 (rknn_model_demo)

验证 OpenHarmony 设备上 `librknnrt.so` 的模型加载与推理功能。
默认使用 `hardware_npu2` 示例自带的 mobilenet_v1 模型 (RK3588) 与
图片资源，也支持指定任意 `.rknn` 模型。

程序流程：

1. 加载 `.rknn` 模型并执行 `rknn_init`
2. 查询输入/输出张量属性 (`RKNN_QUERY_IN_OUT_NUM` 等)
3. 按输入属性构造 **float32** 缓冲 (`rknn_inputs_set`)
4. 完成一次 NPU 推理 (`rknn_run`)，读取输出并打印统计值后正常释放

目录结构：

```
rknn_model_demo/
├── model/RK3588/mobilenet_v1.rknn   # hardware_npu2 示例的 RK3588 模型
├── model/cat_224x224.jpg 等         # 示例图片资源
├── src/main.cc                      # 验证程序源码
├── ohos.toolchain.cmake             # OHOS aarch64 工具链
└── build_ohos.sh                    # 一键交叉编译脚本
```

### 编译 (OpenHarmony aarch64)

```bash
cd third_party/rknn/application/rknn_model_demo
./build_ohos.sh            # 或 ./build_ohos.sh <OHOS_LLVM 路径>
```

### 运行 (RK3588 板卡实测流程)

以 Rockchip RK3588 UnionPi Whale Board V2 为例：

```bash
hdc -t <设备ID> shell "mkdir -p /data/local/tmp/rknn_demo/model"
hdc -t <设备ID> file send build/rknn_model_demo /data/local/tmp/rknn_demo/
hdc -t <设备ID> file send ../../lib/arm64-v8a/librknnrt.so /data/local/tmp/rknn_demo/
hdc -t <设备ID> file send model/RK3588/mobilenet_v1.rknn /data/local/tmp/rknn_demo/model/
hdc -t <设备ID> shell "export LD_LIBRARY_PATH=/data/local/tmp/rknn_demo:\$LD_LIBRARY_PATH; \
    /data/local/tmp/rknn_demo/rknn_model_demo /data/local/tmp/rknn_demo/model/mobilenet_v1.rknn"
```

实测输出 (RK3588 UnionPi Whale Board V2)：

```
model loaded: /data/local/tmp/rknn_demo/model/mobilenet_v1.rknn (4693865 bytes)
rknn_init OK
  input[0]: n=1 c=224 h=224 w=3, fmt=1, type=2, size=150528
  output[0]: n=1 c=1001 h=0 w=0, fmt=3, type=1, size=2002
rknn_run OK (NPU inference)
output tensors: 1001 floats, first values: 0.000058 0.000058 0.000675 ...
rknn_model_demo PASSED
```

说明：输入缓冲按模型属性以 **float32** 构造（尺寸 = 各维度乘积 × 4 字节）。
若按 int8 原始尺寸 (150528) 传入，`rknn_inputs_set` 会返回 -5
(`param input size < model input size`)。

## 模型转换 (ONNX → RKNN)

在 x86 主机上把 Paraformer 流式编码器转换为 `.rknn` (RK3588):

```bash
pip3 install rknn-toolkit2==v2.3.2   # 需要 python 3.8+; 见官方安装文档
python3 convert_paraformer_encoder.py \
    --onnx <encoder.onnx> --output paraformer-encoder.rknn \
    --frames 48 --feat-dim 560
```

注意:

- RKNN 不支持动态 shape, 脚本默认把动态 batch/长度固定为
  `[1, 48, 560]` (48 帧 = 480 ms @10ms 帧, 特征维度 560)；
- 输入特征已在上游归一化, 脚本不再设置 mean/std；
- 主机 onnxruntime 需 >= 1.16 才能加载 opset 18 的 Cast(19) 节点。
