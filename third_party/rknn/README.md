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

验证 OpenHarmony 设备上 `librknnrt.so` 的模型加载与推理功能：

1. 加载 `.rknn` 模型并执行 `rknn_init`
2. 查询输入/输出张量属性
3. 构造输入并完成一次 NPU 推理 (`rknn_run`)
4. 读取输出并打印统计值后正常释放

### 编译 (OpenHarmony aarch64)

```bash
cd third_party/rknn/application/rknn_model_demo
./build_ohos.sh            # 或 ./build_ohos.sh <OHOS_LLVM 路径>
```

### 运行 (设备端)

```bash
hdc file send build/rknn_model_demo /data/local/tmp/
hdc file send ../../lib/arm64-v8a/librknnrt.so /data/local/tmp/
hdc shell "export LD_LIBRARY_PATH=/data/local/tmp:\$LD_LIBRARY_PATH; \
    /data/local/tmp/rknn_model_demo <model.rknn>"
```

输出 `rknn_model_demo PASSED` 即表示 RKNN 运行时加载、模型初始化与
NPU 推理均正常。

## 模型转换 (ONNX → RKNN)

在 x86 主机上把 Paraformer 流式编码器转换为 `.rknn` (RK3588):

```bash
pip3 install rknn-toolkit2==v2.1.0   # 需要 python 3.8+; 见官方安装文档
python3 convert_paraformer_encoder.py \
    --onnx <encoder.onnx> --output paraformer-encoder.rknn \
    --frames 48 --feat-dim 560
```

注意:

- RKNN 不支持动态 shape, 脚本默认把动态 batch/长度固定为
  `[1, 48, 560]` (48 帧 = 480 ms @10ms 帧, 特征维度 560)；
- 输入特征已在上游归一化, 脚本不再设置 mean/std；
- 主机 onnxruntime 需 >= 1.16 才能加载 opset 18 的 Cast(19) 节点。
