# rknn_model_demo — RKNN 运行时验证程序

在 OpenHarmony 设备上验证 `librknnrt.so` 的模型加载与 NPU 推理功能。
默认使用 `model/RK3588/mobilenet_v1.rknn`（取自 hardware_npu2 示例），
也可通过命令行参数指定任意 `.rknn` 模型。

## 编译

```bash
./build_ohos.sh            # 或 ./build_ohos.sh <OHOS_LLVM 路径>
```

## 部署与运行（RK3588 板卡）

```bash
hdc -t <设备ID> shell "mkdir -p /data/local/tmp/rknn_demo/model"
hdc -t <设备ID> file send build/rknn_model_demo /data/local/tmp/rknn_demo/
hdc -t <设备ID> file send ../../lib/arm64-v8a/librknnrt.so /data/local/tmp/rknn_demo/
hdc -t <设备ID> file send model/RK3588/mobilenet_v1.rknn /data/local/tmp/rknn_demo/model/
hdc -t <设备ID> shell "export LD_LIBRARY_PATH=/data/local/tmp/rknn_demo:\$LD_LIBRARY_PATH; \
    /data/local/tmp/rknn_demo/rknn_model_demo /data/local/tmp/rknn_demo/model/mobilenet_v1.rknn"
```

预期输出以 `rknn_model_demo PASSED` 结尾。

## 说明

- 输入缓冲按模型张量属性以 **float32** 构造（各维度乘积 × 4 字节），
  若按 int8 原始尺寸传入，`rknn_inputs_set` 返回 -5。
- 完整文档见上级目录 `../../README.md`。
